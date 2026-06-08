#!/usr/bin/env python3
"""
Mutimer Fashion Store Scraper - Main Orchestrator

Pipeline:
1. Scrape all products from Shopify collections API
2. Fetch existing products from Supabase
3. Classify products: new / changed-image / changed-data / unchanged
4. Generate SIGLIP embeddings only for new + image-changed products
5. Upsert only changed products to Supabase (50/batch, 3 retries)
6. Handle stale products (delete after 2 consecutive missed runs)
7. Print run summary

Usage:
    python main.py [--skip-scrape] [--skip-embeddings] [--skip-upload]

Options:
    --skip-scrape       Skip scraping, load from cache file
    --skip-embeddings   Skip embedding generation
    --skip-upload       Skip uploading to Supabase
    --verify-only       Just verify the current products in Supabase
    --resume            Resume from last checkpoint
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

from config import SOURCE
from shopify_scraper import scrape_all_products
from embeddings import process_product_embeddings, refine_products_after_embedding
from supabase_db import (
    fetch_all_products,
    classify_products,
    batch_upsert,
    handle_stale_products,
    print_run_summary,
    verify_import,
    get_product_count,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

SCRAPE_CACHE = "products_cache.json"
EMBEDDED_SUBSET_CACHE = "products_embedded_cache.json"
CHECKPOINT_FILE = "checkpoint.state"

CHECKPOINT_SCRAPED = "scraped"
CHECKPOINT_CLASSIFIED = "classified"
CHECKPOINT_EMBEDDED = "embedded"
CHECKPOINT_UPLOADED = "uploaded"


def save_cache(products: list[dict], filename: str = SCRAPE_CACHE):
    """Save products to a local cache file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(products)} products to {filename}")


def load_cache(filename: str = SCRAPE_CACHE) -> list[dict]:
    """Load products from local cache file."""
    if not os.path.exists(filename):
        logger.error(f"Cache file {filename} not found.")
        return []
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(stage: str):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(stage)
    logger.debug(f"Checkpoint saved: {stage}")


def load_checkpoint() -> str | None:
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE, "r") as f:
        return f.read().strip()


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.debug("Checkpoint cleared")


def run_pipeline(
    skip_scrape=False,
    skip_embeddings=False,
    skip_upload=False,
    resume=False,
):
    """Run the full smart scraping pipeline."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("MUTIMER FASHION STORE SCRAPER")
    logger.info(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    if resume:
        logger.info("Mode: RESUME from last checkpoint")
    logger.info("=" * 60)

    last_checkpoint = load_checkpoint() if resume else None
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # ── Step 1: Scrape products ──────────────────────────────────────────
    if skip_scrape or last_checkpoint in (CHECKPOINT_CLASSIFIED, CHECKPOINT_EMBEDDED, CHECKPOINT_UPLOADED):
        logger.info("Step 1: Loading products from cache...")
        products = load_cache()
        if not products:
            products = load_cache(EMBEDDED_SUBSET_CACHE)
        if not products:
            logger.error("No cached products found. Run without --skip-scrape first.")
            return
    else:
        logger.info("Step 1: Scraping products from Shopify API...")
        products = scrape_all_products()
        save_cache(products, SCRAPE_CACHE)
        save_checkpoint(CHECKPOINT_SCRAPED)

    logger.info(f"Total scraped products: {len(products)}")

    # ── Step 2: Fetch existing DB products & classify ────────────────────
    db_products = []
    classification = None

    if not skip_upload:
        logger.info("Step 2: Fetching existing products from Supabase...")
        db_products = fetch_all_products()

        logger.info("Classifying products against database...")
        classification = classify_products(products, db_products)
        save_checkpoint(CHECKPOINT_CLASSIFIED)

        # Determine which products need embeddings
        products_to_embed = classification["new"] + classification["changed_image"]
        logger.info(
            f"Products needing embeddings: {len(products_to_embed)} "
            f"({len(classification['new'])} new + {len(classification['changed_image'])} image-changed)"
        )

        # Products that need DB updates but no re-embedding
        products_to_update_db = classification["new"] + classification["changed_image"] + classification["changed_data"]

        logger.info(
            f"Products needing DB updates: {len(products_to_update_db)} "
            f"({len(classification['new'])} new + {len(classification['changed_image'])} image + "
            f"{len(classification['changed_data'])} data)"
        )
    else:
        # When skipping upload, we still embed everything (legacy behavior)
        products_to_embed = products
        products_to_update_db = products

    # ── Step 3: Generate embeddings (only for new + image-changed) ───────
    if skip_embeddings or last_checkpoint == CHECKPOINT_UPLOADED:
        logger.info("Step 3: Skipping embedding generation")
    elif not products_to_embed:
        logger.info("Step 3: No products need embedding — all up to date")
    else:
        logger.info(f"Step 3: Generating SIGLIP embeddings for {len(products_to_embed)} products...")
        process_product_embeddings(products_to_embed)

        # Save the embedded subset cache for resume
        if skip_upload:
            # If not uploading, save what we embedded for later
            all_embedded = []
            # Merge embeddings back into the full set
            url_to_embedded = {}
            for p in products_to_embed:
                if p.get("image_embedding"):
                    url_to_embedded[p.get("product_url")] = {
                        "image_embedding": p.get("image_embedding"),
                        "info_embedding": p.get("info_embedding"),
                    }
            for p in products:
                pu = p.get("product_url")
                if pu in url_to_embedded:
                    p["image_embedding"] = url_to_embedded[pu]["image_embedding"]
                    p["info_embedding"] = url_to_embedded[pu]["info_embedding"]
                elif pu not in {x.get("product_url") for x in products_to_embed}:
                    # Preserve any existing embeddings from cache
                    pass
            save_cache(products, EMBEDDED_SUBSET_CACHE)

        save_checkpoint(CHECKPOINT_EMBEDDED)

    # ── Step 4: Upsert to Supabase ───────────────────────────────────────
    if skip_upload:
        logger.info("Step 4: Skipping upload (--skip-upload)")
        upsert_result = {"upserted": 0, "failed": 0}
        stale_result = {"deleted": 0, "newly_missed": 0}
    else:
        logger.info(f"Step 4: Upserting {len(products_to_update_db)} products to Supabase...")
        upsert_result = batch_upsert(products_to_update_db)

        # ── Step 5: Handle stale products ──────────────────────────────
        logger.info("Step 5: Checking for stale products...")
        stale_result = handle_stale_products(
            classification["seen_urls"],
            db_products,
        )

        save_checkpoint(CHECKPOINT_UPLOADED)

        # ── Step 6: Print summary ──────────────────────────────────────
        print_run_summary(classification, upsert_result, stale_result)

    # ── Elapsed time ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed:.2f}s")

    # ── Verification ──────────────────────────────────────────────────────
    if not skip_upload:
        logger.info("\n--- Verification ---")
        verify_results = verify_import(limit=3)
        if verify_results:
            logger.info("Import verified. Sample records from Supabase:")
            for rec in verify_results:
                title = rec.get("title", "N/A")
                pid = rec.get("id", "N/A")
                has_img = rec.get("image_embedding") is not None
                has_info = rec.get("info_embedding") is not None
                logger.info(
                    f"  - {title} (id: {pid}) | "
                    f"img_emb: {'✓' if has_img else '✗'} | "
                    f"info_emb: {'✓' if has_info else '✗'}"
                )
        else:
            logger.warning("Verification returned no records. Import may have failed.")

        count = get_product_count()
        if count is not None:
            logger.info(f"Total products in Supabase for '{SOURCE}': {count}")

    clear_checkpoint()
    logger.info("=" * 60)


def main():
    """Entry point with CLI argument parsing."""
    args = set(sys.argv[1:])

    if "--verify-only" in args:
        logger.info("Verifying products in Supabase...")
        results = verify_import(limit=10)
        if results:
            logger.info(f"Found {len(results)} products. Sample:")
            for r in results:
                logger.info(
                    f"  - {r.get('title')} | price: {r.get('price')} | "
                    f"img_emb: {r.get('image_embedding') is not None}"
                )
        count = get_product_count()
        if count is not None:
            logger.info(f"Total count: {count}")
        return

    skip_scrape = "--skip-scrape" in args
    skip_embeddings = "--skip-embeddings" in args
    skip_upload = "--skip-upload" in args
    resume = "--resume" in args

    run_pipeline(
        skip_scrape=skip_scrape,
        skip_embeddings=skip_embeddings,
        skip_upload=skip_upload,
        resume=resume,
    )


if __name__ == "__main__":
    main()
