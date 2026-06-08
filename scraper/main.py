#!/usr/bin/env python3
"""
Mutimer Fashion Store Scraper - Main Orchestrator

Pipeline:
1. Scrape all products from Shopify collections API
2. Generate SIGLIP image and text embeddings
3. Upsert everything to Supabase

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
from supabase_db import upsert_products, verify_import, get_product_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

SCRAPE_CACHE = "products_cache.json"
EMBEDDING_CACHE = "products_embedded_cache.json"
CHECKPOINT_FILE = "checkpoint.state"

CHECKPOINT_SCRAPED = "scraped"
CHECKPOINT_EMBEDDED = "embedded"
CHECKPOINT_UPLOADED = "uploaded"


def save_cache(products: list[dict], filename: str = SCRAPE_CACHE):
    """Save scraped products to a local cache file."""
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
    """Save the current pipeline stage as a checkpoint."""
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(stage)
    logger.info(f"Checkpoint saved: {stage}")


def load_checkpoint() -> str | None:
    """Load the last checkpoint stage, or None."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE, "r") as f:
        return f.read().strip()


def clear_checkpoint():
    """Remove the checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("Checkpoint cleared")


def run_pipeline(
    skip_scrape=False,
    skip_embeddings=False,
    skip_upload=False,
    resume=False,
):
    """Run the full scraping pipeline with checkpointing."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("MUTIMER FASHION STORE SCRAPER")
    logger.info(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    if resume:
        logger.info("Mode: RESUME from last checkpoint")
    logger.info("=" * 60)

    # Determine where to resume from
    last_checkpoint = load_checkpoint() if resume else None
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    products = []

    # Step 1: Scrape products
    if skip_scrape or last_checkpoint in (CHECKPOINT_EMBEDDED, CHECKPOINT_UPLOADED):
        logger.info("Step 1: Loading products from cache (--skip-scrape or resuming)...")
        products = load_cache()
        if not products:
            # Try embedded cache
            products = load_cache(EMBEDDING_CACHE)
    else:
        logger.info("Step 1: Scraping products from Shopify API...")
        products = scrape_all_products()
        save_cache(products, SCRAPE_CACHE)
        save_checkpoint(CHECKPOINT_SCRAPED)

    logger.info(f"Total products loaded: {len(products)}")
    if not products:
        logger.error("No products found. Aborting.")
        return

    # Step 2: Generate embeddings
    if skip_embeddings or last_checkpoint == CHECKPOINT_UPLOADED:
        logger.info("Step 2: Loading pre-computed embeddings from cache (--skip-embeddings or resuming)...")
        # If resuming after upload, load the embedded cache
        if last_checkpoint == CHECKPOINT_UPLOADED:
            cached = load_cache(EMBEDDING_CACHE)
            if cached:
                products = cached
            else:
                logger.warning("No embedded cache found, re-generating embeddings...")
                products = process_product_embeddings(products)
                products = refine_products_after_embedding(products)
                save_cache(products, EMBEDDING_CACHE)
                save_checkpoint(CHECKPOINT_EMBEDDED)
    else:
        logger.info("Step 2: Generating SIGLIP embeddings...")
        products = process_product_embeddings(products)
        products = refine_products_after_embedding(products)
        logger.info(f"Products with valid embeddings: {len(products)}")
        save_cache(products, EMBEDDING_CACHE)
        save_checkpoint(CHECKPOINT_EMBEDDED)

    # Step 3: Upload to Supabase
    if skip_upload:
        logger.info("Step 3: Skipping upload (--skip-upload)")
    else:
        logger.info("Step 3: Uploading to Supabase...")
        result = upsert_products(products)
        logger.info(f"Upload result: {result}")
        save_checkpoint(CHECKPOINT_UPLOADED)

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Elapsed time: {elapsed:.2f}s")

    # Verify
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
