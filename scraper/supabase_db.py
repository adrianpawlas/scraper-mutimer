"""
Supabase database operations for the scraper.

Handles batch upserting of scraped product data into the products table
with smart change detection, stale product cleanup, and retry logic.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from supabase import create_client, Client

from config import (
    SOURCE,
    SUPABASE_URL,
    SUPABASE_KEY,
    TABLE_NAME,
    UPLOAD_BATCH_SIZE,
    STALE_RUNS_THRESHOLD,
    MAX_UPSERT_RETRIES,
)

logger = logging.getLogger(__name__)

# Fields that we compare to detect meaningful changes
TRACKED_FIELDS = [
    "title",
    "description",
    "price",
    "sale",
    "image_url",
    "additional_images",
    "category",
    "gender",
    "size",
    "tags",
    "brand",
    "product_url",
    "metadata",       # Contains variant availability (stock status)
]

STALE_LOG_FILE = "failed_stale_log.txt"


def get_client() -> Client:
    """Create and return a Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _normalize_for_comparison(val):
    """Normalize a value for equality comparison (handle None, list ordering, etc.)."""
    if val is None:
        return None
    if isinstance(val, list):
        return sorted(str(v) for v in val)
    return str(val).strip()


def _has_product_changed(scraped: dict, db_record: dict) -> bool:
    """
    Compare a scraped product against its existing DB record.

    Returns True if any tracked field differs (meaningful change detected).
    """
    for field in TRACKED_FIELDS:
        scraped_val = _normalize_for_comparison(scraped.get(field))
        db_val = _normalize_for_comparison(db_record.get(field))
        if scraped_val != db_val:
            logger.debug(f"Field '{field}' changed: '{db_val}' -> '{scraped_val}'")
            return True
    return False


def _prepare_product_for_db(product: dict) -> dict:
    """
    Prepare a product dict for database insertion.
    """
    record = dict(product)

    # Add/update timestamp
    record["created_at"] = datetime.now(timezone.utc).isoformat()

    # Ensure tags is a proper list (not serialized)
    if record.get("tags") and isinstance(record["tags"], str):
        try:
            record["tags"] = json.loads(record["tags"])
        except (json.JSONDecodeError, TypeError):
            record["tags"] = [record["tags"]]

    return record


def _read_stale_log() -> set:
    """Read the set of product IDs that previously failed stale deletion."""
    if not os.path.exists(STALE_LOG_FILE):
        return set()
    with open(STALE_LOG_FILE, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _append_stale_log(product_id: str):
    """Append a product ID to the stale deletion failure log."""
    with open(STALE_LOG_FILE, "a") as f:
        f.write(f"{product_id}\n")


# ─── Fetch existing products ────────────────────────────────────────────────


def fetch_all_products() -> list[dict]:
    """
    Fetch ALL existing products for this source from Supabase.

    Returns:
        List of DB product dicts (includes id, product_url, image_url, etc.)
    """
    client = get_client()
    all_products = []
    page = 0
    page_size = 1000

    while True:
        start = page * page_size
        try:
            response = (
                client.table(TABLE_NAME)
                .select("*")
                .eq("source", SOURCE)
                .range(start, start + page_size - 1)
                .execute()
            )
        except Exception as e:
            logger.error(f"Failed to fetch products page {page}: {e}")
            break

        batch = response.data
        if not batch:
            break

        all_products.extend(batch)
        logger.debug(f"Fetched page {page}: {len(batch)} products (total: {len(all_products)})")

        if len(batch) < page_size:
            break

        page += 1

    logger.info(f"Fetched {len(all_products)} existing products from Supabase")
    return all_products


# ─── Product classification ─────────────────────────────────────────────────


def classify_products(
    scraped_products: list[dict],
    db_products: list[dict],
) -> dict:
    """
    Classify scraped products against existing DB records.

    Args:
        scraped_products: Products from the current scrape run
        db_products: Products already in the database

    Returns:
        Dict with keys:
          - 'new': products not in DB (need full processing + embedding)
          - 'changed_image': products with changed image_url (need re-embedding)
          - 'changed_data': products with other field changes (skip embedding)
          - 'unchanged': products identical to DB (skip entirely)
          - 'seen_urls': set of product_urls seen in this scrape
    """
    # Build lookup: product_url -> DB record
    db_by_url = {}
    for p in db_products:
        url = p.get("product_url")
        if url:
            db_by_url[url] = p

    new_products = []
    changed_image = []
    changed_data = []
    unchanged = []
    seen_urls = set()

    for scraped in scraped_products:
        url = scraped.get("product_url")
        if not url:
            new_products.append(scraped)
            continue

        seen_urls.add(url)
        db_record = db_by_url.get(url)

        if db_record is None:
            # NEW product — not in DB
            new_products.append(scraped)
            continue

        # Check if image changed (triggers re-embedding)
        scraped_image = _normalize_for_comparison(scraped.get("image_url"))
        db_image = _normalize_for_comparison(db_record.get("image_url"))
        image_changed = scraped_image != db_image

        # Check if any other tracked field changed
        data_changed = _has_product_changed(scraped, db_record)

        if image_changed:
            changed_image.append(scraped)
        elif data_changed:
            # Carry over existing embeddings to avoid wiping them on upsert
            scraped["image_embedding"] = db_record.get("image_embedding")
            scraped["info_embedding"] = db_record.get("info_embedding")
            changed_data.append(scraped)
        else:
            unchanged.append(scraped)

    result = {
        "new": new_products,
        "changed_image": changed_image,
        "changed_data": changed_data,
        "unchanged": unchanged,
        "seen_urls": seen_urls,
    }

    logger.info(
        f"Classification: {len(new_products)} new, "
        f"{len(changed_image)} image-changed, "
        f"{len(changed_data)} data-changed, "
        f"{len(unchanged)} unchanged"
    )

    return result


# ─── Stale product handling ────────────────────────────────────────────────


def handle_stale_products(
    seen_urls: set,
    db_products: list[dict],
) -> dict:
    """
    Track and delete stale products.

    Products in the DB that were NOT seen in the current scrape are
    considered potentially stale. Uses a 'missed_runs' counter stored
    in the 'other' JSON field. When missed_runs >= STALE_RUNS_THRESHOLD,
    the product is deleted.

    Args:
        seen_urls: Set of product_urls seen in the current scrape
        db_products: All existing DB products for this source

    Returns:
        Dict with 'deleted' count and 'missed' count
    """
    client = get_client()
    deleted = 0
    newly_missed = 0

    for db_product in db_products:
        url = db_product.get("product_url")
        if not url:
            continue

        # Product was seen in this scrape — reset missed_runs to 0
        if url in seen_urls:
            other_raw = db_product.get("other")
            if other_raw:
                try:
                    other_data = json.loads(other_raw) if isinstance(other_raw, str) else other_raw
                except (json.JSONDecodeError, TypeError):
                    other_data = {}
                if other_data.get("missed_runs", 0) > 0:
                    other_data["missed_runs"] = 0
                    try:
                        client.table(TABLE_NAME).update(
                            {"other": json.dumps(other_data, ensure_ascii=False)}
                        ).eq("id", db_product["id"]).execute()
                    except Exception as e:
                        logger.warning(f"Failed to reset missed_runs for {url}: {e}")
            continue

        # Product NOT seen — increment missed_runs
        other_raw = db_product.get("other")
        try:
            other_data = json.loads(other_raw) if isinstance(other_raw, str) else (other_raw or {})
        except (json.JSONDecodeError, TypeError):
            other_data = {}

        current_missed = other_data.get("missed_runs", 0)
        current_missed += 1
        other_data["missed_runs"] = current_missed

        if current_missed >= STALE_RUNS_THRESHOLD:
            # DELETE — stale for too long
            try:
                client.table(TABLE_NAME).delete().eq("id", db_product["id"]).execute()
                deleted += 1
                logger.info(f"Deleted stale product: {db_product.get('title', 'unknown')} ({url})")
            except Exception as e:
                logger.error(f"Failed to delete stale product {url}: {e}")
                _append_stale_log(db_product.get("id", "unknown"))
        else:
            # Just mark as missed (update the other field)
            try:
                client.table(TABLE_NAME).update(
                    {"other": json.dumps(other_data, ensure_ascii=False)}
                ).eq("id", db_product["id"]).execute()
                newly_missed += 1
                logger.info(
                    f"Missed run {current_missed}/{STALE_RUNS_THRESHOLD}: "
                    f"{db_product.get('title', 'unknown')}"
                )
            except Exception as e:
                logger.warning(f"Failed to update missed_runs for {url}: {e}")

    if deleted:
        logger.info(f"Deleted {deleted} stale products")
    if newly_missed:
        logger.info(f"Marked {newly_missed} products as missed (run 1/{STALE_RUNS_THRESHOLD})")

    return {"deleted": deleted, "newly_missed": newly_missed}


# ─── Batch upsert with retry ────────────────────────────────────────────────


def _log_failed_products(products: list[dict], error: str):
    """Log failed product details to a file for inspection."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with open("failed_products.log", "a") as f:
        f.write(f"\n--- Batch failure at {timestamp} ---\n")
        f.write(f"Error: {error}\n")
        for p in products:
            f.write(f"  - {p.get('id')} | {p.get('title')} | {p.get('product_url')}\n")


def batch_upsert(products: list[dict], batch_size: int = UPLOAD_BATCH_SIZE) -> dict:
    """
    Upsert products to Supabase in batches with retry logic.

    Each batch is retried up to MAX_UPSERT_RETRIES times. If all retries
    fail, individual products are tried one-by-one to isolate failures.

    Args:
        products: List of product dicts to upsert
        batch_size: Number of products per batch

    Returns:
        Dict with 'upserted' and 'failed' counts
    """
    if not products:
        return {"upserted": 0, "failed": 0}

    client = get_client()
    total = len(products)
    upserted = 0
    failed = 0

    logger.info(f"Upserting {total} products in batches of {batch_size}")

    for i in range(0, total, batch_size):
        batch = products[i : i + batch_size]
        records = [_prepare_product_for_db(p) for p in batch]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        # Retry loop for this batch
        success = False
        last_error = None
        for attempt in range(1, MAX_UPSERT_RETRIES + 1):
            try:
                client.table(TABLE_NAME).upsert(
                    records, on_conflict="source,product_url"
                ).execute()
                upserted += len(records)
                logger.info(
                    f"Batch {batch_num}/{total_batches}: upserted {len(records)} products "
                    f"(attempt {attempt})"
                )
                success = True
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_UPSERT_RETRIES:
                    logger.warning(
                        f"Batch {batch_num} failed (attempt {attempt}/{MAX_UPSERT_RETRIES}): {e}. "
                        f"Retrying..."
                    )
                else:
                    logger.error(
                        f"Batch {batch_num} failed after {MAX_UPSERT_RETRIES} attempts: {e}"
                    )

        if not success:
            # Fallback: try one-by-one to isolate the problematic record
            logger.warning(f"Trying individual upsert for batch {batch_num}...")
            for record in records:
                for attempt in range(1, MAX_UPSERT_RETRIES + 1):
                    try:
                        client.table(TABLE_NAME).upsert(
                            record, on_conflict="source,product_url"
                        ).execute()
                        upserted += 1
                        break
                    except Exception as e:
                        if attempt < MAX_UPSERT_RETRIES:
                            continue
                        failed += 1
                        logger.error(
                            f"Failed to upsert product {record.get('id')} "
                            f"({record.get('title')}): {e}"
                        )

            # Log all failed products from this batch
            _log_failed_products(records, str(last_error))

    result = {"upserted": upserted, "failed": failed}
    logger.info(f"Batch upsert complete: {result}")
    return result


# ─── Run summary ────────────────────────────────────────────────────────────


def print_run_summary(
    classification: dict,
    upsert_result: dict,
    stale_result: dict,
):
    """
    Print a formatted run summary at the end of each scraper run.

    Args:
        classification: Result from classify_products()
        upsert_result: Result from batch_upsert()
        stale_result: Result from handle_stale_products()
    """
    new_count = len(classification.get("new", []))
    changed_image = len(classification.get("changed_image", []))
    changed_data = len(classification.get("changed_data", []))
    unchanged = len(classification.get("unchanged", []))
    upserted = upsert_result.get("upserted", 0)
    failed = upsert_result.get("failed", 0)
    deleted = stale_result.get("deleted", 0)

    summary_lines = [
        "\n" + "=" * 60,
        "RUN SUMMARY",
        "=" * 60,
        f"  New products:          {new_count}",
        f"  Updated (image):       {changed_image}",
        f"  Updated (data only):   {changed_data}",
        f"  Unchanged (skipped):   {unchanged}",
        f"  ─────────────────────",
        f"  Upserted to DB:        {upserted}",
    ]

    if failed:
        summary_lines.append(f"  Failed to upsert:      {failed} ⚠️")
    if deleted:
        summary_lines.append(f"  Stale products deleted: {deleted}")

    summary_lines.append("=" * 60)

    summary = "\n".join(summary_lines)
    logger.info(summary)


# ─── Legacy helpers (verification / count) ──────────────────────────────────


def verify_import(limit: int = 5) -> list[dict]:
    """
    Verify that products were successfully imported by fetching a sample.
    """
    client = get_client()
    try:
        response = (
            client.table(TABLE_NAME)
            .select("*")
            .eq("source", SOURCE)
            .limit(limit)
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"Failed to verify import: {e}")
        return []


def get_product_count() -> Optional[int]:
    """Get the total count of products imported for this source."""
    client = get_client()
    try:
        response = (
            client.table(TABLE_NAME)
            .select("*", count="exact")
            .eq("source", SOURCE)
            .execute()
        )
        return response.count
    except Exception as e:
        logger.error(f"Failed to get product count: {e}")
        return None
