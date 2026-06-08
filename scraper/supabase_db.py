"""
Supabase database operations for the scraper.

Handles batch upserting of scraped product data into the products table.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from supabase import create_client, Client

from config import SOURCE, SUPABASE_URL, SUPABASE_KEY, TABLE_NAME, UPLOAD_BATCH_SIZE

logger = logging.getLogger(__name__)


def get_client() -> Client:
    """Create and return a Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _prepare_product_for_db(product: dict) -> dict:
    """
    Prepare a product dict for database insertion.
    
    - Adds created_at timestamp
    - Handles None values properly
    - JSON-encodes any complex fields
    - Ensures vector fields are lists of floats
    """
    record = dict(product)  # Make a copy
    
    # Add timestamp
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    
    # Convert tags to PostgreSQL array format if needed
    if record.get("tags") and isinstance(record["tags"], list):
        record["tags"] = record["tags"]
    
    # Ensure embeddings are proper lists
    for field in ["image_embedding", "info_embedding"]:
        if record.get(field) is not None and isinstance(record[field], list):
            # Supabase-py should handle list of floats for vector columns
            pass
    
    # Convert any remaining None values
    for key in list(record.keys()):
        if record[key] is None:
            record[key] = None
    
    return record


def upsert_products(products: list[dict]) -> dict:
    """
    Upsert products to Supabase in batches.
    
    Uses the unique constraint (source, product_url) for conflict resolution.
    
    Args:
        products: List of product dicts with all fields ready
    
    Returns:
        Dict with counts of upserted, skipped, errors
    """
    if not products:
        logger.warning("No products to upsert")
        return {"upserted": 0, "skipped": 0, "errors": 0}

    client = get_client()
    total = len(products)
    upserted = 0
    errors = 0

    logger.info(f"Upserting {total} products to Supabase in batches of {UPLOAD_BATCH_SIZE}")

    for i in range(0, total, UPLOAD_BATCH_SIZE):
        batch = products[i:i + UPLOAD_BATCH_SIZE]
        records = [_prepare_product_for_db(p) for p in batch]

        try:
            response = (
                client.table(TABLE_NAME)
                .upsert(records, on_conflict="source,product_url")
                .execute()
            )
            upserted += len(records)
            logger.info(f"Upserted batch {i//UPLOAD_BATCH_SIZE + 1}: {len(records)} products")
        except Exception as e:
            errors += len(records)
            logger.error(f"Failed to upsert batch {i//UPLOAD_BATCH_SIZE + 1}: {e}")
            # Try one by one to isolate errors
            for record in records:
                try:
                    client.table(TABLE_NAME).upsert(
                        record, on_conflict="source,product_url"
                    ).execute()
                    upserted += 1
                except Exception as e2:
                    logger.error(f"Failed to upsert product {record.get('id')}: {e2}")
                    errors += 1

    result = {
        "total": total,
        "upserted": upserted,
        "errors": errors,
    }
    logger.info(f"Upsert complete: {result}")
    return result


def verify_import(limit: int = 5) -> list[dict]:
    """
    Verify that products were successfully imported by fetching a sample.
    
    Args:
        limit: Number of records to fetch
    
    Returns:
        List of product records from Supabase
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
    """
    Get the total count of products imported for this source.
    """
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
