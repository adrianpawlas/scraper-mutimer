"""
Shopify products.json scraper for Mutimer.

Fetches all products from specified collections using the Shopify
undocumented products.json API with pagination support.
"""

import json
import logging
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import BASE_URL, COLLECTIONS, SHOPIFY_LIMIT, REQUEST_DELAY, BRAND, SOURCE

logger = logging.getLogger(__name__)


def fetch_collection_products(collection: str, limit: int = SHOPIFY_LIMIT) -> list[dict]:
    """
    Fetch all products from a Shopify collection, handling pagination.
    
    Args:
        collection: Collection slug (e.g., 'clothing', 'accessories')
        limit: Products per page (max 250)
    
    Returns:
        List of raw product dicts from the Shopify API
    """
    all_products = []
    page = 1
    
    while True:
        url = f"{BASE_URL}/collections/{collection}/products.json?limit={limit}&page={page}"
        logger.info(f"Fetching: {url}")
        
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            break
        
        products = data.get("products", [])
        if not products:
            logger.info(f"No more products on page {page} for collection '{collection}'")
            break
        
        logger.info(f"Got {len(products)} products from page {page}")
        
        # Attach collection info to each product for later use
        for product in products:
            product["_collection"] = collection
        
        all_products.extend(products)
        page += 1
        time.sleep(REQUEST_DELAY)
    
    return all_products


def strip_html(html_content: Optional[str]) -> str:
    """Strip HTML tags from a string, returning clean text."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def parse_category(product: dict) -> str:
    """
    Determine the category string from product_type and tags.
    
    Examples:
        product_type="Sweaters" -> "Sweaters"
        Tags containing "Sweaters" and "Hoodies" -> "Sweaters, Hoodies"
    """
    product_type = product.get("product_type", "").strip()
    tags = [tag.strip() for tag in product.get("tags", []) if isinstance(tag, str)]
    
    # Known category-like tags (exclude generic tags like "Active", "Top Sellers")
    category_tags = {"Clothing", "Bottoms", "Outerwear", "Top Sellers"}
    subcategory_tags = {
        "Denim", "Pants", "Sweaters", "Hoodies", "T-Shirts",
        "Shirts", "Jackets", "Coats", "Dresses", "Skirts",
        "Shorts", "Sweatshirts", "Vests", "Blazers", "Suits",
        "Accessories", "Hats", "Bags", "Shoes", "Belts",
        "Scarves", "Gloves", "Socks", "Jewelry",
    }
    
    categories = []
    
    # Add product_type as primary category if it looks like a category
    if product_type and product_type.lower() not in {"default", "general", ""}:
        categories.append(product_type)
    
    # Add matching subcategory tags
    for tag in tags:
        if tag in subcategory_tags and tag not in categories:
            categories.append(tag)
    
    if not categories:
        # Fall back to the collection name
        collection = product.get("_collection", "clothing")
        categories.append(collection.capitalize())
    
    return ", ".join(categories)

def parse_sizes(product: dict) -> Optional[str]:
    """
    Extract size information from product variants.
    
    Returns:
        Comma-separated list of sizes, or None if no size options exist
    """
    options = product.get("options", [])
    for option in options:
        if option.get("name", "").lower() in {"size", "velikost"}:
            values = option.get("values", [])
            if values:
                return ", ".join(str(v) for v in values)
    
    # Try from variants
    variants = product.get("variants", [])
    size_values = set()
    for variant in variants:
        for i in range(1, 4):
            opt = variant.get(f"option{i}")
            if opt and opt.strip():
                size_values.add(opt.strip())
    
    if size_values:
        return ", ".join(sorted(size_values))
    
    return None


def parse_price_info(product: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Parse price and sale information from product variants.
    
    Args:
        product: Raw product dict
    
    Returns:
        Tuple of (price_str, sale_str) where:
        - price_str: CZK price formatted for multiple currencies (e.g., "20.90EUR, 450CZK, 75PLN")
        - sale_str: Sale price string (same format) or None if no sale
    """
    from currency_utils import format_price_with_currency
    
    variants = product.get("variants", [])
    if not variants:
        return None, None
    
    # Get the first variant's prices (usually the main/default)
    first_variant = variants[0]
    
    try:
        price_czk = float(first_variant.get("price", 0))
        compare_at = first_variant.get("compare_at_price")
    except (ValueError, TypeError):
        return None, None
    
    if not price_czk:
        return None, None
    
    # Check if there's a sale (compare_at_price > price)
    is_on_sale = False
    sale_price_czk = None
    if compare_at is not None:
        try:
            compare_at_val = float(compare_at)
            if compare_at_val > 0 and compare_at_val > price_czk:
                is_on_sale = True
                sale_price_czk = price_czk
                price_czk = compare_at_val  # Original price is compare_at_price
        except (ValueError, TypeError):
            pass
    
    # Format prices with currency conversion
    price_str = format_price_with_currency(price_czk)
    sale_str = format_price_with_currency(sale_price_czk) if is_on_sale and sale_price_czk else None
    
    return price_str, sale_str


def parse_images(product: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Extract image URLs from product.
    
    Args:
        product: Raw product dict
    
    Returns:
        Tuple of (main_image_url, additional_images_str)
    """
    images = product.get("images", [])
    if not images:
        return None, None
    
    image_urls = [img.get("src") for img in images if img.get("src")]
    if not image_urls:
        return None, None
    
    main_image = image_urls[0]
    additional = " , ".join(image_urls[1:]) if len(image_urls) > 1 else None
    
    return main_image, additional


def build_metadata(product: dict) -> str:
    """
    Build a comprehensive metadata JSON string with all product info.
    """
    metadata = {
        "shopify_id": product.get("id"),
        "title": product.get("title"),
        "handle": product.get("handle"),
        "description": strip_html(product.get("body_html")),
        "product_type": product.get("product_type"),
        "tags": product.get("tags", []),
        "vendor": product.get("vendor"),
        "variants": [
            {
                "id": v.get("id"),
                "title": v.get("title"),
                "sku": v.get("sku"),
                "price": v.get("price"),
                "compare_at_price": v.get("compare_at_price"),
                "available": v.get("available"),
                "option1": v.get("option1"),
                "option2": v.get("option2"),
                "option3": v.get("option3"),
                "grams": v.get("grams"),
            }
            for v in product.get("variants", [])
        ],
        "options": product.get("options", []),
        "image_count": len(product.get("images", [])),
        "published_at": product.get("published_at"),
        "created_at": product.get("created_at"),
        "updated_at": product.get("updated_at"),
        "collection": product.get("_collection"),
    }
    return json.dumps(metadata, ensure_ascii=False)


def map_product(product: dict) -> dict:
    """
    Map a raw Shopify product dict to our Supabase schema.
    
    Returns:
        Dict ready for Supabase insertion
    """
    # Parse basic fields
    product_id = str(product.get("id"))
    handle = product.get("handle", "")
    collection = product.get("_collection", "clothing")
    title = product.get("title", "")
    description = strip_html(product.get("body_html"))
    
    # Parse images
    image_url, additional_images = parse_images(product)
    
    # Parse prices
    price_str, sale_str = parse_price_info(product)
    
    # Parse sizes
    sizes = parse_sizes(product)
    
    # Build category
    category = parse_category(product)
    
    # Gender is always NULL for this brand
    gender = None
    
    # Build metadata
    metadata = build_metadata(product)
    
    # Build product URL
    product_url = f"{BASE_URL}/collections/{collection}/products/{handle}"
    
    return {
        "id": product_id,
        "source": SOURCE,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": image_url,
        "brand": BRAND,
        "title": title,
        "description": description,
        "category": category,
        "gender": gender,
        "size": sizes,
        "second_hand": False,
        "price": price_str,
        "sale": sale_str,
        "additional_images": additional_images,
        "tags": list(product.get("tags", [])),
        "metadata": metadata,
        "other": None,
        "country": None,
        "compressed_image_url": None,
    }


def scrape_all_products() -> list[dict]:
    """
    Scrape all products from all configured collections.
    
    Returns:
        List of mapped product dicts ready for embedding and upload
    """
    all_raw_products = []
    
    for collection in COLLECTIONS:
        logger.info(f"Scraping collection: {collection}")
        products = fetch_collection_products(collection)
        logger.info(f"Found {len(products)} products in '{collection}'")
        all_raw_products.extend(products)
    
    # Map to our schema
    mapped = [map_product(p) for p in all_raw_products]
    
    # Deduplicate by id (a product might appear in multiple collections)
    seen_ids = set()
    unique_products = []
    for p in mapped:
        if p["id"] not in seen_ids:
            seen_ids.add(p["id"])
            unique_products.append(p)
    
    logger.info(f"Total unique products: {len(unique_products)}")
    return unique_products
