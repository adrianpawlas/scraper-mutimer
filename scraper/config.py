"""
Configuration for the Mutimer Shopify scraper.
All secrets must be provided via environment variables or .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Also looks for .env in parent directory

# Supabase credentials — MUST be set via environment or .env file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validate required credentials
if not SUPABASE_URL:
    raise ValueError(
        "SUPABASE_URL environment variable is required. "
        "Create a .env file in the scraper/ directory or set the env variable."
    )
if not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_KEY environment variable is required. "
        "Create a .env file in the scraper/ directory or set the env variable."
    )

# Scraper settings
SOURCE = "scraper-mutimer"
BRAND = "Mutimer"
BASE_URL = "https://mutimer.co"

# Collections to scrape
COLLECTIONS = [
    "clothing",
    "accessories",
]

# Shopify products.json API settings
SHOPIFY_LIMIT = 250  # Max per page for Shopify

# Embedding model
MODEL_ID = "google/siglip-base-patch16-384"
VECTOR_DIM = 768

# Batch processing
BATCH_SIZE = 10  # Number of products to process in parallel for embeddings
UPLOAD_BATCH_SIZE = 50  # Number of products to upsert to Supabase at once
REQUEST_DELAY = 0.5  # Delay between API requests in seconds
EMBEDDING_DELAY = 0.5  # Delay between embedding generations in seconds
STALE_RUNS_THRESHOLD = 2  # Consecutive missing runs before product is deleted
MAX_UPSERT_RETRIES = 3  # Max retries for batch upsert operations

# Currency conversion
BASE_CURRENCY = "CZK"
TARGET_CURRENCIES = ["EUR", "USD", "PLN"]

# Supabase table name
TABLE_NAME = "products"
