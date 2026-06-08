"""
Embedding generation using google/siglip-base-patch16-384.

Generates 768-dim embeddings for both images and text using the
SIGLIP multimodal model from HuggingFace.

Supports staggered generation with configurable delay between batches
to avoid overwhelming system resources.
"""

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from config import MODEL_ID, BATCH_SIZE, EMBEDDING_DELAY

logger = logging.getLogger(__name__)

# Global model cache (loaded once)
_model = None
_processor = None
_device = None


def _load_model():
    """Load the SIGLIP model and processor (cached)."""
    global _model, _processor, _device

    if _model is not None and _processor is not None:
        return _model, _processor

    logger.info(f"Loading SIGLIP model: {MODEL_ID}")
    start = time.time()

    # Use CUDA if available, otherwise CPU
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {_device}")

    _model = SiglipModel.from_pretrained(MODEL_ID).to(_device)
    _model.eval()
    _processor = AutoProcessor.from_pretrained(MODEL_ID)

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.2f}s")

    return _model, _processor


def download_image(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """
    Download an image from a URL.

    Args:
        url: Image URL
        timeout: Request timeout in seconds

    Returns:
        PIL Image or None on failure
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        # Ensure RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.warning(f"Failed to download image {url}: {e}")
        return None


def get_image_embedding(image_url: str) -> Optional[list[float]]:
    """
    Generate a 768-dim SIGLIP embedding for a single image.

    Args:
        image_url: URL of the image to embed

    Returns:
        List of 768 floats, or None on failure
    """
    model, processor = _load_model()

    image = download_image(image_url)
    if image is None:
        return None

    try:
        inputs = processor(images=image, return_tensors="pt").to(_device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            embeddings = outputs.pooler_output
            # L2 normalize
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        return embeddings[0].cpu().tolist()
    except Exception as e:
        logger.error(f"Failed to generate image embedding for {image_url}: {e}")
        return None


def get_text_embedding(text: str) -> Optional[list[float]]:
    """
    Generate a 768-dim SIGLIP text embedding.

    Args:
        text: Text to embed (e.g., product title + description)

    Returns:
        List of 768 floats, or None on failure
    """
    if not text or not text.strip():
        return None

    model, processor = _load_model()

    try:
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        ).to(_device)

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            embeddings = outputs.pooler_output
            # L2 normalize
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        return embeddings[0].cpu().tolist()
    except Exception as e:
        logger.error(f"Failed to generate text embedding: {e}")
        return None


def _build_info_text(product: dict) -> str:
    """Build a combined info text from product fields for text embedding."""
    info_parts = []
    if product.get("title"):
        info_parts.append(f"Title: {product['title']}")
    if product.get("description"):
        info_parts.append(f"Description: {product['description']}")
    if product.get("category"):
        info_parts.append(f"Category: {product['category']}")
    if product.get("gender"):
        info_parts.append(f"Gender: {product['gender']}")
    if product.get("price"):
        info_parts.append(f"Price: {product['price']}")
    if product.get("size"):
        info_parts.append(f"Sizes: {product['size']}")
    if product.get("tags"):
        info_parts.append(f"Tags: {', '.join(product['tags'])}")
    return " | ".join(info_parts)


def process_product_embeddings(
    products: list[dict],
) -> list[dict]:
    """
    Generate SIGLIP embeddings for a list of products.

    Only generates embeddings for products that have an image_url.
    Adds a staggered delay (EMBEDDING_DELAY) between batches to
    avoid overwhelming system resources.

    Args:
        products: List of product dicts (must have image_url, title, etc.)

    Returns:
        Updated products with 'image_embedding' and 'info_embedding' populated
    """
    if not products:
        return products

    logger.info(f"Generating embeddings for {len(products)} products...")

    model, processor = _load_model()
    batch_size = min(BATCH_SIZE, 4)  # Smaller batches for GPU memory
    total = len(products)
    processed = 0

    for batch_start in range(0, total, batch_size):
        batch = products[batch_start : batch_start + batch_size]

        # --- Image embeddings ---
        downloaded_images = []
        valid_indices = []

        for idx, product in enumerate(batch):
            image_url = product.get("image_url")
            if not image_url:
                continue

            img = download_image(image_url)
            if img is not None:
                downloaded_images.append(img)
                valid_indices.append(idx)

        if downloaded_images:
            try:
                inputs = processor(images=downloaded_images, return_tensors="pt").to(_device)
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                    embeddings = outputs.pooler_output
                    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

                for pos, orig_idx in enumerate(valid_indices):
                    product_idx = batch_start + orig_idx
                    products[product_idx]["image_embedding"] = embeddings[pos].cpu().tolist()

                logger.debug(
                    f"Image embeddings: batch {batch_start // batch_size + 1}/"
                    f"{(total + batch_size - 1) // batch_size} "
                    f"({len(downloaded_images)} images)"
                )
            except Exception as e:
                logger.error(f"Image embedding batch failed at {batch_start}: {e}")

        # --- Text embeddings ---
        text_inputs = []
        text_indices = []
        for idx, product in enumerate(batch):
            info_text = _build_info_text(product)
            if info_text.strip():
                text_inputs.append(info_text)
                text_indices.append(idx)

        if text_inputs:
            try:
                inputs = processor(
                    text=text_inputs,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                ).to(_device)

                with torch.no_grad():
                    outputs = model.get_text_features(**inputs)
                    embeddings = outputs.pooler_output
                    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

                for pos, orig_idx in enumerate(text_indices):
                    product_idx = batch_start + orig_idx
                    products[product_idx]["info_embedding"] = embeddings[pos].cpu().tolist()
            except Exception as e:
                logger.error(f"Text embedding batch failed at {batch_start}: {e}")

        processed += len(batch)
        logger.debug(f"Embedding progress: {processed}/{total}")

        # Staggered delay between batches (except after the last one)
        if batch_start + batch_size < total:
            logger.debug(f"Waiting {EMBEDDING_DELAY}s before next batch...")
            time.sleep(EMBEDDING_DELAY)

    # Stats
    with_img = sum(1 for p in products if p.get("image_embedding"))
    with_txt = sum(1 for p in products if p.get("info_embedding"))
    logger.info(f"Image embeddings: {with_img}/{total} | Text embeddings: {with_txt}/{total}")

    return products


def refine_products_after_embedding(products: list[dict]) -> list[dict]:
    """
    Remove products that failed to generate their image embedding.
    Returns only products with valid image_embedding.
    """
    valid = [p for p in products if p.get("image_embedding") is not None]
    skipped = len(products) - len(valid)
    if skipped:
        logger.warning(f"Skipping {skipped} products without valid image embeddings")
    return valid
