"""
Embedding generation using google/siglip-base-patch16-384.

Generates 768-dim embeddings for both images and text using the
SIGLIP multimodal model from HuggingFace.
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

from config import MODEL_ID, BATCH_SIZE, REQUEST_DELAY

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
            # get_image_features returns BaseModelOutputWithPooling, pooler_output has the 768-dim embedding
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
            # get_text_features returns BaseModelOutputWithPooling, pooler_output has the 768-dim embedding
            embeddings = outputs.pooler_output
            # L2 normalize
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        return embeddings[0].cpu().tolist()
    except Exception as e:
        logger.error(f"Failed to generate text embedding: {e}")
        return None


def process_product_embeddings(products: list[dict]) -> list[dict]:
    """
    Process embeddings for a list of products concurrently.
    
    For each product:
        - Downloads the main image and generates image_embedding
        - Generates info_embedding from product text data
    
    Args:
        products: List of product dicts (must have image_url, title, etc.)
    
    Returns:
        Updated products with embedding fields populated
    """
    logger.info(f"Processing embeddings for {len(products)} products...")

    # Process in batches using ThreadPoolExecutor for I/O-bound image downloads
    with ThreadPoolExecutor(max_workers=min(BATCH_SIZE, 10)) as executor:
        future_to_product = {}

        for idx, product in enumerate(products):
            # Build info text for text embedding
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
            product["_info_text"] = " | ".join(info_parts)

            # Submit image download task
            future = executor.submit(download_image, product.get("image_url", ""))
            future_to_product[future] = idx

        # Collect downloaded images
        downloaded_images = {}
        for future in as_completed(future_to_product):
            idx = future_to_product[future]
            try:
                img = future.result()
                if img:
                    downloaded_images[idx] = img
            except Exception as e:
                logger.warning(f"Failed to download image for product {idx}: {e}")

    # Now generate embeddings in batches (batched GPU processing is more efficient)
    model, processor = _load_model()
    batch_size = min(BATCH_SIZE, 4)  # Smaller batches for GPU memory

    # Image embeddings
    product_indices = list(downloaded_images.keys())
    for start_idx in range(0, len(product_indices), batch_size):
        batch_indices = product_indices[start_idx:start_idx + batch_size]
        batch_images = [downloaded_images[i] for i in batch_indices]

        try:
            inputs = processor(images=batch_images, return_tensors="pt").to(_device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                # BaseModelOutputWithPooling -> pooler_output gives 768-dim embeddings
                embeddings = outputs.pooler_output
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

            for batch_pos, orig_idx in enumerate(batch_indices):
                products[orig_idx]["image_embedding"] = embeddings[batch_pos].cpu().tolist()
        except Exception as e:
            logger.error(f"Batch image embedding failed at batch {start_idx}: {e}")
            for orig_idx in batch_indices:
                products[orig_idx]["image_embedding"] = None

    # Text embeddings (batch processing)
    text_indices = [
        i for i, p in enumerate(products)
        if p.get("_info_text")
    ]

    for start_idx in range(0, len(text_indices), batch_size):
        batch_indices = text_indices[start_idx:start_idx + batch_size]
        batch_texts = [products[i]["_info_text"] for i in batch_indices]

        try:
            inputs = processor(
                text=batch_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=64,
                truncation=True,
            ).to(_device)

            with torch.no_grad():
                outputs = model.get_text_features(**inputs)
                # BaseModelOutputWithPooling -> pooler_output gives 768-dim embeddings
                embeddings = outputs.pooler_output
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

            for batch_pos, orig_idx in enumerate(batch_indices):
                products[orig_idx]["info_embedding"] = embeddings[batch_pos].cpu().tolist()
        except Exception as e:
            logger.error(f"Batch text embedding failed at batch {start_idx}: {e}")
            for orig_idx in batch_indices:
                products[orig_idx]["info_embedding"] = None

    # Clean up temp fields
    for p in products:
        p.pop("_info_text", None)

    # Log stats
    with_embeddings = sum(1 for p in products if p.get("image_embedding"))
    logger.info(f"Generated image embeddings for {with_embeddings}/{len(products)} products")

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
