"""
Currency conversion utilities for the scraper.

Converts base CZK prices to EUR, USD, PLN using a free exchange rate API.
Caches rates to avoid repeated API calls.
"""

import time
import logging
from typing import Optional

import requests

from config import BASE_CURRENCY, TARGET_CURRENCIES, REQUEST_DELAY

logger = logging.getLogger(__name__)

# Cache for exchange rates
_exchange_rates_cache: Optional[dict] = None
_cache_timestamp: float = 0
_CACHE_TTL = 3600  # 1 hour


def _fetch_exchange_rates() -> dict:
    """
    Fetch exchange rates from a free API.
    Uses frankfurter.app which is free and doesn't require an API key.
    Falls back to open.er-api.com if frankfurter fails.
    """
    global _exchange_rates_cache, _cache_timestamp

    now = time.time()
    if _exchange_rates_cache and (now - _cache_timestamp) < _CACHE_TTL:
        return _exchange_rates_cache

    rates = {}

    # Try frankfurter.app first
    try:
        symbols = ",".join(TARGET_CURRENCIES)
        url = f"https://api.frankfurter.app/latest?from={BASE_CURRENCY}&to={symbols}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates", {})
        logger.info(f"Fetched exchange rates from frankfurter.app: {rates}")
    except Exception as e:
        logger.warning(f"Frankfurter API failed: {e}. Trying fallback...")

    # Fallback to open.er-api.com
    if not rates:
        try:
            url = f"https://open.er-api.com/v6/latest/{BASE_CURRENCY}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            rates = {k: v for k, v in data.get("rates", {}).items() if k in TARGET_CURRENCIES}
            logger.info(f"Fetched exchange rates from open.er-api.com: {rates}")
        except Exception as e:
            logger.error(f"Fallback exchange rate API also failed: {e}")
            # Use approximate hardcoded rates as last resort
            rates = {"EUR": 0.041, "USD": 0.044, "PLN": 0.18}

    # Also add base currency itself with rate 1.0
    rates[BASE_CURRENCY] = 1.0

    _exchange_rates_cache = rates
    _cache_timestamp = now
    return rates


def get_aud_to_czk_rate() -> float:
    """
    Fetch the AUD to CZK exchange rate from the API.
    
    Ensures the main CZK→target rates cache is populated first to avoid
    cache collisions with _fetch_exchange_rates.
    """
    global _exchange_rates_cache, _cache_timestamp

    now = time.time()
    if _exchange_rates_cache and (now - _cache_timestamp) < _CACHE_TTL:
        if "_AUD_TO_CZK" in _exchange_rates_cache:
            return _exchange_rates_cache["_AUD_TO_CZK"]

    # Ensure the main CZK→target rates are cached first
    _fetch_exchange_rates()

    # Now try to fetch AUD→CZK rate
    rate = None

    # Try frankfurter.app first
    try:
        url = "https://api.frankfurter.app/latest?from=AUD&to=CZK"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates", {})
        if "CZK" in rates:
            rate = rates["CZK"]
    except Exception as e:
        logger.warning(f"Failed to fetch AUD→CZK rate: {e}")

    # Try open.er-api.com as fallback
    if rate is None:
        try:
            url = "https://open.er-api.com/v6/latest/AUD"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            rates = data.get("rates", {})
            if "CZK" in rates:
                rate = rates["CZK"]
        except Exception as e:
            logger.error(f"Fallback AUD→CZK API also failed: {e}")

    # Hardcoded fallback: approximately 1 AUD = 14.5 CZK
    if rate is None:
        rate = 14.5
        logger.warning(f"Using fallback AUD→CZK rate: {rate}")
    else:
        logger.info(f"Fetched AUD→CZK rate: {rate}")

    # Store in the existing cache dict
    if _exchange_rates_cache is not None:
        _exchange_rates_cache["_AUD_TO_CZK"] = rate
    return rate


def convert_aud_to_czk(price_aud: float) -> float:
    """
    Convert a price from AUD to CZK using the live exchange rate.
    """
    rate = get_aud_to_czk_rate()
    return price_aud * rate


def format_price_with_currency(price_czk: float) -> str:
    """
    Convert a CZK price to multiple currencies and format the output.
    
    Args:
        price_czk: Price in CZK (e.g., 3980.00)
    
    Returns:
        Formatted string like "20.90USD, 450CZK, 75PLN"
        Prices are formatted to 2 decimal places and trailing zeros are kept.
    """
    rates = _fetch_exchange_rates()
    
    parts = []
    for currency in TARGET_CURRENCIES:
        rate = rates.get(currency)
        if rate:
            converted = price_czk * rate
            parts.append(f"{converted:.2f}{currency}")
    
    # Add CZK last
    parts.append(f"{price_czk:.2f}{BASE_CURRENCY}")
    
    return ", ".join(parts)


def format_price_single(price_czk: float, currency: str = "EUR") -> str:
    """
    Convert a CZK price to a single target currency.
    
    Args:
        price_czk: Price in CZK
        currency: Target currency code (default: EUR)
    
    Returns:
        Formatted string like "20.90EUR"
    """
    rates = _fetch_exchange_rates()
    rate = rates.get(currency)
    if rate:
        converted = price_czk * rate
        return f"{converted:.2f}{currency}"
    return f"{price_czk:.2f}{BASE_CURRENCY}"
