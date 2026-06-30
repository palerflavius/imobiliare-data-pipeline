import os
import re


# Runtime target metadata comes from GitHub Actions matrix values or local env vars.
SITE_NAME = os.getenv("SITE_NAME", "imobiliare.ro")
COUNTY_SLUG = os.getenv("COUNTY_SLUG", "brasov")
CITY_SLUG = os.getenv("CITY_SLUG", "brasov")
AREA_SLUG = os.getenv("AREA_SLUG", "")
OFFER_TYPE = os.getenv("OFFER_TYPE", "sale")
PROPERTY_TYPE = os.getenv("PROPERTY_TYPE", "apartments")

# Tuning knobs control crawl speed and batch size without code changes.
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "2"))
DETAIL_REQUEST_DELAY_SECONDS = float(os.getenv("DETAIL_REQUEST_DELAY_SECONDS", "0.2"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "0")) or None
MAX_DETAIL_PAGES = int(os.getenv("MAX_DETAIL_PAGES", "0")) or None
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "300"))
DETAIL_WORKERS = int(os.getenv("DETAIL_WORKERS", "4"))
PAGE_WORKERS = int(os.getenv("PAGE_WORKERS", "4"))

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
}

BROWSER_IMPERSONATE = os.getenv("BROWSER_IMPERSONATE", "chrome")
HTTP_PROXY = os.getenv("SCRAPER_HTTP_PROXY", "")


def safe_path_part(value: str) -> str:
    """Convert arbitrary metadata into a stable partition path segment."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "-", value)
    return value.strip("-") or "unknown"


START_URL = os.getenv("START_URL", "")


def partition_path() -> str:
    """Build the default Hugging Face partition path for the current target."""
    parts = {
        "site": SITE_NAME,
        "county": COUNTY_SLUG,
        "city": CITY_SLUG,
        "offer": OFFER_TYPE,
        "property": PROPERTY_TYPE,
    }
    if AREA_SLUG:
        parts["area"] = AREA_SLUG
    return "/".join(f"{key}={safe_path_part(value)}" for key, value in parts.items())


PARTITION_PATH = os.getenv("PARTITION_PATH", partition_path())
HF_INDEX_PATH = os.getenv("HF_INDEX_PATH", f"raw/{PARTITION_PATH}/index/listing_price_index.parquet")
