import os
import re
import tempfile
from pathlib import Path


SITE_NAME = os.getenv("SITE_NAME", "imobiliare.ro")
COUNTY_SLUG = os.getenv("COUNTY_SLUG", "brasov")
CITY_SLUG = os.getenv("CITY_SLUG", "brasov")
AREA_SLUG = os.getenv("AREA_SLUG", "")
OFFER_TYPE = os.getenv("OFFER_TYPE", "sale")
PROPERTY_TYPE = os.getenv("PROPERTY_TYPE", "apartments")

REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "2"))
DETAIL_REQUEST_DELAY_SECONDS = float(os.getenv("DETAIL_REQUEST_DELAY_SECONDS", "0.2"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "0")) or None
MAX_DETAIL_PAGES = int(os.getenv("MAX_DETAIL_PAGES", "0")) or None
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "300"))
DETAIL_WORKERS = int(os.getenv("DETAIL_WORKERS", "4"))
PAGE_WORKERS = int(os.getenv("PAGE_WORKERS", "4"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(Path(tempfile.gettempdir()) / "imobiliare-output")))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; imobiliare-data-pipeline/1.0; educational project)"
}


def safe_path_part(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "-", value)
    return value.strip("-") or "unknown"


START_URL = os.getenv("START_URL", "")


def partition_path() -> str:
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
