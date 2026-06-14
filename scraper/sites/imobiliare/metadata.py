import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from scraper.core.config import AREA_SLUG, CITY_SLUG, COUNTY_SLUG, OFFER_TYPE, PROPERTY_TYPE, SITE_NAME
from scraper.sites.imobiliare.parser import listing_id_from_url


STOP_WORDS = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "camera",
    "camere",
    "mobilat",
    "mobilata",
    "utilat",
    "utilata",
    "decomandat",
    "semidecomandat",
    "nedecomandat",
    "mp",
}


def is_blank(value) -> bool:
    return value is None or value == "" or str(value).lower() in {"nan", "nat", "none"}


def slug_to_label(value: str | None) -> str | None:
    if not value:
        return None
    return " ".join(part.capitalize() for part in value.split("-") if part)


def listing_slug(listing_url: str | None) -> str | None:
    if not listing_url:
        return None

    path = urlparse(listing_url).path
    if "/oferta/" not in path:
        return None

    slug = path.split("/oferta/", 1)[1].strip("/")
    return slug or None


def tokens_until_stop(tokens: list[str]) -> list[str]:
    result = []
    for token in tokens:
        if token in STOP_WORDS:
            break
        result.append(token)
    return result


def rooms_from_slug(slug: str) -> float | None:
    match = re.search(r"-(\d+)-camere(?:-|$)", slug)
    if match:
        return float(match.group(1))

    if slug.startswith("garsoniera-") or "-garsoniera-" in slug:
        return 1.0

    return None


def offer_type_from_slug(slug: str) -> str | None:
    if "-de-vanzare-" in slug:
        return "sale"
    if "-de-inchiriat-" in slug:
        return "rent"
    return None


def property_type_from_slug(slug: str) -> str | None:
    if slug.startswith(("apartament-", "penthouse-", "garsoniera-", "studio-")):
        return "apartments"
    return None


def location_tokens_from_slug(slug: str) -> list[str]:
    if "-de-vanzare-" in slug:
        return slug.split("-de-vanzare-", 1)[1].split("-")
    if "-de-inchiriat-" in slug:
        return slug.split("-de-inchiriat-", 1)[1].split("-")
    return []


def infer_metadata_from_listing_url(listing_url: str | None) -> dict:
    slug = listing_slug(listing_url)
    if not slug:
        return {}

    tokens = location_tokens_from_slug(slug)
    metadata = {
        "source": "imobiliare.ro",
        "site": SITE_NAME or "imobiliare.ro",
        "offer_type": offer_type_from_slug(slug) or OFFER_TYPE,
        "property_type": property_type_from_slug(slug) or PROPERTY_TYPE,
        "listing_id": listing_id_from_url(listing_url),
        "rooms": rooms_from_slug(slug),
    }

    if len(tokens) >= 2 and tokens[0] == "sector" and tokens[1].isdigit():
        sector = f"sector-{tokens[1]}"
        neighborhood_tokens = tokens_until_stop(tokens[2:])
        metadata.update(
            {
                "county": "bucuresti",
                "city": "bucuresti",
                "area": sector,
                "address_locality": slug_to_label("-".join(neighborhood_tokens)),
                "address_region": "Bucuresti",
            }
        )
        return metadata

    if tokens and tokens[0] == "bucuresti":
        neighborhood_tokens = tokens_until_stop(tokens[1:])
        neighborhood_slug = "-".join(neighborhood_tokens) or None
        metadata.update(
            {
                "county": "bucuresti",
                "city": "bucuresti",
                "area": neighborhood_slug,
                "address_locality": slug_to_label("-".join(neighborhood_tokens)),
                "address_region": "Bucuresti",
            }
        )
        return metadata

    if tokens and tokens[0] == "brasov":
        neighborhood_tokens = tokens_until_stop(tokens[1:])
        neighborhood_slug = "-".join(neighborhood_tokens) or None
        metadata.update(
            {
                "county": "brasov",
                "city": "brasov",
                "area": neighborhood_slug,
                "address_locality": slug_to_label("-".join(neighborhood_tokens)),
                "address_region": "Brasov",
            }
        )
        return metadata

    if COUNTY_SLUG or CITY_SLUG or AREA_SLUG:
        metadata.update(
            {
                "county": COUNTY_SLUG or None,
                "city": CITY_SLUG or None,
                "area": AREA_SLUG or None,
            }
        )

    return metadata


def backfill_listing_metadata(listing: dict, *, fill_scraped_at: bool = False) -> dict:
    metadata = infer_metadata_from_listing_url(listing.get("listing_url") or listing.get("final_listing_url"))
    result = dict(listing)

    for field, value in metadata.items():
        if value is not None and is_blank(result.get(field)):
            result[field] = value

    if fill_scraped_at and is_blank(result.get("scraped_at")):
        result["scraped_at"] = datetime.now(timezone.utc).isoformat()

    return result


def backfill_dataframe(df, *, fill_scraped_at: bool = False):
    if df.empty or "listing_url" not in df.columns:
        return df

    required_fields = ["source", "site", "county", "city", "offer_type", "property_type"]
    before_missing = None
    if all(field in df.columns for field in required_fields):
        before_missing = df[required_fields].apply(lambda column: column.map(is_blank)).any(axis=1)

    rows = [backfill_listing_metadata(row, fill_scraped_at=fill_scraped_at) for row in df.to_dict("records")]
    result = type(df)(rows)
    if "metadata_backfilled_at" not in result.columns:
        result["metadata_backfilled_at"] = None

    if before_missing is not None:
        result.loc[before_missing, "metadata_backfilled_at"] = datetime.now(timezone.utc).isoformat()

    return result
