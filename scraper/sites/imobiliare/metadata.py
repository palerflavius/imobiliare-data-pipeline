import re
import unicodedata
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
    """Return true for null-like values coming from Python or pandas."""
    return value is None or value == "" or str(value).lower() in {"nan", "nat", "none"}


def slug_to_label(value: str | None) -> str | None:
    """Convert a URL slug into a human-readable label."""
    if not value:
        return None
    return " ".join(part.capitalize() for part in value.split("-") if part)


def label_to_slug(value: str | None) -> str | None:
    """Convert a location label into an ASCII partition-safe slug."""
    if not value:
        return None

    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower()
    ascii_value = re.sub(r"[^a-z0-9]+", "-", ascii_value)
    return ascii_value.strip("-") or None


def locality_label_from_listing(listing: dict, *, allow_location_fallback: bool) -> str | None:
    """Return the best locality-like label available on a listing row."""
    fields = ["address_locality", "full_address_text"]
    if allow_location_fallback:
        fields.append("location")

    for field in fields:
        value = listing.get(field)
        if is_blank(value):
            continue

        label = str(value).split(",", 1)[0].strip()
        if label:
            return label

    return None


def normalized_slug(value: str | None) -> str | None:
    """Return a slug for comparisons without treating blanks as real values."""
    slug = label_to_slug(value)
    return slug or None


def ignored_city_slug_values(listing: dict) -> set[str]:
    """Return region labels that should not be treated as city values."""
    ignored = {"romania"}

    for field in ("county", "address_country"):
        slug = normalized_slug(str(listing.get(field))) if not is_blank(listing.get(field)) else None
        if slug:
            ignored.add(slug)
            ignored.add(f"judetul-{slug}")
            ignored.add(f"judet-{slug}")

    return ignored


def first_usable_city_label(labels: list[str], listing: dict) -> str | None:
    """Pick the first label that does not look like county/country metadata."""
    ignored = ignored_city_slug_values(listing)

    for label in labels:
        slug = normalized_slug(label)
        if slug and slug not in ignored:
            return label

    return None


def city_label_from_region_or_full_address(listing: dict, *, allow_location_fallback: bool) -> str | None:
    """Infer city for county-wide targets from region/full address, then optional card location."""
    region = None if is_blank(listing.get("address_region")) else str(listing.get("address_region")).strip()
    city_label = first_usable_city_label([region] if region else [], listing)
    if city_label:
        return city_label

    full_address = None if is_blank(listing.get("full_address_text")) else str(listing.get("full_address_text"))
    if full_address:
        parts = [part.strip() for part in full_address.split(",") if part.strip()]
        # For addresses like "Pipera, Voluntari, Ilfov", the city is usually the last non-county part.
        city_label = first_usable_city_label(list(reversed(parts)), listing)
        if city_label:
            return city_label

    if allow_location_fallback:
        location = None if is_blank(listing.get("location")) else str(listing.get("location")).split(",", 1)[0].strip()
        city_label = first_usable_city_label([location] if location else [], listing)
        if city_label:
            return city_label

        # If the only region value is "Judetul Ilfov", fall back to address_locality.
        locality = None if is_blank(listing.get("address_locality")) else str(listing.get("address_locality")).strip()
        return first_usable_city_label([locality] if locality else [], listing)

    return None


def city_slug_from_location(listing: dict, *, allow_location_fallback: bool = True) -> str | None:
    """Infer a city slug from address or card location fields."""
    # County-wide targets start as city=all; use region/full address to route output by city.
    return label_to_slug(
        city_label_from_region_or_full_address(
            listing,
            allow_location_fallback=allow_location_fallback,
        )
    )


def area_slug_from_address_locality(listing: dict) -> str | None:
    """Use address_locality as the safest area source for fixed-city targets."""
    locality_label = locality_label_from_listing(listing, allow_location_fallback=False)
    return label_to_slug(locality_label)


def area_slug_from_full_address(listing: dict) -> str | None:
    """Use the first full_address_text segment as the stored area value."""
    full_address = None if is_blank(listing.get("full_address_text")) else str(listing.get("full_address_text"))
    if not full_address:
        return None

    first_part = full_address.split(",", 1)[0].strip()
    return label_to_slug(first_part)


def listing_slug(listing_url: str | None) -> str | None:
    """Return the offer slug portion of a listing URL."""
    if not listing_url:
        return None

    path = urlparse(listing_url).path
    if "/oferta/" not in path:
        return None

    slug = path.split("/oferta/", 1)[1].strip("/")
    return slug or None


def tokens_until_stop(tokens: list[str]) -> list[str]:
    """Collect location slug tokens until property descriptors begin."""
    result = []
    for token in tokens:
        if token in STOP_WORDS:
            break
        result.append(token)
    return result


def rooms_from_slug(slug: str) -> float | None:
    """Infer room count from the listing slug when card parsing missed it."""
    match = re.search(r"-(\d+)-camere(?:-|$)", slug)
    if match:
        return float(match.group(1))

    if slug.startswith("garsoniera-") or "-garsoniera-" in slug:
        return 1.0

    return None


def offer_type_from_slug(slug: str) -> str | None:
    """Infer normalized offer type from the listing slug."""
    if "-de-vanzare-" in slug:
        return "sale"
    if "-de-inchiriat-" in slug:
        return "rent"
    return None


def property_type_from_slug(slug: str) -> str | None:
    """Infer normalized property type from the listing slug."""
    if slug.startswith(("apartament-", "penthouse-", "garsoniera-", "studio-")):
        return "apartments"
    return None


def location_tokens_from_slug(slug: str) -> list[str]:
    """Return the slug tokens that describe the listing location."""
    if "-de-vanzare-" in slug:
        return slug.split("-de-vanzare-", 1)[1].split("-")
    if "-de-inchiriat-" in slug:
        return slug.split("-de-inchiriat-", 1)[1].split("-")
    return []


def infer_metadata_from_listing_url(listing_url: str | None) -> dict:
    """Infer missing scraper metadata from the canonical listing URL."""
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
        # Bucharest sector URLs encode the sector before the neighborhood.
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
        metadata.update(
            {
                "county": "bucuresti",
                "city": "bucuresti",
                "address_locality": slug_to_label("-".join(neighborhood_tokens)),
                "address_region": "Bucuresti",
            }
        )
        return metadata

    if tokens and tokens[0] == "brasov":
        metadata.update(
            {
                "county": "brasov",
                "city": "brasov",
                "address_region": "Brasov",
            }
        )
        return metadata

    if COUNTY_SLUG or CITY_SLUG or AREA_SLUG:
        # Fall back to the configured target when the URL does not encode a known city pattern.
        metadata.update(
            {
                "county": COUNTY_SLUG or None,
                "city": CITY_SLUG or None,
                "area": AREA_SLUG or None,
            }
        )

    return metadata


def backfill_listing_metadata(
    listing: dict,
    *,
    fill_scraped_at: bool = False,
    allow_location_partition_fallback: bool = True,
) -> dict:
    """Fill missing metadata fields on one listing row."""
    metadata = infer_metadata_from_listing_url(listing.get("listing_url") or listing.get("final_listing_url"))
    result = dict(listing)

    for field, value in metadata.items():
        if value is not None and is_blank(result.get(field)):
            result[field] = value

    if str(result.get("city")).lower() == "all":
        # Keep configured city values, but split "all" targets into real city partitions.
        city_slug = city_slug_from_location(result, allow_location_fallback=allow_location_partition_fallback)
        if city_slug:
            result["city"] = city_slug
            area_slug = area_slug_from_full_address(result)
            result["area"] = area_slug if area_slug != city_slug else None
    elif str(result.get("county")).lower() != "bucuresti":
        # For fixed-city targets, keep city fixed and store area from the visible full address.
        area_slug = area_slug_from_full_address(result)
        if area_slug:
            result["area"] = area_slug

    if fill_scraped_at and is_blank(result.get("scraped_at")):
        result["scraped_at"] = datetime.now(timezone.utc).isoformat()

    return result


def backfill_dataframe(df, *, fill_scraped_at: bool = False):
    """Apply metadata backfill to a pandas DataFrame of listing rows."""
    if df.empty or "listing_url" not in df.columns:
        return df

    required_fields = ["source", "site", "county", "city", "offer_type", "property_type"]
    before_missing = None
    if all(field in df.columns for field in required_fields):
        before_missing = df[required_fields].apply(lambda column: column.map(is_blank)).any(axis=1)

    rows = [
        backfill_listing_metadata(
            row,
            fill_scraped_at=fill_scraped_at,
            allow_location_partition_fallback=True,
        )
        for row in df.to_dict("records")
    ]
    result = type(df)(rows)
    if "metadata_backfilled_at" not in result.columns:
        result["metadata_backfilled_at"] = None

    if before_missing is not None:
        result.loc[before_missing, "metadata_backfilled_at"] = datetime.now(timezone.utc).isoformat()

    return result
