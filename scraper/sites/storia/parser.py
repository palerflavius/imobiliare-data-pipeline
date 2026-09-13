import html
import json
import re
import unicodedata
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from scraper.core.config import AREA_SLUG, CITY_SLUG, COUNTY_SLUG, OFFER_TYPE, PROPERTY_TYPE, SITE_NAME


BASE_URL = "https://www.storia.ro"
ROOM_VALUES = {
    "ONE": 1.0,
    "TWO": 2.0,
    "THREE": 3.0,
    "FOUR": 4.0,
    "FIVE": 5.0,
    "SIX": 6.0,
    "SEVEN": 7.0,
    "EIGHT": 8.0,
    "NINE": 9.0,
    "TEN": 10.0,
}
FLOOR_VALUES = {
    "BASEMENT": "demisol",
    "GROUND": "parter",
    "FIRST": "1",
    "SECOND": "2",
    "THIRD": "3",
    "FOURTH": "4",
    "FIFTH": "5",
    "SIXTH": "6",
    "SEVENTH": "7",
    "EIGHTH": "8",
    "NINTH": "9",
    "TENTH": "10",
}


def clean_text(value: str | None) -> str | None:
    """Normalize whitespace and empty strings from scraped text."""
    if not value:
        return None
    return re.sub(r"\s+", " ", value).strip()


def label_to_slug(value: str | None) -> str | None:
    """Convert a Storia location label into the local partition slug format."""
    if not value:
        return None

    value = re.sub(r"\s*\(judet\)\s*$", "", value, flags=re.IGNORECASE)
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower()
    ascii_value = re.sub(r"[^a-z0-9]+", "-", ascii_value)
    return ascii_value.strip("-") or None


def load_next_data(html_text: str) -> dict:
    """Extract the Next.js SSR data blob from a Storia result page."""
    match = re.search(
        r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        html_text,
        re.DOTALL,
    )
    if not match:
        return {}
    return json.loads(html.unescape(match.group(1)))


def search_ads_data(html_text: str) -> dict:
    """Return the search result payload from the Next.js page data."""
    data = load_next_data(html_text)
    return (
        data.get("props", {})
        .get("pageProps", {})
        .get("data", {})
        .get("searchAds", {})
    )


def extract_last_page(html_text: str) -> int:
    """Detect the last result page from Storia SSR pagination."""
    search_ads = search_ads_data(html_text)
    total_pages = search_ads.get("pagination", {}).get("totalPages")
    if isinstance(total_pages, int) and total_pages > 0:
        return total_pages

    page_numbers = [1]
    for match in re.finditer(r"[?&]page=(\d+)", html_text):
        page_numbers.append(int(match.group(1)))
    return max(page_numbers)


def page_url(base_url: str, page_number: int) -> str:
    """Return the URL for a requested result page."""
    if page_number == 1:
        return base_url

    parsed = urlparse(base_url)
    query = parse_qs(parsed.query)
    query["page"] = [str(page_number)]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def money_value(item: dict) -> float | None:
    """Read the EUR listing price exposed by Storia."""
    for field in ("totalPrice", "rentPrice"):
        money = item.get(field)
        if not isinstance(money, dict) or money.get("currency") != "EUR":
            continue
        value = money.get("value")
        if value is not None:
            return float(value)
    return None


def reverse_locations(item: dict) -> list[dict]:
    """Return Storia reverse-geocoding locations for a listing."""
    return (
        item.get("location", {})
        .get("reverseGeocoding", {})
        .get("locations", [])
    )


def location_by_level(item: dict, levels: set[str]) -> dict | None:
    """Find the most specific reverse-geocoding location matching one of the levels."""
    for location in reversed(reverse_locations(item)):
        if location.get("locationLevel") in levels:
            return location
    return None


def address_data(item: dict) -> dict:
    """Return the listing address dictionary from Storia's payload."""
    return item.get("location", {}).get("address", {})


def address_location(item: dict) -> str | None:
    """Build the visible location text from reverse geocoding or address fields."""
    district = location_by_level(item, {"district", "neighbourhood"})
    city = location_by_level(item, {"city", "county_capital"})
    county = location_by_level(item, {"county"})
    names = [part.get("name") for part in (district, city, county) if part and part.get("name")]
    if names:
        return ", ".join(names)

    address = address_data(item)
    address_city = address.get("city", {}).get("name") if isinstance(address.get("city"), dict) else None
    province = address.get("province", {}).get("name") if isinstance(address.get("province"), dict) else None
    return ", ".join(part for part in (address_city, province) if part) or None


def street_name(item: dict) -> str | None:
    """Read the street name when Storia exposes it."""
    street = address_data(item).get("street")
    if not isinstance(street, dict):
        return None
    return clean_text(street.get("name"))


def county_slug(item: dict) -> str | None:
    """Infer county slug from reverse geocoding, falling back to configured target."""
    county = location_by_level(item, {"county"})
    return label_to_slug(county.get("name")) if county else (COUNTY_SLUG or None)


def city_slug(item: dict) -> str | None:
    """Infer city slug from reverse geocoding, falling back to configured target."""
    if CITY_SLUG and CITY_SLUG != "all":
        return CITY_SLUG

    city = location_by_level(item, {"city", "county_capital"})
    if city:
        location_id = city.get("id") or ""
        parts = [part for part in location_id.split("/") if part]
        if len(parts) >= 2:
            return parts[1]
        return label_to_slug(city.get("name"))

    return CITY_SLUG or None


def area_slug(item: dict) -> str | None:
    """Infer area slug from reverse geocoding or configured area."""
    district = location_by_level(item, {"district", "neighbourhood"})
    if district:
        location_id = district.get("id") or ""
        parts = [part for part in location_id.split("/") if part]
        if parts:
            return parts[-1]
        return label_to_slug(district.get("name"))
    return AREA_SLUG or None


def listing_url(item: dict) -> str:
    """Build a public Storia listing URL from the href/slug fields."""
    href = item.get("href")
    if href:
        href = href.replace("[lang]", "ro")
        return urljoin(BASE_URL, href)
    return f"{BASE_URL}/ro/oferta/{item['slug']}"


def listing_event_key(listing: dict) -> str:
    """Create a key that changes when the listing price changes."""
    listing_id = listing.get("listing_id") or listing.get("listing_url")
    price_eur = listing.get("price_eur")
    price_key = "" if price_eur is None else f"{float(price_eur):.2f}"
    return f"{listing_id}|{price_key}"


def parse_listing_item(item: dict, page_url: str) -> dict | None:
    """Convert one Storia AdvertListItem into the shared pipeline schema."""
    price_eur = money_value(item)
    title = clean_text(item.get("title"))
    slug = item.get("slug")
    if price_eur is None or not title or not slug:
        return None

    url = listing_url(item)
    address_city = address_data(item).get("city")
    address_province = address_data(item).get("province")
    locality = address_city.get("name") if isinstance(address_city, dict) else None
    region = address_province.get("name") if isinstance(address_province, dict) else None

    return {
        "source": "storia.ro",
        "site": SITE_NAME,
        "county": county_slug(item),
        "city": city_slug(item),
        "area": area_slug(item),
        "offer_type": OFFER_TYPE,
        "property_type": PROPERTY_TYPE,
        "title": title,
        "price_eur": price_eur,
        "location": address_location(item),
        "street_address": street_name(item),
        "address_locality": clean_text(locality),
        "address_region": clean_text(region),
        "address_country": "Romania",
        "full_address_text": address_location(item),
        "rooms": ROOM_VALUES.get(str(item.get("roomsNumber") or "").upper()),
        "area_sqm": item.get("areaInSquareMeters") or item.get("terrainAreaInSquareMeters"),
        "floor": FLOOR_VALUES.get(str(item.get("floorNumber") or "").upper()),
        "page_url": page_url,
        "listing_url": url,
        "final_listing_url": url,
        "detail_error": None,
        "listing_id": str(item.get("id") or slug),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "dateCreated": item.get("dateCreated"),
        "datePublished": item.get("createdAtFirst"),
        "dateModified": item.get("pushedUpAt"),
        "data_bi_listing_agency": item.get("agency", {}).get("name") if isinstance(item.get("agency"), dict) else None,
        "data_bi_listing_agent": item.get("advertOwner", {}).get("name")
        if isinstance(item.get("advertOwner"), dict)
        else None,
        "seller_type": "owner" if item.get("isPrivateOwner") else "agency",
    }


def parse_listings(html_text: str, page_url: str) -> list[dict]:
    """Parse Storia result-page listings into normalized row dictionaries."""
    items = search_ads_data(html_text).get("items", [])
    listings = [listing for item in items if (listing := parse_listing_item(item, page_url))]

    unique = {}
    for item in listings:
        unique[item["listing_url"]] = item
    return list(unique.values())
