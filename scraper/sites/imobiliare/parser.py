import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from selectolax.parser import HTMLParser

from scraper.core.config import AREA_SLUG, CITY_SLUG, COUNTY_SLUG, OFFER_TYPE, PROPERTY_TYPE, SITE_NAME


def clean_text(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", " ", value).strip()


def normalize_number(value: str | None) -> float | None:
    if not value:
        return None

    value = value.strip()
    value = value.replace(".", "")
    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def extract_price_eur(text: str) -> float | None:
    match = re.search(
        r"(\d{2,3}(?:[.\s]\d{3})+|\d{4,})\s*(?:EUR|euro|\u20ac|\u00e2\u201a\u00ac)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return normalize_number(match.group(1))


def extract_rooms(text: str) -> float | None:
    match = re.search(r"(\d+(?:[,.]\d+)?)\s+camere?", text, re.IGNORECASE)
    if not match:
        return None
    return normalize_number(match.group(1))


def extract_area_sqm(text: str) -> float | None:
    match = re.search(r"(\d+(?:[,.]\d+)?)\s*mp", text, re.IGNORECASE)
    if not match:
        return None
    return normalize_number(match.group(1))


def extract_floor(text: str) -> str | None:
    match = re.search(
        r"(Etaj\s+[^\n\r]+|Mansard[aa]\s*/\s*\d+|Mansard\u0103\s*/\s*\d+|Parter\s*/\s*\d+)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return clean_text(match.group(1))


def looks_like_location(line: str) -> bool:
    line = clean_text(line) or ""
    if "," not in line:
        return False
    bad_words = ["EUR", "euro", "\u20ac", "\u00e2\u201a\u00ac", "camere", "mp", "Etaj", "Loading", "Trimite mesaj"]
    return not any(word.lower() in line.lower() for word in bad_words)


def extract_location(lines: list[str]) -> str | None:
    for line in lines:
        if looks_like_location(line):
            return clean_text(line)
    return None


def node_to_lines(node) -> list[str]:
    text = node.text(separator="\n")
    lines = [clean_text(line) for line in text.splitlines()]
    return [line for line in lines if line]


def is_listing_title(line: str) -> bool:
    line = clean_text(line) or ""

    if len(line) < 20:
        return False

    bad_keywords = [
        "apartamente de vanzare",
        "apartamente de v\u00e2nzare",
        "anunturi disponibile",
        "anun\u021buri disponibile",
        "reseteaza filtrele",
        "reseteaz\u0103 filtrele",
        "aplica filtrele",
        "aplic\u0103 filtrele",
        "salveaza cautarea",
        "salveaz\u0103 c\u0103utarea",
        "activeaza notificarile",
        "activeaz\u0103 notific\u0103rile",
        "dezactiveaza notificarile",
        "dezactiveaz\u0103 notific\u0103rile",
        "trimite mesaj",
        "loading",
        "sortare",
        "top listing",
        "cele mai recente",
    ]

    if any(word in line.lower() for word in bad_keywords):
        return False

    good_keywords = [
        "apartament",
        "penthouse",
        "garsonier",
        "studio",
        "camere",
        "rezidence",
        "residence",
        "bloc",
        "zona",
    ]

    return any(word in line.lower() for word in good_keywords)


def listing_id_from_url(url: str) -> str | None:
    match = re.search(r"-(\d+)(?:[/?#]|$)", url)
    return match.group(1) if match else None


def listing_event_key(listing: dict) -> str:
    listing_id = listing.get("listing_id") or listing.get("listing_url")
    price_eur = listing.get("price_eur")
    price_key = "" if price_eur is None else f"{float(price_eur):.2f}"
    return f"{listing_id}|{price_key}"


def find_card_container(anchor):
    node = anchor
    for _ in range(8):
        if node is None:
            break

        text = node.text(separator="\n")
        if extract_price_eur(text) is not None and extract_location(node_to_lines(node)):
            return node

        node = node.parent

    return anchor.parent or anchor


def extract_title_from_card(lines: list[str]) -> str | None:
    for line in lines:
        if is_listing_title(line):
            return clean_text(line)
    return None


def extract_last_page(html_text: str) -> int:
    page_numbers = [1]

    for match in re.finditer(r"[?&]page=(\d+)", html_text):
        page_numbers.append(int(match.group(1)))

    for pattern in [r"&quot;last_page&quot;:(\d+)", r'"last_page"\s*:\s*(\d+)']:
        match = re.search(pattern, html_text)
        if match:
            page_numbers.append(int(match.group(1)))

    return max(page_numbers)


def page_url(base_url: str, page_number: int) -> str:
    if page_number == 1:
        return base_url

    parsed = urlparse(base_url)
    query = parse_qs(parsed.query)
    query["page"] = [str(page_number)]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def parse_listings(html_text: str, page_url: str) -> list[dict]:
    tree = HTMLParser(html_text)
    listings = []

    for anchor in tree.css('a[data-cy="listing-information-link"], a[id^="listing-link-"]'):
        href = anchor.attributes.get("href")
        if not href or "/oferta/" not in href:
            continue

        listing_url = urljoin(page_url, href)
        card = find_card_container(anchor)
        block_lines = node_to_lines(card)
        block_text = "\n".join(block_lines)
        title = extract_title_from_card(block_lines)

        if not title:
            continue

        price_eur = extract_price_eur(block_text)
        if price_eur is None:
            continue

        listings.append(
            {
                "source": "imobiliare.ro",
                "site": SITE_NAME,
                "county": COUNTY_SLUG or None,
                "city": CITY_SLUG,
                "area": AREA_SLUG or None,
                "offer_type": OFFER_TYPE,
                "property_type": PROPERTY_TYPE,
                "title": title,
                "price_eur": price_eur,
                "location": extract_location(block_lines),
                "street_address": None,
                "address_locality": None,
                "address_region": None,
                "address_country": None,
                "full_address_text": None,
                "rooms": extract_rooms(block_text),
                "area_sqm": extract_area_sqm(block_text),
                "floor": extract_floor(block_text),
                "page_url": page_url,
                "listing_url": listing_url,
                "final_listing_url": None,
                "detail_error": None,
                "listing_id": listing_id_from_url(listing_url),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    unique = {}
    for item in listings:
        unique[item["listing_url"]] = item

    return list(unique.values())
