import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import httpx
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential


BASE_URL = "https://www.imobiliare.ro"
START_URL = "https://www.imobiliare.ro/vanzare-apartamente/judetul-brasov/brasov"

REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "2"))
DETAIL_REQUEST_DELAY_SECONDS = float(os.getenv("DETAIL_REQUEST_DELAY_SECONDS", "1"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "0")) or None
MAX_DETAIL_PAGES = int(os.getenv("MAX_DETAIL_PAGES", "0")) or None

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; imobiliare-data-pipeline/1.0; educational project)"
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def fetch_response(client: httpx.Client, url: str) -> httpx.Response:
    response = client.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response


def fetch(client: httpx.Client, url: str) -> str:
    response = fetch_response(client, url)
    return response.text


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


def html_to_lines(html_text: str) -> list[str]:
    tree = HTMLParser(html_text)

    for node in tree.css("script, style, noscript"):
        node.decompose()

    text = tree.body.text(separator="\n") if tree.body else tree.text(separator="\n")
    lines = [clean_text(line) for line in text.splitlines()]
    return [line for line in lines if line]


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


def extract_listing_links(html_text: str, page_url: str) -> list[str]:
    tree = HTMLParser(html_text)
    links = []

    for anchor in tree.css('a[data-cy="listing-information-link"], a[id^="listing-link-"]'):
        href = anchor.attributes.get("href")
        if not href or "/oferta/" not in href:
            continue
        links.append(urljoin(page_url, href))

    return list(dict.fromkeys(links))


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
                "title": title,
                "price_eur": price_eur,
                "location": extract_location(block_lines),
                "rooms": extract_rooms(block_text),
                "area_sqm": extract_area_sqm(block_text),
                "floor": extract_floor(block_text),
                "page_url": page_url,
                "listing_url": listing_url,
                "final_listing_url": None,
                "listing_id": listing_id_from_url(listing_url),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    unique = {}
    for item in listings:
        unique[item["listing_url"]] = item

    return list(unique.values())


def resolve_listing_url(client: httpx.Client, listing_url: str) -> str:
    response = fetch_response(client, listing_url)
    tree = HTMLParser(response.text)
    canonical = tree.css_first('link[rel="canonical"]')

    if canonical:
        href = canonical.attributes.get("href")
        if href:
            return urljoin(listing_url, href)

    return str(response.url)


def upload_to_hugging_face(file_path: Path) -> None:
    from huggingface_hub import HfApi

    hf_token = os.getenv("HF_TOKEN")
    hf_repo_id = os.getenv("HF_REPO_ID")

    if not hf_token or not hf_repo_id:
        print("Skipping Hugging Face upload: HF_TOKEN or HF_REPO_ID is not set.")
        return

    api = HfApi(token=hf_token)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path_in_repo = f"raw/imobiliare/listings_{date_str}.parquet"

    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=path_in_repo,
        repo_id=hf_repo_id,
        repo_type="dataset",
    )

    print(f"Uploaded to Hugging Face: {path_in_repo}")


def main() -> None:
    import pandas as pd

    all_listings = []

    with httpx.Client(follow_redirects=True) as client:
        print(f"Scraping start page: {START_URL}")
        first_html = fetch(client, START_URL)
        last_page = extract_last_page(first_html)
        if MAX_PAGES is not None:
            last_page = min(last_page, MAX_PAGES)

        print(f"Pages to scrape: {last_page}")

        for page_number in range(1, last_page + 1):
            url = page_url(START_URL, page_number)
            print(f"Scraping page {page_number}/{last_page}: {url}")
            html_text = first_html if page_number == 1 else fetch(client, url)

            listings = parse_listings(html_text, url)
            print(f"Found listings on page {page_number}: {len(listings)}")

            all_listings.extend(listings)
            time.sleep(REQUEST_DELAY_SECONDS)

    if not all_listings:
        raise RuntimeError("No listings found. Page structure may have changed.")

    df = pd.DataFrame(all_listings).drop_duplicates(subset=["listing_url"])

    with httpx.Client(follow_redirects=True) as client:
        detail_limit = len(df) if MAX_DETAIL_PAGES is None else min(len(df), MAX_DETAIL_PAGES)
        print(f"Resolving detail URLs: {detail_limit}/{len(df)}")

        for position, index in enumerate(df.index[:detail_limit], start=1):
            listing_url = df.at[index, "listing_url"]
            print(f"Resolving detail URL {position}/{detail_limit}: {listing_url}")
            df.at[index, "final_listing_url"] = resolve_listing_url(client, listing_url)
            time.sleep(DETAIL_REQUEST_DELAY_SECONDS)

    output_dir = Path("/tmp/imobiliare-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "listings.parquet"
    df.to_parquet(output_file, index=False)

    print("Preview:")
    print(df.head(10).to_string())
    print(f"Rows scraped: {len(df)}")

    upload_to_hugging_face(output_file)


if __name__ == "__main__":
    main()
