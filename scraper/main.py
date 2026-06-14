import os
import re
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import httpx
from selectolax.parser import HTMLParser
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


BASE_URL = "https://www.imobiliare.ro"
SITE_NAME = os.getenv("SITE_NAME", "imobiliare.ro")
CITY_SLUG = os.getenv("CITY_SLUG", "brasov")
OFFER_TYPE = os.getenv("OFFER_TYPE", "sale")
PROPERTY_TYPE = os.getenv("PROPERTY_TYPE", "apartments")
START_URL = os.getenv(
    "START_URL",
    "https://www.imobiliare.ro/vanzare-apartamente/judetul-brasov/brasov",
)

REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "2"))
DETAIL_REQUEST_DELAY_SECONDS = float(os.getenv("DETAIL_REQUEST_DELAY_SECONDS", "0.2"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "0")) or None
MAX_DETAIL_PAGES = int(os.getenv("MAX_DETAIL_PAGES", "0")) or None
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "300"))
DETAIL_WORKERS = int(os.getenv("DETAIL_WORKERS", "4"))
PAGE_WORKERS = int(os.getenv("PAGE_WORKERS", "4"))


def partition_path() -> str:
    parts = {
        "site": SITE_NAME,
        "city": CITY_SLUG,
        "offer": OFFER_TYPE,
        "property": PROPERTY_TYPE,
    }
    return "/".join(f"{key}={safe_path_part(value)}" for key, value in parts.items())


def safe_path_part(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "-", value)
    return value.strip("-") or "unknown"


PARTITION_PATH = os.getenv("PARTITION_PATH", partition_path())
HF_INDEX_PATH = os.getenv("HF_INDEX_PATH", f"raw/{PARTITION_PATH}/index/listing_price_index.parquet")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; imobiliare-data-pipeline/1.0; educational project)"
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def is_retryable_fetch_error(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code == 429 or status_code >= 500

    return isinstance(error, (httpx.TimeoutException, httpx.TransportError))


@retry(
    retry=retry_if_exception(is_retryable_fetch_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True,
)
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
                "site": SITE_NAME,
                "city": CITY_SLUG,
                "offer_type": OFFER_TYPE,
                "property_type": PROPERTY_TYPE,
                "title": title,
                "price_eur": price_eur,
                "location": extract_location(block_lines),
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


def resolve_listing_url(client: httpx.Client, listing_url: str) -> tuple[str, str | None]:
    try:
        response = fetch_response(client, listing_url)
        tree = HTMLParser(response.text)
        canonical = tree.css_first('link[rel="canonical"]')

        if canonical:
            href = canonical.attributes.get("href")
            if href:
                return urljoin(listing_url, href), None

        return str(response.url), None
    except httpx.HTTPStatusError as error:
        status_code = error.response.status_code
        message = f"HTTP {status_code}: {error.response.reason_phrase}"
        print(f"Warning: detail URL failed ({message}): {listing_url}")
        return listing_url, message
    except httpx.HTTPError as error:
        message = f"{type(error).__name__}: {error}"
        print(f"Warning: detail URL failed ({message}): {listing_url}")
        return listing_url, message


def resolve_listing_detail(index_and_url: tuple[int, str]) -> tuple[int, str, str | None]:
    index, listing_url = index_and_url

    if DETAIL_REQUEST_DELAY_SECONDS > 0:
        time.sleep(DETAIL_REQUEST_DELAY_SECONDS)

    with httpx.Client(follow_redirects=True) as client:
        final_url, detail_error = resolve_listing_url(client, listing_url)

    return index, final_url, detail_error


def resolve_detail_urls(df):
    detail_limit = len(df) if MAX_DETAIL_PAGES is None else min(len(df), MAX_DETAIL_PAGES)
    print(f"Resolving detail URLs: {detail_limit}/{len(df)} with {DETAIL_WORKERS} workers")

    tasks = [(index, df.at[index, "listing_url"]) for index in df.index[:detail_limit]]

    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as executor:
        futures = [executor.submit(resolve_listing_detail, task) for task in tasks]

        for position, future in enumerate(as_completed(futures), start=1):
            index, final_url, detail_error = future.result()
            df.at[index, "final_listing_url"] = final_url
            df.at[index, "detail_error"] = detail_error
            print(f"Resolved detail URL {position}/{detail_limit}: {final_url}")

    return df


def hf_config() -> tuple[str | None, str | None]:
    hf_token = os.getenv("HF_TOKEN")
    hf_repo_id = os.getenv("HF_REPO_ID")
    return hf_token, hf_repo_id


def load_existing_index(output_dir: Path):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    hf_token, hf_repo_id = hf_config()
    if not hf_token or not hf_repo_id:
        print("Skipping Hugging Face index load: HF_TOKEN or HF_REPO_ID is not set.")
        return pd.DataFrame()

    try:
        index_file = hf_hub_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            filename=HF_INDEX_PATH,
            token=hf_token,
            local_dir=output_dir / "hf-cache",
        )
    except Exception as error:
        print(f"No existing Hugging Face index loaded ({type(error).__name__}: {error}).")
        return pd.DataFrame()

    df = pd.read_parquet(index_file)
    print(f"Loaded Hugging Face index rows: {len(df)}")
    return df


def index_lookup(index_df) -> tuple[set[str], dict[str, float]]:
    if index_df.empty:
        return set(), {}

    event_keys = set(index_df["event_key"].dropna().astype(str))
    latest_prices = {}

    for row in index_df.sort_values("last_seen_at").itertuples(index=False):
        listing_id = getattr(row, "listing_id", None)
        price_eur = getattr(row, "price_eur", None)
        if listing_id is not None and price_eur is not None:
            latest_prices[str(listing_id)] = float(price_eur)

    return event_keys, latest_prices


def upload_file_to_hugging_face(file_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    hf_token, hf_repo_id = hf_config()
    if not hf_token or not hf_repo_id:
        print(f"Skipping Hugging Face upload for {path_in_repo}: HF_TOKEN or HF_REPO_ID is not set.")
        return

    api = HfApi(token=hf_token)

    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=path_in_repo,
        repo_id=hf_repo_id,
        repo_type="dataset",
    )

    print(f"Uploaded to Hugging Face: {path_in_repo}")


def upload_batch_to_hugging_face(file_path: Path, batch_number: int) -> None:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path_in_repo = f"raw/{PARTITION_PATH}/date={date_str}/listings_batch_{batch_number:04d}.parquet"
    upload_file_to_hugging_face(file_path, path_in_repo)


def update_index(index_df, batch_df):
    import pandas as pd

    now = datetime.now(timezone.utc).isoformat()
    index_rows = []

    for row in batch_df.itertuples(index=False):
        index_rows.append(
            {
                "event_key": row.event_key,
                "listing_id": str(row.listing_id),
                "price_eur": float(row.price_eur),
                "listing_url": row.listing_url,
                "final_listing_url": row.final_listing_url,
                "title": row.title,
                "location": row.location,
                "first_seen_at": row.scraped_at,
                "last_seen_at": now,
            }
        )

    new_index_df = pd.DataFrame(index_rows)
    if index_df.empty:
        return new_index_df.drop_duplicates(subset=["event_key"], keep="last")

    combined = pd.concat([index_df, new_index_df], ignore_index=True)
    return combined.drop_duplicates(subset=["event_key"], keep="last")


def upload_index(index_df, output_dir: Path) -> None:
    if index_df.empty:
        return

    index_file = output_dir / "listing_price_index.parquet"
    index_df.to_parquet(index_file, index=False)
    upload_file_to_hugging_face(index_file, HF_INDEX_PATH)


def save_and_upload_batch(listings: list[dict], output_dir: Path, batch_number: int, index_df):
    import pandas as pd

    if not listings:
        return index_df

    df = pd.DataFrame(listings).drop_duplicates(subset=["listing_url"])
    df = resolve_detail_urls(df)

    output_file = output_dir / f"listings_batch_{batch_number:04d}.parquet"
    df.to_parquet(output_file, index=False)

    print("Preview:")
    print(df.head(10).to_string())
    print(f"Rows in batch {batch_number}: {len(df)}")

    upload_batch_to_hugging_face(output_file, batch_number)

    index_df = update_index(index_df, df)
    upload_index(index_df, output_dir)
    return index_df


def scrape_page(page_number: int, last_page: int, first_html: str | None = None) -> list[dict]:
    url = page_url(START_URL, page_number)
    print(f"Scraping page {page_number}/{last_page}: {url}")

    if first_html is not None:
        html_text = first_html
    else:
        with httpx.Client(follow_redirects=True) as client:
            html_text = fetch(client, url)

    listings = parse_listings(html_text, url)
    print(f"Found listings on page {page_number}: {len(listings)}")
    return listings


def batched(items: Iterable[dict], size: int) -> Iterable[list[dict]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []

    if batch:
        yield batch


def main() -> None:
    output_dir = Path("/tmp/imobiliare-output") / PARTITION_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    index_df = load_existing_index(output_dir)
    existing_event_keys, latest_prices = index_lookup(index_df)
    new_or_changed_listings = []
    seen_event_keys = set(existing_event_keys)
    seen_listing_urls_this_run = set()
    batch_number = 1

    with httpx.Client(follow_redirects=True) as client:
        print(f"Scraping start page: {START_URL}")
        first_html = fetch(client, START_URL)
        last_page = extract_last_page(first_html)
        if MAX_PAGES is not None:
            last_page = min(last_page, MAX_PAGES)

        print(f"Pages to scrape: {last_page}")

        all_page_listings = scrape_page(1, last_page, first_html=first_html)

    page_numbers = list(range(2, last_page + 1))
    if page_numbers:
        print(f"Scraping remaining pages with {PAGE_WORKERS} workers")
        with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as executor:
            futures = [executor.submit(scrape_page, page_number, last_page) for page_number in page_numbers]

            for future in as_completed(futures):
                all_page_listings.extend(future.result())
                if REQUEST_DELAY_SECONDS > 0:
                    time.sleep(REQUEST_DELAY_SECONDS)

    for listing in all_page_listings:
        listing_url = listing["listing_url"]
        if listing_url in seen_listing_urls_this_run:
            continue
        seen_listing_urls_this_run.add(listing_url)

        event_key = listing_event_key(listing)
        listing_id = str(listing.get("listing_id") or listing_url)
        previous_price = latest_prices.get(listing_id)
        current_price = float(listing["price_eur"])

        if event_key in seen_event_keys:
            continue

        listing["event_key"] = event_key
        listing["previous_price_eur"] = previous_price
        listing["price_changed"] = previous_price is not None and previous_price != current_price
        new_or_changed_listings.append(listing)
        seen_event_keys.add(event_key)

    if not seen_listing_urls_this_run:
        raise RuntimeError("No listings found. Page structure may have changed.")

    print(f"Unique listings scraped this run: {len(seen_listing_urls_this_run)}")
    print(f"New or price-changed listings to upload: {len(new_or_changed_listings)}")

    for batch_listings in batched(new_or_changed_listings, BATCH_SIZE):
        index_df = save_and_upload_batch(batch_listings, output_dir, batch_number, index_df)
        batch_number += 1

    print(f"Rows scraped: {len(seen_listing_urls_this_run)}")


if __name__ == "__main__":
    main()
