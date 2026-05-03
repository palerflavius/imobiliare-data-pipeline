import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
import pandas as pd
from huggingface_hub import HfApi
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential


BASE_URL = "https://www.imobiliare.ro"

START_URLS = [
    "https://www.imobiliare.ro/vanzare-apartamente/judetul-brasov/brasov",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; imobiliare-data-pipeline/1.0; educational project)"
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def fetch(client: httpx.Client, url: str) -> str:
    response = client.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
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
    match = re.search(r"(\d{2,3}(?:[.\s]\d{3})+|\d{4,})\s*€", text)
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
    match = re.search(r"(Etaj\s+[^\n\r]+|Mansardă\s*/\s*\d+|Parter\s*/\s*\d+)", text, re.IGNORECASE)
    if not match:
        return None
    return clean_text(match.group(1))


def looks_like_location(line: str) -> bool:
    line = clean_text(line) or ""
    if "," not in line:
        return False
    bad_words = ["€", "camere", "mp", "Etaj", "Loading", "Trimite mesaj"]
    return not any(word.lower() in line.lower() for word in bad_words)


def extract_location(lines: list[str]) -> str | None:
    for line in lines:
        if looks_like_location(line):
            return clean_text(line)
    return None


def html_to_lines(html_text: str) -> list[str]:
    tree = HTMLParser(html_text)

    # eliminăm bucăți care pot murdări textul
    for node in tree.css("script, style, noscript"):
        node.decompose()

    text = tree.body.text(separator="\n") if tree.body else tree.text(separator="\n")
    lines = [clean_text(line) for line in text.splitlines()]
    return [line for line in lines if line]


def parse_listings(html_text: str, page_url: str) -> list[dict]:
    lines = html_to_lines(html_text)
    listings = []

    for i, line in enumerate(lines):
        # Pe pagina publică, titlurile anunțurilor apar ca headings.
        if not line.startswith("###"):
            continue

        title = clean_text(line.replace("###", ""))
        if not title or len(title) < 15:
            continue

        # Evităm titlurile duplicate/scurtate care apar uneori după primul titlu.
        if title.endswith("...") or "Caută" in title:
            continue

        block_lines = lines[i : i + 18]
        block_text = "\n".join(block_lines)

        price_eur = extract_price_eur(block_text)
        rooms = extract_rooms(block_text)
        area_sqm = extract_area_sqm(block_text)
        floor = extract_floor(block_text)
        location = extract_location(block_lines)

        # Dacă nu avem preț, probabil nu e card valid.
        if price_eur is None:
            continue

        listings.append(
            {
                "source": "imobiliare.ro",
                "title": title,
                "price_eur": price_eur,
                "location": location,
                "rooms": rooms,
                "area_sqm": area_sqm,
                "floor": floor,
                "page_url": page_url,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # deduplicare după titlu + preț + suprafață
    unique = {}
    for item in listings:
        key = (
            item.get("title"),
            item.get("price_eur"),
            item.get("area_sqm"),
            item.get("location"),
        )
        unique[key] = item

    return list(unique.values())


def upload_to_hugging_face(file_path: Path) -> None:
    hf_token = os.environ["HF_TOKEN"]
    hf_repo_id = os.environ["HF_REPO_ID"]

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
    all_listings = []

    with httpx.Client(follow_redirects=True) as client:
        for url in START_URLS:
            print(f"Scraping: {url}")
            html_text = fetch(client, url)

            listings = parse_listings(html_text, url)
            print(f"Found listings on page: {len(listings)}")

            all_listings.extend(listings)
            time.sleep(5)

    if not all_listings:
        raise RuntimeError("No listings found. Page structure may have changed.")

    df = pd.DataFrame(all_listings).drop_duplicates(
        subset=["title", "price_eur", "area_sqm", "location"]
    )

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