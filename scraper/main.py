import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
import pandas as pd
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential
from huggingface_hub import HfApi


BASE_URL = "https://www.imobiliare.ro"
START_URLS = [
    "https://www.imobiliare.ro/vanzare-apartamente/brasov",
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


def extract_price(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"([\d\.\s]+)\s*€", text)
    if not match:
        return None
    number = match.group(1).replace(".", "").replace(" ", "")
    try:
        return float(number)
    except ValueError:
        return None


def parse_listings(html_text: str, page_url: str) -> list[dict]:
    tree = HTMLParser(html_text)
    listings = []

    # Selector generic: căutăm linkuri care par anunțuri.
    links = tree.css("a[href]")

    seen = set()

    for link in links:
        href = link.attributes.get("href", "")
        full_url = urljoin(BASE_URL, href)

        if "/vanzare-" not in full_url and "/inchiriere-" not in full_url:
            continue

        title = clean_text(link.text())
        if not title or len(title) < 15:
            continue

        if full_url in seen:
            continue

        seen.add(full_url)

        parent_text = clean_text(link.parent.text() if link.parent else "")
        price_eur = extract_price(parent_text)

        listings.append(
            {
                "source": "imobiliare.ro",
                "url": full_url,
                "title": title,
                "price_eur": price_eur,
                "raw_text": parent_text,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "page_url": page_url,
            }
        )

    return listings


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
            all_listings.extend(listings)

            # politicos: nu lovim site-ul agresiv
            time.sleep(5)

    if not all_listings:
        raise RuntimeError("No listings found. Selectors may need adjustment.")

    df = pd.DataFrame(all_listings).drop_duplicates(subset=["url"])

    output_dir = Path("/tmp/imobiliare-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "listings.parquet"
    df.to_parquet(output_file, index=False)

    print(f"Rows scraped: {len(df)}")
    upload_to_hugging_face(output_file)


if __name__ == "__main__":
    main()
