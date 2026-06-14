import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from scraper.core.config import BATCH_SIZE, MAX_PAGES, OUTPUT_DIR, PAGE_WORKERS, PARTITION_PATH, REQUEST_DELAY_SECONDS
from scraper.core.http_client import fetch
from scraper.core.site import SiteAdapter
from scraper.storage.huggingface import (
    index_lookup,
    load_existing_index,
    update_index,
    upload_batch_to_hugging_face,
    upload_index,
)


def save_and_upload_batch(
    site: SiteAdapter,
    listings: list[dict],
    output_dir: Path,
    batch_number: int,
    index_df,
):
    import pandas as pd

    if not listings:
        return index_df

    df = pd.DataFrame(listings).drop_duplicates(subset=["listing_url"])
    df = site.resolve_detail_urls(df)

    output_file = output_dir / f"listings_batch_{batch_number:04d}.parquet"
    df.to_parquet(output_file, index=False)

    print("Preview:")
    print(df.head(10).to_string())
    print(f"Rows in batch {batch_number}: {len(df)}")

    upload_batch_to_hugging_face(output_file, batch_number)

    index_df = update_index(index_df, df)
    upload_index(index_df, output_dir)
    return index_df


def scrape_page(site: SiteAdapter, page_number: int, last_page: int, first_html: str | None = None) -> list[dict]:
    url = site.page_url(site.start_url, page_number)
    print(f"Scraping page {page_number}/{last_page}: {url}")

    if first_html is not None:
        html_text = first_html
    else:
        with httpx.Client(follow_redirects=True) as client:
            html_text = fetch(client, url)

    listings = site.parse_listings(html_text, url)
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


def run_site_pipeline(site: SiteAdapter) -> None:
    output_dir = OUTPUT_DIR / PARTITION_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    index_df = load_existing_index(output_dir)
    existing_event_keys, latest_prices = index_lookup(index_df)
    new_or_changed_listings = []
    seen_event_keys = set(existing_event_keys)
    seen_listing_urls_this_run = set()
    batch_number = 1

    with httpx.Client(follow_redirects=True) as client:
        print(f"Scraping start page for {site.name}: {site.start_url}")
        first_html = fetch(client, site.start_url)
        last_page = site.extract_last_page(first_html)
        if MAX_PAGES is not None:
            last_page = min(last_page, MAX_PAGES)

        print(f"Pages to scrape: {last_page}")
        all_page_listings = scrape_page(site, 1, last_page, first_html=first_html)

    page_numbers = list(range(2, last_page + 1))
    if page_numbers:
        print(f"Scraping remaining pages with {PAGE_WORKERS} workers")
        with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as executor:
            futures = [executor.submit(scrape_page, site, page_number, last_page) for page_number in page_numbers]

            for future in as_completed(futures):
                all_page_listings.extend(future.result())
                if REQUEST_DELAY_SECONDS > 0:
                    time.sleep(REQUEST_DELAY_SECONDS)

    for listing in all_page_listings:
        listing_url = listing["listing_url"]
        if listing_url in seen_listing_urls_this_run:
            continue
        seen_listing_urls_this_run.add(listing_url)

        event_key = site.listing_event_key(listing)
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
        index_df = save_and_upload_batch(site, batch_listings, output_dir, batch_number, index_df)
        batch_number += 1

    print(f"Rows scraped: {len(seen_listing_urls_this_run)}")
