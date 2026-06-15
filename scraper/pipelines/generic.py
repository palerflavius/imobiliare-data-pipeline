import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from scraper.core.config import BATCH_SIZE, MAX_PAGES, PAGE_WORKERS, REQUEST_DELAY_SECONDS
from scraper.core.http_client import fetch
from scraper.core.site import SiteAdapter
from scraper.storage.huggingface import (
    add_index_operation,
    batch_path_in_repo,
    deleted_listings_dataframe,
    index_lookup,
    load_existing_index,
    partition_path_for_row,
    parquet_bytes,
    update_index,
    upload_operations_to_hugging_face,
)


def save_batch(
    site: SiteAdapter,
    listings: list[dict],
    batch_number: int,
    index_df,
    upload_operations: list,
):
    """Prepare, enrich, partition, and stage one upload batch."""
    import pandas as pd
    from huggingface_hub import CommitOperationAdd

    if not listings:
        return index_df

    # Deleted records already have the final schema; active records still need detail-page enrichment.
    active_listings = [listing for listing in listings if listing.get("record_status") != "deleted"]
    deleted_listings = [listing for listing in listings if listing.get("record_status") == "deleted"]
    frames = []

    if active_listings:
        active_df = pd.DataFrame(active_listings).drop_duplicates(subset=["listing_url"])
        active_df = site.resolve_detail_urls(active_df)
        active_df["record_status"] = "active"
        if "deleted_detected_at" not in active_df.columns:
            active_df["deleted_detected_at"] = None
        frames.append(active_df)

    if deleted_listings:
        frames.append(pd.DataFrame(deleted_listings))

    df = pd.concat(frames, ignore_index=True)
    output_columns = list(df.columns)

    print("Preview:")
    print(df.head(10).to_string())
    print(f"Rows in batch {batch_number}: {len(df)}")

    # Rows scraped from a county-wide target can land in different city partitions.
    df["_partition_path"] = [partition_path_for_row(row) for row in df.to_dict("records")]
    for partition_path, partition_df in df.groupby("_partition_path", sort=True):
        upload_operations.append(
            CommitOperationAdd(
                path_in_repo=batch_path_in_repo(batch_number, partition_path),
                path_or_fileobj=parquet_bytes(partition_df[output_columns]),
            )
        )

    index_df = update_index(index_df, df[output_columns])
    return index_df


def scrape_page(site: SiteAdapter, page_number: int, last_page: int, first_html: str | None = None) -> list[dict]:
    """Scrape one listing result page."""
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
    """Yield fixed-size lists from an iterable of listing rows."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []

    if batch:
        yield batch


def run_site_pipeline(site: SiteAdapter) -> None:
    """Run one full scrape target from listing pages through Hugging Face upload."""
    import pandas as pd

    # The index is the compact state used for change detection and deletion detection.
    index_df = load_existing_index()
    existing_event_keys, latest_prices = index_lookup(index_df)
    new_or_changed_listings = []
    seen_event_keys = set(existing_event_keys)
    seen_listing_urls_this_run = set()
    seen_listing_ids_this_run = set()
    upload_operations = []
    batch_number = 1

    with httpx.Client(follow_redirects=True) as client:
        # Fetch page 1 first because it also tells us how many pages exist.
        print(f"Scraping start page for {site.name}: {site.start_url}")
        first_html = fetch(client, site.start_url)
        last_page = site.extract_last_page(first_html)
        if MAX_PAGES is not None:
            last_page = min(last_page, MAX_PAGES)

        print(f"Pages to scrape: {last_page}")
        all_page_listings = scrape_page(site, 1, last_page, first_html=first_html)

    page_numbers = list(range(2, last_page + 1))
    if page_numbers:
        # Listing pages are independent, so they can be scraped concurrently.
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
        seen_listing_ids_this_run.add(listing_id)
        previous_price = latest_prices.get(listing_id)
        current_price = float(listing["price_eur"])

        # Upload only first-seen listings or price changes; unchanged rows stay in the index.
        if event_key in seen_event_keys:
            continue

        listing["event_key"] = event_key
        listing["previous_price_eur"] = previous_price
        listing["price_changed"] = previous_price is not None and previous_price != current_price
        listing["record_status"] = "active"
        listing["deleted_detected_at"] = None
        new_or_changed_listings.append(listing)
        seen_event_keys.add(event_key)

    if not seen_listing_urls_this_run:
        raise RuntimeError("No listings found. Page structure may have changed.")

    print(f"Unique listings scraped this run: {len(seen_listing_urls_this_run)}")
    print(f"New or price-changed listings to upload: {len(new_or_changed_listings)}")

    output_columns = list(pd.DataFrame(new_or_changed_listings).columns) if new_or_changed_listings else list(index_df.columns)
    for column in ("record_status", "deleted_detected_at"):
        if column not in output_columns:
            output_columns.append(column)
    # Missing listings are written back as normal rows, so downstream consumers see one schema.
    deleted_df = deleted_listings_dataframe(index_df, seen_listing_ids_this_run, output_columns)
    print(f"Listings missing from this run: {len(deleted_df)}")
    output_listings = new_or_changed_listings + deleted_df.to_dict("records")

    for batch_listings in batched(output_listings, BATCH_SIZE):
        index_df = save_batch(site, batch_listings, batch_number, index_df, upload_operations)
        batch_number += 1

    if batch_number > 1:
        add_index_operation(index_df, upload_operations)

    if upload_operations:
        upload_operations_to_hugging_face(upload_operations)
    else:
        print("No new, price-changed, or missing listings to upload.")

    print(f"Rows scraped: {len(seen_listing_urls_this_run)}")
