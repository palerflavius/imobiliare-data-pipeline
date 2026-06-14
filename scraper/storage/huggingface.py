from datetime import datetime, timezone
import os
import re
import time

from scraper.core.config import HF_INDEX_PATH, PARTITION_PATH


def hf_config() -> tuple[str | None, str | None]:
    import os

    hf_token = os.getenv("HF_TOKEN")
    hf_repo_id = os.getenv("HF_REPO_ID")
    return hf_token, hf_repo_id


def load_existing_index():
    import pandas as pd
    from huggingface_hub import HfFileSystem

    hf_token, hf_repo_id = hf_config()
    if not hf_token or not hf_repo_id:
        print("Skipping Hugging Face index load: HF_TOKEN or HF_REPO_ID is not set.")
        return pd.DataFrame()

    try:
        fs = HfFileSystem(token=hf_token)
        with fs.open(f"datasets/{hf_repo_id}/{HF_INDEX_PATH}", "rb") as index_file:
            df = pd.read_parquet(index_file)
    except Exception as error:
        print(f"No existing Hugging Face index loaded ({type(error).__name__}: {error}).")
        return pd.DataFrame()

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


def parquet_bytes(df) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def batch_path_in_repo(batch_number: int) -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"raw/{PARTITION_PATH}/date={date_str}/listings_batch_{batch_number:04d}.parquet"


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


def add_index_operation(index_df, operations: list) -> None:
    if index_df.empty:
        return

    from huggingface_hub import CommitOperationAdd

    operations.append(
        CommitOperationAdd(
            path_in_repo=HF_INDEX_PATH,
            path_or_fileobj=parquet_bytes(index_df),
        )
    )


def retry_delay_seconds(error: Exception) -> int:
    response = getattr(error, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after and retry_after.isdigit():
            return int(retry_after)

    message = str(error)
    match = re.search(r"Retry after (\d+) seconds", message)
    if match:
        return int(match.group(1))

    if "repository commits" in message or "128 per hour" in message:
        return int(os.getenv("HF_COMMIT_RETRY_FALLBACK_SECONDS", "3600"))

    return 300


def upload_operations_to_hugging_face(operations: list) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    hf_token, hf_repo_id = hf_config()

    if not hf_token or not hf_repo_id:
        print(f"Skipping Hugging Face upload for raw/{PARTITION_PATH}: HF_TOKEN or HF_REPO_ID is not set.")
        return

    if not operations:
        print("Skipping Hugging Face upload: no files to commit.")
        return

    api = HfApi(token=hf_token)
    max_retries = int(os.getenv("HF_COMMIT_RETRIES", "3"))

    for attempt in range(1, max_retries + 1):
        try:
            api.create_commit(
                repo_id=hf_repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Upload scraper batch for {PARTITION_PATH}",
            )
            break
        except HfHubHTTPError as error:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            if status_code != 429 or attempt == max_retries:
                raise

            delay = retry_delay_seconds(error)
            print(f"Hugging Face rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}.")
            time.sleep(delay)

    print(f"Uploaded {len(operations)} files to Hugging Face in one commit.")
