from datetime import datetime, timezone
import os
import re
import time

from scraper.core.config import CITY_SLUG, HF_INDEX_PATH, PARTITION_PATH, PROPERTY_TYPE, safe_path_part


def hf_config() -> tuple[str | None, str | None]:
    """Read Hugging Face credentials from environment variables."""
    import os

    hf_token = os.getenv("HF_TOKEN")
    hf_repo_id = os.getenv("HF_REPO_ID")
    return hf_token, hf_repo_id


def index_path_in_repo(partition_path: str = PARTITION_PATH) -> str:
    """Return the index path beside date= folders for one partition."""
    return f"raw/{partition_path}/index/listing_price_index.parquet"


def index_glob_for_current_target() -> str:
    """Return the Hugging Face glob used to load index state for this target."""
    if CITY_SLUG != "all":
        return HF_INDEX_PATH

    parts = PARTITION_PATH.split("/")
    glob_parts = ["city=*" if part == "city=all" else part for part in parts]
    return f"raw/{'/'.join(glob_parts)}/index/listing_price_index.parquet"


def load_existing_index():
    """Load the partition-level listing index from Hugging Face if available."""
    import pandas as pd
    from huggingface_hub import HfFileSystem

    hf_token, hf_repo_id = hf_config()
    if not hf_token or not hf_repo_id:
        print("Skipping Hugging Face index load: HF_TOKEN or HF_REPO_ID is not set.")
        return pd.DataFrame()

    try:
        fs = HfFileSystem(token=hf_token)
        index_glob = index_glob_for_current_target()
        index_paths = sorted(fs.glob(f"datasets/{hf_repo_id}/{index_glob}"))
        if not index_paths:
            print(f"No existing Hugging Face index files matched {index_glob}.")
            return pd.DataFrame()

        frames = []
        for index_path in index_paths:
            with fs.open(index_path, "rb") as index_file:
                frames.append(pd.read_parquet(index_file))
        df = pd.concat(frames, ignore_index=True)
    except Exception as error:
        print(f"No existing Hugging Face index loaded ({type(error).__name__}: {error}).")
        return pd.DataFrame()

    print(f"Loaded Hugging Face index rows: {len(df)}")
    return df


def index_lookup(index_df) -> tuple[set[str], dict[str, float]]:
    """Build fast lookup structures for event and price-change detection."""
    if index_df.empty:
        return set(), {}

    lookup_df = index_df
    if "record_status" in lookup_df.columns:
        lookup_df = lookup_df[lookup_df["record_status"] != "deleted"]

    event_keys = set(lookup_df["event_key"].dropna().astype(str))
    latest_prices = {}

    for row in lookup_df.sort_values("last_seen_at").itertuples(index=False):
        listing_id = getattr(row, "listing_id", None)
        price_eur = getattr(row, "price_eur", None)
        if listing_id is not None and price_eur is not None:
            latest_prices[str(listing_id)] = float(price_eur)

    return event_keys, latest_prices


def parquet_bytes(df) -> bytes:
    """Serialize a DataFrame into in-memory parquet bytes for upload."""
    from io import BytesIO

    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def is_blank(value) -> bool:
    """Return true for null-like values coming from Python or pandas."""
    return value is None or value == "" or str(value).lower() in {"nan", "nat", "none"}


def value_from_row(row, field: str, default: str = "") -> str:
    """Read a field from a dict, pandas Series, or row-like object."""
    if hasattr(row, "get"):
        value = row.get(field)
    else:
        value = getattr(row, field, None)
    return default if is_blank(value) else str(value)


def partition_path_for_row(row) -> str:
    """Build the Hugging Face partition path for an individual row."""
    # Build the destination partition from row metadata, not only from the scraper target.
    county = value_from_row(row, "county", "unknown")
    parts = {
        "site": value_from_row(row, "site", "imobiliare.ro"),
        "county": county,
        "city": value_from_row(row, "city", "unknown"),
        "offer": value_from_row(row, "offer_type", "sale"),
        "property": value_from_row(row, "property_type", PROPERTY_TYPE),
    }
    area = value_from_row(row, "area")
    # Only Bucharest sectors are partition dimensions; other area values stay as columns.
    if safe_path_part(county) == "bucuresti" and safe_path_part(area).startswith("sector-"):
        parts["area"] = area
    return "/".join(f"{key}={safe_path_part(value)}" for key, value in parts.items())


def batch_path_in_repo(batch_number: int, partition_path: str = PARTITION_PATH) -> str:
    """Return the raw parquet path for one batch inside a partition."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"raw/{partition_path}/date={date_str}/listings_batch_{batch_number:04d}.parquet"


def deleted_listings_dataframe(index_df, seen_listing_ids: set[str], output_columns: list[str]):
    """Create deleted rows for indexed listings missing from the current run."""
    import pandas as pd

    if index_df.empty or "listing_id" not in index_df.columns:
        return pd.DataFrame()

    latest_df = index_df.copy()
    latest_df = latest_df.dropna(subset=["listing_id"])
    latest_df["listing_id"] = latest_df["listing_id"].astype(str)

    if latest_df.empty:
        return pd.DataFrame()

    if "last_seen_at" in latest_df.columns:
        latest_df = latest_df.sort_values("last_seen_at")

    # Only emit a deletion once: if the latest index state is already deleted, skip it.
    latest_df = latest_df.drop_duplicates(subset=["listing_id"], keep="last")
    if "record_status" in latest_df.columns:
        latest_df = latest_df[latest_df["record_status"] != "deleted"]
    latest_df = latest_df[~latest_df["listing_id"].isin(seen_listing_ids)]

    if latest_df.empty:
        return pd.DataFrame()

    latest_df["record_status"] = "deleted"
    latest_df["deleted_detected_at"] = datetime.now(timezone.utc).isoformat()
    latest_df["scraped_at"] = latest_df["deleted_detected_at"]

    for column in output_columns:
        if column not in latest_df.columns:
            latest_df[column] = None

    return latest_df[output_columns]


def update_index(index_df, batch_df):
    """Merge uploaded rows back into the listing index."""
    import pandas as pd

    now = datetime.now(timezone.utc).isoformat()
    index_rows = []

    for row in batch_df.to_dict("records"):
        index_row = dict(row)
        index_row["listing_id"] = str(index_row.get("listing_id"))
        if is_blank(index_row.get("first_seen_at")):
            index_row["first_seen_at"] = index_row.get("scraped_at")
        index_row["last_seen_at"] = now
        index_rows.append(index_row)

    new_index_df = pd.DataFrame(index_rows)
    if index_df.empty:
        return new_index_df.drop_duplicates(subset=["event_key"], keep="last")

    combined = pd.concat([index_df, new_index_df], ignore_index=True)
    return combined.drop_duplicates(subset=["event_key"], keep="last")


def add_index_operation(index_df, operations: list) -> None:
    """Stage updated index parquet files beside each real partition's date= folders."""
    if index_df.empty:
        return

    from huggingface_hub import CommitOperationAdd

    index_df = index_df.copy()
    index_df["_partition_path"] = [partition_path_for_row(row) for row in index_df.to_dict("records")]
    for partition_path, partition_df in index_df.groupby("_partition_path", sort=True):
        operations.append(
            CommitOperationAdd(
                path_in_repo=index_path_in_repo(partition_path),
                path_or_fileobj=parquet_bytes(partition_df.drop(columns=["_partition_path"])),
            )
        )


def retry_delay_seconds(error: Exception) -> int:
    """Choose a wait time after a Hugging Face rate-limit response."""
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


def operation_chunks(operations: list, chunk_size: int) -> list[list]:
    """Split staged files into smaller commits to reduce Hugging Face preupload pressure."""
    if chunk_size <= 0:
        return [operations]
    return [operations[index : index + chunk_size] for index in range(0, len(operations), chunk_size)]


def create_commit_with_retries(api, hf_repo_id: str, operations: list, commit_message: str) -> None:
    """Create one Hugging Face commit with explicit retry handling for 429 responses."""
    from huggingface_hub.errors import HfHubHTTPError

    max_retries = int(os.getenv("HF_COMMIT_RETRIES", "3"))

    for attempt in range(1, max_retries + 1):
        try:
            api.create_commit(
                repo_id=hf_repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
            )
            return
        except HfHubHTTPError as error:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            if status_code != 429 or attempt == max_retries:
                raise

            delay = retry_delay_seconds(error)
            print(f"Hugging Face rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}.")
            time.sleep(delay)


def upload_operations_to_hugging_face(operations: list) -> None:
    """Upload staged parquet files to Hugging Face in rate-limit-friendly commits."""
    from huggingface_hub import HfApi

    hf_token, hf_repo_id = hf_config()

    if not hf_token or not hf_repo_id:
        print(f"Skipping Hugging Face upload for raw/{PARTITION_PATH}: HF_TOKEN or HF_REPO_ID is not set.")
        return

    if not operations:
        print("Skipping Hugging Face upload: no files to commit.")
        return

    api = HfApi(token=hf_token)
    chunk_size = int(os.getenv("HF_UPLOAD_OPERATION_CHUNK_SIZE", "12"))
    chunk_delay = float(os.getenv("HF_UPLOAD_CHUNK_DELAY_SECONDS", "20"))
    chunks = operation_chunks(operations, chunk_size)

    for index, chunk in enumerate(chunks, start=1):
        commit_message = f"Upload scraper batch for {PARTITION_PATH} ({index}/{len(chunks)})"
        create_commit_with_retries(api, hf_repo_id, chunk, commit_message)
        print(f"Uploaded Hugging Face commit {index}/{len(chunks)} with {len(chunk)} files.")

        if index < len(chunks) and chunk_delay > 0:
            print(f"Waiting {chunk_delay} seconds before next Hugging Face commit chunk.")
            time.sleep(chunk_delay)

    print(f"Uploaded {len(operations)} files to Hugging Face in {len(chunks)} commit(s).")
