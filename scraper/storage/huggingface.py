from datetime import datetime, timezone
from pathlib import Path

from scraper.core.config import HF_INDEX_PATH, PARTITION_PATH


def hf_config() -> tuple[str | None, str | None]:
    import os

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
