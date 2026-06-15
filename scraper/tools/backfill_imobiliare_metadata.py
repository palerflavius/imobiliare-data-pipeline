import argparse
from pathlib import Path

import pandas as pd

from scraper.sites.imobiliare.metadata import backfill_dataframe
from scraper.storage.huggingface import parquet_bytes


def has_changed(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """Return true when metadata repair changed any cell."""
    return not before.fillna("").astype(str).equals(after.fillna("").astype(str))


def parquet_files(path: Path) -> list[Path]:
    """Return one parquet file or all parquet files under a directory."""
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.parquet"))


def backfill_local(input_path: Path, output_path: Path | None, overwrite: bool) -> None:
    """Repair metadata in local parquet files."""
    files = parquet_files(input_path)
    changed = 0

    for file_path in files:
        df = pd.read_parquet(file_path)
        fixed = backfill_dataframe(df)
        if not has_changed(df, fixed):
            continue

        # Preserve directory layout when writing repaired files to a separate output root.
        if overwrite:
            target = file_path
        elif input_path.is_file():
            target = output_path / file_path.name
        else:
            target = output_path / file_path.relative_to(input_path)

        target.parent.mkdir(parents=True, exist_ok=True)
        fixed.to_parquet(target, index=False)
        changed += 1
        print(f"Backfilled {file_path} -> {target}")

    print(f"Changed parquet files: {changed}/{len(files)}")


def backfill_huggingface(repo_id: str, prefix: str, commit: bool, max_files: int | None) -> None:
    """Repair metadata in Hugging Face parquet files, optionally committing changes."""
    import os
    from huggingface_hub import CommitOperationAdd, HfApi, HfFileSystem

    token = os.getenv("HF_TOKEN")
    fs = HfFileSystem(token=token)
    api = HfApi(token=token)
    paths = sorted(fs.glob(f"datasets/{repo_id}/{prefix}/**/*.parquet"))
    operations = []

    for path in paths:
        # Index files are derived state and should be managed by the main scraper.
        if "/index/" in path:
            continue
        if max_files is not None and len(operations) >= max_files:
            break

        with fs.open(path, "rb") as handle:
            df = pd.read_parquet(handle)

        fixed = backfill_dataframe(df)
        if not has_changed(df, fixed):
            continue

        path_in_repo = path.split(f"datasets/{repo_id}/", 1)[1]
        operations.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=parquet_bytes(fixed)))
        print(f"Will backfill {path_in_repo}")

    if not operations:
        print("No parquet files need backfill.")
        return

    if not commit:
        print(f"Dry run only. Files that would be committed: {len(operations)}")
        return

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Backfill imobiliare.ro metadata for {prefix}",
    )
    print(f"Committed {len(operations)} repaired parquet files.")


def main() -> None:
    """Parse CLI options and run either local or Hugging Face backfill."""
    parser = argparse.ArgumentParser(description="Backfill missing imobiliare.ro metadata from listing_url.")
    parser.add_argument("--input", type=Path, help="Local parquet file or directory.")
    parser.add_argument("--output", type=Path, help="Output directory for local repair.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite local parquet files in place.")
    parser.add_argument("--hf-repo-id", help="Hugging Face dataset repo id.")
    parser.add_argument("--hf-prefix", default="raw/site=imobiliare.ro", help="HF repo prefix to scan.")
    parser.add_argument("--commit", action="store_true", help="Commit repaired HF parquet files. Default is dry-run.")
    parser.add_argument("--max-files", type=int, help="Limit changed HF files processed in one run.")
    args = parser.parse_args()

    if args.hf_repo_id:
        backfill_huggingface(args.hf_repo_id, args.hf_prefix, args.commit, args.max_files)
        return

    if not args.input:
        raise SystemExit("Use --input for local files or --hf-repo-id for Hugging Face.")
    if not args.overwrite and not args.output:
        raise SystemExit("Use --output or --overwrite for local backfill.")

    backfill_local(args.input, args.output, args.overwrite)


if __name__ == "__main__":
    main()
