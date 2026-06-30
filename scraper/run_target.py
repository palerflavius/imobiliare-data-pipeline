import json
import os
import subprocess
import sys


def configured_searches() -> list[dict]:
    """Read the searches attached to the current geographic matrix target."""
    raw_searches = os.getenv("SEARCHES_JSON", "")
    if not raw_searches:
        raise RuntimeError("SEARCHES_JSON is required.")

    searches = json.loads(raw_searches)
    if not isinstance(searches, list) or not searches:
        raise RuntimeError("SEARCHES_JSON must contain a non-empty list.")
    return searches


def search_label(search: dict) -> str:
    """Return a concise label for GitHub Actions log groups."""
    return f"{search['offer_type']} / {search['property_type']}"


def run_search(search: dict) -> int:
    """Run one offer/property search in a fresh Python process."""
    env = os.environ.copy()
    env["OFFER_TYPE"] = search["offer_type"]
    env["PROPERTY_TYPE"] = search["property_type"]
    env["START_URL"] = search["start_url"]

    label = search_label(search)
    print(f"::group::Scrape {label}", flush=True)
    print(f"Start URL: {search['start_url']}", flush=True)
    try:
        result = subprocess.run([sys.executable, "-m", "scraper.main"], env=env, check=False)
        return result.returncode
    finally:
        print("::endgroup::", flush=True)


def main() -> None:
    """Run all searches for one geographic target and report combined failures."""
    failures = []

    for search in configured_searches():
        return_code = run_search(search)
        if return_code:
            failures.append(f"{search_label(search)} (exit {return_code})")

    if failures:
        raise SystemExit(f"Searches failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
