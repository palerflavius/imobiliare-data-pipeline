import sys

from scraper.core.config import UPSTREAM_BLOCKED_EXIT_CODE
from scraper.core.http_client import UpstreamBlockedError
from scraper.pipelines.generic import run_site_pipeline
from scraper.sites.registry import get_site_adapter


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    try:
        run_site_pipeline(get_site_adapter())
    except UpstreamBlockedError as error:
        print(f"Upstream blocked scraper request: {error}", flush=True)
        raise SystemExit(UPSTREAM_BLOCKED_EXIT_CODE) from error
