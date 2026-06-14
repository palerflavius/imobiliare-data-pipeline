import sys

from scraper.pipelines.generic import run_site_pipeline
from scraper.sites.registry import get_site_adapter


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    run_site_pipeline(get_site_adapter())
