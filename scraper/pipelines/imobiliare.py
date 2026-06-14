from scraper.pipelines.generic import run_site_pipeline
from scraper.sites.imobiliare.adapter import ImobiliareSiteAdapter


def run_pipeline() -> None:
    run_site_pipeline(ImobiliareSiteAdapter())
