from scraper.core.config import SITE_NAME
from scraper.core.site import SiteAdapter
from scraper.sites.imobiliare.adapter import ImobiliareSiteAdapter


def get_site_adapter(site_name: str | None = None) -> SiteAdapter:
    selected_site = site_name or SITE_NAME

    if selected_site == "imobiliare.ro":
        return ImobiliareSiteAdapter()

    raise ValueError(f"Unsupported SITE_NAME: {selected_site}")
