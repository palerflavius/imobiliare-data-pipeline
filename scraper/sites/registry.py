from scraper.core.config import SITE_NAME
from scraper.core.site import SiteAdapter
from scraper.sites.imobiliare.adapter import ImobiliareSiteAdapter


def get_site_adapter(site_name: str | None = None) -> SiteAdapter:
    """Return the adapter implementation for the selected real-estate site."""
    selected_site = site_name or SITE_NAME

    # Keeping site selection centralized makes future sites plug into the same pipeline.
    if selected_site == "imobiliare.ro":
        return ImobiliareSiteAdapter()

    raise ValueError(f"Unsupported SITE_NAME: {selected_site}")
