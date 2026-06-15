from typing import Protocol


class SiteAdapter(Protocol):
    """Minimal interface required by the generic scraping pipeline."""

    name: str
    start_url: str

    def page_url(self, base_url: str, page_number: int) -> str:
        """Return the URL for a paginated listing page."""
        ...

    def extract_last_page(self, html_text: str) -> int:
        """Detect how many listing pages should be scraped."""
        ...

    def parse_listings(self, html_text: str, page_url: str) -> list[dict]:
        """Extract listing rows from a listing-page HTML document."""
        ...

    def listing_event_key(self, listing: dict) -> str:
        """Return a stable key for detecting new or changed listing events."""
        ...

    def resolve_detail_urls(self, df):
        """Enrich listing rows using their detail pages."""
        ...
