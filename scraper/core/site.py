from typing import Protocol


class SiteAdapter(Protocol):
    name: str
    start_url: str

    def page_url(self, base_url: str, page_number: int) -> str:
        ...

    def extract_last_page(self, html_text: str) -> int:
        ...

    def parse_listings(self, html_text: str, page_url: str) -> list[dict]:
        ...

    def listing_event_key(self, listing: dict) -> str:
        ...

    def resolve_detail_urls(self, df):
        ...
