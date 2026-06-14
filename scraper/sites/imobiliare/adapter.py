from scraper.core import config
from scraper.core.config import safe_path_part
from scraper.sites.imobiliare import details, parser
from scraper.sites.imobiliare.metadata import backfill_dataframe, backfill_listing_metadata


BASE_URL = "https://www.imobiliare.ro"


def offer_property_path() -> str:
    if config.OFFER_TYPE == "sale" and config.PROPERTY_TYPE == "apartments":
        return "vanzare-apartamente"
    return f"{safe_path_part(config.OFFER_TYPE)}-{safe_path_part(config.PROPERTY_TYPE)}"


def default_start_url() -> str:
    path = offer_property_path()

    if config.AREA_SLUG:
        return f"{BASE_URL}/{path}/{safe_path_part(config.CITY_SLUG)}/{safe_path_part(config.AREA_SLUG)}"

    if config.CITY_SLUG and config.CITY_SLUG != "all":
        return f"{BASE_URL}/{path}/judetul-{safe_path_part(config.COUNTY_SLUG)}/{safe_path_part(config.CITY_SLUG)}"

    return f"{BASE_URL}/{path}/judetul-{safe_path_part(config.COUNTY_SLUG)}"


class ImobiliareSiteAdapter:
    name = "imobiliare.ro"

    def __init__(self, start_url: str | None = None) -> None:
        self.start_url = start_url or config.START_URL or default_start_url()

    def page_url(self, base_url: str, page_number: int) -> str:
        return parser.page_url(base_url, page_number)

    def extract_last_page(self, html_text: str) -> int:
        return parser.extract_last_page(html_text)

    def parse_listings(self, html_text: str, page_url: str) -> list[dict]:
        return [backfill_listing_metadata(listing) for listing in parser.parse_listings(html_text, page_url)]

    def listing_event_key(self, listing: dict) -> str:
        return parser.listing_event_key(listing)

    def resolve_detail_urls(self, df):
        return backfill_dataframe(details.resolve_detail_urls(df))
