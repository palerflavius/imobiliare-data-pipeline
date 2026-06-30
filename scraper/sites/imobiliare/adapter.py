from scraper.core import config
from scraper.core.config import safe_path_part
from scraper.sites.imobiliare import details, parser
from scraper.sites.imobiliare.metadata import backfill_dataframe, backfill_listing_metadata


BASE_URL = "https://www.imobiliare.ro"


def offer_property_path() -> str:
    """Map configured offer/property values to the imobiliare.ro URL path."""
    paths = {
        ("sale", "apartments"): "vanzare-apartamente",
        ("sale", "houses-villas"): "vanzare-case-vile",
        ("sale", "lands"): "vanzare-terenuri",
        ("rent", "apartments"): "inchirieri-apartamente",
        ("rent", "houses-villas"): "inchirieri-case-vile",
    }
    try:
        return paths[(config.OFFER_TYPE, config.PROPERTY_TYPE)]
    except KeyError as error:
        raise ValueError(
            f"Unsupported search: offer_type={config.OFFER_TYPE}, property_type={config.PROPERTY_TYPE}"
        ) from error


def default_start_url() -> str:
    """Build the start URL when the matrix does not provide START_URL."""
    path = offer_property_path()

    # Area pages use /city/area, while county/city pages use /judetul-county/city.
    if config.AREA_SLUG:
        return f"{BASE_URL}/{path}/{safe_path_part(config.CITY_SLUG)}/{safe_path_part(config.AREA_SLUG)}"

    if config.CITY_SLUG and config.CITY_SLUG != "all":
        return f"{BASE_URL}/{path}/judetul-{safe_path_part(config.COUNTY_SLUG)}/{safe_path_part(config.CITY_SLUG)}"

    return f"{BASE_URL}/{path}/judetul-{safe_path_part(config.COUNTY_SLUG)}"


class ImobiliareSiteAdapter:
    """Adapter that connects imobiliare.ro parsing details to the generic pipeline."""

    name = "imobiliare.ro"

    def __init__(self, start_url: str | None = None) -> None:
        """Store the target start URL for this scraper run."""
        self.start_url = start_url or config.START_URL or default_start_url()

    def page_url(self, base_url: str, page_number: int) -> str:
        """Delegate pagination URL generation to the site parser."""
        return parser.page_url(base_url, page_number)

    def extract_last_page(self, html_text: str) -> int:
        """Delegate pagination detection to the site parser."""
        return parser.extract_last_page(html_text)

    def parse_listings(self, html_text: str, page_url: str) -> list[dict]:
        """Parse listing cards and fill any metadata inferable from URLs."""
        return [
            backfill_listing_metadata(listing, allow_location_partition_fallback=False)
            for listing in parser.parse_listings(html_text, page_url)
        ]

    def listing_event_key(self, listing: dict) -> str:
        """Use the site-specific event key for price-change detection."""
        return parser.listing_event_key(listing)

    def resolve_detail_urls(self, df):
        """Resolve detail pages, then backfill metadata from final URLs and addresses."""
        return backfill_dataframe(details.resolve_detail_urls(df))
