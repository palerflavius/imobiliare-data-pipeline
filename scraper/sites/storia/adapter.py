from scraper.core import config
from scraper.core.config import safe_path_part
from scraper.sites.storia import parser


BASE_URL = "https://www.storia.ro"


def offer_property_path() -> str:
    """Map configured offer/property values to the storia.ro URL path."""
    paths = {
        ("sale", "apartments"): "vanzare/apartament",
        ("sale", "houses-villas"): "vanzare/casa",
        ("sale", "lands"): "vanzare/teren",
        ("rent", "apartments"): "inchiriere/apartament",
        ("rent", "houses-villas"): "inchiriere/casa",
    }
    try:
        return paths[(config.OFFER_TYPE, config.PROPERTY_TYPE)]
    except KeyError as error:
        raise ValueError(
            f"Unsupported search: offer_type={config.OFFER_TYPE}, property_type={config.PROPERTY_TYPE}"
        ) from error


def storia_area_slug(area_slug: str) -> str:
    """Convert local area slugs to Storia URL slugs."""
    if area_slug.startswith("sector-"):
        return area_slug.replace("sector-", "sectorul-", 1)
    return area_slug


def default_start_url() -> str:
    """Build the start URL when the matrix does not provide START_URL."""
    path = offer_property_path()

    if config.COUNTY_SLUG == "bucuresti":
        url = f"{BASE_URL}/ro/rezultate/{path}/bucuresti"
        if config.AREA_SLUG:
            url = f"{url}/{storia_area_slug(safe_path_part(config.AREA_SLUG))}"
        return url

    if config.CITY_SLUG and config.CITY_SLUG != "all":
        return (
            f"{BASE_URL}/ro/rezultate/{path}/"
            f"{safe_path_part(config.COUNTY_SLUG)}/{safe_path_part(config.CITY_SLUG)}"
        )

    return f"{BASE_URL}/ro/rezultate/{path}/{safe_path_part(config.COUNTY_SLUG)}"


class StoriaSiteAdapter:
    """Adapter that connects storia.ro parsing details to the generic pipeline."""

    name = "storia.ro"

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
        """Parse listing rows from Storia SSR data."""
        return parser.parse_listings(html_text, page_url)

    def listing_event_key(self, listing: dict) -> str:
        """Use the site-specific event key for price-change detection."""
        return parser.listing_event_key(listing)

    def resolve_detail_urls(self, df):
        """Storia result pages already expose canonical listing URLs and metadata."""
        return df
