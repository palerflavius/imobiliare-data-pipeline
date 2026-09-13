import json
import unittest

from scraper.sites.storia import parser


def storia_html(payload: dict) -> str:
    """Wrap a payload in the Next.js script shape used by Storia."""
    return f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'


class StoriaParserTests(unittest.TestCase):
    """Cover Storia SSR payload parsing."""

    def test_extracts_last_page_from_next_data(self) -> None:
        html = storia_html({"props": {"pageProps": {"data": {"searchAds": {"pagination": {"totalPages": 7}}}}}})
        self.assertEqual(parser.extract_last_page(html), 7)

    def test_parses_listing_item(self) -> None:
        html = storia_html(
            {
                "props": {
                    "pageProps": {
                        "data": {
                            "searchAds": {
                                "items": [
                                    {
                                        "id": 123,
                                        "title": "Apartament 2 camere Centrul Civic",
                                        "slug": "apartament-2-camere-centrul-civic-IDabc",
                                        "href": "[lang]/ad/apartament-2-camere-centrul-civic-IDabc",
                                        "totalPrice": {"value": 101000, "currency": "EUR"},
                                        "roomsNumber": "TWO",
                                        "floorNumber": "SECOND",
                                        "areaInSquareMeters": 50,
                                        "location": {
                                            "address": {
                                                "street": {"name": "Calea Bucuresti"},
                                                "city": {"name": "Brasov"},
                                                "province": {"name": "Brasov (judet)"},
                                            },
                                            "reverseGeocoding": {
                                                "locations": [
                                                    {"id": "brasov", "name": "Brasov", "locationLevel": "county"},
                                                    {
                                                        "id": "brasov/brasov",
                                                        "name": "Brasov",
                                                        "locationLevel": "county_capital",
                                                    },
                                                    {
                                                        "id": "brasov/brasov/tractorul",
                                                        "name": "Tractorul",
                                                        "locationLevel": "district",
                                                    },
                                                ]
                                            },
                                        },
                                        "agency": {"name": "Test Agency"},
                                        "advertOwner": {"name": "Test Agent"},
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        )

        listings = parser.parse_listings(html, "https://www.storia.ro/ro/rezultate/vanzare/apartament/brasov/brasov")

        self.assertEqual(len(listings), 1)
        self.assertEqual(listings[0]["source"], "storia.ro")
        self.assertEqual(listings[0]["price_eur"], 101000.0)
        self.assertEqual(listings[0]["rooms"], 2.0)
        self.assertEqual(listings[0]["area"], "tractorul")
        self.assertEqual(listings[0]["listing_url"], "https://www.storia.ro/ro/ad/apartament-2-camere-centrul-civic-IDabc")


if __name__ == "__main__":
    unittest.main()
