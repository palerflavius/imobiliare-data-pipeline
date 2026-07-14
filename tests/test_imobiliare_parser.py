import unittest

from scraper.sites.imobiliare.parser import extract_price_eur


class ExtractPriceEurTests(unittest.TestCase):
    """Cover both rental prices and thousands-formatted sale prices."""

    def test_extracts_three_digit_rent(self) -> None:
        self.assertEqual(extract_price_eur("380 € / lună"), 380.0)

    def test_extracts_thousands_formatted_rent(self) -> None:
        self.assertEqual(extract_price_eur("1.490 € / lună"), 1490.0)

    def test_extracts_sale_price(self) -> None:
        self.assertEqual(extract_price_eur("240.000 EUR"), 240000.0)

    def test_extracts_real_euro_symbol_after_year(self) -> None:
        text = "FINALIZARE 2026\n135.360 €\nDezvoltator"
        self.assertEqual(extract_price_eur(text), 135360.0)


if __name__ == "__main__":
    unittest.main()
