import unittest

from scraper.build_matrix import start_url


class BuildMatrixTests(unittest.TestCase):
    """Cover site-specific result URL generation."""

    def test_builds_imobiliare_brasov_url(self) -> None:
        self.assertEqual(
            start_url("imobiliare.ro", "brasov", "brasov", None, "sale", "apartments"),
            "https://www.imobiliare.ro/vanzare-apartamente/judetul-brasov/brasov",
        )

    def test_builds_storia_brasov_url(self) -> None:
        self.assertEqual(
            start_url("storia.ro", "brasov", "brasov", None, "sale", "apartments"),
            "https://www.storia.ro/ro/rezultate/vanzare/apartament/brasov/brasov",
        )

    def test_builds_storia_bucharest_sector_url(self) -> None:
        self.assertEqual(
            start_url("storia.ro", "bucuresti", "bucuresti", "sector-1", "rent", "houses-villas"),
            "https://www.storia.ro/ro/rezultate/inchiriere/casa/bucuresti/sectorul-1",
        )


if __name__ == "__main__":
    unittest.main()
