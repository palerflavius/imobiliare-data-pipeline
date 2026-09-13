import os
import unittest
from unittest.mock import patch

from scraper.core.config import UPSTREAM_BLOCKED_EXIT_CODE
from scraper import run_target


SEARCHES_JSON = """[
  {
    "offer_type": "sale",
    "property_type": "apartments",
    "start_url": "https://www.imobiliare.ro/vanzare-apartamente/judetul-brasov/brasov"
  }
]"""


class RunTargetTests(unittest.TestCase):
    """Cover aggregation of scraper subprocess outcomes."""

    def test_upstream_blocked_search_is_skipped(self) -> None:
        with patch.dict(os.environ, {"SEARCHES_JSON": SEARCHES_JSON}):
            with patch.object(run_target, "run_search", return_value=UPSTREAM_BLOCKED_EXIT_CODE):
                run_target.main()

    def test_non_blocked_search_failure_exits(self) -> None:
        with patch.dict(os.environ, {"SEARCHES_JSON": SEARCHES_JSON}):
            with patch.object(run_target, "run_search", return_value=1):
                with self.assertRaises(SystemExit):
                    run_target.main()


if __name__ == "__main__":
    unittest.main()
