import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.providers import (
    CSVExchangeRateProvider,
    InvalidExchangeRateDataError,
    UnsupportedCurrencyError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_exchange.csv"


class CSVExchangeRateProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = CSVExchangeRateProvider(
            SAMPLE_DATA_PATH,
            foreign_currency="USD",
        )

    def test_returns_only_standard_columns_sorted_by_date(self) -> None:
        source_data = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-01", "2025-01-02"],
                "rate": [200.0, 100.0, 220.0],
                "unused": [1, 2, 3],
            }
        )
        provider = CSVExchangeRateProvider(
            "mocked.csv",
            foreign_currency="JPY",
            foreign_units_per_rate=100,
        )

        with patch("src.providers.pd.read_csv", return_value=source_data):
            result = provider.fetch_daily_rates(
                "jpy",
                "2025-01-01",
                "2025-01-02",
            )

        self.assertEqual(result.columns.tolist(), ["date", "rate"])
        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2025-01-01", "2025-01-02"],
        )
        self.assertEqual(result["rate"].tolist(), [1.0, 2.2])

    def test_filters_the_sample_csv_by_inclusive_date_range(self) -> None:
        result = self.provider.fetch_daily_rates(
            "USD",
            "2025-02-01",
            "2025-02-10",
        )

        self.assertEqual(len(result), 10)
        self.assertEqual(result.iloc[0]["date"], pd.Timestamp("2025-02-01"))
        self.assertEqual(result.iloc[-1]["date"], pd.Timestamp("2025-02-10"))

    def test_rejects_unsupported_or_malformed_currency_codes(self) -> None:
        for currency in ("EUR", "US", "123"):
            with self.subTest(currency=currency):
                with self.assertRaises(UnsupportedCurrencyError):
                    self.provider.fetch_daily_rates(
                        currency,
                        "2025-01-01",
                        "2025-01-02",
                    )

    def test_rejects_invalid_rate_data(self) -> None:
        invalid_values = (None, "not-a-rate", 0, -1, np.inf)

        for invalid_value in invalid_values:
            with self.subTest(rate=invalid_value):
                source_data = pd.DataFrame(
                    {"date": ["2025-01-01"], "rate": [invalid_value]}
                )
                with patch(
                    "src.providers.pd.read_csv",
                    return_value=source_data,
                ):
                    with self.assertRaises(InvalidExchangeRateDataError):
                        self.provider.fetch_daily_rates(
                            "USD",
                            "2025-01-01",
                            "2025-01-01",
                        )

    def test_does_not_modify_the_input_csv(self) -> None:
        contents_before = SAMPLE_DATA_PATH.read_bytes()

        self.provider.fetch_daily_rates(
            "USD",
            "2025-01-01",
            "2025-07-19",
        )

        self.assertEqual(SAMPLE_DATA_PATH.read_bytes(), contents_before)

    def test_rejects_invalid_source_dates_and_date_ranges(self) -> None:
        source_data = pd.DataFrame({"date": ["invalid"], "rate": [1300]})
        with patch("src.providers.pd.read_csv", return_value=source_data):
            with self.assertRaises(InvalidExchangeRateDataError):
                self.provider.fetch_daily_rates(
                    "USD",
                    "2025-01-01",
                    "2025-01-01",
                )

        with self.assertRaisesRegex(ValueError, "start_date"):
            self.provider.fetch_daily_rates(
                "USD",
                "2025-01-02",
                "2025-01-01",
            )


if __name__ == "__main__":
    unittest.main()
