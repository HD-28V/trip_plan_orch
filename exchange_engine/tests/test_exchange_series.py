import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.exchange_series import merge_historical_and_latest
from src.indicators import calculate_indicators


class ExchangeSeriesMergeTests(unittest.TestCase):
    def test_sorts_selects_currency_and_latest_wins_duplicate_date(self) -> None:
        historical = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-01"],
                "rate": [1350.0, 1340.0],
            }
        )
        latest = pd.DataFrame(
            {
                "date": ["2025-01-03", "2025-01-02", "2025-01-03"],
                "currency": ["EUR", "USD", "USD"],
                "rate": [1500.0, 1335.0, 1345.0],
            }
        )
        historical_before = historical.copy(deep=True)
        latest_before = latest.copy(deep=True)

        result = merge_historical_and_latest(historical, latest, "USD")

        self.assertEqual(result.columns.tolist(), ["date", "rate"])
        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2025-01-01", "2025-01-02", "2025-01-03"],
        )
        self.assertEqual(result["rate"].tolist(), [1340.0, 1335.0, 1345.0])
        assert_frame_equal(historical, historical_before)
        assert_frame_equal(latest, latest_before)

    def test_rejects_historical_currency_mismatch(self) -> None:
        historical = pd.DataFrame(
            {
                "date": ["2025-01-01"],
                "rate": [1350.0],
                "currency": ["EUR"],
            }
        )
        latest = pd.DataFrame(
            {
                "date": ["2025-01-02"],
                "currency": ["USD"],
                "rate": [1340.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "different currency"):
            merge_historical_and_latest(historical, latest, "USD")

    def test_merged_result_can_feed_indicator_engine(self) -> None:
        historical = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=180, freq="D"),
                "rate": range(1300, 1480),
            }
        )
        latest = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-06-30")],
                "currency": ["USD"],
                "rate": [1400.0],
            }
        )

        merged = merge_historical_and_latest(historical, latest, "USD")
        indicators = calculate_indicators(merged)

        self.assertFalse(pd.isna(indicators.iloc[-1]["SMA120"]))
        self.assertEqual(indicators.columns[:2].tolist(), ["date", "rate"])


if __name__ == "__main__":
    unittest.main()
