import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.indicators import (
    INDICATOR_COLUMNS,
    calculate_bollinger_bands,
    calculate_distance_percent,
    calculate_indicators,
    calculate_percentile_rank,
    prepare_exchange_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_exchange.csv"


class IndicatorEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_data = pd.read_csv(SAMPLE_DATA_PATH)

    def test_sample_data_calculates_all_expected_indicators(self) -> None:
        result = calculate_indicators(self.sample_data)
        latest = result.iloc[-1]
        rates = result["rate"]

        expected_sma60 = rates.tail(60).mean()
        expected_sma120 = rates.tail(120).mean()
        expected_percentile = (
            rates.tail(180).rank(method="average", pct=True).iloc[-1] * 100
        )
        bollinger_window = rates.tail(20)
        expected_middle = bollinger_window.mean()
        expected_std = bollinger_window.std(ddof=0)

        self.assertEqual(len(result), 200)
        self.assertTrue(set(INDICATOR_COLUMNS).issubset(result.columns))
        self.assertAlmostEqual(latest["SMA60"], expected_sma60)
        self.assertAlmostEqual(latest["SMA120"], expected_sma120)
        self.assertAlmostEqual(
            latest["SMA60_distance_pct"],
            (latest["rate"] - expected_sma60) / expected_sma60 * 100,
        )
        self.assertAlmostEqual(
            latest["SMA120_distance_pct"],
            (latest["rate"] - expected_sma120) / expected_sma120 * 100,
        )
        self.assertAlmostEqual(latest["percentile_rank_180"], expected_percentile)
        self.assertAlmostEqual(latest["BB_middle"], expected_middle)
        self.assertAlmostEqual(latest["BB_upper"], expected_middle + 2 * expected_std)
        self.assertAlmostEqual(latest["BB_lower"], expected_middle - 2 * expected_std)

    def test_prepares_dates_sorts_rows_and_does_not_modify_input(self) -> None:
        original = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-01"],
                "rate": [1350, 1340],
            }
        )

        result = calculate_indicators(original)

        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2025-01-01", "2025-01-02"],
        )
        self.assertEqual(result["rate"].tolist(), [1340.0, 1350.0])
        self.assertNotIn("SMA60", original.columns)
        self.assertEqual(original["date"].tolist(), ["2025-01-02", "2025-01-01"])

    def test_insufficient_history_returns_nan_indicators(self) -> None:
        short_data = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=19, freq="D"),
                "rate": range(1300, 1319),
            }
        )

        result = calculate_indicators(short_data)

        self.assertTrue(result[list(INDICATOR_COLUMNS)].isna().all().all())

    def test_missing_and_non_finite_rates_are_handled_as_nan(self) -> None:
        data = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=200, freq="D"),
                "rate": [1300.0] * 197 + [None, "not-a-rate", np.inf],
            }
        )

        result = calculate_indicators(data)

        self.assertTrue(result["rate"].tail(3).isna().all())
        self.assertTrue(result[list(INDICATOR_COLUMNS)].tail(3).isna().all().all())

    def test_zero_average_produces_nan_distance(self) -> None:
        distance = calculate_distance_percent(
            pd.Series([100.0]),
            pd.Series([0.0]),
        )

        self.assertTrue(pd.isna(distance.iloc[0]))

    def test_percentile_rank_uses_average_rank_for_ties(self) -> None:
        rates = pd.Series([100.0] * 180)

        percentile = calculate_percentile_rank(rates)

        self.assertAlmostEqual(percentile.iloc[-1], 90.5 / 180 * 100)

    def test_bollinger_bands_require_a_complete_window(self) -> None:
        bands = calculate_bollinger_bands(pd.Series(range(1, 20)))

        self.assertTrue(bands.isna().all().all())

    def test_rejects_missing_columns_and_invalid_dates(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns: rate"):
            prepare_exchange_data(pd.DataFrame({"date": ["2025-01-01"]}))

        with self.assertRaisesRegex(ValueError, "invalid date"):
            prepare_exchange_data(
                pd.DataFrame({"date": ["invalid"], "rate": [1300]})
            )


if __name__ == "__main__":
    unittest.main()
