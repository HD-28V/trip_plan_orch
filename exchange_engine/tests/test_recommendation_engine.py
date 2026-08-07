import unittest
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from src.dday_policy import DdayBand, DdayUrgency, DdayValidationError, evaluate_production_dday_signal
from src.production_config import evaluate_production_signal
from src.recommendation_engine import (
    RecommendationValidationError,
    _validate_metadata_consistency,
    evaluate_exchange_recommendation,
)
from src.signal_engine import Signal
from src.split_exchange import SplitExchangeValidationError, evaluate_production_split_exchange


TOTAL = Decimal("1000000")


def make_indicator_row(**overrides: object) -> pd.Series:
    values: dict[str, object] = {
        "date": pd.Timestamp("2026-08-07"),
        "rate": 102.0,
        "SMA60": 103.0,
        "SMA120": 104.0,
        "SMA60_distance_pct": 0.0,
        "SMA120_distance_pct": 0.0,
        "percentile_rank_180": 50.0,
        "BB_middle": 105.0,
        "BB_upper": 110.0,
        "BB_lower": 100.0,
    }
    values.update(overrides)
    return pd.Series(values)


def row_for_signal(signal: Signal, scale: float = 1.0) -> pd.Series:
    values: dict[str, object] = {
        "rate": 100.0 * scale,
        "SMA60": 101.0 * scale,
        "SMA120": 102.0 * scale,
        "BB_middle": 105.0 * scale,
        "BB_upper": 110.0 * scale,
        "BB_lower": 100.0 * scale,
        "SMA60_distance_pct": 0.0,
        "SMA120_distance_pct": 0.0,
        "percentile_rank_180": 50.0,
    }
    if signal is Signal.WATCH:
        values["SMA60_distance_pct"] = -1.5
    elif signal is Signal.GOOD:
        values.update(
            SMA60_distance_pct=-1.5,
            SMA120_distance_pct=-3.0,
            percentile_rank_180=20.0,
        )
    elif signal is Signal.STRONG:
        values.update(
            SMA60_distance_pct=-2.0,
            SMA120_distance_pct=-4.0,
            percentile_rank_180=20.0,
        )
    elif signal is Signal.WAIT:
        values["rate"] = 102.0 * scale
    return make_indicator_row(**values)


class RecommendationOrchestrationTests(unittest.TestCase):
    def test_matches_the_existing_three_production_entry_points(self) -> None:
        row = row_for_signal(Signal.GOOD)
        expected_market = evaluate_production_signal(row)
        expected_dday = evaluate_production_dday_signal(expected_market, 14)
        expected_split = evaluate_production_split_exchange(
            expected_dday, TOTAL, Decimal("300000")
        )

        result = evaluate_exchange_recommendation(
            row, 14, TOTAL, Decimal("300000")
        )

        self.assertEqual(result.market_result, expected_market)
        self.assertEqual(result.dday_result, expected_dday)
        self.assertEqual(result.split_result, expected_split)
        self.assertEqual(result.base_signal, Signal.GOOD)
        self.assertEqual(result.adjusted_signal, Signal.STRONG)
        self.assertIn("sma60_condition", result.satisfied_conditions)
        self.assertEqual(result.trace.dday_reason, expected_dday.reason)
        self.assertEqual(result.trace.split_reason, expected_split.reason)

    def test_all_market_signal_levels_and_required_dday_boundaries(self) -> None:
        expected_bands = {
            90: (DdayBand.D61_PLUS, DdayUrgency.LOW),
            30: (DdayBand.D15_TO_D30, DdayUrgency.HIGH),
            14: (DdayBand.D7_TO_D14, DdayUrgency.HIGH),
            7: (DdayBand.D7_TO_D14, DdayUrgency.HIGH),
            1: (DdayBand.D1_TO_D6, DdayUrgency.CRITICAL),
            0: (DdayBand.D0, DdayUrgency.DEADLINE),
        }
        for signal in Signal:
            for days, (band, urgency) in expected_bands.items():
                with self.subTest(signal=signal.value, days=days):
                    result = evaluate_exchange_recommendation(
                        row_for_signal(signal), days, TOTAL, Decimal("300000")
                    )
                    self.assertEqual(result.base_signal, signal)
                    self.assertEqual(result.dday_band, band)
                    self.assertEqual(result.urgency, urgency)

    def test_cumulative_amount_cases_are_preserved(self) -> None:
        cases = (
            (Decimal("0"), Decimal("500000")),
            (Decimal("300000"), Decimal("200000")),
            (Decimal("500000"), Decimal("0")),
            (Decimal("800000"), Decimal("0")),
            (Decimal("1000000"), Decimal("0")),
        )
        for already, expected_additional in cases:
            with self.subTest(already=already):
                result = evaluate_exchange_recommendation(
                    row_for_signal(Signal.GOOD), 90, TOTAL, already
                )
                self.assertEqual(result.target_cumulative_ratio, Decimal("0.50"))
                self.assertEqual(result.recommended_additional_krw, expected_additional)

    def test_deadline_completes_the_remaining_budget(self) -> None:
        result = evaluate_exchange_recommendation(
            row_for_signal(Signal.WAIT), 0, TOTAL, Decimal("300000")
        )
        self.assertEqual(result.target_cumulative_ratio, Decimal("1.00"))
        self.assertEqual(result.recommended_additional_krw, Decimal("700000"))
        self.assertEqual(result.remaining_after_recommendation_krw, Decimal("0"))

    def test_scale_independence_for_usd_jpy_and_eur_style_rates(self) -> None:
        baseline = evaluate_exchange_recommendation(
            row_for_signal(Signal.GOOD, 1.0), 30, TOTAL, Decimal("300000")
        )
        for currency, scale in (("USD", 1400.0), ("JPY", 9.0), ("EUR", 1600.0)):
            with self.subTest(currency=currency):
                result = evaluate_exchange_recommendation(
                    row_for_signal(Signal.GOOD, scale), 30, TOTAL, Decimal("300000")
                )
                self.assertEqual(result.base_signal, baseline.base_signal)
                self.assertEqual(result.adjusted_signal, baseline.adjusted_signal)
                self.assertEqual(result.target_cumulative_ratio, baseline.target_cumulative_ratio)
                self.assertEqual(result.recommended_additional_krw, baseline.recommended_additional_krw)


class RecommendationContractTests(unittest.TestCase):
    def test_nan_indicators_are_reported_without_crashing(self) -> None:
        result = evaluate_exchange_recommendation(
            make_indicator_row(
                rate=np.nan,
                SMA60=np.nan,
                SMA120=np.nan,
                SMA60_distance_pct=np.nan,
                SMA120_distance_pct=np.nan,
                percentile_rank_180=np.nan,
                BB_middle=np.nan,
                BB_upper=np.nan,
                BB_lower=np.nan,
            ),
            90,
            TOTAL,
            Decimal("0"),
        )
        self.assertEqual(result.base_signal, Signal.WAIT)
        self.assertIn("SMA60", result.unavailable_indicators)
        self.assertIn("percentile_rank_180", result.trace.unavailable_indicators)

    def test_input_series_and_stage_results_remain_immutable(self) -> None:
        row = row_for_signal(Signal.WATCH)
        row_before = row.copy(deep=True)
        market = evaluate_production_signal(row)
        dday = evaluate_production_dday_signal(market, 14)
        split = evaluate_production_split_exchange(dday, TOTAL, Decimal("300000"))

        result = evaluate_exchange_recommendation(row, 14, TOTAL, Decimal("300000"))

        assert_series_equal(row, row_before)
        self.assertEqual(market, evaluate_production_signal(row))
        self.assertEqual(dday, evaluate_production_dday_signal(market, 14))
        self.assertEqual(split, evaluate_production_split_exchange(dday, TOTAL, Decimal("300000")))
        self.assertEqual(result.base_signal, Signal.WATCH)

    def test_invalid_budget_and_dday_are_rejected_by_existing_contracts(self) -> None:
        row = row_for_signal(Signal.WAIT)
        with self.assertRaises(DdayValidationError):
            evaluate_exchange_recommendation(row, -1, TOTAL, Decimal("0"))
        for total, already in ((Decimal("0"), Decimal("0")), (TOTAL, Decimal("1000001"))):
            with self.subTest(total=total, already=already):
                with self.assertRaises(SplitExchangeValidationError):
                    evaluate_exchange_recommendation(row, 30, total, already)

    def test_metadata_mismatch_is_rejected_explicitly(self) -> None:
        market = evaluate_production_signal(row_for_signal(Signal.GOOD))
        dday = evaluate_production_dday_signal(market, 30)
        split = evaluate_production_split_exchange(dday, TOTAL, Decimal("0"))
        with self.assertRaisesRegex(RecommendationValidationError, "split result metadata"):
            _validate_metadata_consistency(
                market, dday, replace(split, split_policy_version="v2")
            )


if __name__ == "__main__":
    unittest.main()
