import unittest
from dataclasses import FrozenInstanceError
from datetime import date

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.calibration_candidates import build_initial_candidate_plan
from src.production_config import (
    ProductionSignalConfiguration,
    evaluate_production_signal,
    get_production_signal_configuration,
    get_production_signal_policy,
    get_production_signal_thresholds,
)
from src.signal_engine import Signal, evaluate_signal


def make_indicator_row(**overrides: object) -> pd.Series:
    values: dict[str, object] = {
        "date": pd.Timestamp("2026-08-06"),
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


class ProductionConfigurationTests(unittest.TestCase):
    def test_metadata_identifies_the_adopted_v1_configuration(self) -> None:
        configuration = get_production_signal_configuration()

        self.assertIsInstance(configuration, ProductionSignalConfiguration)
        self.assertEqual(
            configuration.configuration_id,
            "sma60_sensitive__balanced",
        )
        self.assertEqual(configuration.threshold_profile_id, "sma60_sensitive")
        self.assertEqual(configuration.decision_policy_id, "balanced")
        self.assertEqual(configuration.version, "v1")
        self.assertEqual(
            configuration.calibration_data_range,
            (date(2018, 1, 1), date(2023, 12, 29)),
        )
        self.assertEqual(configuration.validation_start_date, date(2024, 1, 1))

    def test_thresholds_are_the_existing_sma60_sensitive_candidate(self) -> None:
        plan = build_initial_candidate_plan()
        matching_configuration = next(
            candidate
            for candidate in plan.configurations
            if candidate.configuration_id == "sma60_sensitive__balanced"
        )

        self.assertEqual(
            get_production_signal_thresholds(),
            matching_configuration.thresholds,
        )
        self.assertEqual(
            (
                get_production_signal_thresholds().sma60_good,
                get_production_signal_thresholds().sma60_strong,
                get_production_signal_thresholds().sma120_good,
                get_production_signal_thresholds().sma120_strong,
                get_production_signal_thresholds().percentile_good,
                get_production_signal_thresholds().percentile_strong,
                get_production_signal_thresholds().bollinger_near_lower_pct,
            ),
            (-1.0, -2.0, -2.0, -4.0, 25.0, 10.0, 1.0),
        )

    def test_policy_is_the_existing_balanced_candidate(self) -> None:
        plan = build_initial_candidate_plan()
        matching_configuration = next(
            candidate
            for candidate in plan.configurations
            if candidate.configuration_id == "sma60_sensitive__balanced"
        )

        self.assertEqual(
            get_production_signal_policy(),
            matching_configuration.policy,
        )
        policy = get_production_signal_policy()
        self.assertEqual(
            (
                policy.minimum_available_conditions,
                policy.watch_min_satisfied_conditions,
                policy.good_min_satisfied_conditions,
                policy.strong_min_satisfied_conditions,
                policy.strong_min_strong_conditions,
            ),
            (3, 1, 3, 4, 3),
        )

    def test_production_configuration_and_nested_rules_are_immutable(self) -> None:
        configuration = get_production_signal_configuration()

        with self.assertRaises(FrozenInstanceError):
            configuration.version = "v2"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            configuration.thresholds.sma60_good = -99.0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            configuration.policy.good_min_satisfied_conditions = 1  # type: ignore[misc]


class ProductionSignalEntryPointTests(unittest.TestCase):
    def test_entry_point_matches_evaluate_signal_exactly(self) -> None:
        row = make_indicator_row(
            rate=100.5,
            SMA60_distance_pct=-1.5,
            SMA120_distance_pct=-3.0,
            percentile_rank_180=20.0,
        )

        actual = evaluate_production_signal(row)
        expected = evaluate_signal(
            row,
            thresholds=get_production_signal_thresholds(),
            policy=get_production_signal_policy(),
        )

        self.assertEqual(actual, expected)

    def test_wait_state(self) -> None:
        result = evaluate_production_signal(make_indicator_row())

        self.assertEqual(result.signal, Signal.WAIT)
        self.assertEqual(result.satisfied_conditions, ())

    def test_watch_state(self) -> None:
        result = evaluate_production_signal(
            make_indicator_row(SMA60_distance_pct=-1.5)
        )

        self.assertEqual(result.signal, Signal.WATCH)
        self.assertEqual(result.satisfied_conditions, ("sma60_condition",))

    def test_good_state(self) -> None:
        result = evaluate_production_signal(
            make_indicator_row(
                SMA60_distance_pct=-1.5,
                SMA120_distance_pct=-3.0,
                percentile_rank_180=20.0,
            )
        )

        self.assertEqual(result.signal, Signal.GOOD)
        self.assertEqual(result.satisfied_condition_count, 3)

    def test_strong_state(self) -> None:
        result = evaluate_production_signal(
            make_indicator_row(
                rate=100.0,
                SMA60_distance_pct=-2.0,
                SMA120_distance_pct=-4.0,
                percentile_rank_180=20.0,
            )
        )

        self.assertEqual(result.signal, Signal.STRONG)
        self.assertEqual(result.satisfied_condition_count, 4)
        self.assertEqual(result.strong_condition_count, 3)

    def test_nan_and_insufficient_indicators_return_wait_with_details(self) -> None:
        result = evaluate_production_signal(
            make_indicator_row(
                rate=np.nan,
                SMA60=np.nan,
                SMA120=np.nan,
                SMA60_distance_pct=np.nan,
                SMA120_distance_pct=np.nan,
                percentile_rank_180=np.nan,
                BB_lower=np.nan,
            )
        )

        self.assertEqual(result.signal, Signal.WAIT)
        self.assertEqual(result.available_condition_count, 0)
        self.assertIn("rate", result.unavailable_indicators)
        self.assertIn("percentile_rank_180", result.unavailable_indicators)

    def test_signal_is_independent_of_jpy_usd_and_eur_rate_scale(self) -> None:
        signals = []
        for currency, lower_band in (
            ("JPY", 9.0),
            ("USD", 1300.0),
            ("EUR", 1500.0),
        ):
            with self.subTest(currency=currency):
                result = evaluate_production_signal(
                    make_indicator_row(
                        rate=lower_band * 1.005,
                        SMA60=lower_band * 1.02,
                        SMA120=lower_band * 1.03,
                        SMA60_distance_pct=-1.5,
                        SMA120_distance_pct=-3.0,
                        percentile_rank_180=20.0,
                        BB_middle=lower_band * 1.02,
                        BB_upper=lower_band * 1.05,
                        BB_lower=lower_band,
                    )
                )
                signals.append(result.signal)

        self.assertEqual(signals, [Signal.GOOD, Signal.GOOD, Signal.GOOD])

    def test_input_series_and_parent_dataframe_are_not_modified(self) -> None:
        frame = pd.DataFrame(
            [
                make_indicator_row().to_dict(),
                make_indicator_row(SMA60_distance_pct=-1.5).to_dict(),
            ]
        )
        frame_before = frame.copy(deep=True)
        latest_row = frame.iloc[-1]
        latest_row_before = latest_row.copy(deep=True)

        evaluate_production_signal(latest_row)

        assert_series_equal(latest_row, latest_row_before)
        assert_frame_equal(frame, frame_before)


if __name__ == "__main__":
    unittest.main()
