import unittest
from dataclasses import FrozenInstanceError, replace

import pandas as pd
from pandas.testing import assert_series_equal

from src.dday_policy import (
    DdayBand,
    DdayPolicy,
    DdayUrgency,
    DdayValidationError,
    evaluate_dday_signal,
    evaluate_production_dday_signal,
    get_production_dday_policy,
)
from src.production_config import (
    evaluate_production_signal,
    get_production_signal_configuration,
    get_production_signal_policy,
    get_production_signal_thresholds,
)
from src.signal_engine import Signal, evaluate_signal


def make_indicator_row(**overrides: object) -> pd.Series:
    values: dict[str, object] = {
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


def make_production_signal_result(signal: Signal):
    rows = {
        Signal.WAIT: make_indicator_row(),
        Signal.WATCH: make_indicator_row(SMA60_distance_pct=-1.5),
        Signal.GOOD: make_indicator_row(
            SMA60_distance_pct=-1.5,
            SMA120_distance_pct=-3.0,
            percentile_rank_180=20.0,
        ),
        Signal.STRONG: make_indicator_row(
            rate=100.0,
            SMA60_distance_pct=-2.0,
            SMA120_distance_pct=-4.0,
            percentile_rank_180=20.0,
        ),
    }
    result = evaluate_production_signal(rows[signal])
    assert result.signal is signal
    return result


class ProductionDdayPolicyTests(unittest.TestCase):
    def test_policy_metadata_and_rules_are_immutable(self) -> None:
        policy = get_production_dday_policy()

        self.assertIsInstance(policy, DdayPolicy)
        self.assertEqual(policy.policy_id, "dday_policy_v1")
        self.assertEqual(policy.version, "v1")
        self.assertEqual(
            [rule.band for rule in policy.band_rules],
            list(DdayBand),
        )
        with self.assertRaises(FrozenInstanceError):
            policy.version = "v2"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            policy.band_rules[0].promotion_steps = 3  # type: ignore[misc]

    def test_all_required_boundaries_and_signals(self) -> None:
        cases = (
            (90, DdayBand.D61_PLUS, DdayUrgency.LOW, 0),
            (61, DdayBand.D61_PLUS, DdayUrgency.LOW, 0),
            (60, DdayBand.D31_TO_D60, DdayUrgency.NORMAL, 0),
            (31, DdayBand.D31_TO_D60, DdayUrgency.NORMAL, 0),
            (30, DdayBand.D15_TO_D30, DdayUrgency.HIGH, 1),
            (15, DdayBand.D15_TO_D30, DdayUrgency.HIGH, 1),
            (14, DdayBand.D7_TO_D14, DdayUrgency.HIGH, 1),
            (7, DdayBand.D7_TO_D14, DdayUrgency.HIGH, 1),
            (6, DdayBand.D1_TO_D6, DdayUrgency.CRITICAL, 2),
            (1, DdayBand.D1_TO_D6, DdayUrgency.CRITICAL, 2),
            (0, DdayBand.D0, DdayUrgency.DEADLINE, 0),
        )
        policy = get_production_dday_policy()

        for days, expected_band, expected_urgency, promotion in cases:
            for base_signal in Signal:
                with self.subTest(days=days, base_signal=base_signal.value):
                    result = evaluate_dday_signal(
                        make_production_signal_result(base_signal),
                        days,
                        policy=policy,
                    )
                    expected_rank = min(
                        tuple(Signal).index(base_signal) + promotion,
                        len(Signal) - 1,
                    )
                    expected_adjusted = tuple(Signal)[expected_rank]

                    self.assertEqual(result.base_signal, base_signal)
                    self.assertEqual(result.adjusted_signal, expected_adjusted)
                    self.assertEqual(result.dday_band, expected_band)
                    self.assertEqual(result.urgency, expected_urgency)
                    self.assertEqual(result.promotion_steps, promotion)
                    self.assertEqual(
                        result.applied_promotion_steps,
                        expected_rank - tuple(Signal).index(base_signal),
                    )

    def test_deadline_and_reason_preserve_market_quality(self) -> None:
        base_result = make_production_signal_result(Signal.WAIT)

        result = evaluate_production_dday_signal(base_result, 0)

        self.assertEqual(result.base_signal, Signal.WAIT)
        self.assertEqual(result.adjusted_signal, Signal.WAIT)
        self.assertEqual(result.urgency, DdayUrgency.DEADLINE)
        self.assertIn("deadline", result.reason)
        self.assertEqual(
            result.production_configuration_id,
            "sma60_sensitive__balanced",
        )
        self.assertEqual(result.production_configuration_version, "v1")

    def test_examples_for_deadline_adjustment(self) -> None:
        cases = (
            (90, Signal.WATCH, Signal.WATCH),
            (30, Signal.WAIT, Signal.WATCH),
            (14, Signal.WATCH, Signal.GOOD),
            (7, Signal.WAIT, Signal.WATCH),
            (1, Signal.WAIT, Signal.GOOD),
            (0, Signal.GOOD, Signal.GOOD),
        )

        for days, base_signal, adjusted_signal in cases:
            with self.subTest(days=days, base_signal=base_signal.value):
                result = evaluate_production_dday_signal(
                    make_production_signal_result(base_signal),
                    days,
                )
                self.assertEqual(result.adjusted_signal, adjusted_signal)
                self.assertTrue(result.reason)


class DdayEvaluationContractTests(unittest.TestCase):
    def test_signal_result_and_source_series_are_not_modified(self) -> None:
        source_row = make_indicator_row(SMA60_distance_pct=-1.5)
        source_before = source_row.copy(deep=True)
        signal_result = evaluate_production_signal(source_row)
        signal_before = replace(signal_result)

        result = evaluate_production_dday_signal(signal_result, 12)

        assert_series_equal(source_row, source_before)
        self.assertEqual(signal_result, signal_before)
        self.assertEqual(result.base_signal, Signal.WATCH)
        self.assertEqual(result.adjusted_signal, Signal.GOOD)
        self.assertEqual(result.urgency, DdayUrgency.HIGH)
        self.assertIn("D7_TO_D14", result.reason)

    def test_strong_is_clamped_and_never_exceeds_the_signal_range(self) -> None:
        result = evaluate_production_dday_signal(
            make_production_signal_result(Signal.STRONG),
            1,
        )

        self.assertEqual(result.base_signal, Signal.STRONG)
        self.assertEqual(result.adjusted_signal, Signal.STRONG)
        self.assertEqual(result.promotion_steps, 2)
        self.assertEqual(result.applied_promotion_steps, 0)

    def test_invalid_days_signal_and_policy_inputs_are_rejected(self) -> None:
        signal_result = make_production_signal_result(Signal.WAIT)
        policy = get_production_dday_policy()

        with self.assertRaises(DdayValidationError):
            evaluate_dday_signal(signal_result, -1, policy=policy)
        for invalid_days in (True, 1.5, "1", None):
            with self.subTest(days=invalid_days):
                with self.assertRaises(TypeError):
                    evaluate_dday_signal(
                        signal_result,
                        invalid_days,
                        policy=policy,
                    )
        with self.assertRaises(TypeError):
            evaluate_dday_signal(None, 10, policy=policy)
        with self.assertRaises(TypeError):
            evaluate_dday_signal(signal_result, 10, policy=object())

    def test_very_large_dday_uses_far_band(self) -> None:
        result = evaluate_production_dday_signal(
            make_production_signal_result(Signal.WAIT),
            100_000,
        )

        self.assertEqual(result.dday_band, DdayBand.D61_PLUS)
        self.assertEqual(result.adjusted_signal, Signal.WAIT)

    def test_generic_policy_is_independent_from_market_production_config(self) -> None:
        row = make_indicator_row(SMA60_distance_pct=-1.5)
        custom_thresholds = replace(
            get_production_signal_thresholds(),
            sma60_good=-0.5,
        )
        custom_signal = evaluate_signal(
            row,
            thresholds=custom_thresholds,
            policy=get_production_signal_policy(),
        )
        configuration_before = get_production_signal_configuration()

        generic_result = evaluate_dday_signal(
            custom_signal,
            12,
            policy=get_production_dday_policy(),
        )

        self.assertIsNone(generic_result.production_configuration_id)
        self.assertIsNone(generic_result.production_configuration_version)
        self.assertEqual(get_production_signal_configuration(), configuration_before)
        with self.assertRaisesRegex(ValueError, "production market configuration"):
            evaluate_production_dday_signal(custom_signal, 12)


if __name__ == "__main__":
    unittest.main()
