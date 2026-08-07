import unittest
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pandas as pd
from pandas.testing import assert_series_equal

from src.dday_policy import (
    DdayBand,
    DdaySignalResult,
    DdayUrgency,
    evaluate_production_dday_signal,
    get_production_dday_policy,
)
from src.production_config import (
    evaluate_production_signal,
    get_production_signal_configuration,
)
from src.signal_engine import Signal
from src.split_exchange import (
    SplitExchangePolicy,
    SplitExchangeValidationError,
    evaluate_production_split_exchange,
    evaluate_split_exchange,
    get_production_split_exchange_policy,
)


TOTAL = Decimal("1000000")


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


def make_dday_result(
    adjusted_signal: Signal,
    urgency: DdayUrgency,
    *,
    with_production_metadata: bool = False,
) -> DdaySignalResult:
    configuration = get_production_signal_configuration()
    dday_policy = get_production_dday_policy()
    return DdaySignalResult(
        base_signal=adjusted_signal,
        adjusted_signal=adjusted_signal,
        urgency=urgency,
        days_until_departure=10,
        dday_band=DdayBand.D7_TO_D14,
        promotion_steps=1,
        applied_promotion_steps=0,
        reason="test fixture",
        policy_id=dday_policy.policy_id,
        policy_version=dday_policy.version,
        production_configuration_id=(
            configuration.configuration_id if with_production_metadata else None
        ),
        production_configuration_version=(
            configuration.version if with_production_metadata else None
        ),
    )


class SplitExchangePolicyTests(unittest.TestCase):
    def test_policy_metadata_ratio_tables_and_immutability(self) -> None:
        policy = get_production_split_exchange_policy()

        self.assertIsInstance(policy, SplitExchangePolicy)
        self.assertEqual(policy.policy_id, "split_exchange_policy_v1")
        self.assertEqual(policy.version, "v1")
        self.assertEqual(
            [rule.target_ratio for rule in policy.signal_target_rules],
            [Decimal("0.00"), Decimal("0.25"), Decimal("0.50"), Decimal("0.75")],
        )
        self.assertEqual(
            [rule.minimum_ratio for rule in policy.urgency_minimum_rules],
            [
                Decimal("0.00"),
                Decimal("0.00"),
                Decimal("0.25"),
                Decimal("0.75"),
                Decimal("1.00"),
            ],
        )
        with self.assertRaises(FrozenInstanceError):
            policy.version = "v2"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            policy.signal_target_rules[0].target_ratio = Decimal("1")  # type: ignore[misc]

    def test_signal_target_ratios_use_adjusted_signal(self) -> None:
        policy = get_production_split_exchange_policy()
        expected = {
            Signal.WAIT: Decimal("0.00"),
            Signal.WATCH: Decimal("0.25"),
            Signal.GOOD: Decimal("0.50"),
            Signal.STRONG: Decimal("0.75"),
        }

        for signal, ratio in expected.items():
            with self.subTest(signal=signal.value):
                result = evaluate_split_exchange(
                    make_dday_result(signal, DdayUrgency.LOW),
                    TOTAL,
                    0,
                    policy=policy,
                )
                self.assertEqual(result.adjusted_signal, signal)
                self.assertEqual(result.signal_target_ratio, ratio)
                self.assertEqual(result.target_cumulative_ratio, ratio)

    def test_urgency_minimum_ratios(self) -> None:
        policy = get_production_split_exchange_policy()
        expected = {
            DdayUrgency.LOW: Decimal("0.00"),
            DdayUrgency.NORMAL: Decimal("0.00"),
            DdayUrgency.HIGH: Decimal("0.25"),
            DdayUrgency.CRITICAL: Decimal("0.75"),
            DdayUrgency.DEADLINE: Decimal("1.00"),
        }

        for urgency, ratio in expected.items():
            with self.subTest(urgency=urgency.value):
                result = evaluate_split_exchange(
                    make_dday_result(Signal.WAIT, urgency),
                    TOTAL,
                    0,
                    policy=policy,
                )
                self.assertEqual(result.urgency_minimum_ratio, ratio)
                self.assertEqual(result.target_cumulative_ratio, ratio)


class SplitExchangeCalculationTests(unittest.TestCase):
    def test_required_signal_urgency_combinations(self) -> None:
        cases = (
            (Signal.GOOD, DdayUrgency.HIGH, Decimal("0.50")),
            (Signal.WAIT, DdayUrgency.CRITICAL, Decimal("0.75")),
            (Signal.STRONG, DdayUrgency.LOW, Decimal("0.75")),
            (Signal.WATCH, DdayUrgency.DEADLINE, Decimal("1.00")),
        )

        for signal, urgency, target_ratio in cases:
            with self.subTest(signal=signal.value, urgency=urgency.value):
                result = evaluate_split_exchange(
                    make_dday_result(signal, urgency),
                    TOTAL,
                    Decimal("300000"),
                    policy=get_production_split_exchange_policy(),
                )
                self.assertEqual(result.target_cumulative_ratio, target_ratio)
                self.assertEqual(
                    result.target_cumulative_amount_krw,
                    TOTAL * target_ratio,
                )

    def test_cumulative_amounts_for_required_already_exchanged_values(self) -> None:
        dday_result = make_dday_result(Signal.GOOD, DdayUrgency.HIGH)
        cases = (
            (Decimal("0"), Decimal("500000"), Decimal("500000")),
            (Decimal("300000"), Decimal("200000"), Decimal("500000")),
            (Decimal("500000"), Decimal("0"), Decimal("500000")),
            (Decimal("800000"), Decimal("0"), Decimal("200000")),
        )

        for already, expected_additional, expected_remaining in cases:
            with self.subTest(already=already):
                result = evaluate_split_exchange(
                    dday_result,
                    TOTAL,
                    already,
                    policy=get_production_split_exchange_policy(),
                )
                self.assertEqual(result.recommended_additional_krw, expected_additional)
                self.assertEqual(
                    result.remaining_after_recommendation_krw,
                    expected_remaining,
                )

    def test_over_exchange_protections_and_full_exchange(self) -> None:
        policy = get_production_split_exchange_policy()
        target_exceeded = evaluate_split_exchange(
            make_dday_result(Signal.GOOD, DdayUrgency.HIGH),
            TOTAL,
            Decimal("800000"),
            policy=policy,
        )
        fully_exchanged = evaluate_split_exchange(
            make_dday_result(Signal.WATCH, DdayUrgency.DEADLINE),
            TOTAL,
            TOTAL,
            policy=policy,
        )

        self.assertEqual(target_exceeded.recommended_additional_krw, Decimal("0"))
        self.assertEqual(fully_exchanged.recommended_additional_krw, Decimal("0"))
        self.assertEqual(
            fully_exchanged.remaining_after_recommendation_krw,
            Decimal("0"),
        )
        for result in (target_exceeded, fully_exchanged):
            self.assertGreaterEqual(result.recommended_additional_krw, Decimal("0"))
            self.assertLessEqual(
                result.recommended_additional_krw,
                result.total_target_krw - result.already_exchanged_krw,
            )

    def test_d0_deadline_target_is_one_hundred_percent(self) -> None:
        dday_result = evaluate_production_dday_signal(
            make_production_signal_result(Signal.WAIT),
            0,
        )

        result = evaluate_production_split_exchange(
            dday_result,
            TOTAL,
            Decimal("300000"),
        )

        self.assertEqual(result.urgency, DdayUrgency.DEADLINE)
        self.assertEqual(result.target_cumulative_ratio, Decimal("1.00"))
        self.assertEqual(result.recommended_additional_krw, Decimal("700000"))


class SplitExchangeContractTests(unittest.TestCase):
    def test_input_dday_result_and_signal_source_are_not_modified(self) -> None:
        source_row = make_indicator_row(SMA60_distance_pct=-1.5)
        source_before = source_row.copy(deep=True)
        signal_result = evaluate_production_signal(source_row)
        dday_result = evaluate_production_dday_signal(signal_result, 12)
        dday_before = replace(dday_result)

        result = evaluate_production_split_exchange(
            dday_result,
            TOTAL,
            Decimal("300000"),
        )

        assert_series_equal(source_row, source_before)
        self.assertEqual(dday_result, dday_before)
        self.assertEqual(result.base_signal, Signal.WATCH)
        self.assertEqual(result.adjusted_signal, Signal.GOOD)
        self.assertEqual(result.recommended_additional_krw, Decimal("200000"))

    def test_production_and_dday_metadata_are_transferred(self) -> None:
        dday_result = evaluate_production_dday_signal(
            make_production_signal_result(Signal.WATCH),
            12,
        )

        result = evaluate_production_split_exchange(
            dday_result,
            TOTAL,
            0,
        )

        self.assertEqual(
            result.production_configuration_id,
            "sma60_sensitive__balanced",
        )
        self.assertEqual(result.production_configuration_version, "v1")
        self.assertEqual(result.dday_policy_id, "dday_policy_v1")
        self.assertEqual(result.dday_policy_version, "v1")
        self.assertEqual(result.split_policy_id, "split_exchange_policy_v1")
        self.assertEqual(result.split_policy_version, "v1")
        self.assertIn("target_cumulative_ratio", result.reason)

    def test_invalid_inputs_and_nonproduction_dday_results_are_rejected(self) -> None:
        valid_result = make_dday_result(Signal.GOOD, DdayUrgency.HIGH)
        policy = get_production_split_exchange_policy()

        for total in (0, -1, True, float("nan"), float("inf"), "1000000"):
            with self.subTest(total=total):
                with self.assertRaises((TypeError, SplitExchangeValidationError, ValueError)):
                    evaluate_split_exchange(valid_result, total, 0, policy=policy)
        for already in (-1, TOTAL + 1, True, float("nan"), float("-inf"), "0"):
            with self.subTest(already=already):
                with self.assertRaises((TypeError, SplitExchangeValidationError, ValueError)):
                    evaluate_split_exchange(valid_result, TOTAL, already, policy=policy)
        with self.assertRaises(TypeError):
            evaluate_split_exchange(None, TOTAL, 0, policy=policy)
        with self.assertRaises(TypeError):
            evaluate_split_exchange(valid_result, TOTAL, 0, policy=object())
        with self.assertRaisesRegex(ValueError, "production market configuration"):
            evaluate_production_split_exchange(valid_result, TOTAL, 0)


if __name__ == "__main__":
    unittest.main()
