import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.indicators import calculate_indicators
from src.signal_engine import (
    CONDITION_NAMES,
    SIGNAL_INPUT_COLUMNS,
    ConditionStatus,
    Signal,
    SignalDecisionPolicy,
    SignalThresholds,
    evaluate_signal,
)


TEST_THRESHOLDS = SignalThresholds(
    sma60_good=-2.0,
    sma60_strong=-5.0,
    sma120_good=-3.0,
    sma120_strong=-6.0,
    percentile_good=25.0,
    percentile_strong=10.0,
    bollinger_near_lower_pct=1.0,
)

CONSERVATIVE_POLICY = SignalDecisionPolicy(
    minimum_available_conditions=3,
    watch_min_satisfied_conditions=1,
    good_min_satisfied_conditions=4,
    strong_min_satisfied_conditions=4,
    strong_min_strong_conditions=4,
)


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


def evaluate(
    row: pd.Series,
    *,
    thresholds: SignalThresholds = TEST_THRESHOLDS,
    policy: SignalDecisionPolicy = CONSERVATIVE_POLICY,
):
    return evaluate_signal(row, thresholds=thresholds, policy=policy)


class SignalClassificationTests(unittest.TestCase):
    def test_wait_when_no_condition_is_satisfied(self) -> None:
        result = evaluate(make_indicator_row())

        self.assertEqual(result.signal, Signal.WAIT)
        self.assertEqual(result.current_rate, 102.0)
        self.assertEqual(
            tuple(condition.name for condition in result.conditions),
            CONDITION_NAMES,
        )
        self.assertTrue(
            all(
                condition.status is ConditionStatus.NOT_MET
                for condition in result.conditions
            )
        )
        self.assertEqual(result.satisfied_conditions, ())
        self.assertEqual(result.unavailable_indicators, ())
        self.assertEqual(result.available_condition_count, 4)
        self.assertEqual(result.satisfied_condition_count, 0)

    def test_watch_when_only_one_condition_is_satisfied(self) -> None:
        result = evaluate(
            make_indicator_row(SMA60_distance_pct=-2.5)
        )

        self.assertEqual(result.signal, Signal.WATCH)
        self.assertEqual(result.sma60_condition.status, ConditionStatus.GOOD)
        self.assertEqual(
            result.satisfied_conditions,
            ("sma60_condition",),
        )

    def test_good_when_all_conditions_are_at_least_good(self) -> None:
        result = evaluate(
            make_indicator_row(
                rate=100.5,
                SMA60_distance_pct=-2.5,
                SMA120_distance_pct=-3.5,
                percentile_rank_180=20.0,
            )
        )

        self.assertEqual(result.signal, Signal.GOOD)
        self.assertTrue(
            all(
                condition.status is ConditionStatus.GOOD
                for condition in result.conditions
            )
        )
        self.assertEqual(result.satisfied_condition_count, 4)
        self.assertEqual(result.strong_condition_count, 0)
        self.assertEqual(result.sma60_condition.observed_value, -2.5)
        self.assertEqual(
            result.sma60_condition.good_threshold,
            TEST_THRESHOLDS.sma60_good,
        )
        self.assertEqual(
            result.sma60_condition.strong_threshold,
            TEST_THRESHOLDS.sma60_strong,
        )
        self.assertAlmostEqual(
            result.bollinger_condition.observed_value,
            0.5,
        )
        self.assertEqual(result.bollinger_condition.good_threshold, 1.0)
        self.assertEqual(result.bollinger_condition.strong_threshold, 0.0)

    def test_strong_when_all_conditions_are_strong(self) -> None:
        result = evaluate(
            make_indicator_row(
                rate=100.0,
                SMA60_distance_pct=-5.0,
                SMA120_distance_pct=-6.0,
                percentile_rank_180=10.0,
            )
        )

        self.assertEqual(result.signal, Signal.STRONG)
        self.assertTrue(
            all(
                condition.status is ConditionStatus.STRONG
                for condition in result.conditions
            )
        )
        self.assertEqual(result.strong_condition_count, 4)

    def test_three_strong_and_one_good_is_not_strong(self) -> None:
        result = evaluate(
            make_indicator_row(
                rate=100.5,
                SMA60_distance_pct=-5.0,
                SMA120_distance_pct=-6.0,
                percentile_rank_180=10.0,
            )
        )

        self.assertEqual(result.signal, Signal.GOOD)
        self.assertEqual(result.satisfied_condition_count, 4)
        self.assertEqual(result.strong_condition_count, 3)

    def test_each_condition_is_evaluated_independently(self) -> None:
        cases = (
            (
                "sma60_condition",
                {"SMA60_distance_pct": -2.5},
            ),
            (
                "sma120_condition",
                {"SMA120_distance_pct": -3.5},
            ),
            (
                "percentile_condition",
                {"percentile_rank_180": 20.0},
            ),
            (
                "bollinger_condition",
                {"rate": 100.5},
            ),
        )

        for expected_name, overrides in cases:
            with self.subTest(condition=expected_name):
                result = evaluate(make_indicator_row(**overrides))
                statuses = {
                    condition.name: condition.status
                    for condition in result.conditions
                }

                self.assertEqual(result.signal, Signal.WATCH)
                self.assertEqual(
                    result.satisfied_conditions,
                    (expected_name,),
                )
                self.assertEqual(
                    statuses[expected_name],
                    ConditionStatus.GOOD,
                )
                self.assertTrue(
                    all(
                        status is ConditionStatus.NOT_MET
                        for name, status in statuses.items()
                        if name != expected_name
                    )
                )

    def test_sma_and_percentile_boundaries_are_inclusive(self) -> None:
        cases = (
            (
                "SMA60_distance_pct",
                "sma60_condition",
                TEST_THRESHOLDS.sma60_good,
                TEST_THRESHOLDS.sma60_strong,
            ),
            (
                "SMA120_distance_pct",
                "sma120_condition",
                TEST_THRESHOLDS.sma120_good,
                TEST_THRESHOLDS.sma120_strong,
            ),
            (
                "percentile_rank_180",
                "percentile_condition",
                TEST_THRESHOLDS.percentile_good,
                TEST_THRESHOLDS.percentile_strong,
            ),
        )

        for field_name, condition_name, good, strong in cases:
            with self.subTest(field=field_name, boundary="strong"):
                result = evaluate(make_indicator_row(**{field_name: strong}))
                self.assertEqual(
                    getattr(result, condition_name).status,
                    ConditionStatus.STRONG,
                )
            with self.subTest(field=field_name, boundary="above-strong"):
                result = evaluate(
                    make_indicator_row(
                        **{field_name: np.nextafter(strong, np.inf)}
                    )
                )
                self.assertEqual(
                    getattr(result, condition_name).status,
                    ConditionStatus.GOOD,
                )
            with self.subTest(field=field_name, boundary="good"):
                result = evaluate(make_indicator_row(**{field_name: good}))
                self.assertEqual(
                    getattr(result, condition_name).status,
                    ConditionStatus.GOOD,
                )
            with self.subTest(field=field_name, boundary="above-good"):
                result = evaluate(
                    make_indicator_row(
                        **{field_name: np.nextafter(good, np.inf)}
                    )
                )
                self.assertEqual(
                    getattr(result, condition_name).status,
                    ConditionStatus.NOT_MET,
                )

    def test_bollinger_lower_and_tolerance_boundaries(self) -> None:
        cases = (
            (99.0, ConditionStatus.STRONG),
            (100.0, ConditionStatus.STRONG),
            (np.nextafter(100.0, np.inf), ConditionStatus.GOOD),
            (100.5, ConditionStatus.GOOD),
            (101.0, ConditionStatus.GOOD),
            (np.nextafter(101.0, np.inf), ConditionStatus.NOT_MET),
        )

        for rate, expected_status in cases:
            with self.subTest(rate=rate):
                result = evaluate(make_indicator_row(rate=rate))
                self.assertEqual(
                    result.bollinger_condition.status,
                    expected_status,
                )

        zero_tolerance = replace(
            TEST_THRESHOLDS,
            bollinger_near_lower_pct=0.0,
        )
        just_above_lower = np.nextafter(100.0, np.inf)
        exact_lower_result = evaluate(
            make_indicator_row(rate=100.0),
            thresholds=zero_tolerance,
        )
        self.assertEqual(
            exact_lower_result.bollinger_condition.status,
            ConditionStatus.STRONG,
        )
        result = evaluate(
            make_indicator_row(rate=just_above_lower),
            thresholds=zero_tolerance,
        )
        self.assertEqual(
            result.bollinger_condition.status,
            ConditionStatus.NOT_MET,
        )


class SignalMissingDataTests(unittest.TestCase):
    def test_nan_and_infinite_dependencies_are_unavailable(self) -> None:
        cases = (
            ("SMA60", "sma60_condition"),
            (
                "SMA60_distance_pct",
                "sma60_condition",
            ),
            ("SMA120", "sma120_condition"),
            (
                "SMA120_distance_pct",
                "sma120_condition",
            ),
            (
                "percentile_rank_180",
                "percentile_condition",
            ),
            ("rate", "bollinger_condition"),
            ("BB_lower", "bollinger_condition"),
        )

        for field_name, condition_name in cases:
            for unavailable_value in (np.nan, np.inf):
                with self.subTest(
                    field=field_name,
                    value=unavailable_value,
                ):
                    result = evaluate(
                        make_indicator_row(
                            **{field_name: unavailable_value}
                        )
                    )
                    condition = getattr(result, condition_name)
                    self.assertEqual(
                        condition.status,
                        ConditionStatus.UNAVAILABLE,
                    )
                    self.assertIsNone(condition.observed_value)
                    self.assertIn(
                        field_name,
                        result.unavailable_indicators,
                    )

        for lower_band in (0.0, -1.0):
            with self.subTest(BB_lower=lower_band):
                nonpositive_lower = evaluate(
                    make_indicator_row(BB_lower=lower_band)
                )
                self.assertEqual(
                    nonpositive_lower.bollinger_condition.status,
                    ConditionStatus.UNAVAILABLE,
                )
                self.assertIn(
                    "BB_lower",
                    nonpositive_lower.unavailable_indicators,
                )

    def test_missing_condition_caps_signal_and_too_few_force_wait(self) -> None:
        one_missing = make_indicator_row(
            rate=100.0,
            SMA60_distance_pct=-5.0,
            SMA120_distance_pct=-6.0,
            percentile_rank_180=np.nan,
        )
        result = evaluate(one_missing)

        self.assertEqual(result.signal, Signal.WATCH)
        self.assertEqual(result.available_condition_count, 3)
        self.assertEqual(result.strong_condition_count, 3)
        self.assertEqual(
            result.percentile_condition.status,
            ConditionStatus.UNAVAILABLE,
        )

        permissive_policy = SignalDecisionPolicy(
            minimum_available_conditions=1,
            watch_min_satisfied_conditions=1,
            good_min_satisfied_conditions=1,
            strong_min_satisfied_conditions=1,
            strong_min_strong_conditions=1,
        )
        permissive_result = evaluate(
            one_missing,
            policy=permissive_policy,
        )
        self.assertEqual(permissive_result.signal, Signal.GOOD)

        too_few = one_missing.copy(deep=True)
        too_few["SMA120_distance_pct"] = np.nan
        result = evaluate(too_few)

        self.assertEqual(result.signal, Signal.WAIT)
        self.assertEqual(result.available_condition_count, 2)

    def test_all_core_conditions_unavailable_return_wait(self) -> None:
        result = evaluate(
            make_indicator_row(
                rate=np.nan,
                SMA60_distance_pct=np.nan,
                SMA120_distance_pct=np.nan,
                percentile_rank_180=np.nan,
            )
        )

        self.assertEqual(result.signal, Signal.WAIT)
        self.assertEqual(result.available_condition_count, 0)
        self.assertEqual(
            result.unavailable_indicators,
            (
                "rate",
                "SMA60_distance_pct",
                "SMA120_distance_pct",
                "percentile_rank_180",
            ),
        )
        self.assertTrue(
            all(
                condition.status is ConditionStatus.UNAVAILABLE
                for condition in result.conditions
            )
        )

    def test_short_indicator_rows_are_supported_without_mutation(self) -> None:
        for periods, expected_available in ((19, 0), (60, 2)):
            with self.subTest(periods=periods):
                source = pd.DataFrame(
                    {
                        "date": pd.date_range(
                            "2026-01-01",
                            periods=periods,
                            freq="D",
                        ),
                        "rate": np.linspace(100.0, 110.0, periods),
                    }
                )
                indicator_rows = calculate_indicators(source)
                indicator_rows_before = indicator_rows.copy(deep=True)
                latest_row = indicator_rows.iloc[-1]
                latest_row_before = latest_row.copy(deep=True)

                result = evaluate(latest_row)

                self.assertEqual(result.signal, Signal.WAIT)
                self.assertEqual(
                    result.available_condition_count,
                    expected_available,
                )
                assert_series_equal(latest_row, latest_row_before)
                assert_frame_equal(indicator_rows, indicator_rows_before)


class SignalConfigurationTests(unittest.TestCase):
    def test_thresholds_and_policy_are_explicitly_injected(self) -> None:
        row = make_indicator_row(
            SMA60_distance_pct=-1.0,
            SMA120_distance_pct=-1.0,
            percentile_rank_180=30.0,
        )
        relaxed_thresholds = replace(
            TEST_THRESHOLDS,
            sma60_good=-0.5,
            sma120_good=-0.5,
            percentile_good=40.0,
        )
        three_condition_good_policy = replace(
            CONSERVATIVE_POLICY,
            good_min_satisfied_conditions=3,
        )

        strict_result = evaluate(
            row,
            policy=three_condition_good_policy,
        )
        relaxed_result = evaluate(
            row,
            thresholds=relaxed_thresholds,
            policy=three_condition_good_policy,
        )

        self.assertEqual(strict_result.signal, Signal.WAIT)
        self.assertEqual(relaxed_result.signal, Signal.GOOD)
        self.assertEqual(relaxed_result.thresholds, relaxed_thresholds)
        self.assertEqual(relaxed_result.policy, three_condition_good_policy)
        with self.assertRaises(TypeError):
            evaluate_signal(row)

    def test_decision_policy_injection_changes_aggregation(self) -> None:
        row = make_indicator_row(
            SMA60_distance_pct=-2.5,
            SMA120_distance_pct=-3.5,
        )
        relaxed_policy = replace(
            CONSERVATIVE_POLICY,
            good_min_satisfied_conditions=2,
        )

        self.assertEqual(evaluate(row).signal, Signal.WATCH)
        self.assertEqual(
            evaluate(row, policy=relaxed_policy).signal,
            Signal.GOOD,
        )

    def test_rejects_invalid_indicator_thresholds(self) -> None:
        cases = (
            ({"sma60_strong": -1.0}, ValueError),
            ({"sma120_strong": -1.0}, ValueError),
            ({"percentile_strong": 30.0}, ValueError),
            ({"percentile_good": 101.0}, ValueError),
            ({"bollinger_near_lower_pct": -0.1}, ValueError),
            ({"sma60_good": np.nan}, ValueError),
            ({"sma60_good": True}, TypeError),
        )

        for updates, expected_error in cases:
            with self.subTest(updates=updates):
                with self.assertRaises(expected_error):
                    replace(TEST_THRESHOLDS, **updates)

    def test_rejects_invalid_decision_policy(self) -> None:
        cases = (
            ({"minimum_available_conditions": 0}, ValueError),
            ({"minimum_available_conditions": 5}, ValueError),
            (
                {
                    "watch_min_satisfied_conditions": 2,
                    "good_min_satisfied_conditions": 1,
                },
                ValueError,
            ),
            (
                {
                    "good_min_satisfied_conditions": 3,
                    "strong_min_satisfied_conditions": 3,
                    "strong_min_strong_conditions": 4,
                },
                ValueError,
            ),
            ({"minimum_available_conditions": True}, TypeError),
        )

        for updates, expected_error in cases:
            with self.subTest(updates=updates):
                with self.assertRaises(expected_error):
                    replace(CONSERVATIVE_POLICY, **updates)

    def test_rejects_wrong_configuration_object_types(self) -> None:
        row = make_indicator_row()

        with self.assertRaisesRegex(TypeError, "thresholds"):
            evaluate_signal(
                row,
                thresholds=object(),
                policy=CONSERVATIVE_POLICY,
            )
        with self.assertRaisesRegex(TypeError, "policy"):
            evaluate_signal(
                row,
                thresholds=TEST_THRESHOLDS,
                policy=object(),
            )


class SignalInputContractTests(unittest.TestCase):
    def test_input_series_and_mapping_are_not_modified(self) -> None:
        row = make_indicator_row(SMA60_distance_pct=-2.5)
        row_before = row.copy(deep=True)
        mapping = row.to_dict()
        mapping_before = mapping.copy()

        series_result = evaluate(row)
        mapping_result = evaluate_signal(
            mapping,
            thresholds=TEST_THRESHOLDS,
            policy=CONSERVATIVE_POLICY,
        )

        assert_series_equal(row, row_before)
        self.assertEqual(mapping, mapping_before)
        self.assertEqual(series_result.signal, mapping_result.signal)
        self.assertEqual(series_result.conditions, mapping_result.conditions)

    def test_signal_is_independent_of_currency_rate_scale(self) -> None:
        cases = (
            ("JPY", 9.0),
            ("USD", 1300.0),
            ("EUR", 1500.0),
        )

        for currency, lower_band in cases:
            with self.subTest(currency=currency):
                row = make_indicator_row(
                    rate=lower_band * 1.005,
                    SMA60=lower_band * 1.02,
                    SMA120=lower_band * 1.03,
                    SMA60_distance_pct=-2.5,
                    SMA120_distance_pct=-3.5,
                    percentile_rank_180=20.0,
                    BB_lower=lower_band,
                    BB_middle=lower_band * 1.05,
                    BB_upper=lower_band * 1.10,
                    currency=currency,
                )

                result = evaluate(row)

                self.assertEqual(result.signal, Signal.GOOD)
                self.assertTrue(
                    all(
                        condition.status is ConditionStatus.GOOD
                        for condition in result.conditions
                    )
                )

    def test_rejects_missing_required_columns(self) -> None:
        for column in SIGNAL_INPUT_COLUMNS:
            with self.subTest(column=column):
                row = make_indicator_row().drop(labels=[column])
                with self.assertRaisesRegex(ValueError, column):
                    evaluate(row)

    def test_rejects_malformed_indicator_values(self) -> None:
        cases = (
            ({"SMA60_distance_pct": "not-a-number"}, "SMA60"),
            ({"SMA60_distance_pct": [1.0]}, "SMA60"),
            ({"rate": True}, "rate"),
            ({"rate": 0.0}, "rate"),
            ({"rate": -1.0}, "rate"),
            ({"percentile_rank_180": -1.0}, "percentile"),
            ({"percentile_rank_180": 101.0}, "percentile"),
            (
                {
                    "BB_lower": 106.0,
                    "BB_middle": 105.0,
                    "BB_upper": 110.0,
                },
                "Bollinger",
            ),
            (
                {
                    "BB_lower": 100.0,
                    "BB_middle": 105.0,
                    "BB_upper": 104.0,
                },
                "Bollinger",
            ),
        )

        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate(make_indicator_row(**overrides))

    def test_rejects_dataframe_input_without_modifying_it(self) -> None:
        frame = pd.DataFrame([make_indicator_row()])
        frame_before = frame.copy(deep=True)

        with self.assertRaisesRegex(TypeError, "Series or mapping"):
            evaluate_signal(
                frame,
                thresholds=TEST_THRESHOLDS,
                policy=CONSERVATIVE_POLICY,
            )

        assert_frame_equal(frame, frame_before)


if __name__ == "__main__":
    unittest.main()
