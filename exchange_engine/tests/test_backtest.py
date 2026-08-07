import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.backtest import (
    STANDARD_EVALUATION_HORIZONS,
    BacktestConfiguration,
    advantage_column,
    compare_configurations,
    forward_mean_column,
    run_backtest,
    summarize_by_signal,
)
from src.indicators import INDICATOR_COLUMNS
from src.signal_engine import SignalDecisionPolicy, SignalThresholds


STRICT_THRESHOLDS = SignalThresholds(
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

RELAXED_THRESHOLDS = SignalThresholds(
    sma60_good=0.0,
    sma60_strong=-1.0,
    sma120_good=0.0,
    sma120_strong=-1.0,
    percentile_good=50.0,
    percentile_strong=25.0,
    bollinger_near_lower_pct=1.0,
)


def make_rates(values: object) -> pd.DataFrame:
    rates = list(values)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(rates), freq="D"),
            "rate": rates,
        }
    )


def fixed_indicator_calculator(data: pd.DataFrame) -> pd.DataFrame:
    """Fast deterministic test adapter that preserves the indicator contract."""
    result = data.copy(deep=True)
    rate = pd.to_numeric(result["rate"], errors="coerce").astype(float)
    available = rate.notna()
    result["rate"] = rate
    result["SMA60"] = rate
    result["SMA120"] = rate
    result["SMA60_distance_pct"] = np.where(available, 0.0, np.nan)
    result["SMA120_distance_pct"] = np.where(available, 0.0, np.nan)
    result["percentile_rank_180"] = np.where(available, 50.0, np.nan)
    result["BB_middle"] = rate
    result["BB_upper"] = rate
    result["BB_lower"] = rate
    return result


class RecordingIndicatorCalculator:
    def __init__(self) -> None:
        self.prefix_lengths: list[int] = []
        self.maximum_dates: list[pd.Timestamp] = []

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        self.prefix_lengths.append(len(data))
        self.maximum_dates.append(data["date"].max())
        return fixed_indicator_calculator(data)


class BacktestForwardEvaluationTests(unittest.TestCase):
    def test_calculates_injected_forward_horizons_and_tail_nan(self) -> None:
        source = make_rates(range(100, 170))
        result = run_backtest(
            source,
            thresholds=STRICT_THRESHOLDS,
            policy=CONSERVATIVE_POLICY,
            horizons=STANDARD_EVALUATION_HORIZONS,
            indicator_calculator=fixed_indicator_calculator,
        )

        first_expected = {
            5: (103.0, 3.0),
            10: (105.5, 5.5),
            20: (110.5, 10.5),
            60: (130.5, 30.5),
        }
        last_expected = {
            5: (64, 167.0, 3.0 / 164.0 * 100),
            10: (59, 164.5, 5.5 / 159.0 * 100),
            20: (49, 159.5, 10.5 / 149.0 * 100),
            60: (9, 139.5, 30.5 / 109.0 * 100),
        }

        self.assertEqual(result.horizons, STANDARD_EVALUATION_HORIZONS)
        for horizon in STANDARD_EVALUATION_HORIZONS:
            with self.subTest(horizon=horizon):
                mean_column = forward_mean_column(horizon)
                pct_column = advantage_column(horizon)
                expected_mean, expected_pct = first_expected[horizon]
                self.assertAlmostEqual(result.daily_results.loc[0, mean_column], expected_mean)
                self.assertAlmostEqual(result.daily_results.loc[0, pct_column], expected_pct)

                last_index, last_mean, last_pct = last_expected[horizon]
                self.assertAlmostEqual(
                    result.daily_results.loc[last_index, mean_column],
                    last_mean,
                )
                self.assertAlmostEqual(
                    result.daily_results.loc[last_index, pct_column],
                    last_pct,
                )
                self.assertTrue(
                    result.daily_results.loc[last_index + 1 :, mean_column]
                    .isna()
                    .all()
                )
                self.assertTrue(
                    result.daily_results.loc[last_index + 1 :, pct_column]
                    .isna()
                    .all()
                )

    def test_advantage_direction_matches_lower_entry_rate(self) -> None:
        result = run_backtest(
            make_rates([100.0, 110.0, 110.0, 120.0, 110.0, 110.0]),
            thresholds=STRICT_THRESHOLDS,
            policy=CONSERVATIVE_POLICY,
            horizons=(2,),
            indicator_calculator=fixed_indicator_calculator,
        )

        advantages = result.daily_results[advantage_column(2)]
        self.assertAlmostEqual(advantages.iloc[0], 10.0)
        self.assertGreater(advantages.iloc[0], 0)
        self.assertAlmostEqual(advantages.iloc[3], -10.0 / 120.0 * 100)
        self.assertLess(advantages.iloc[3], 0)

        watch_summary = result.summary.query("signal == 'WATCH'").iloc[0]
        self.assertEqual(watch_summary["favorable_count"], 3)
        self.assertEqual(watch_summary["unfavorable_count"], 1)

    def test_nan_requires_a_complete_future_window_and_valid_entry(self) -> None:
        result = run_backtest(
            make_rates([100.0, 101.0, np.nan, 103.0, 104.0, 105.0, 106.0]),
            thresholds=STRICT_THRESHOLDS,
            policy=CONSERVATIVE_POLICY,
            horizons=(2,),
            indicator_calculator=fixed_indicator_calculator,
        )

        expected_means = pd.Series(
            [np.nan, np.nan, 103.5, 104.5, 105.5, np.nan, np.nan],
            name=forward_mean_column(2),
        )
        expected_advantages = pd.Series(
            [
                np.nan,
                np.nan,
                np.nan,
                1.5 / 103.0 * 100,
                1.5 / 104.0 * 100,
                np.nan,
                np.nan,
            ],
            name=advantage_column(2),
        )
        assert_series_equal(
            result.daily_results[forward_mean_column(2)],
            expected_means,
        )
        assert_series_equal(
            result.daily_results[advantage_column(2)],
            expected_advantages,
        )
        self.assertEqual(result.daily_results.loc[2, "signal"], "WAIT")
        self.assertEqual(
            result.daily_results.loc[2, "available_condition_count"],
            0,
        )


class BacktestLookAheadTests(unittest.TestCase):
    def test_future_mutation_changes_labels_but_not_past_indicators_or_signals(self) -> None:
        source = make_rates(1000.0 + np.arange(185))
        modified = source.copy(deep=True)
        cutoff = 179
        modified.loc[cutoff + 1 :, "rate"] += 10000.0

        original_result = run_backtest(
            source,
            thresholds=STRICT_THRESHOLDS,
            policy=CONSERVATIVE_POLICY,
            horizons=(5,),
        )
        modified_result = run_backtest(
            modified,
            thresholds=STRICT_THRESHOLDS,
            policy=CONSERVATIVE_POLICY,
            horizons=(5,),
        )

        causal_columns = [
            "date",
            "rate",
            "signal",
            *INDICATOR_COLUMNS,
            "available_condition_count",
            "satisfied_condition_count",
            "strong_condition_count",
        ]
        assert_frame_equal(
            original_result.daily_results.loc[:cutoff, causal_columns],
            modified_result.daily_results.loc[:cutoff, causal_columns],
        )
        row = original_result.daily_results.loc[cutoff]
        self.assertAlmostEqual(row["SMA60"], 1149.5)
        self.assertAlmostEqual(row["SMA120"], 1119.5)
        self.assertAlmostEqual(row["percentile_rank_180"], 100.0)
        self.assertEqual(row["signal"], "WAIT")
        self.assertAlmostEqual(row[forward_mean_column(5)], 1182.0)
        self.assertAlmostEqual(
            modified_result.daily_results.loc[cutoff, forward_mean_column(5)],
            11182.0,
        )

    def test_indicator_calculator_receives_only_each_sorted_prefix(self) -> None:
        source = make_rates([106, 101, 104, 102, 105, 103]).iloc[
            [5, 1, 4, 0, 3, 2]
        ]
        recorder = RecordingIndicatorCalculator()

        result = run_backtest(
            source,
            thresholds=STRICT_THRESHOLDS,
            policy=CONSERVATIVE_POLICY,
            horizons=(1,),
            indicator_calculator=recorder,
        )

        self.assertEqual(recorder.prefix_lengths, [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            recorder.maximum_dates,
            result.daily_results["date"].tolist(),
        )
        self.assertTrue(result.daily_results["date"].is_monotonic_increasing)


class BacktestSummaryTests(unittest.TestCase):
    def test_summary_covers_every_signal_horizon_and_valid_denominators(self) -> None:
        daily = pd.DataFrame(
            {
                "signal": [
                    "WAIT",
                    "WAIT",
                    "WATCH",
                    "WATCH",
                    "GOOD",
                    "GOOD",
                    "STRONG",
                    "STRONG",
                ],
                advantage_column(5): [-2, 0, 1, 3, -1, 4, 2, 6],
                advantage_column(10): [-4, np.nan, 2, 4, 0, -2, 5, np.nan],
            }
        )
        daily_before = daily.copy(deep=True)

        summary = summarize_by_signal(daily, horizons=(5, 10))

        assert_frame_equal(daily, daily_before)
        self.assertEqual(len(summary), 8)
        self.assertEqual(
            list(zip(summary["signal"], summary["horizon"])),
            [
                ("WAIT", 5),
                ("WAIT", 10),
                ("WATCH", 5),
                ("WATCH", 10),
                ("GOOD", 5),
                ("GOOD", 10),
                ("STRONG", 5),
                ("STRONG", 10),
            ],
        )
        self.assertTrue(summary["occurrence_count"].eq(2).all())
        self.assertTrue(summary["occurrence_ratio"].eq(0.25).all())

        wait_5 = summary.query("signal == 'WAIT' and horizon == 5").iloc[0]
        self.assertAlmostEqual(wait_5["mean_advantage_pct"], -1.0)
        self.assertAlmostEqual(wait_5["median_advantage_pct"], -1.0)
        self.assertEqual(wait_5["favorable_ratio"], 0.0)
        self.assertEqual(wait_5["neutral_ratio"], 0.5)
        self.assertEqual(wait_5["unfavorable_ratio"], 0.5)

        strong_10 = summary.query(
            "signal == 'STRONG' and horizon == 10"
        ).iloc[0]
        self.assertEqual(strong_10["evaluable_count"], 1)
        self.assertEqual(strong_10["unavailable_count"], 1)
        self.assertEqual(strong_10["evaluation_coverage_ratio"], 0.5)
        self.assertEqual(strong_10["favorable_ratio"], 1.0)

        good_10 = summary.query("signal == 'GOOD' and horizon == 10").iloc[0]
        self.assertAlmostEqual(good_10["mean_advantage_pct"], -1.0)
        self.assertEqual(good_10["neutral_ratio"], 0.5)
        self.assertEqual(good_10["unfavorable_ratio"], 0.5)

    def test_summary_keeps_unseen_signals_with_unavailable_metrics(self) -> None:
        daily = pd.DataFrame(
            {
                "signal": ["WAIT", "WAIT"],
                advantage_column(5): [np.nan, np.nan],
            }
        )

        summary = summarize_by_signal(daily, horizons=(5,))

        self.assertEqual(summary["signal"].tolist(), ["WAIT", "WATCH", "GOOD", "STRONG"])
        absent = summary.query("signal == 'GOOD'").iloc[0]
        self.assertEqual(absent["occurrence_count"], 0)
        self.assertEqual(absent["occurrence_ratio"], 0.0)
        self.assertEqual(absent["evaluable_count"], 0)
        self.assertTrue(pd.isna(absent["mean_advantage_pct"]))
        self.assertTrue(pd.isna(absent["favorable_ratio"]))

        wait = summary.query("signal == 'WAIT'").iloc[0]
        self.assertEqual(wait["occurrence_count"], 2)
        self.assertEqual(wait["evaluable_count"], 0)
        self.assertEqual(wait["unavailable_count"], 2)
        self.assertEqual(wait["evaluation_coverage_ratio"], 0.0)
        self.assertTrue(pd.isna(wait["favorable_ratio"]))
        self.assertTrue(pd.isna(wait["unfavorable_ratio"]))


class BacktestConfigurationTests(unittest.TestCase):
    def test_candidates_share_one_replay_and_use_injected_thresholds_and_policy(self) -> None:
        relaxed_policy = replace(
            CONSERVATIVE_POLICY,
            good_min_satisfied_conditions=1,
        )
        candidates = (
            BacktestConfiguration(
                "strict",
                STRICT_THRESHOLDS,
                CONSERVATIVE_POLICY,
            ),
            BacktestConfiguration(
                "relaxed-thresholds",
                RELAXED_THRESHOLDS,
                CONSERVATIVE_POLICY,
            ),
            BacktestConfiguration(
                "relaxed-policy",
                STRICT_THRESHOLDS,
                relaxed_policy,
            ),
        )
        recorder = RecordingIndicatorCalculator()
        source = make_rates([100.0, 101.0, 102.0, 103.0])

        comparison = compare_configurations(
            source,
            candidates,
            horizons=(2,),
            indicator_calculator=recorder,
        )

        self.assertEqual(comparison.configurations, candidates)
        self.assertEqual(comparison.horizons, (2,))
        self.assertEqual(recorder.prefix_lengths, [1, 2, 3, 4])
        self.assertEqual(
            comparison.summary["configuration_id"].value_counts().to_dict(),
            {
                "strict": 4,
                "relaxed-thresholds": 4,
                "relaxed-policy": 4,
            },
        )
        grouped = comparison.daily_results.groupby(
            "configuration_id",
            sort=False,
        )
        self.assertTrue(grouped.get_group("strict")["signal"].eq("WATCH").all())
        self.assertTrue(
            grouped.get_group("relaxed-thresholds")["signal"].eq("GOOD").all()
        )
        self.assertTrue(
            grouped.get_group("relaxed-policy")["signal"].eq("GOOD").all()
        )

        shared_columns = [
            "date",
            "rate",
            *INDICATOR_COLUMNS,
            forward_mean_column(2),
            advantage_column(2),
        ]
        strict_shared = grouped.get_group("strict")[shared_columns].reset_index(drop=True)
        for identifier in ("relaxed-thresholds", "relaxed-policy"):
            assert_frame_equal(
                strict_shared,
                grouped.get_group(identifier)[shared_columns].reset_index(drop=True),
            )

    def test_rejects_duplicate_or_malformed_configurations(self) -> None:
        candidate = BacktestConfiguration(
            "candidate",
            STRICT_THRESHOLDS,
            CONSERVATIVE_POLICY,
        )
        source = make_rates([100.0, 101.0])

        with self.assertRaisesRegex(ValueError, "unique"):
            compare_configurations(
                source,
                (candidate, candidate),
                horizons=(1,),
                indicator_calculator=fixed_indicator_calculator,
            )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            compare_configurations(
                source,
                (),
                horizons=(1,),
                indicator_calculator=fixed_indicator_calculator,
            )
        with self.assertRaisesRegex(ValueError, "whitespace"):
            BacktestConfiguration(
                " candidate ",
                STRICT_THRESHOLDS,
                CONSERVATIVE_POLICY,
            )


class BacktestInputContractTests(unittest.TestCase):
    def test_input_is_unchanged_and_output_uses_chronological_order(self) -> None:
        source = pd.DataFrame(
            {
                "date": ["2025-01-03", "2025-01-01", "2025-01-02"],
                "rate": [103, 101, 102],
                "ignored": [3, 1, 2],
            },
            index=[30, 10, 20],
        )
        source_before = source.copy(deep=True)

        single = run_backtest(
            source,
            thresholds=STRICT_THRESHOLDS,
            policy=CONSERVATIVE_POLICY,
            horizons=(1,),
            indicator_calculator=fixed_indicator_calculator,
        )
        compare_configurations(
            source,
            (
                BacktestConfiguration(
                    "only",
                    STRICT_THRESHOLDS,
                    CONSERVATIVE_POLICY,
                ),
            ),
            horizons=(1,),
            indicator_calculator=fixed_indicator_calculator,
        )

        assert_frame_equal(source, source_before)
        self.assertEqual(
            single.daily_results["rate"].tolist(),
            [101.0, 102.0, 103.0],
        )
        self.assertTrue(single.daily_results["date"].is_monotonic_increasing)

    def test_rate_scale_does_not_change_signals_or_percentage_metrics(self) -> None:
        relative_path = 1.0 - np.arange(185) * 0.001
        scales = {"JPY": 9.0, "USD": 1300.0, "EUR": 1500.0}
        scale_thresholds = SignalThresholds(
            sma60_good=100.0,
            sma60_strong=-100.0,
            sma120_good=100.0,
            sma120_strong=-100.0,
            percentile_good=100.0,
            percentile_strong=0.0,
            bollinger_near_lower_pct=100.0,
        )
        results = {
            currency: run_backtest(
                make_rates(relative_path * scale),
                thresholds=scale_thresholds,
                policy=CONSERVATIVE_POLICY,
                horizons=(5,),
            ).daily_results
            for currency, scale in scales.items()
        }
        baseline = results["USD"]
        percentage_columns = [
            "SMA60_distance_pct",
            "SMA120_distance_pct",
            "percentile_rank_180",
            advantage_column(5),
        ]
        categorical_columns = [
            "signal",
            "available_condition_count",
            "satisfied_condition_count",
            "strong_condition_count",
        ]
        raw_columns = [
            "rate",
            "SMA60",
            "SMA120",
            "BB_middle",
            "BB_upper",
            "BB_lower",
            forward_mean_column(5),
        ]

        for currency, scale in scales.items():
            with self.subTest(currency=currency):
                current = results[currency]
                assert_frame_equal(
                    current[categorical_columns],
                    baseline[categorical_columns],
                )
                np.testing.assert_allclose(
                    current[percentage_columns],
                    baseline[percentage_columns],
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                )
                np.testing.assert_allclose(
                    current[raw_columns] / scale,
                    baseline[raw_columns] / scales["USD"],
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                )

    def test_rejects_invalid_horizons_and_accepts_long_horizon_as_nan(self) -> None:
        source = make_rates([100.0, 101.0, 102.0])
        invalid_cases = (
            ((), ValueError),
            ((0,), ValueError),
            ((-1,), ValueError),
            ((True,), TypeError),
            ((5.0,), TypeError),
            (("5",), TypeError),
            ((5, 5), ValueError),
            (None, TypeError),
        )

        for horizons, expected_error in invalid_cases:
            with self.subTest(horizons=horizons):
                with self.assertRaises(expected_error):
                    run_backtest(
                        source,
                        thresholds=STRICT_THRESHOLDS,
                        policy=CONSERVATIVE_POLICY,
                        horizons=horizons,
                        indicator_calculator=fixed_indicator_calculator,
                    )

        with self.assertRaises(TypeError):
            run_backtest(
                source,
                thresholds=STRICT_THRESHOLDS,
                policy=CONSERVATIVE_POLICY,
                indicator_calculator=fixed_indicator_calculator,
            )

        result = run_backtest(
            source,
            thresholds=STRICT_THRESHOLDS,
            policy=CONSERVATIVE_POLICY,
            horizons=(10,),
            indicator_calculator=fixed_indicator_calculator,
        )
        self.assertTrue(result.daily_results[forward_mean_column(10)].isna().all())
        self.assertTrue(result.daily_results[advantage_column(10)].isna().all())

    def test_rejects_invalid_settings_and_indicator_contract(self) -> None:
        source = make_rates([100.0, 101.0])

        with self.assertRaisesRegex(TypeError, "thresholds"):
            run_backtest(
                source,
                thresholds=object(),
                policy=CONSERVATIVE_POLICY,
                horizons=(1,),
                indicator_calculator=fixed_indicator_calculator,
            )
        with self.assertRaisesRegex(TypeError, "policy"):
            run_backtest(
                source,
                thresholds=STRICT_THRESHOLDS,
                policy=object(),
                horizons=(1,),
                indicator_calculator=fixed_indicator_calculator,
            )
        with self.assertRaisesRegex(TypeError, "callable"):
            run_backtest(
                source,
                thresholds=STRICT_THRESHOLDS,
                policy=CONSERVATIVE_POLICY,
                horizons=(1,),
                indicator_calculator=None,
            )

        def malformed_calculator(data: pd.DataFrame) -> pd.DataFrame:
            return fixed_indicator_calculator(data).drop(columns=["BB_lower"])

        with self.assertRaisesRegex(ValueError, "BB_lower"):
            run_backtest(
                source,
                thresholds=STRICT_THRESHOLDS,
                policy=CONSERVATIVE_POLICY,
                horizons=(1,),
                indicator_calculator=malformed_calculator,
            )

    def test_rejects_empty_duplicate_dates_and_nonpositive_rates(self) -> None:
        cases = (
            (pd.DataFrame(columns=["date", "rate"]), "must not be empty"),
            (
                pd.DataFrame(
                    {"date": ["2025-01-01", "2025-01-01"], "rate": [100, 101]}
                ),
                "one observation per date",
            ),
            (
                pd.DataFrame(
                    {
                        "date": [
                            "2025-01-01 09:00:00",
                            "2025-01-01 16:00:00",
                        ],
                        "rate": [100, 101],
                    }
                ),
                "one observation per date",
            ),
            (make_rates([100.0, 0.0]), "greater than zero"),
        )
        for source, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    run_backtest(
                        source,
                        thresholds=STRICT_THRESHOLDS,
                        policy=CONSERVATIVE_POLICY,
                        horizons=(1,),
                        indicator_calculator=fixed_indicator_calculator,
                    )


if __name__ == "__main__":
    unittest.main()
