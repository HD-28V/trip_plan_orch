import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.backtest import advantage_column, forward_mean_column
from src.calibration import (
    CONFIGURATION_SUMMARY_COLUMNS,
    CROSS_CURRENCY_SUMMARY_COLUMNS,
    PERIOD_DATA_SUMMARY_COLUMNS,
    REPORT_METADATA_COLUMNS,
    REPORT_FILES,
    SIGNAL_HORIZON_REPORT_COLUMNS,
    CalibrationDataError,
    build_configuration_summary,
    run_calibration,
    write_calibration_reports,
)
from src.calibration_candidates import (
    CandidateConfigurationSpec,
    PolicyCandidate,
    ThresholdCandidate,
    build_candidate_plan,
)
from src.signal_engine import SignalDecisionPolicy, SignalThresholds


BASELINE_THRESHOLDS = SignalThresholds(
    sma60_good=-2.0,
    sma60_strong=-4.0,
    sma120_good=-2.0,
    sma120_strong=-4.0,
    percentile_good=25.0,
    percentile_strong=10.0,
    bollinger_near_lower_pct=1.0,
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

BALANCED_POLICY = SignalDecisionPolicy(
    minimum_available_conditions=3,
    watch_min_satisfied_conditions=1,
    good_min_satisfied_conditions=3,
    strong_min_satisfied_conditions=4,
    strong_min_strong_conditions=3,
)


def make_candidate_plan(*, include_relaxed: bool = False):
    thresholds = [ThresholdCandidate("baseline", BASELINE_THRESHOLDS)]
    specs = [
        CandidateConfigurationSpec(
            "baseline__balanced",
            "baseline",
            "balanced",
        )
    ]
    if include_relaxed:
        thresholds.append(ThresholdCandidate("relaxed", RELAXED_THRESHOLDS))
        specs.append(
            CandidateConfigurationSpec(
                "relaxed__balanced",
                "relaxed",
                "balanced",
            )
        )
    return build_candidate_plan(
        tuple(thresholds),
        (PolicyCandidate("balanced", BALANCED_POLICY),),
        tuple(specs),
        max_candidate_count=len(specs),
    )


def make_rates(
    values: object,
    *,
    start: str = "2025-01-01",
) -> pd.DataFrame:
    rates = list(values)
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(rates)),
            "rate": rates,
        }
    )


def fixed_indicator_calculator(data: pd.DataFrame) -> pd.DataFrame:
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


class CalibrationPeriodTests(unittest.TestCase):
    def test_validation_mutation_cannot_change_calibration_results(self) -> None:
        source = make_rates(1000.0 + np.arange(200))
        validation_start = source.loc[190, "date"]
        modified = source.copy(deep=True)
        modified.loc[190:, "rate"] += 10000.0
        plan = make_candidate_plan()

        original = run_calibration(
            {"USD": source},
            validation_start=validation_start,
            candidate_plan=plan,
            horizons=(5,),
            indicator_calculator=fixed_indicator_calculator,
        )
        changed = run_calibration(
            {"USD": modified},
            validation_start=validation_start,
            candidate_plan=plan,
            horizons=(5,),
            indicator_calculator=fixed_indicator_calculator,
        )

        original_calibration = original.daily_results.query(
            "period_type == 'CALIBRATION'"
        ).reset_index(drop=True)
        changed_calibration = changed.daily_results.query(
            "period_type == 'CALIBRATION'"
        ).reset_index(drop=True)
        assert_frame_equal(original_calibration, changed_calibration)
        assert_frame_equal(
            original.signal_horizon_summary.query(
                "period_type == 'CALIBRATION'"
            ).reset_index(drop=True),
            changed.signal_horizon_summary.query(
                "period_type == 'CALIBRATION'"
            ).reset_index(drop=True),
        )
        self.assertTrue(
            original_calibration[forward_mean_column(5)].tail(5).isna().all()
        )
        self.assertTrue(
            original_calibration[advantage_column(5)].tail(5).isna().all()
        )
        original_validation = original.daily_results.query(
            "period_type == 'VALIDATION'"
        )["rate"].reset_index(drop=True)
        changed_validation = changed.daily_results.query(
            "period_type == 'VALIDATION'"
        )["rate"].reset_index(drop=True)
        self.assertFalse(original_validation.equals(changed_validation))

    def test_validation_first_row_uses_calibration_history_as_warmup(self) -> None:
        source = make_rates(1000.0 + np.arange(185))
        validation_start = source.loc[180, "date"]

        report = run_calibration(
            {"USD": source},
            validation_start=validation_start,
            candidate_plan=make_candidate_plan(),
            horizons=(1,),
        )

        first_validation = report.daily_results.query(
            "period_type == 'VALIDATION'"
        ).iloc[0]
        self.assertEqual(first_validation["date"], validation_start)
        self.assertFalse(pd.isna(first_validation["SMA120"]))
        self.assertFalse(pd.isna(first_validation["percentile_rank_180"]))
        validation_metadata = report.period_data_summary.query(
            "period_type == 'VALIDATION'"
        ).iloc[0]
        self.assertEqual(validation_metadata["warmup_row_count"], 180)
        self.assertEqual(validation_metadata["replay_row_count"], 185)
        self.assertEqual(validation_metadata["evaluation_row_count"], 5)

    def test_period_split_uses_actual_observations_and_preserves_input(self) -> None:
        source = make_rates([100, 101, 102, 103, 104, 105])
        source_before = source.copy(deep=True)
        historical = {"usd": source}
        validation_start = source.loc[3, "date"]

        report = run_calibration(
            historical,
            validation_start=validation_start,
            candidate_plan=make_candidate_plan(),
            horizons=(1,),
            indicator_calculator=fixed_indicator_calculator,
        )

        assert_frame_equal(source, source_before)
        self.assertEqual(list(historical), ["usd"])
        self.assertEqual(report.currencies, ("USD",))
        counts = report.period_data_summary.set_index("period_type")[
            "evaluation_row_count"
        ].to_dict()
        self.assertEqual(counts, {"CALIBRATION": 3, "VALIDATION": 3})
        summary_counts = report.signal_horizon_summary.groupby("period_type")[
            "total_date_count"
        ].unique()
        self.assertEqual(summary_counts["CALIBRATION"].tolist(), [3])
        self.assertEqual(summary_counts["VALIDATION"].tolist(), [3])


class CalibrationReportTests(unittest.TestCase):
    def test_report_schema_and_row_counts_cover_all_dimensions(self) -> None:
        relative = np.array([1.00, 1.01, 0.99, 1.02, 0.98, 1.03, 0.97, 1.04])
        histories = {
            "USD": make_rates(relative * 1300),
            "JPY": make_rates(relative * 9),
            "EUR": make_rates(relative * 1500),
        }
        plan = make_candidate_plan(include_relaxed=True)
        split = histories["USD"].loc[4, "date"]

        report = run_calibration(
            histories,
            validation_start=split,
            candidate_plan=plan,
            horizons=(1, 2),
            indicator_calculator=fixed_indicator_calculator,
        )

        self.assertEqual(
            report.signal_horizon_summary.columns.tolist(),
            list(SIGNAL_HORIZON_REPORT_COLUMNS),
        )
        self.assertEqual(
            report.configuration_summary.columns.tolist(),
            list(CONFIGURATION_SUMMARY_COLUMNS),
        )
        self.assertEqual(
            report.cross_currency_summary.columns.tolist(),
            list(CROSS_CURRENCY_SUMMARY_COLUMNS),
        )
        self.assertEqual(
            report.period_data_summary.columns.tolist(),
            list(PERIOD_DATA_SUMMARY_COLUMNS),
        )
        self.assertEqual(
            report.report_metadata.columns.tolist(),
            list(REPORT_METADATA_COLUMNS),
        )
        self.assertEqual(len(report.signal_horizon_summary), 2 * 3 * 2 * 4 * 2)
        self.assertEqual(len(report.configuration_summary), 2 * 3 * 2 * 2)
        self.assertEqual(len(report.cross_currency_summary), 2 * 2 * 4 * 2)
        self.assertEqual(len(report.period_data_summary), 2 * 3)
        self.assertEqual(len(report.candidate_configurations), 2)

        for frame in (
            report.signal_horizon_summary,
            report.configuration_summary,
            report.cross_currency_summary,
        ):
            self.assertFalse(
                any(
                    word in column.lower()
                    for column in frame.columns
                    for word in ("winner", "rank", "score", "best", "selected")
                )
            )

    def test_same_relative_path_is_scale_independent_across_currencies(self) -> None:
        relative = np.array([1.0, 0.98, 1.01, 0.97, 1.02, 0.96, 1.03, 0.95])
        histories = {
            "USD": make_rates(relative * 1300),
            "JPY": make_rates(relative * 9),
            "EUR": make_rates(relative * 1500),
        }
        report = run_calibration(
            histories,
            validation_start=histories["USD"].loc[4, "date"],
            candidate_plan=make_candidate_plan(),
            horizons=(1,),
            indicator_calculator=fixed_indicator_calculator,
        )

        for period_type in ("CALIBRATION", "VALIDATION"):
            period_daily = report.daily_results.query(
                "period_type == @period_type"
            )
            baseline = period_daily.query("currency == 'USD'").reset_index(drop=True)
            for currency in ("JPY", "EUR"):
                current = period_daily.query(
                    "currency == @currency"
                ).reset_index(drop=True)
                self.assertEqual(current["signal"].tolist(), baseline["signal"].tolist())
                np.testing.assert_allclose(
                    current[advantage_column(1)],
                    baseline[advantage_column(1)],
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                )

        cross = report.cross_currency_summary.query(
            "period_type == 'CALIBRATION' and signal == 'WATCH'"
        ).iloc[0]
        self.assertEqual(cross["reported_currency_count"], 3)
        self.assertEqual(cross["mean_advantage_defined_currency_count"], 3)
        self.assertAlmostEqual(
            cross["USD_mean_advantage_pct"],
            cross["JPY_mean_advantage_pct"],
        )
        self.assertAlmostEqual(
            cross["USD_mean_advantage_pct"],
            cross["EUR_mean_advantage_pct"],
        )

    def test_configuration_summary_keeps_sample_coverage_and_nan_denominators(self) -> None:
        daily = pd.DataFrame(
            {
                "period_type": [
                    "CALIBRATION",
                    "CALIBRATION",
                    "CALIBRATION",
                    "VALIDATION",
                    "VALIDATION",
                ],
                "currency": ["USD"] * 5,
                "configuration_id": ["config"] * 5,
                "signal": ["GOOD", "STRONG", "WAIT", "GOOD", "WAIT"],
                advantage_column(5): [2.0, np.nan, -1.0, np.nan, 3.0],
            }
        )
        daily_before = daily.copy(deep=True)

        summary = build_configuration_summary(
            daily,
            currencies=("USD",),
            configuration_ids=("config",),
            horizons=(5,),
        )

        assert_frame_equal(daily, daily_before)
        calibration = summary.query("period_type == 'CALIBRATION'").iloc[0]
        self.assertEqual(calibration["total_date_count"], 3)
        self.assertEqual(calibration["good_strong_occurrence_count"], 2)
        self.assertEqual(calibration["good_strong_evaluable_count"], 1)
        self.assertEqual(calibration["good_strong_unavailable_count"], 1)
        self.assertEqual(calibration["good_strong_evaluation_coverage_ratio"], 0.5)
        self.assertEqual(calibration["good_strong_favorable_ratio"], 1.0)
        self.assertEqual(calibration["strong_occurrence_count"], 1)
        self.assertEqual(calibration["strong_evaluable_count"], 0)

        validation = summary.query("period_type == 'VALIDATION'").iloc[0]
        self.assertEqual(validation["good_strong_occurrence_count"], 1)
        self.assertEqual(validation["good_strong_evaluable_count"], 0)
        self.assertEqual(validation["good_strong_evaluation_coverage_ratio"], 0.0)
        self.assertTrue(pd.isna(validation["good_strong_favorable_ratio"]))

    def test_nan_rates_remain_unavailable_without_crashing_reports(self) -> None:
        source = make_rates([100.0, 101.0, np.nan, 103.0, 104.0, 105.0])
        report = run_calibration(
            {"USD": source},
            validation_start=source.loc[3, "date"],
            candidate_plan=make_candidate_plan(),
            horizons=(1,),
            indicator_calculator=fixed_indicator_calculator,
        )

        nan_row = report.daily_results.loc[report.daily_results["rate"].isna()].iloc[0]
        self.assertEqual(nan_row["signal"], "WAIT")
        self.assertTrue(pd.isna(nan_row[advantage_column(1)]))
        absent_cross = report.cross_currency_summary.query(
            "signal == 'STRONG'"
        ).iloc[0]
        self.assertEqual(absent_cross["minimum_occurrence_count"], 0)
        self.assertEqual(absent_cross["mean_advantage_defined_currency_count"], 0)
        self.assertTrue(
            pd.isna(absent_cross["mean_advantage_across_currencies_pct"])
        )

    def test_all_configurations_use_the_same_historical_labels(self) -> None:
        source = make_rates([100.0, 102.0, 101.0, 103.0, 99.0, 104.0])
        report = run_calibration(
            {"USD": source},
            validation_start=source.loc[3, "date"],
            candidate_plan=make_candidate_plan(include_relaxed=True),
            horizons=(1,),
            indicator_calculator=fixed_indicator_calculator,
        )

        grouped = report.daily_results.groupby(
            ["period_type", "configuration_id"],
            sort=False,
        )
        for period in ("CALIBRATION", "VALIDATION"):
            baseline = grouped.get_group(
                (period, "baseline__balanced")
            )[["date", "rate", forward_mean_column(1), advantage_column(1)]].reset_index(
                drop=True
            )
            relaxed = grouped.get_group(
                (period, "relaxed__balanced")
            )[["date", "rate", forward_mean_column(1), advantage_column(1)]].reset_index(
                drop=True
            )
            assert_frame_equal(baseline, relaxed)
        self.assertTrue(
            report.daily_results.query(
                "configuration_id == 'baseline__balanced'"
            )["signal"].eq("WATCH").all()
        )
        self.assertTrue(
            report.daily_results.query(
                "configuration_id == 'relaxed__balanced'"
            )["signal"].eq("GOOD").all()
        )

    def test_writer_creates_only_compact_reports_and_refuses_overwrite(self) -> None:
        source = make_rates([100.0, 101.0, 102.0, 103.0])
        report = run_calibration(
            {"USD": source},
            validation_start=source.loc[2, "date"],
            candidate_plan=make_candidate_plan(),
            horizons=(1,),
            indicator_calculator=fixed_indicator_calculator,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "nested" / "reports"
            paths = write_calibration_reports(report, output_dir)

            self.assertEqual(
                [path.name for path in paths],
                [filename for filename, _ in REPORT_FILES],
            )
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                sorted(filename for filename, _ in REPORT_FILES),
            )
            for path in paths:
                loaded = pd.read_csv(path)
                self.assertNotIn("Unnamed: 0", loaded.columns)
            with self.assertRaises(FileExistsError):
                write_calibration_reports(report, output_dir)


class CalibrationValidationTests(unittest.TestCase):
    def test_rejects_invalid_split_currency_and_historical_contracts(self) -> None:
        source = make_rates([100.0, 101.0, 102.0])
        plan = make_candidate_plan()
        cases = (
            ({"USD": source}, source["date"].iloc[0], "calibration period"),
            (
                {"USD": source},
                source["date"].iloc[-1] + pd.Timedelta(days=1),
                "validation period",
            ),
            ({"CHF": source}, source["date"].iloc[1], "currency"),
            (
                {"usd": source, "USD": source},
                source["date"].iloc[1],
                "unique",
            ),
        )
        for histories, split, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    run_calibration(
                        histories,
                        validation_start=split,
                        candidate_plan=plan,
                        horizons=(1,),
                        indicator_calculator=fixed_indicator_calculator,
                    )

        with self.assertRaisesRegex(ValueError, "valid date"):
            run_calibration(
                {"USD": source},
                validation_start="not-a-date",
                candidate_plan=plan,
                horizons=(1,),
                indicator_calculator=fixed_indicator_calculator,
            )


if __name__ == "__main__":
    unittest.main()
