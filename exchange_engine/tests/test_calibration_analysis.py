import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.calibration import (
    CONFIGURATION_SUMMARY_COLUMNS,
    PERIOD_DATA_SUMMARY_COLUMNS,
    REPORT_METADATA_COLUMNS,
    SIGNAL_HORIZON_REPORT_COLUMNS,
    build_cross_currency_summary,
)
from src.calibration_analysis import (
    ANALYSIS_REPORT_FILES,
    AnalysisHeuristics,
    CalibrationAnalysisError,
    CalibrationInputTables,
    analyze_calibration_results,
    load_calibration_results,
    validate_calibration_results,
    write_analysis_reports,
)
from src.calibration_analysis_configuration import CONFIGURATION_REVIEW_COLUMNS
from src.calibration_analysis_output import SHORTLIST_COLUMNS
from src.calibration_analysis_reviews import (
    CURRENCY_STABILITY_REVIEW_COLUMNS,
    HORIZON_STABILITY_REVIEW_COLUMNS,
    SIGNAL_SEPARATION_REVIEW_COLUMNS,
    STABILITY_REVIEW_COLUMNS,
)
from src.calibration_candidates import build_initial_candidate_plan


PERIODS = ("CALIBRATION", "VALIDATION")
CURRENCIES = ("USD", "JPY", "EUR")
HORIZONS = (5, 10, 20, 60)
SIGNALS = ("WAIT", "WATCH", "GOOD", "STRONG")


def _signal_metrics(
    configuration_index: int,
    period: str,
    currency: str,
    signal: str,
    horizon: int,
) -> tuple[int, float, float, int]:
    occurrence = {"WAIT": 20, "WATCH": 20, "GOOD": 40, "STRONG": 40}[signal]
    calibration = {
        "WAIT": (-0.40, -0.40, 8),
        "WATCH": (0.00, 0.00, 10),
        "GOOD": (0.50, 0.40, 26),
        "STRONG": (1.00, 0.90, 30),
    }
    mean, median, favorable = calibration[signal]

    if period == "VALIDATION" and configuration_index < 2:
        mean += 0.20
        median += 0.10
        favorable += {"WAIT": 1, "WATCH": 1, "GOOD": 2, "STRONG": 2}[signal]
    elif period == "VALIDATION" and configuration_index == 2:
        if currency == "EUR" and signal == "STRONG":
            return 0, np.nan, np.nan, 0
        if signal == "STRONG":
            occurrence = 5
            favorable = 4
        if signal == "WAIT":
            occurrence = 60 if currency == "EUR" else 55
        if currency == "EUR" and signal in {"GOOD", "STRONG"}:
            mean, median = -0.40, -0.30
            favorable = int(occurrence * 0.30)
    elif period == "VALIDATION" and configuration_index >= 3:
        if horizon in {20, 60}:
            values = {
                "WAIT": (-0.10, -0.10, 8),
                "WATCH": (-0.20, -0.20, 8),
                "GOOD": (-0.40, -0.30, 12),
                "STRONG": (-0.80, -0.70, 8),
            }
        elif currency == "EUR":
            values = {
                "WAIT": (-0.10, -0.10, 8),
                "WATCH": (-0.20, -0.20, 8),
                "GOOD": (-0.40, -0.30, 12),
                "STRONG": (-0.80, -0.70, 8),
            }
        else:
            values = {
                "WAIT": (-0.30, -0.20, 8),
                "WATCH": (0.10, 0.10, 12),
                "GOOD": (0.60, 0.50, 28),
                "STRONG": (0.20, 0.10, 16),
            }
        mean, median, favorable = values[signal]
    return occurrence, mean, median, favorable


def make_signal_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period in PERIODS:
        for currency in CURRENCIES:
            for configuration_index, configuration_id in enumerate(
                manifest["configuration_id"]
            ):
                for signal in SIGNALS:
                    for horizon in HORIZONS:
                        occurrence, mean, median, favorable = _signal_metrics(
                            configuration_index,
                            period,
                            currency,
                            signal,
                            horizon,
                        )
                        evaluable = occurrence
                        unavailable = 0
                        unfavorable = evaluable - favorable
                        rows.append(
                            {
                                "period_type": period,
                                "currency": currency,
                                "configuration_id": configuration_id,
                                "signal": signal,
                                "horizon": horizon,
                                "total_date_count": 120,
                                "occurrence_count": occurrence,
                                "occurrence_ratio": occurrence / 120,
                                "evaluable_count": evaluable,
                                "unavailable_count": unavailable,
                                "evaluation_coverage_ratio": (
                                    1.0 if occurrence else np.nan
                                ),
                                "mean_advantage_pct": mean,
                                "median_advantage_pct": median,
                                "favorable_count": favorable,
                                "favorable_ratio": (
                                    favorable / evaluable if evaluable else np.nan
                                ),
                                "neutral_count": 0,
                                "neutral_ratio": 0.0 if evaluable else np.nan,
                                "unfavorable_count": unfavorable,
                                "unfavorable_ratio": (
                                    unfavorable / evaluable if evaluable else np.nan
                                ),
                            }
                        )
    return pd.DataFrame.from_records(rows, columns=SIGNAL_HORIZON_REPORT_COLUMNS)


def make_configuration_summary(signal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["period_type", "currency", "configuration_id", "horizon"]
    selected = signal.loc[signal["signal"].isin(("GOOD", "STRONG"))]
    for key, group in selected.groupby(keys, sort=False):
        period, currency, configuration_id, horizon = key
        total = int(group["total_date_count"].iloc[0])
        occurrence = int(group["occurrence_count"].sum())
        evaluable = int(group["evaluable_count"].sum())
        unavailable = int(group["unavailable_count"].sum())
        favorable = int(group["favorable_count"].sum())
        neutral = int(group["neutral_count"].sum())
        unfavorable = int(group["unfavorable_count"].sum())
        weighted_mean = (
            float(
                (
                    group["mean_advantage_pct"].fillna(0)
                    * group["evaluable_count"]
                ).sum()
                / evaluable
            )
            if evaluable
            else np.nan
        )
        weighted_median = (
            float(
                (
                    group["median_advantage_pct"].fillna(0)
                    * group["evaluable_count"]
                ).sum()
                / evaluable
            )
            if evaluable
            else np.nan
        )
        strong = group.loc[group["signal"].eq("STRONG")].iloc[0]
        rows.append(
            {
                "period_type": period,
                "currency": currency,
                "configuration_id": configuration_id,
                "horizon": horizon,
                "total_date_count": total,
                "good_strong_occurrence_count": occurrence,
                "good_strong_occurrence_ratio": occurrence / total,
                "good_strong_evaluable_count": evaluable,
                "good_strong_unavailable_count": unavailable,
                "good_strong_evaluation_coverage_ratio": (
                    evaluable / occurrence if occurrence else np.nan
                ),
                "good_strong_mean_advantage_pct": weighted_mean,
                "good_strong_median_advantage_pct": weighted_median,
                "good_strong_favorable_count": favorable,
                "good_strong_favorable_ratio": (
                    favorable / evaluable if evaluable else np.nan
                ),
                "good_strong_neutral_count": neutral,
                "good_strong_neutral_ratio": (
                    neutral / evaluable if evaluable else np.nan
                ),
                "good_strong_unfavorable_count": unfavorable,
                "good_strong_unfavorable_ratio": (
                    unfavorable / evaluable if evaluable else np.nan
                ),
                "strong_occurrence_count": int(strong["occurrence_count"]),
                "strong_evaluable_count": int(strong["evaluable_count"]),
            }
        )
    return pd.DataFrame.from_records(rows, columns=CONFIGURATION_SUMMARY_COLUMNS)


def make_metadata() -> pd.DataFrame:
    rows = (
        ("report_purpose", "historical candidate comparison; no winner selected"),
        ("advantage_pct", "(forward_mean_rate - entry_rate) / entry_rate * 100"),
        ("favorable", "advantage_pct > 0"),
        ("neutral", "advantage_pct == 0"),
        ("unfavorable", "advantage_pct < 0"),
        ("horizon_unit", "subsequent trading observations; entry excluded"),
        ("calibration_period", "date < 2020-07-01"),
        ("validation_period", "date >= 2020-07-01"),
        ("occurrence_ratio_denominator", "all evaluation dates in period"),
        ("coverage_denominator", "signal occurrence_count"),
        ("direction_ratio_denominator", "non-NaN evaluable_count"),
        ("cross_currency_weighting", "equal weight per defined currency metric"),
        ("cross_currency_stddev", "population standard deviation (ddof=0)"),
        ("currencies", "USD,JPY,EUR"),
        ("horizons", "5,10,20,60"),
        ("candidate_configuration_count", "12"),
        ("total_runtime_seconds", "1.000000"),
    )
    return pd.DataFrame(rows, columns=REPORT_METADATA_COLUMNS)


def make_period_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period in PERIODS:
        for currency in CURRENCIES:
            calibration = period == "CALIBRATION"
            rows.append(
                {
                    "period_type": period,
                    "currency": currency,
                    "source_start_date": "2020-01-01",
                    "source_end_date": "2020-12-31",
                    "source_row_count": 240,
                    "replay_row_count": 120 if calibration else 240,
                    "warmup_row_count": 0 if calibration else 120,
                    "evaluation_start_date": (
                        "2020-01-01" if calibration else "2020-07-01"
                    ),
                    "evaluation_end_date": (
                        "2020-06-30" if calibration else "2020-12-31"
                    ),
                    "evaluation_row_count": 120,
                    "candidate_configuration_count": 12,
                    "elapsed_seconds": 0.1,
                }
            )
    return pd.DataFrame.from_records(rows, columns=PERIOD_DATA_SUMMARY_COLUMNS)


def make_tables(*, source_dir: Path | None = None) -> CalibrationInputTables:
    manifest = build_initial_candidate_plan().to_manifest()
    signal = make_signal_summary(manifest)
    configuration = make_configuration_summary(signal)
    cross = build_cross_currency_summary(
        signal,
        currencies=CURRENCIES,
        configuration_ids=manifest["configuration_id"].tolist(),
        horizons=HORIZONS,
    )
    return CalibrationInputTables(
        report_metadata=make_metadata(),
        candidate_configurations=manifest,
        period_data_summary=make_period_data(),
        configuration_summary=configuration,
        signal_horizon_summary=signal,
        cross_currency_summary=cross,
        source_dir=source_dir,
    )


def replace_tables(
    tables: CalibrationInputTables,
    **updates: object,
) -> CalibrationInputTables:
    values = {
        "report_metadata": tables.report_metadata.copy(deep=True),
        "candidate_configurations": tables.candidate_configurations.copy(deep=True),
        "period_data_summary": tables.period_data_summary.copy(deep=True),
        "configuration_summary": tables.configuration_summary.copy(deep=True),
        "signal_horizon_summary": tables.signal_horizon_summary.copy(deep=True),
        "cross_currency_summary": tables.cross_currency_summary.copy(deep=True),
        "source_dir": tables.source_dir,
    }
    values.update(updates)
    return CalibrationInputTables(**values)


class CalibrationAnalysisValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = make_tables()

    def test_valid_input_returns_deep_copies_without_mutating_source(self) -> None:
        before = self.tables.signal_horizon_summary.copy(deep=True)

        validated = validate_calibration_results(self.tables)

        assert_frame_equal(self.tables.signal_horizon_summary, before)
        self.assertIsNot(validated.signal_horizon_summary, self.tables.signal_horizon_summary)
        validated.signal_horizon_summary.loc[0, "mean_advantage_pct"] = 999
        self.assertNotEqual(
            validated.signal_horizon_summary.loc[0, "mean_advantage_pct"],
            self.tables.signal_horizon_summary.loc[0, "mean_advantage_pct"],
        )

    def test_rejects_duplicate_and_incomplete_cartesian_dimensions(self) -> None:
        source = self.tables.signal_horizon_summary
        corruptions = {
            "duplicate": pd.concat([source, source.iloc[[0]]], ignore_index=True),
            "currency": source.drop(source.query("currency == 'USD'").index[0]),
            "horizon": source.drop(source.query("horizon == 60").index[0]),
            "signal": source.drop(source.query("signal == 'STRONG'").index[0]),
            "configuration": source.drop(
                source.query("configuration_id == 'baseline__balanced'").index[0]
            ),
        }
        for name, corrupted in corruptions.items():
            with self.subTest(name=name):
                with self.assertRaises(CalibrationAnalysisError):
                    validate_calibration_results(
                        replace_tables(self.tables, signal_horizon_summary=corrupted)
                    )

        missing_period = self.tables.period_data_summary.iloc[1:].reset_index(drop=True)
        with self.assertRaises(CalibrationAnalysisError):
            validate_calibration_results(
                replace_tables(self.tables, period_data_summary=missing_period)
            )

    def test_rejects_wrong_candidate_count_and_metric_contracts(self) -> None:
        manifest = self.tables.candidate_configurations.iloc[:11].copy(deep=True)
        with self.assertRaises(CalibrationAnalysisError):
            validate_calibration_results(
                replace_tables(self.tables, candidate_configurations=manifest)
            )

        for column, value in (
            ("occurrence_ratio", 0.999),
            ("evaluable_count", 999),
            ("mean_advantage_pct", np.nan),
        ):
            corrupted = self.tables.signal_horizon_summary.copy(deep=True)
            corrupted.loc[0, column] = value
            with self.subTest(column=column):
                with self.assertRaises(CalibrationAnalysisError):
                    validate_calibration_results(
                        replace_tables(
                            self.tables,
                            signal_horizon_summary=corrupted,
                        )
                    )

        unavailable = self.tables.signal_horizon_summary.query(
            "period_type == 'VALIDATION' and currency == 'EUR' "
            "and configuration_id == 'baseline__sensitive' "
            "and signal == 'STRONG' and horizon == 5"
        ).index[0]
        corrupted = self.tables.signal_horizon_summary.copy(deep=True)
        self.assertTrue(pd.isna(corrupted.loc[unavailable, "favorable_ratio"]))
        corrupted.loc[unavailable, "favorable_ratio"] = 0.0
        with self.assertRaises(CalibrationAnalysisError):
            validate_calibration_results(
                replace_tables(self.tables, signal_horizon_summary=corrupted)
            )

    def test_rejects_configuration_and_cross_report_corruption(self) -> None:
        configuration = self.tables.configuration_summary.copy(deep=True)
        configuration.loc[0, "good_strong_occurrence_count"] += 1
        with self.assertRaises(CalibrationAnalysisError):
            validate_calibration_results(
                replace_tables(self.tables, configuration_summary=configuration)
            )

        cross = self.tables.cross_currency_summary.copy(deep=True)
        cross.loc[0, "minimum_occurrence_count"] += 1
        with self.assertRaises(CalibrationAnalysisError):
            validate_calibration_results(
                replace_tables(self.tables, cross_currency_summary=cross)
            )

    def test_rejects_duplicate_manifest_id_and_split_date_mismatch(self) -> None:
        manifest = self.tables.candidate_configurations.copy(deep=True)
        manifest.loc[1, "configuration_id"] = manifest.loc[0, "configuration_id"]
        with self.assertRaises(CalibrationAnalysisError):
            validate_calibration_results(
                replace_tables(self.tables, candidate_configurations=manifest)
            )

        metadata = self.tables.report_metadata.copy(deep=True)
        metadata.loc[
            metadata["field"].eq("calibration_period"), "value"
        ] = "date < 2020-07-02"
        metadata.loc[
            metadata["field"].eq("validation_period"), "value"
        ] = "date >= 2020-07-02"
        with self.assertRaises(CalibrationAnalysisError):
            validate_calibration_results(
                replace_tables(self.tables, report_metadata=metadata)
            )

    def test_rejects_non_partitioning_exclusive_signal_counts(self) -> None:
        signal = self.tables.signal_horizon_summary.copy(deep=True)
        mask = (
            signal["period_type"].eq("VALIDATION")
            & signal["currency"].eq("USD")
            & signal["configuration_id"].eq("baseline__balanced")
            & signal["signal"].eq("WAIT")
        )
        signal.loc[mask, "occurrence_count"] += 1
        signal.loc[mask, "unavailable_count"] += 1
        signal.loc[mask, "occurrence_ratio"] = (
            signal.loc[mask, "occurrence_count"]
            / signal.loc[mask, "total_date_count"]
        )
        signal.loc[mask, "evaluation_coverage_ratio"] = (
            signal.loc[mask, "evaluable_count"]
            / signal.loc[mask, "occurrence_count"]
        )
        with self.assertRaises(CalibrationAnalysisError):
            validate_calibration_results(
                replace_tables(self.tables, signal_horizon_summary=signal)
            )


class CalibrationAnalysisReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = make_tables()
        cls.report = analyze_calibration_results(cls.tables)
        cls.ids = cls.tables.candidate_configurations["configuration_id"].tolist()

    def test_analysis_emits_exact_review_dimensions_and_schemas(self) -> None:
        self.assertEqual(len(self.report.configuration_review), 12)
        self.assertEqual(len(self.report.stability_review), 576)
        self.assertEqual(len(self.report.currency_stability_review), 192)
        self.assertEqual(len(self.report.horizon_stability_review), 144)
        self.assertEqual(len(self.report.signal_separation_review), 288)
        self.assertEqual(
            self.report.configuration_review.columns.tolist(),
            list(CONFIGURATION_REVIEW_COLUMNS),
        )
        self.assertEqual(
            self.report.stability_review.columns.tolist(),
            list(STABILITY_REVIEW_COLUMNS),
        )
        self.assertEqual(
            self.report.currency_stability_review.columns.tolist(),
            list(CURRENCY_STABILITY_REVIEW_COLUMNS),
        )
        self.assertEqual(
            self.report.horizon_stability_review.columns.tolist(),
            list(HORIZON_STABILITY_REVIEW_COLUMNS),
        )
        self.assertEqual(
            self.report.signal_separation_review.columns.tolist(),
            list(SIGNAL_SEPARATION_REVIEW_COLUMNS),
        )
        required_compact_columns = {
            "calibration_good_strong_sample_count",
            "validation_good_strong_sample_count",
            "calibration_good_strong_evaluable_label_count",
            "validation_good_strong_evaluable_label_count",
            "calibration_validation_common_supported_cell_count",
            "matched_calibration_good_strong_favorable_ratio",
            "matched_validation_good_strong_favorable_ratio",
            "validation_minimum_currency_pooled_favorable_ratio",
            "validation_currency_pooled_favorable_stddev",
            "validation_minimum_currency_good_strong_sample_count",
            "risk_flags",
            "analysis_group",
        }
        self.assertTrue(
            required_compact_columns.issubset(
                self.report.configuration_review.columns
            )
        )
        for column in required_compact_columns.difference(
            {"risk_flags", "analysis_group"}
        ):
            self.assertFalse(self.report.configuration_review[column].isna().all())

    def test_stability_preserves_signed_and_absolute_gaps(self) -> None:
        row = self.report.stability_review.query(
            "configuration_id == 'baseline__conservative' and currency == 'USD' "
            "and signal == 'GOOD' and horizon == 5"
        ).iloc[0]

        self.assertAlmostEqual(row["favorable_ratio_gap"], 0.05)
        self.assertAlmostEqual(row["favorable_ratio_abs_gap"], 0.05)
        self.assertAlmostEqual(row["mean_advantage_gap_pct"], 0.20)
        self.assertAlmostEqual(row["mean_advantage_abs_gap_pct"], 0.20)
        self.assertAlmostEqual(row["median_advantage_gap_pct"], 0.10)
        self.assertAlmostEqual(row["median_advantage_abs_gap_pct"], 0.10)

    def test_currency_horizon_separation_and_sample_flags_are_evidence_aware(self) -> None:
        weak_id = self.ids[3]
        currency = self.report.currency_stability_review.query(
            "period_type == 'VALIDATION' and configuration_id == @weak_id "
            "and signal == 'GOOD' and horizon == 5"
        ).iloc[0]
        self.assertTrue(currency["currency_instability"])
        self.assertIn("CURRENCY_INSTABILITY", currency["risk_flags"])

        horizon = self.report.horizon_stability_review.query(
            "period_type == 'VALIDATION' and currency == 'USD' "
            "and configuration_id == @weak_id and signal == 'GOOD'"
        ).iloc[0]
        self.assertTrue(horizon["horizon_instability"])
        self.assertIn("HORIZON_INSTABILITY", horizon["risk_flags"])
        self.assertEqual(horizon["signal_occurrence_count"], 40)

        separation = self.report.signal_separation_review.query(
            "period_type == 'VALIDATION' and currency == 'USD' "
            "and configuration_id == @weak_id and horizon == 5"
        ).iloc[0]
        self.assertTrue(separation["strong_not_better_than_good"])
        self.assertIn("STRONG_NOT_BETTER", separation["risk_flags"])

        unavailable = self.report.signal_separation_review.query(
            "period_type == 'VALIDATION' and currency == 'EUR' "
            "and configuration_id == 'baseline__sensitive' and horizon == 5"
        ).iloc[0]
        self.assertEqual(unavailable["good_to_strong_status"], "UNAVAILABLE")
        self.assertTrue(pd.isna(unavailable["good_to_strong_favorable_ratio_gap"]))
        self.assertIn("SIGNAL_SEPARATION_UNAVAILABLE", unavailable["risk_flags"])

    def test_configuration_sample_is_not_summed_across_horizons(self) -> None:
        row = self.report.configuration_review.query(
            "configuration_id == 'baseline__conservative'"
        ).iloc[0]
        self.assertEqual(row["validation_good_strong_sample_count"], 240)
        self.assertNotEqual(row["validation_good_strong_sample_count"], 240 * 4)

    def test_groups_and_shortlist_are_unranked_and_not_filled(self) -> None:
        groups = self.report.configuration_review.set_index("configuration_id")[
            "analysis_group"
        ]
        self.assertEqual(groups.loc[self.ids[0]], "PROMISING")
        self.assertEqual(groups.loc[self.ids[1]], "PROMISING")
        self.assertEqual(groups.loc[self.ids[2]], "MIXED")
        self.assertEqual(groups.loc[self.ids[3]], "WEAK")
        self.assertEqual(
            self.report.candidate_shortlist["configuration_id"].tolist(),
            self.ids[:2],
        )
        self.assertGreaterEqual(len(self.report.candidate_shortlist), 2)
        self.assertLessEqual(len(self.report.candidate_shortlist), 5)
        forbidden = ("winner", "rank", "score", "best", "optimal")
        for frame in (
            self.report.configuration_review,
            self.report.candidate_shortlist,
            self.report.stability_review,
        ):
            self.assertFalse(
                any(word in column.lower() for column in frame for word in forbidden)
            )

    def test_compact_and_detailed_numeric_review_values_are_auditable(self) -> None:
        baseline = self.report.configuration_review.query(
            "configuration_id == 'baseline__conservative'"
        ).iloc[0]
        self.assertEqual(baseline["calibration_good_strong_sample_count"], 240)
        self.assertEqual(baseline["calibration_good_strong_evaluable_label_count"], 960)
        self.assertEqual(
            baseline["calibration_validation_common_supported_cell_count"], 12
        )
        self.assertAlmostEqual(baseline["favorable_ratio_gap"], 0.05)
        self.assertAlmostEqual(baseline["mean_advantage_gap_pct"], 0.20)
        self.assertAlmostEqual(baseline["median_advantage_gap_pct"], 0.10)
        self.assertAlmostEqual(
            baseline["validation_minimum_currency_pooled_favorable_ratio"],
            0.75,
        )
        self.assertAlmostEqual(
            baseline["validation_currency_pooled_favorable_stddev"],
            0.0,
        )
        self.assertEqual(
            baseline["validation_minimum_currency_good_strong_sample_count"],
            80,
        )

        weak_id = self.ids[3]
        currency = self.report.currency_stability_review.query(
            "period_type == 'VALIDATION' and configuration_id == @weak_id "
            "and signal == 'GOOD' and horizon == 5"
        ).iloc[0]
        self.assertAlmostEqual(currency["minimum_favorable_ratio"], 0.30)
        self.assertAlmostEqual(currency["favorable_ratio_range"], 0.40)
        self.assertAlmostEqual(
            currency["favorable_ratio_stddev"],
            0.18856180831641264,
        )
        horizon = self.report.horizon_stability_review.query(
            "period_type == 'VALIDATION' and currency == 'USD' "
            "and configuration_id == @weak_id and signal == 'GOOD'"
        ).iloc[0]
        self.assertAlmostEqual(horizon["short_to_medium_favorable_drop"], 0.40)
        self.assertAlmostEqual(horizon["short_to_medium_mean_gap_pct"], -1.00)
        self.assertAlmostEqual(horizon["short_to_medium_median_gap_pct"], -0.80)
        separation = self.report.signal_separation_review.query(
            "period_type == 'VALIDATION' and currency == 'USD' "
            "and configuration_id == @weak_id and horizon == 5"
        ).iloc[0]
        self.assertEqual(separation["wait_to_watch_status"], "IMPROVED")
        self.assertEqual(separation["watch_to_good_status"], "IMPROVED")
        self.assertEqual(separation["good_to_strong_status"], "WORSE")

        rare_id = self.ids[2]
        rare = self.report.configuration_review.query(
            "configuration_id == @rare_id"
        ).iloc[0]
        self.assertEqual(rare["validation_strong_sample_count"], 10)
        self.assertEqual(rare["validation_minimum_strong_currency_count"], 0)
        self.assertTrue(rare["strong_too_rare"])
        self.assertIn("STRONG_TOO_RARE", rare["risk_flags"])

    def test_completed_local_run_has_three_unranked_review_candidates(self) -> None:
        input_dir = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "calibration_2018_2026_validation_2024"
        )
        report = analyze_calibration_results(load_calibration_results(input_dir))

        self.assertGreaterEqual(len(report.candidate_shortlist), 2)
        self.assertLessEqual(len(report.candidate_shortlist), 5)
        self.assertEqual(
            report.candidate_shortlist["configuration_id"].tolist(),
            [
                "baseline__sensitive",
                "sma60_sensitive__balanced",
                "sma120_sensitive__balanced",
            ],
        )
        self.assertEqual(
            report.configuration_review["analysis_group"].value_counts().to_dict(),
            {"MIXED": 8, "PROMISING": 3, "WEAK": 1},
        )
        self.assertEqual(
            report.candidate_shortlist.columns.tolist(),
            list(SHORTLIST_COLUMNS),
        )
        metadata = report.analysis_metadata.set_index("field")["value"]
        self.assertIn(str(metadata["analysis_only"]).lower(), ("true", "1"))
        self.assertIn(str(metadata["winner_selected"]).lower(), ("false", "0"))
        self.assertEqual(str(metadata["ranking_or_score"]), "none")
        sma60_horizon = report.horizon_stability_review.query(
            "period_type == 'VALIDATION' and currency == 'USD' "
            "and configuration_id == 'sma60_sensitive__balanced' "
            "and signal == 'GOOD'"
        ).iloc[0]
        self.assertFalse(sma60_horizon["horizon_instability"])

    def test_analysis_does_not_mutate_any_input_frame(self) -> None:
        tables = make_tables()
        before = {
            name: getattr(tables, name).copy(deep=True)
            for name in (
                "report_metadata",
                "candidate_configurations",
                "period_data_summary",
                "configuration_summary",
                "signal_horizon_summary",
                "cross_currency_summary",
            )
        }

        analyze_calibration_results(tables, heuristics=AnalysisHeuristics())

        for name, expected in before.items():
            assert_frame_equal(getattr(tables, name), expected)


class CalibrationAnalysisWriterTests(unittest.TestCase):
    def test_writer_emits_exact_files_and_refuses_overwrite_or_source_dir(self) -> None:
        test_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=test_root) as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source"
            source_dir.mkdir()
            report = analyze_calibration_results(make_tables(source_dir=source_dir))
            output_dir = root / "analysis"

            paths = write_analysis_reports(report, output_dir)

            self.assertEqual(
                [path.name for path in paths],
                [filename for filename, _ in ANALYSIS_REPORT_FILES],
            )
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                sorted(filename for filename, _ in ANALYSIS_REPORT_FILES),
            )
            for filename, attribute in ANALYSIS_REPORT_FILES:
                path = output_dir / filename
                if filename.endswith(".csv"):
                    loaded = pd.read_csv(path)
                    self.assertNotIn("Unnamed: 0", loaded.columns)
                    self.assertEqual(
                        loaded.columns.tolist(),
                        getattr(report, attribute).columns.tolist(),
                    )
                else:
                    self.assertIn("최종", path.read_text(encoding="utf-8"))

            stability_csv = pd.read_csv(output_dir / "stability_review.csv")
            unavailable = stability_csv.query(
                "configuration_id == 'baseline__sensitive' "
                "and currency == 'EUR' and signal == 'STRONG' and horizon == 5"
            ).iloc[0]
            self.assertTrue(pd.isna(unavailable["validation_favorable_ratio"]))
            self.assertNotEqual(unavailable["validation_favorable_ratio"], 0.0)

            invalid_output = root / "invalid_analysis"
            invalid_report = replace(report, candidate_shortlist="not-a-frame")
            with self.assertRaises(TypeError):
                write_analysis_reports(invalid_report, invalid_output)
            self.assertFalse(invalid_output.exists())
            self.assertFalse((root / ".invalid_analysis.pending").exists())

            with self.assertRaises(FileExistsError):
                write_analysis_reports(report, output_dir)
            with self.assertRaises((CalibrationAnalysisError, ValueError)):
                write_analysis_reports(report, source_dir)


if __name__ == "__main__":
    unittest.main()
