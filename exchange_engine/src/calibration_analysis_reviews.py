"""Deterministic review tables for completed calibration summaries.

The functions in this module are deliberately duck typed so the orchestration
module can own its input-table and heuristic dataclasses without creating a
circular import.  Every function copies caller-owned frames, emits a stable
schema, and produces descriptive review flags only.  No score, ranking,
winner, or production setting is created here.
"""

from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd

from src.calibration_analysis_helpers import *  # noqa: F403


PERIODS = ("CALIBRATION", "VALIDATION")
CURRENCIES = ("USD", "JPY", "EUR")
HORIZONS = (5, 10, 20, 60)
SIGNALS = ("WAIT", "WATCH", "GOOD", "STRONG")
REVIEW_SIGNALS = ("GOOD", "STRONG")
SEPARATION_PAIRS = (
    ("wait_to_watch", "WAIT", "WATCH"),
    ("watch_to_good", "WATCH", "GOOD"),
    ("good_to_strong", "GOOD", "STRONG"),
)

_SIGNAL_REQUIRED_COLUMNS = {
    "period_type",
    "currency",
    "configuration_id",
    "signal",
    "horizon",
    "total_date_count",
    "occurrence_count",
    "occurrence_ratio",
    "evaluable_count",
    "unavailable_count",
    "evaluation_coverage_ratio",
    "mean_advantage_pct",
    "median_advantage_pct",
    "favorable_count",
    "favorable_ratio",
    "neutral_count",
    "neutral_ratio",
    "unfavorable_count",
    "unfavorable_ratio",
}

_CONFIGURATION_REQUIRED_COLUMNS = {
    "period_type",
    "currency",
    "configuration_id",
    "horizon",
    "total_date_count",
    "good_strong_occurrence_count",
    "good_strong_occurrence_ratio",
    "good_strong_evaluable_count",
    "good_strong_unavailable_count",
    "good_strong_evaluation_coverage_ratio",
    "good_strong_mean_advantage_pct",
    "good_strong_median_advantage_pct",
    "good_strong_favorable_count",
    "good_strong_favorable_ratio",
    "good_strong_unfavorable_count",
    "good_strong_unfavorable_ratio",
    "strong_occurrence_count",
    "strong_evaluable_count",
}

STABILITY_REVIEW_COLUMNS = (
    "configuration_id",
    "currency",
    "signal",
    "horizon",
    "calibration_occurrence_count",
    "validation_occurrence_count",
    "occurrence_count_gap",
    "occurrence_count_abs_gap",
    "calibration_occurrence_ratio",
    "validation_occurrence_ratio",
    "occurrence_ratio_gap",
    "occurrence_ratio_abs_gap",
    "calibration_evaluable_count",
    "validation_evaluable_count",
    "evaluable_count_gap",
    "evaluable_count_abs_gap",
    "calibration_evaluation_coverage_ratio",
    "validation_evaluation_coverage_ratio",
    "coverage_gap",
    "coverage_abs_gap",
    "calibration_favorable_ratio",
    "validation_favorable_ratio",
    "favorable_ratio_gap",
    "favorable_ratio_abs_gap",
    "calibration_mean_advantage_pct",
    "validation_mean_advantage_pct",
    "mean_advantage_gap_pct",
    "mean_advantage_abs_gap_pct",
    "calibration_median_advantage_pct",
    "validation_median_advantage_pct",
    "median_advantage_gap_pct",
    "median_advantage_abs_gap_pct",
    "calibration_unfavorable_ratio",
    "validation_unfavorable_ratio",
    "unfavorable_ratio_gap",
    "unfavorable_ratio_abs_gap",
    "calibration_supported",
    "validation_supported",
    "calibration_adequate",
    "validation_adequate",
    "performance_gap_available",
    "calibration_very_low_sample",
    "validation_very_low_sample",
    "calibration_low_sample",
    "validation_low_sample",
    "calibration_low_coverage",
    "validation_low_coverage",
    "calibration_severe_outcome_censoring",
    "validation_severe_outcome_censoring",
    "calibration_validation_shift",
    "calibration_validation_drift",
    "risk_flags",
)

_CURRENCY_VALUE_COLUMNS = (
    "occurrence_count",
    "evaluable_count",
    "evaluation_coverage_ratio",
    "favorable_count",
    "favorable_ratio",
    "mean_advantage_pct",
    "median_advantage_pct",
    "supported",
    "adequate",
)
CURRENCY_STABILITY_REVIEW_COLUMNS = (
    "period_type",
    "configuration_id",
    "signal",
    "horizon",
    *tuple(
        f"{currency}_{column}"
        for currency in CURRENCIES
        for column in _CURRENCY_VALUE_COLUMNS
    ),
    "reported_currency_count",
    "supported_currency_count",
    "adequate_currency_count",
    "signal_occurrence_count",
    "total_evaluable_count",
    "supported_evaluable_count",
    "supported_favorable_count",
    "pooled_favorable_ratio",
    "evaluable_weighted_mean_advantage_pct",
    "median_of_supported_currency_medians_pct",
    "minimum_evaluable_count",
    "minimum_coverage_ratio",
    "minimum_favorable_ratio",
    "maximum_favorable_ratio",
    "favorable_ratio_range",
    "favorable_ratio_stddev",
    "minimum_mean_advantage_pct",
    "maximum_mean_advantage_pct",
    "mean_advantage_range_pct",
    "minimum_median_advantage_pct",
    "maximum_median_advantage_pct",
    "median_advantage_range_pct",
    "mean_direction_conflict",
    "median_direction_conflict",
    "common_currency_evidence_unavailable",
    "low_currency_sample",
    "very_low_currency_sample",
    "low_currency_coverage",
    "severe_currency_censoring",
    "currency_instability",
    "risk_flags",
)

_HORIZON_VALUE_COLUMNS = (
    "evaluable_count",
    "evaluation_coverage_ratio",
    "favorable_ratio",
    "mean_advantage_pct",
    "median_advantage_pct",
    "supported",
    "adequate",
)
HORIZON_STABILITY_REVIEW_COLUMNS = (
    "period_type",
    "currency",
    "configuration_id",
    "signal",
    "signal_occurrence_count",
    *tuple(
        f"h{horizon}_{column}"
        for horizon in HORIZONS
        for column in _HORIZON_VALUE_COLUMNS
    ),
    "supported_horizon_count",
    "adequate_horizon_count",
    "minimum_evaluable_count",
    "minimum_coverage_ratio",
    "short_supported_horizon_count",
    "medium_supported_horizon_count",
    "short_favorable_ratio_median",
    "medium_favorable_ratio_median",
    "short_to_medium_favorable_drop",
    "short_mean_advantage_pct_median",
    "medium_mean_advantage_pct_median",
    "short_to_medium_mean_gap_pct",
    "short_median_advantage_pct_median",
    "medium_median_advantage_pct_median",
    "short_to_medium_median_gap_pct",
    "mean_direction_reversal",
    "median_direction_reversal",
    "horizon_evidence_unavailable",
    "low_horizon_sample",
    "very_low_horizon_sample",
    "low_horizon_coverage",
    "severe_horizon_censoring",
    "horizon_instability",
    "risk_flags",
)

_PAIR_VALUE_COLUMNS = (
    "weaker_evaluable_count",
    "stronger_evaluable_count",
    "weaker_coverage_ratio",
    "stronger_coverage_ratio",
    "comparable",
    "favorable_ratio_gap",
    "mean_advantage_gap_pct",
    "median_advantage_gap_pct",
    "improvement_metric_count",
    "deterioration_metric_count",
    "status",
)
SIGNAL_SEPARATION_REVIEW_COLUMNS = (
    "period_type",
    "currency",
    "configuration_id",
    "horizon",
    *tuple(
        f"{pair_name}_{column}"
        for pair_name, _, _ in SEPARATION_PAIRS
        for column in _PAIR_VALUE_COLUMNS
    ),
    "watch_good_separation_failed",
    "good_strong_separation_failed",
    "strong_not_better_than_good",
    "signal_separation_unavailable",
    "risk_flags",
)

def build_stability_review(
    signal_summary: pd.DataFrame,
    *,
    heuristics: object,
    configuration_ids: Iterable[str],
) -> pd.DataFrame:
    """Compare matched CALIBRATION and VALIDATION signal cells."""

    source = _copy_frame(signal_summary, _SIGNAL_REQUIRED_COLUMNS, "signal_summary")
    identifiers = _configuration_ids(configuration_ids)
    rows: list[dict[str, object]] = []
    for configuration_id in identifiers:
        for currency in CURRENCIES:
            for signal in SIGNALS:
                for horizon in HORIZONS:
                    calibration = _one_row(
                        source,
                        period_type="CALIBRATION",
                        currency=currency,
                        configuration_id=configuration_id,
                        signal=signal,
                        horizon=horizon,
                    )
                    validation = _one_row(
                        source,
                        period_type="VALIDATION",
                        currency=currency,
                        configuration_id=configuration_id,
                        signal=signal,
                        horizon=horizon,
                    )
                    rows.append(
                        _stability_row(
                            calibration,
                            validation,
                            heuristics=heuristics,
                            configuration_id=configuration_id,
                            currency=currency,
                            signal=signal,
                            horizon=horizon,
                        )
                    )
    return pd.DataFrame.from_records(rows, columns=STABILITY_REVIEW_COLUMNS)


def build_currency_stability_review(
    signal_summary: pd.DataFrame,
    *,
    heuristics: object,
    configuration_ids: Iterable[str],
) -> pd.DataFrame:
    """Compare supported GOOD/STRONG evidence across the three currencies."""

    source = _copy_frame(signal_summary, _SIGNAL_REQUIRED_COLUMNS, "signal_summary")
    identifiers = _configuration_ids(configuration_ids)
    rows: list[dict[str, object]] = []
    for period_type in PERIODS:
        for configuration_id in identifiers:
            for signal in REVIEW_SIGNALS:
                for horizon in HORIZONS:
                    currency_rows = {
                        currency: _one_row(
                            source,
                            period_type=period_type,
                            currency=currency,
                            configuration_id=configuration_id,
                            signal=signal,
                            horizon=horizon,
                        )
                        for currency in CURRENCIES
                    }
                    rows.append(
                        _currency_row(
                            currency_rows,
                            heuristics=heuristics,
                            period_type=period_type,
                            configuration_id=configuration_id,
                            signal=signal,
                            horizon=horizon,
                        )
                    )
    return pd.DataFrame.from_records(
        rows,
        columns=CURRENCY_STABILITY_REVIEW_COLUMNS,
    )


def build_horizon_stability_review(
    signal_summary: pd.DataFrame,
    *,
    heuristics: object,
    configuration_ids: Iterable[str],
) -> pd.DataFrame:
    """Compare short and medium forward horizons without summing samples."""

    source = _copy_frame(signal_summary, _SIGNAL_REQUIRED_COLUMNS, "signal_summary")
    identifiers = _configuration_ids(configuration_ids)
    rows: list[dict[str, object]] = []
    for period_type in PERIODS:
        for currency in CURRENCIES:
            for configuration_id in identifiers:
                for signal in REVIEW_SIGNALS:
                    horizon_rows = {
                        horizon: _one_row(
                            source,
                            period_type=period_type,
                            currency=currency,
                            configuration_id=configuration_id,
                            signal=signal,
                            horizon=horizon,
                        )
                        for horizon in HORIZONS
                    }
                    rows.append(
                        _horizon_row(
                            horizon_rows,
                            heuristics=heuristics,
                            period_type=period_type,
                            currency=currency,
                            configuration_id=configuration_id,
                            signal=signal,
                        )
                    )
    return pd.DataFrame.from_records(
        rows,
        columns=HORIZON_STABILITY_REVIEW_COLUMNS,
    )


def build_signal_separation_review(
    signal_summary: pd.DataFrame,
    *,
    heuristics: object,
    configuration_ids: Iterable[str],
) -> pd.DataFrame:
    """Compare adjacent exclusive Signal groups within matched cells."""

    source = _copy_frame(signal_summary, _SIGNAL_REQUIRED_COLUMNS, "signal_summary")
    identifiers = _configuration_ids(configuration_ids)
    rows: list[dict[str, object]] = []
    for period_type in PERIODS:
        for currency in CURRENCIES:
            for configuration_id in identifiers:
                for horizon in HORIZONS:
                    signal_rows = {
                        signal: _one_row(
                            source,
                            period_type=period_type,
                            currency=currency,
                            configuration_id=configuration_id,
                            signal=signal,
                            horizon=horizon,
                        )
                        for signal in SIGNALS
                    }
                    rows.append(
                        _separation_row(
                            signal_rows,
                            heuristics=heuristics,
                            period_type=period_type,
                            currency=currency,
                            configuration_id=configuration_id,
                            horizon=horizon,
                        )
                    )
    return pd.DataFrame.from_records(
        rows,
        columns=SIGNAL_SEPARATION_REVIEW_COLUMNS,
    )


def _stability_row(
    calibration: pd.Series,
    validation: pd.Series,
    *,
    heuristics: object,
    configuration_id: str,
    currency: str,
    signal: str,
    horizon: int,
) -> dict[str, object]:
    calibration_supported = _supported(calibration, heuristics)
    validation_supported = _supported(validation, heuristics)
    calibration_adequate = _adequate(calibration, heuristics)
    validation_adequate = _adequate(validation, heuristics)
    matched_support = calibration_supported and validation_supported

    row: dict[str, object] = {
        "configuration_id": configuration_id,
        "currency": currency,
        "signal": signal,
        "horizon": horizon,
    }
    for column in ("occurrence_count", "occurrence_ratio", "evaluable_count"):
        calibration_value = _number(calibration[column])
        validation_value = _number(validation[column])
        row[f"calibration_{column}"] = calibration_value
        row[f"validation_{column}"] = validation_value
        row[f"{column}_gap"] = validation_value - calibration_value
        row[f"{column}_abs_gap"] = abs(validation_value - calibration_value)

    calibration_coverage = _number(calibration["evaluation_coverage_ratio"])
    validation_coverage = _number(validation["evaluation_coverage_ratio"])
    row.update(
        {
            "calibration_evaluation_coverage_ratio": calibration_coverage,
            "validation_evaluation_coverage_ratio": validation_coverage,
            "coverage_gap": _difference(validation_coverage, calibration_coverage),
            "coverage_abs_gap": _abs_difference(
                validation_coverage,
                calibration_coverage,
            ),
        }
    )

    metric_names = (
        "favorable_ratio",
        "mean_advantage_pct",
        "median_advantage_pct",
        "unfavorable_ratio",
    )
    for metric in metric_names:
        calibration_value = _number(calibration[metric])
        validation_value = _number(validation[metric])
        row[f"calibration_{metric}"] = calibration_value
        row[f"validation_{metric}"] = validation_value
        gap_name = {
            "favorable_ratio": "favorable_ratio_gap",
            "mean_advantage_pct": "mean_advantage_gap_pct",
            "median_advantage_pct": "median_advantage_gap_pct",
            "unfavorable_ratio": "unfavorable_ratio_gap",
        }[metric]
        abs_gap_name = gap_name.replace("_gap", "_abs_gap")
        if matched_support:
            row[gap_name] = _difference(validation_value, calibration_value)
            row[abs_gap_name] = _abs_difference(validation_value, calibration_value)
        else:
            row[gap_name] = np.nan
            row[abs_gap_name] = np.nan

    row.update(
        {
            "calibration_supported": calibration_supported,
            "validation_supported": validation_supported,
            "calibration_adequate": calibration_adequate,
            "validation_adequate": validation_adequate,
            "performance_gap_available": matched_support,
            "calibration_very_low_sample": _very_low_sample(calibration, heuristics),
            "validation_very_low_sample": _very_low_sample(validation, heuristics),
            "calibration_low_sample": _low_sample(calibration, heuristics),
            "validation_low_sample": _low_sample(validation, heuristics),
            "calibration_low_coverage": _low_coverage(calibration, heuristics),
            "validation_low_coverage": _low_coverage(validation, heuristics),
            "calibration_severe_outcome_censoring": _severe_censoring(
                calibration,
                heuristics,
            ),
            "validation_severe_outcome_censoring": _severe_censoring(
                validation,
                heuristics,
            ),
        }
    )
    shift_votes = _threshold_votes(
        (
            (row["favorable_ratio_abs_gap"], _h(heuristics, "material_favorable_gap")),
            (row["mean_advantage_abs_gap_pct"], _h(heuristics, "material_mean_gap_pct")),
            (row["median_advantage_abs_gap_pct"], _h(heuristics, "material_mean_gap_pct")),
        )
    )
    deterioration_votes = _threshold_votes(
        (
            (_negated(row["favorable_ratio_gap"]), _h(heuristics, "material_favorable_gap")),
            (_negated(row["mean_advantage_gap_pct"]), _h(heuristics, "material_mean_gap_pct")),
            (_negated(row["median_advantage_gap_pct"]), _h(heuristics, "material_mean_gap_pct")),
        )
    )
    row["calibration_validation_shift"] = bool(matched_support and shift_votes >= 2)
    row["calibration_validation_drift"] = bool(
        matched_support and deterioration_votes >= 2
    )
    flags: list[str] = []
    if not matched_support:
        flags.append("PERFORMANCE_GAP_UNAVAILABLE")
    if row["validation_very_low_sample"]:
        flags.append("VERY_LOW_SAMPLE")
    elif row["validation_low_sample"]:
        flags.append("LOW_SAMPLE")
    if row["validation_severe_outcome_censoring"]:
        flags.append("SEVERE_OUTCOME_CENSORING")
    elif row["validation_low_coverage"]:
        flags.append("LOW_COVERAGE")
    if row["calibration_validation_shift"]:
        flags.append("CALIBRATION_VALIDATION_SHIFT")
    if row["calibration_validation_drift"]:
        flags.append("CALIBRATION_VALIDATION_DRIFT")
    row["risk_flags"] = _flags(flags)
    return row


def _currency_row(
    currency_rows: dict[str, pd.Series],
    *,
    heuristics: object,
    period_type: str,
    configuration_id: str,
    signal: str,
    horizon: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "period_type": period_type,
        "configuration_id": configuration_id,
        "signal": signal,
        "horizon": horizon,
    }
    supported_rows: list[pd.Series] = []
    adequate_count = 0
    for currency, source in currency_rows.items():
        supported = _supported(source, heuristics)
        adequate = _adequate(source, heuristics)
        if supported:
            supported_rows.append(source)
        adequate_count += int(adequate)
        for column in _CURRENCY_VALUE_COLUMNS:
            if column == "supported":
                value: object = supported
            elif column == "adequate":
                value = adequate
            else:
                value = _number(source[column])
            row[f"{currency}_{column}"] = value

    supported_evaluable = sum(
        _number(source["evaluable_count"]) for source in supported_rows
    )
    supported_favorable = sum(
        _number(source["favorable_count"]) for source in supported_rows
    )
    favorable_values = _finite_values(
        source["favorable_ratio"] for source in supported_rows
    )
    mean_values = _finite_values(
        source["mean_advantage_pct"] for source in supported_rows
    )
    median_values = _finite_values(
        source["median_advantage_pct"] for source in supported_rows
    )
    mean_weighted = _weighted_mean(
        (
            _number(source["mean_advantage_pct"]),
            _number(source["evaluable_count"]),
        )
        for source in supported_rows
    )
    mean_conflict = _direction_conflict(mean_values)
    median_conflict = _direction_conflict(median_values)
    favorable_range = _range(favorable_values)
    common_unavailable = len(supported_rows) < len(CURRENCIES)
    currency_instability = bool(
        not common_unavailable
        and _at_least(
            favorable_range,
            _h(heuristics, "material_currency_favorable_range"),
        )
        and (mean_conflict or median_conflict)
    )

    row.update(
        {
            "reported_currency_count": len(currency_rows),
            "supported_currency_count": len(supported_rows),
            "adequate_currency_count": adequate_count,
            "signal_occurrence_count": sum(
                _number(source["occurrence_count"])
                for source in currency_rows.values()
            ),
            "total_evaluable_count": sum(
                _number(source["evaluable_count"])
                for source in currency_rows.values()
            ),
            "supported_evaluable_count": supported_evaluable,
            "supported_favorable_count": supported_favorable,
            "pooled_favorable_ratio": _safe_ratio(
                supported_favorable,
                supported_evaluable,
            ),
            "evaluable_weighted_mean_advantage_pct": mean_weighted,
            "median_of_supported_currency_medians_pct": _median(median_values),
            "minimum_evaluable_count": min(
                _number(source["evaluable_count"])
                for source in currency_rows.values()
            ),
            "minimum_coverage_ratio": _nan_min(
                source["evaluation_coverage_ratio"]
                for source in currency_rows.values()
            ),
            "minimum_favorable_ratio": _nan_min(favorable_values),
            "maximum_favorable_ratio": _nan_max(favorable_values),
            "favorable_ratio_range": favorable_range,
            "favorable_ratio_stddev": _population_std(favorable_values),
            "minimum_mean_advantage_pct": _nan_min(mean_values),
            "maximum_mean_advantage_pct": _nan_max(mean_values),
            "mean_advantage_range_pct": _range(mean_values),
            "minimum_median_advantage_pct": _nan_min(median_values),
            "maximum_median_advantage_pct": _nan_max(median_values),
            "median_advantage_range_pct": _range(median_values),
            "mean_direction_conflict": mean_conflict,
            "median_direction_conflict": median_conflict,
            "common_currency_evidence_unavailable": common_unavailable,
            "low_currency_sample": any(
                _low_sample(source, heuristics) for source in currency_rows.values()
            ),
            "very_low_currency_sample": any(
                _very_low_sample(source, heuristics)
                for source in currency_rows.values()
            ),
            "low_currency_coverage": any(
                _low_coverage(source, heuristics) for source in currency_rows.values()
            ),
            "severe_currency_censoring": any(
                _severe_censoring(source, heuristics)
                for source in currency_rows.values()
            ),
            "currency_instability": currency_instability,
        }
    )
    flags: list[str] = []
    if row["very_low_currency_sample"]:
        flags.append("VERY_LOW_SAMPLE")
    elif row["low_currency_sample"]:
        flags.append("LOW_CURRENCY_SAMPLE")
    if row["severe_currency_censoring"]:
        flags.append("SEVERE_OUTCOME_CENSORING")
    elif row["low_currency_coverage"]:
        flags.append("LOW_COVERAGE")
    if common_unavailable:
        flags.append("COMMON_CURRENCY_EVIDENCE_UNAVAILABLE")
    if currency_instability:
        flags.append("CURRENCY_INSTABILITY")
    row["risk_flags"] = _flags(flags)
    return row


def _horizon_row(
    horizon_rows: dict[int, pd.Series],
    *,
    heuristics: object,
    period_type: str,
    currency: str,
    configuration_id: str,
    signal: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "period_type": period_type,
        "currency": currency,
        "configuration_id": configuration_id,
        "signal": signal,
        # Occurrence is the same signal sample at every horizon; take it once.
        "signal_occurrence_count": _number(horizon_rows[HORIZONS[0]]["occurrence_count"]),
    }
    supported: dict[int, bool] = {}
    adequate: dict[int, bool] = {}
    for horizon, source in horizon_rows.items():
        supported[horizon] = _supported(source, heuristics)
        adequate[horizon] = _adequate(source, heuristics)
        for column in _HORIZON_VALUE_COLUMNS:
            if column == "supported":
                value: object = supported[horizon]
            elif column == "adequate":
                value = adequate[horizon]
            else:
                value = _number(source[column])
            row[f"h{horizon}_{column}"] = value

    short = (5, 10)
    medium = (20, 60)
    short_supported = [horizon for horizon in short if supported[horizon]]
    medium_supported = [horizon for horizon in medium if supported[horizon]]
    evidence_unavailable = not short_supported or not medium_supported

    def segment_median(horizons: list[int], column: str) -> float:
        return _median(
            _finite_values(horizon_rows[horizon][column] for horizon in horizons)
        )

    short_favorable = segment_median(short_supported, "favorable_ratio")
    medium_favorable = segment_median(medium_supported, "favorable_ratio")
    short_mean = segment_median(short_supported, "mean_advantage_pct")
    medium_mean = segment_median(medium_supported, "mean_advantage_pct")
    short_median = segment_median(short_supported, "median_advantage_pct")
    medium_median = segment_median(medium_supported, "median_advantage_pct")
    favorable_drop = _difference(short_favorable, medium_favorable)
    mean_gap = _difference(medium_mean, short_mean)
    median_gap = _difference(medium_median, short_median)
    mean_reversal = _direction_reversal(short_mean, medium_mean)
    median_reversal = _direction_reversal(short_median, medium_median)
    instability = bool(
        not evidence_unavailable
        and (
            _at_least(
                favorable_drop,
                _h(heuristics, "material_horizon_favorable_drop"),
            )
            or mean_reversal
            or median_reversal
        )
    )
    row.update(
        {
            "supported_horizon_count": sum(supported.values()),
            "adequate_horizon_count": sum(adequate.values()),
            "minimum_evaluable_count": min(
                _number(source["evaluable_count"])
                for source in horizon_rows.values()
            ),
            "minimum_coverage_ratio": _nan_min(
                source["evaluation_coverage_ratio"]
                for source in horizon_rows.values()
            ),
            "short_supported_horizon_count": len(short_supported),
            "medium_supported_horizon_count": len(medium_supported),
            "short_favorable_ratio_median": short_favorable,
            "medium_favorable_ratio_median": medium_favorable,
            "short_to_medium_favorable_drop": favorable_drop,
            "short_mean_advantage_pct_median": short_mean,
            "medium_mean_advantage_pct_median": medium_mean,
            "short_to_medium_mean_gap_pct": mean_gap,
            "short_median_advantage_pct_median": short_median,
            "medium_median_advantage_pct_median": medium_median,
            "short_to_medium_median_gap_pct": median_gap,
            "mean_direction_reversal": mean_reversal,
            "median_direction_reversal": median_reversal,
            "horizon_evidence_unavailable": evidence_unavailable,
            "low_horizon_sample": any(
                _low_sample(source, heuristics) for source in horizon_rows.values()
            ),
            "very_low_horizon_sample": any(
                _very_low_sample(source, heuristics)
                for source in horizon_rows.values()
            ),
            "low_horizon_coverage": any(
                _low_coverage(source, heuristics) for source in horizon_rows.values()
            ),
            "severe_horizon_censoring": any(
                _severe_censoring(source, heuristics)
                for source in horizon_rows.values()
            ),
            "horizon_instability": instability,
        }
    )
    flags: list[str] = []
    if row["very_low_horizon_sample"]:
        flags.append("VERY_LOW_SAMPLE")
    elif row["low_horizon_sample"]:
        flags.append("LOW_SAMPLE")
    if row["severe_horizon_censoring"]:
        flags.append("SEVERE_OUTCOME_CENSORING")
    elif row["low_horizon_coverage"]:
        flags.append("LOW_COVERAGE")
    if evidence_unavailable:
        flags.append("HORIZON_EVIDENCE_UNAVAILABLE")
    if instability:
        flags.append("HORIZON_INSTABILITY")
    row["risk_flags"] = _flags(flags)
    return row


def _separation_row(
    signal_rows: dict[str, pd.Series],
    *,
    heuristics: object,
    period_type: str,
    currency: str,
    configuration_id: str,
    horizon: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "period_type": period_type,
        "currency": currency,
        "configuration_id": configuration_id,
        "horizon": horizon,
    }
    for pair_name, weaker_name, stronger_name in SEPARATION_PAIRS:
        weaker = signal_rows[weaker_name]
        stronger = signal_rows[stronger_name]
        comparable = _separation_supported(weaker, heuristics) and _separation_supported(
            stronger,
            heuristics,
        )
        row[f"{pair_name}_weaker_evaluable_count"] = _number(
            weaker["evaluable_count"]
        )
        row[f"{pair_name}_stronger_evaluable_count"] = _number(
            stronger["evaluable_count"]
        )
        row[f"{pair_name}_weaker_coverage_ratio"] = _number(
            weaker["evaluation_coverage_ratio"]
        )
        row[f"{pair_name}_stronger_coverage_ratio"] = _number(
            stronger["evaluation_coverage_ratio"]
        )
        row[f"{pair_name}_comparable"] = comparable
        if comparable:
            favorable_gap = _difference(
                _number(stronger["favorable_ratio"]),
                _number(weaker["favorable_ratio"]),
            )
            mean_gap = _difference(
                _number(stronger["mean_advantage_pct"]),
                _number(weaker["mean_advantage_pct"]),
            )
            median_gap = _difference(
                _number(stronger["median_advantage_pct"]),
                _number(weaker["median_advantage_pct"]),
            )
            improvement_count = sum(
                (
                    _at_least(favorable_gap, _h(heuristics, "favorable_tie_band")),
                    _at_least(mean_gap, _h(heuristics, "advantage_tie_band_pct")),
                    _at_least(median_gap, _h(heuristics, "advantage_tie_band_pct")),
                )
            )
            deterioration_count = sum(
                (
                    _at_most(favorable_gap, -_h(heuristics, "favorable_tie_band")),
                    _at_most(mean_gap, -_h(heuristics, "advantage_tie_band_pct")),
                    _at_most(median_gap, -_h(heuristics, "advantage_tie_band_pct")),
                )
            )
            if improvement_count >= 2:
                status = "IMPROVED"
            elif deterioration_count >= 2:
                status = "WORSE"
            else:
                status = "NOT_SEPARATED"
        else:
            favorable_gap = np.nan
            mean_gap = np.nan
            median_gap = np.nan
            improvement_count = 0
            deterioration_count = 0
            status = "UNAVAILABLE"
        row[f"{pair_name}_favorable_ratio_gap"] = favorable_gap
        row[f"{pair_name}_mean_advantage_gap_pct"] = mean_gap
        row[f"{pair_name}_median_advantage_gap_pct"] = median_gap
        row[f"{pair_name}_improvement_metric_count"] = improvement_count
        row[f"{pair_name}_deterioration_metric_count"] = deterioration_count
        row[f"{pair_name}_status"] = status

    watch_good_status = row["watch_to_good_status"]
    good_strong_status = row["good_to_strong_status"]
    row["watch_good_separation_failed"] = bool(
        watch_good_status in {"WORSE", "NOT_SEPARATED"}
    )
    row["good_strong_separation_failed"] = bool(
        good_strong_status in {"WORSE", "NOT_SEPARATED"}
    )
    row["strong_not_better_than_good"] = bool(good_strong_status == "WORSE")
    row["signal_separation_unavailable"] = bool(
        watch_good_status == "UNAVAILABLE" or good_strong_status == "UNAVAILABLE"
    )
    flags: list[str] = []
    if row["signal_separation_unavailable"]:
        flags.append("SIGNAL_SEPARATION_UNAVAILABLE")
    if row["watch_good_separation_failed"] or row["good_strong_separation_failed"]:
        flags.append("WEAK_SIGNAL_SEPARATION_CELL")
    if row["strong_not_better_than_good"]:
        flags.append("STRONG_NOT_BETTER_THAN_GOOD_CELL")
    row["risk_flags"] = _flags(flags)
    return row
