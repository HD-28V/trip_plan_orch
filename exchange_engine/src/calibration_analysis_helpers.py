"""Shared primitives for calibration analysis review modules.

This internal module imports neither calibration-analysis module, so both the
detailed and configuration review builders can import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Iterable
from statistics import median

import numpy as np
import pandas as pd


__all__ = [
    "_abs_difference", "_absolute", "_adequate", "_aggregate_gap",
    "_any_true", "_at_least", "_at_most", "_bool_value",
    "_configuration_flags", "_configuration_ids", "_configuration_record",
    "_copy_frame", "_difference", "_direction_conflict",
    "_direction_reversal", "_evidence_adequate", "_evidence_supported",
    "_finite", "_finite_values", "_flags", "_h", "_low_coverage",
    "_low_sample", "_median", "_nan_max", "_nan_min", "_negated",
    "_number", "_one_row", "_population_std", "_positive_performance_cell",
    "_range", "_require_group_size", "_review_gap_votes", "_row_at",
    "_safe_ratio", "_separation_supported", "_severe_censoring",
    "_supported", "_table_frame", "_threshold_votes", "_validate_signal_input",
    "_very_low_sample", "_weighted_mean",
]


def _validate_signal_input(data: pd.DataFrame) -> pd.DataFrame:
    return _copy_frame(
        data,
        {
            "period_type", "currency", "configuration_id", "signal", "horizon",
            "occurrence_count", "occurrence_ratio", "evaluable_count",
            "evaluation_coverage_ratio", "mean_advantage_pct",
            "median_advantage_pct", "favorable_count", "favorable_ratio",
            "unfavorable_ratio",
        },
        "signal_summary",
    )


def _copy_frame(
    data: pd.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{frame_name} must be a pandas DataFrame")
    if not data.columns.is_unique:
        raise ValueError(f"{frame_name} must not contain duplicate columns")
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: "
            + ", ".join(sorted(missing))
        )
    return data.copy(deep=True)


def _table_frame(
    tables: object,
    attribute_name: str,
    required_columns: set[str],
) -> pd.DataFrame:
    try:
        data = getattr(tables, attribute_name)
    except AttributeError as error:
        raise TypeError(f"tables must expose {attribute_name}") from error
    return _copy_frame(data, required_columns, attribute_name)


def _configuration_ids(values: Iterable[str]) -> tuple[str, ...]:
    if values is None or isinstance(values, (str, bytes)):
        raise TypeError("configuration_ids must be an iterable")
    identifiers = tuple(values)
    if not identifiers:
        raise ValueError("configuration_ids must not be empty")
    if not all(isinstance(value, str) and value for value in identifiers):
        raise ValueError("configuration_ids must contain non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("configuration_ids must be unique")
    return identifiers


def _one_row(data: pd.DataFrame, **key: object) -> pd.Series:
    matched = data
    for column, value in key.items():
        matched = matched.loc[matched[column].eq(value)]
    if len(matched) != 1:
        label = "/".join(str(value) for value in key.values())
        raise ValueError(f"expected exactly one summary row for {label}")
    return matched.iloc[0].copy(deep=True)


def _row_at(data: pd.DataFrame, **key: object) -> pd.Series:
    return _one_row(data, **key)


def _require_group_size(data: pd.DataFrame, expected: int, label: str) -> None:
    if len(data) != expected:
        raise ValueError(f"{label} must contain exactly {expected} review rows")


def _h(heuristics: object, field_name: str) -> float:
    try:
        value = getattr(heuristics, field_name)
    except AttributeError as error:
        raise TypeError(f"heuristics must expose {field_name}") from error
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"heuristic {field_name} must be numeric")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"heuristic {field_name} must be finite")
    return numeric


def _number(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("review metric must be numeric or missing")
    if value is None:
        return np.nan
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return np.nan
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError("review metric must be scalar")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("review metric must be numeric or missing") from error
    return numeric if np.isfinite(numeric) else np.nan


def _metric_value(row: pd.Series, signal_name: str, combined_name: str) -> float:
    if signal_name in row.index:
        return _number(row[signal_name])
    if combined_name in row.index:
        return _number(row[combined_name])
    raise ValueError(f"review row is missing {signal_name} or {combined_name}")


def _supported(row: pd.Series, heuristics: object) -> bool:
    evaluable = _metric_value(row, "evaluable_count", "good_strong_evaluable_count")
    coverage = _metric_value(
        row, "evaluation_coverage_ratio", "good_strong_evaluation_coverage_ratio"
    )
    return bool(
        _finite(evaluable) and _finite(coverage)
        and evaluable >= _h(heuristics, "signal_comparison_min_evaluable_count")
        and coverage >= _h(heuristics, "severe_coverage_ratio")
    )


def _evidence_supported(row: pd.Series, heuristics: object) -> bool:
    return _supported(row, heuristics)


def _adequate(row: pd.Series, heuristics: object) -> bool:
    evaluable = _metric_value(row, "evaluable_count", "good_strong_evaluable_count")
    coverage = _metric_value(
        row, "evaluation_coverage_ratio", "good_strong_evaluation_coverage_ratio"
    )
    return bool(
        _finite(evaluable) and _finite(coverage)
        and evaluable >= _h(heuristics, "low_sample_count")
        and coverage >= _h(heuristics, "low_coverage_ratio")
    )


def _evidence_adequate(row: pd.Series, heuristics: object) -> bool:
    return _adequate(row, heuristics)


def _separation_supported(row: pd.Series, heuristics: object) -> bool:
    return _supported(row, heuristics)


def _very_low_sample(row: pd.Series, heuristics: object) -> bool:
    value = _metric_value(row, "evaluable_count", "good_strong_evaluable_count")
    return bool(_finite(value) and value < _h(heuristics, "very_low_sample_count"))


def _low_sample(row: pd.Series, heuristics: object) -> bool:
    value = _metric_value(row, "evaluable_count", "good_strong_evaluable_count")
    return bool(_finite(value) and value < _h(heuristics, "low_sample_count"))


def _low_coverage(row: pd.Series, heuristics: object) -> bool:
    value = _metric_value(
        row, "evaluation_coverage_ratio", "good_strong_evaluation_coverage_ratio"
    )
    return bool(_finite(value) and value < _h(heuristics, "low_coverage_ratio"))


def _severe_censoring(row: pd.Series, heuristics: object) -> bool:
    value = _metric_value(
        row, "evaluation_coverage_ratio", "good_strong_evaluation_coverage_ratio"
    )
    return bool(_finite(value) and value < _h(heuristics, "severe_coverage_ratio"))


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _difference(left: object, right: object) -> float:
    left_number, right_number = _number(left), _number(right)
    if not _finite(left_number) or not _finite(right_number):
        return np.nan
    return left_number - right_number


def _abs_difference(left: object, right: object) -> float:
    return _absolute(_difference(left, right))


def _aggregate_gap(left: object, right: object, available: bool) -> float:
    return _difference(left, right) if available else np.nan


def _absolute(value: object) -> float:
    numeric = _number(value)
    return abs(numeric) if _finite(numeric) else np.nan


def _negated(value: object) -> float:
    numeric = _number(value)
    return -numeric if _finite(numeric) else np.nan


def _finite_values(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        numeric = _number(value)
        if _finite(numeric):
            result.append(numeric)
    return result


def _safe_ratio(numerator: object, denominator: object) -> float:
    left, right = _number(numerator), _number(denominator)
    return left / right if _finite(left) and _finite(right) and right > 0 else np.nan


def _weighted_mean(values: Iterable[tuple[object, object]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        numeric, numeric_weight = _number(value), _number(weight)
        if _finite(numeric) and _finite(numeric_weight) and numeric_weight > 0:
            numerator += numeric * numeric_weight
            denominator += numeric_weight
    return numerator / denominator if denominator else np.nan


def _median(values: Iterable[object]) -> float:
    finite = _finite_values(values)
    return float(median(finite)) if finite else np.nan


def _nan_min(values: Iterable[object]) -> float:
    finite = _finite_values(values)
    return min(finite) if finite else np.nan


def _nan_max(values: Iterable[object]) -> float:
    finite = _finite_values(values)
    return max(finite) if finite else np.nan


def _range(values: Iterable[object]) -> float:
    finite = _finite_values(values)
    return max(finite) - min(finite) if finite else np.nan


def _population_std(values: Iterable[object]) -> float:
    finite = _finite_values(values)
    return float(np.std(finite, ddof=0)) if len(finite) >= 2 else np.nan


def _direction_conflict(values: Iterable[object]) -> bool:
    finite = _finite_values(values)
    return bool(any(value > 0 for value in finite) and any(value < 0 for value in finite))


def _direction_reversal(first: object, second: object) -> bool:
    left, right = _number(first), _number(second)
    # Only a positive short-horizon value becoming negative is deterioration.
    # A negative-to-positive change is improvement, not an instability risk.
    return bool(_finite(left) and _finite(right) and left >= 0 and right < 0)


def _at_least(value: object, threshold: object) -> bool:
    numeric, boundary = _number(value), _number(threshold)
    return bool(_finite(numeric) and _finite(boundary) and numeric >= boundary)


def _at_most(value: object, threshold: object) -> bool:
    numeric, boundary = _number(value), _number(threshold)
    return bool(_finite(numeric) and _finite(boundary) and numeric <= boundary)


def _threshold_votes(values: Iterable[tuple[object, object]]) -> int:
    return sum(_at_least(value, threshold) for value, threshold in values)


def _flags(values: Iterable[str]) -> str:
    return ";".join(dict.fromkeys(value for value in values if value))


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (0, 1):
        return bool(value)
    raise ValueError("review flag columns must contain booleans")


def _any_true(values: Iterable[object]) -> bool:
    return any(_bool_value(value) for value in values)


def _positive_performance_cell(row: pd.Series, heuristics: object) -> bool:
    favorable = _metric_value(
        row, "favorable_ratio", "good_strong_favorable_ratio"
    )
    mean_value = _metric_value(
        row, "mean_advantage_pct", "good_strong_mean_advantage_pct"
    )
    median_value = _metric_value(
        row, "median_advantage_pct", "good_strong_median_advantage_pct"
    )
    return bool(
        sum(
            (
                _finite(favorable)
                and favorable
                > _h(heuristics, "positive_favorable_ratio_floor"),
                _finite(mean_value)
                and mean_value
                > _h(heuristics, "positive_advantage_floor_pct"),
                _finite(median_value)
                and median_value
                > _h(heuristics, "positive_advantage_floor_pct"),
            )
        )
        >= int(_h(heuristics, "positive_metric_vote_count"))
    )


def _review_gap_votes(
    favorable_gap: object,
    mean_gap: object,
    median_gap: object,
    heuristics: object,
    deterioration: bool,
) -> int:
    transform = _negated if deterioration else _absolute
    return _threshold_votes(
        (
            (transform(favorable_gap), _h(heuristics, "material_favorable_gap")),
            (transform(mean_gap), _h(heuristics, "material_mean_gap_pct")),
            (transform(median_gap), _h(heuristics, "material_mean_gap_pct")),
        )
    )


def _configuration_flags(
    very_low_sample: bool,
    low_sample: bool,
    low_coverage: bool,
    severe_censoring: bool,
    gap_available: bool,
    shift: bool,
    drift: bool,
    currency_unavailable: bool,
    currency_instability: bool,
    horizon_unavailable: bool,
    horizon_instability: bool,
    separation_unavailable: bool,
    weak_separation: bool,
    strong_too_rare: bool,
    strong_not_better: bool,
) -> str:
    flags: list[str] = []
    if very_low_sample:
        flags.append("VERY_LOW_SAMPLE")
    elif low_sample:
        flags.append("LOW_SAMPLE")
    if severe_censoring:
        flags.append("SEVERE_OUTCOME_CENSORING")
    elif low_coverage:
        flags.append("LOW_COVERAGE")
    if not gap_available:
        flags.append("PERFORMANCE_GAP_UNAVAILABLE")
    if shift:
        flags.append("CALIBRATION_VALIDATION_SHIFT")
    if drift:
        flags.append("CALIBRATION_VALIDATION_DRIFT")
    if currency_unavailable:
        flags.append("COMMON_CURRENCY_EVIDENCE_UNAVAILABLE")
    if currency_instability:
        flags.append("CURRENCY_INSTABILITY")
    if horizon_unavailable:
        flags.append("HORIZON_EVIDENCE_UNAVAILABLE")
    if horizon_instability:
        flags.append("HORIZON_INSTABILITY")
    if separation_unavailable:
        flags.append("SIGNAL_SEPARATION_UNAVAILABLE")
    if weak_separation:
        flags.append("WEAK_SIGNAL_SEPARATION")
    if strong_too_rare:
        flags.append("STRONG_TOO_RARE")
    if strong_not_better:
        flags.append("STRONG_NOT_BETTER_THAN_GOOD")
    return _flags(flags)


def _configuration_record(
    manifest: dict[str, object],
    calibration: dict[str, object],
    validation: dict[str, object],
    matched: dict[str, object],
    currency: dict[str, object],
    horizon: dict[str, object],
    positive_cells: int,
    positive_ratio: float,
    strong_count: int,
    minimum_strong_count: int,
    favorable_gap: float,
    mean_gap: float,
    median_gap: float,
    shift: bool,
    drift: bool,
    currency_instability: bool,
    horizon_instability: bool,
    weak_separation: bool,
    strong_not_better: bool,
    currency_unavailable: bool,
    horizon_unavailable: bool,
    separation_unavailable: bool,
    very_low_sample: bool,
    low_sample: bool,
    low_coverage: bool,
    severe_censoring: bool,
    strong_too_rare: bool,
    core_risk_axis_count: int,
    group: str,
    risk_flags: str,
) -> dict[str, object]:
    evidence_axis = (
        "UNAVAILABLE" if currency["supported_currency_count"] < 3
        else "CAUTION" if currency["adequate_currency_count"] < 3
        else "PASS"
    )
    gap_available = all(
        _finite(value) for value in (favorable_gap, mean_gap, median_gap)
    )
    period_axis = (
        "UNAVAILABLE"
        if not gap_available
        else "RISK"
        if drift
        else "CAUTION"
        if shift
        else "PASS"
    )
    currency_axis = (
        "RISK" if currency_instability else "UNAVAILABLE" if currency_unavailable else "PASS"
    )
    horizon_axis = (
        "RISK" if horizon_instability else "UNAVAILABLE" if horizon_unavailable else "PASS"
    )
    separation_axis = (
        "RISK" if weak_separation or strong_not_better
        else "UNAVAILABLE" if separation_unavailable else "PASS"
    )
    reasons = (
        f"historical review group={group}; supported currencies="
        f"{currency['supported_currency_count']}/3; positive supported cells="
        f"{positive_cells}/{validation['supported_cell_count']}; core risk axes="
        f"{core_risk_axis_count}; human review required"
    )
    return {
        "configuration_id": manifest["configuration_id"],
        "threshold_id": manifest["threshold_id"],
        "policy_id": manifest["policy_id"],
        "calibration_good_strong_sample_count": calibration["sample_count"],
        "validation_good_strong_sample_count": validation["sample_count"],
        "good_strong_sample_count_gap": _difference(
            validation["sample_count"], calibration["sample_count"]
        ),
        "good_strong_sample_count_abs_gap": _abs_difference(
            validation["sample_count"], calibration["sample_count"]
        ),
        "calibration_good_strong_occurrence_ratio": calibration["occurrence_ratio"],
        "validation_good_strong_occurrence_ratio": validation["occurrence_ratio"],
        "good_strong_occurrence_ratio_gap": _difference(
            validation["occurrence_ratio"], calibration["occurrence_ratio"]
        ),
        "good_strong_occurrence_ratio_abs_gap": _abs_difference(
            validation["occurrence_ratio"], calibration["occurrence_ratio"]
        ),
        "calibration_good_strong_evaluable_label_count": calibration["total_evaluable_count"],
        "validation_good_strong_evaluable_label_count": validation["total_evaluable_count"],
        "calibration_supported_label_evaluable_count": calibration["supported_evaluable_count"],
        "validation_supported_label_evaluable_count": validation["supported_evaluable_count"],
        "calibration_supported_cell_count": calibration["supported_cell_count"],
        "validation_supported_cell_count": validation["supported_cell_count"],
        "calibration_validation_common_supported_cell_count": matched["cell_count"],
        "calibration_adequate_cell_count": calibration["adequate_cell_count"],
        "validation_adequate_cell_count": validation["adequate_cell_count"],
        "calibration_good_strong_favorable_ratio": calibration["pooled_favorable_ratio"],
        "validation_good_strong_favorable_ratio": validation["pooled_favorable_ratio"],
        "matched_calibration_good_strong_favorable_ratio": matched["calibration"]["pooled_favorable_ratio"],
        "matched_validation_good_strong_favorable_ratio": matched["validation"]["pooled_favorable_ratio"],
        "favorable_ratio_gap": favorable_gap,
        "favorable_ratio_abs_gap": _absolute(favorable_gap),
        "calibration_good_strong_mean_advantage_pct": calibration["weighted_mean_advantage_pct"],
        "validation_good_strong_mean_advantage_pct": validation["weighted_mean_advantage_pct"],
        "matched_calibration_good_strong_mean_advantage_pct": matched["calibration"]["weighted_mean_advantage_pct"],
        "matched_validation_good_strong_mean_advantage_pct": matched["validation"]["weighted_mean_advantage_pct"],
        "mean_advantage_gap_pct": mean_gap,
        "mean_advantage_abs_gap_pct": _absolute(mean_gap),
        "calibration_good_strong_median_advantage_pct": calibration["median_of_cell_medians_pct"],
        "validation_good_strong_median_advantage_pct": validation["median_of_cell_medians_pct"],
        "matched_calibration_good_strong_median_advantage_pct": matched["calibration"]["median_of_cell_medians_pct"],
        "matched_validation_good_strong_median_advantage_pct": matched["validation"]["median_of_cell_medians_pct"],
        "median_advantage_gap_pct": median_gap,
        "median_advantage_abs_gap_pct": _absolute(median_gap),
        "calibration_minimum_coverage_ratio": calibration["minimum_coverage_ratio"],
        "validation_minimum_coverage_ratio": validation["minimum_coverage_ratio"],
        "minimum_coverage_ratio_gap": _difference(
            validation["minimum_coverage_ratio"],
            calibration["minimum_coverage_ratio"],
        ),
        "minimum_coverage_ratio_abs_gap": _abs_difference(
            validation["minimum_coverage_ratio"],
            calibration["minimum_coverage_ratio"],
        ),
        "validation_supported_currency_count": currency["supported_currency_count"],
        "validation_adequate_currency_count": currency["adequate_currency_count"],
        "validation_positive_currency_count": currency["positive_currency_count"],
        "validation_minimum_currency_pooled_favorable_ratio": currency["minimum_currency_pooled_favorable_ratio"],
        "validation_currency_pooled_favorable_stddev": currency["currency_pooled_favorable_stddev"],
        "validation_currency_pooled_favorable_range": currency["currency_pooled_favorable_range"],
        "validation_minimum_currency_good_strong_sample_count": currency["minimum_currency_sample_count"],
        "validation_horizon_comparable_currency_count": horizon["comparable_currency_count"],
        "validation_maximum_short_to_medium_favorable_drop": horizon["maximum_favorable_drop"],
        "validation_positive_supported_cell_count": positive_cells,
        "validation_positive_supported_cell_ratio": positive_ratio,
        "validation_minimum_evaluable_count": validation["minimum_evaluable_count"],
        "validation_strong_sample_count": strong_count,
        "validation_minimum_strong_currency_count": minimum_strong_count,
        "calibration_validation_shift": shift,
        "calibration_validation_drift": drift,
        "currency_instability": currency_instability,
        "horizon_instability": horizon_instability,
        "weak_signal_separation": weak_separation,
        "strong_not_better_than_good": strong_not_better,
        "common_currency_evidence_unavailable": currency_unavailable,
        "horizon_evidence_unavailable": horizon_unavailable,
        "signal_separation_unavailable": separation_unavailable,
        "very_low_sample": very_low_sample,
        "low_sample": low_sample,
        "low_coverage": low_coverage,
        "severe_outcome_censoring": severe_censoring,
        "strong_too_rare": strong_too_rare,
        "core_risk_axis_count": core_risk_axis_count,
        "evidence_axis": evidence_axis,
        "period_stability_axis": period_axis,
        "currency_stability_axis": currency_axis,
        "horizon_stability_axis": horizon_axis,
        "signal_separation_axis": separation_axis,
        "analysis_group": group,
        "risk_flags": risk_flags,
        "review_reasons": reasons,
    }
