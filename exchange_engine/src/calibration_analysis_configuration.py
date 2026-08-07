"""Configuration-level calibration review and shared analysis primitives.

This module is independent from the calibration-analysis orchestrator and its
dataclasses.  ``tables`` and ``heuristics`` are intentionally duck typed.  The
private helpers are also the primitives used by the four detailed review
builders; keeping them here avoids importing the orchestrator and therefore
avoids a circular dependency.
"""

from __future__ import annotations

import pandas as pd

from src.calibration_analysis_helpers import *  # noqa: F403


PERIODS = ("CALIBRATION", "VALIDATION")
CURRENCIES = ("USD", "JPY", "EUR")
HORIZONS = (5, 10, 20, 60)

CONFIGURATION_REVIEW_COLUMNS = (
    "configuration_id",
    "threshold_id",
    "policy_id",
    "calibration_good_strong_sample_count",
    "validation_good_strong_sample_count",
    "good_strong_sample_count_gap",
    "good_strong_sample_count_abs_gap",
    "calibration_good_strong_occurrence_ratio",
    "validation_good_strong_occurrence_ratio",
    "good_strong_occurrence_ratio_gap",
    "good_strong_occurrence_ratio_abs_gap",
    "calibration_good_strong_evaluable_label_count",
    "validation_good_strong_evaluable_label_count",
    "calibration_supported_label_evaluable_count",
    "validation_supported_label_evaluable_count",
    "calibration_supported_cell_count",
    "validation_supported_cell_count",
    "calibration_validation_common_supported_cell_count",
    "calibration_adequate_cell_count",
    "validation_adequate_cell_count",
    "calibration_good_strong_favorable_ratio",
    "validation_good_strong_favorable_ratio",
    "matched_calibration_good_strong_favorable_ratio",
    "matched_validation_good_strong_favorable_ratio",
    "favorable_ratio_gap",
    "favorable_ratio_abs_gap",
    "calibration_good_strong_mean_advantage_pct",
    "validation_good_strong_mean_advantage_pct",
    "matched_calibration_good_strong_mean_advantage_pct",
    "matched_validation_good_strong_mean_advantage_pct",
    "mean_advantage_gap_pct",
    "mean_advantage_abs_gap_pct",
    "calibration_good_strong_median_advantage_pct",
    "validation_good_strong_median_advantage_pct",
    "matched_calibration_good_strong_median_advantage_pct",
    "matched_validation_good_strong_median_advantage_pct",
    "median_advantage_gap_pct",
    "median_advantage_abs_gap_pct",
    "calibration_minimum_coverage_ratio",
    "validation_minimum_coverage_ratio",
    "minimum_coverage_ratio_gap",
    "minimum_coverage_ratio_abs_gap",
    "validation_supported_currency_count",
    "validation_adequate_currency_count",
    "validation_positive_currency_count",
    "validation_minimum_currency_pooled_favorable_ratio",
    "validation_currency_pooled_favorable_stddev",
    "validation_currency_pooled_favorable_range",
    "validation_minimum_currency_good_strong_sample_count",
    "validation_horizon_comparable_currency_count",
    "validation_maximum_short_to_medium_favorable_drop",
    "validation_positive_supported_cell_count",
    "validation_positive_supported_cell_ratio",
    "validation_minimum_evaluable_count",
    "validation_strong_sample_count",
    "validation_minimum_strong_currency_count",
    "calibration_validation_shift",
    "calibration_validation_drift",
    "currency_instability",
    "horizon_instability",
    "weak_signal_separation",
    "strong_not_better_than_good",
    "common_currency_evidence_unavailable",
    "horizon_evidence_unavailable",
    "signal_separation_unavailable",
    "very_low_sample",
    "low_sample",
    "low_coverage",
    "severe_outcome_censoring",
    "strong_too_rare",
    "core_risk_axis_count",
    "evidence_axis",
    "period_stability_axis",
    "currency_stability_axis",
    "horizon_stability_axis",
    "signal_separation_axis",
    "analysis_group",
    "risk_flags",
    "review_reasons",
)

_CONFIGURATION_REQUIRED = {
    "period_type",
    "currency",
    "configuration_id",
    "horizon",
    "total_date_count",
    "good_strong_occurrence_count",
    "good_strong_evaluable_count",
    "good_strong_evaluation_coverage_ratio",
    "good_strong_mean_advantage_pct",
    "good_strong_median_advantage_pct",
    "good_strong_favorable_count",
    "good_strong_favorable_ratio",
}
_SIGNAL_REQUIRED = {
    "period_type",
    "currency",
    "configuration_id",
    "signal",
    "horizon",
    "occurrence_count",
}


def build_configuration_review(
    tables: object,
    *,
    stability_review: pd.DataFrame,
    currency_stability_review: pd.DataFrame,
    horizon_stability_review: pd.DataFrame,
    signal_separation_review: pd.DataFrame,
    heuristics: object,
) -> pd.DataFrame:
    """Return 12 unranked review rows in candidate-manifest order."""

    manifest = _table_frame(
        tables,
        "candidate_configurations",
        {"configuration_id", "threshold_id", "policy_id"},
    )
    if len(manifest) != 12:
        raise ValueError("candidate_configurations must contain exactly 12 rows")
    identifiers = _configuration_ids(manifest["configuration_id"].tolist())
    combined = _table_frame(
        tables, "configuration_summary", _CONFIGURATION_REQUIRED
    )
    signal_summary = _table_frame(
        tables, "signal_horizon_summary", _SIGNAL_REQUIRED
    )
    stability = _copy_frame(
        stability_review,
        {"configuration_id", "calibration_validation_drift"},
        "stability_review",
    )
    currencies = _copy_frame(
        currency_stability_review,
        {
            "period_type",
            "configuration_id",
            "currency_instability",
            "common_currency_evidence_unavailable",
        },
        "currency_stability_review",
    )
    horizons = _copy_frame(
        horizon_stability_review,
        {
            "period_type",
            "configuration_id",
            "horizon_instability",
            "horizon_evidence_unavailable",
        },
        "horizon_stability_review",
    )
    separation = _copy_frame(
        signal_separation_review,
        {
            "period_type",
            "configuration_id",
            "watch_to_good_comparable",
            "watch_to_good_status",
            "good_to_strong_comparable",
            "good_to_strong_status",
        },
        "signal_separation_review",
    )

    rows: list[dict[str, object]] = []
    for manifest_row, configuration_id in zip(
        manifest.to_dict("records"), identifiers, strict=True
    ):
        config_rows = combined.loc[
            combined["configuration_id"].eq(configuration_id)
        ].copy(deep=True)
        _require_group_size(config_rows, 24, configuration_id)
        _require_group_size(
            stability.loc[stability["configuration_id"].eq(configuration_id)],
            48,
            configuration_id,
        )
        calibration_rows = config_rows.loc[
            config_rows["period_type"].eq("CALIBRATION")
        ]
        validation_rows = config_rows.loc[
            config_rows["period_type"].eq("VALIDATION")
        ]
        calibration = _aggregate_good_strong_period(
            calibration_rows, heuristics
        )
        validation = _aggregate_good_strong_period(validation_rows, heuristics)
        matched = _matched_supported_outcomes(
            calibration_rows,
            validation_rows,
            heuristics,
        )
        gap_available = all(
            _finite(value)
            for value in (
                matched["calibration"]["pooled_favorable_ratio"],
                matched["validation"]["pooled_favorable_ratio"],
                matched["calibration"]["weighted_mean_advantage_pct"],
                matched["validation"]["weighted_mean_advantage_pct"],
                matched["calibration"]["median_of_cell_medians_pct"],
                matched["validation"]["median_of_cell_medians_pct"],
            )
        )
        favorable_gap = _aggregate_gap(
            matched["validation"]["pooled_favorable_ratio"],
            matched["calibration"]["pooled_favorable_ratio"],
            gap_available,
        )
        mean_gap = _aggregate_gap(
            matched["validation"]["weighted_mean_advantage_pct"],
            matched["calibration"]["weighted_mean_advantage_pct"],
            gap_available,
        )
        median_gap = _aggregate_gap(
            matched["validation"]["median_of_cell_medians_pct"],
            matched["calibration"]["median_of_cell_medians_pct"],
            gap_available,
        )
        shift = bool(
            gap_available
            and _review_gap_votes(
                favorable_gap, mean_gap, median_gap, heuristics, False
            )
            >= 2
        )
        drift = bool(
            gap_available
            and _review_gap_votes(
                favorable_gap, mean_gap, median_gap, heuristics, True
            )
            >= 2
        )

        currency_evidence = _configuration_currency_evidence(
            validation_rows, heuristics
        )
        horizon_evidence = _configuration_horizon_evidence(
            validation_rows, heuristics
        )
        supported_validation = [
            source
            for _, source in validation_rows.iterrows()
            if _supported(source, heuristics)
        ]
        positive_cells = sum(
            _positive_performance_cell(source, heuristics)
            for source in supported_validation
        )
        positive_ratio = _safe_ratio(positive_cells, len(supported_validation))

        signal_rows = signal_summary.loc[
            signal_summary["configuration_id"].eq(configuration_id)
            & signal_summary["period_type"].eq("VALIDATION")
            & signal_summary["signal"].eq("STRONG")
            & signal_summary["horizon"].eq(HORIZONS[0])
        ]
        _require_group_size(signal_rows, len(CURRENCIES), configuration_id)
        strong_counts = [int(_number(value)) for value in signal_rows["occurrence_count"]]
        strong_count = sum(strong_counts)
        minimum_strong_count = min(strong_counts)
        strong_too_rare = minimum_strong_count < int(
            _h(heuristics, "very_low_sample_count")
        )

        currency_rows = currencies.loc[
            currencies["configuration_id"].eq(configuration_id)
            & currencies["period_type"].eq("VALIDATION")
        ]
        horizon_rows = horizons.loc[
            horizons["configuration_id"].eq(configuration_id)
            & horizons["period_type"].eq("VALIDATION")
        ]
        separation_rows = separation.loc[
            separation["configuration_id"].eq(configuration_id)
            & separation["period_type"].eq("VALIDATION")
        ]
        _require_group_size(currency_rows, 8, configuration_id)
        _require_group_size(horizon_rows, 6, configuration_id)
        _require_group_size(separation_rows, 12, configuration_id)
        currency_instability = bool(currency_evidence["currency_instability"])
        currency_unavailable = bool(
            currency_evidence["common_currency_evidence_unavailable"]
        )
        horizon_instability = bool(horizon_evidence["horizon_instability"])
        horizon_unavailable = bool(horizon_evidence["horizon_evidence_unavailable"])
        separation_result = _configuration_separation(
            separation_rows, heuristics
        )
        weak_separation = bool(
            separation_result["watch_good_failed"]
            or separation_result["good_strong_failed"]
        )
        strong_not_better = bool(separation_result["strong_not_better"])
        separation_unavailable = bool(separation_result["unavailable"])

        very_low_sample = any(
            _very_low_sample(source, heuristics)
            for _, source in validation_rows.iterrows()
        )
        low_sample = any(
            _low_sample(source, heuristics)
            for _, source in validation_rows.iterrows()
        )
        low_coverage = any(
            _low_coverage(source, heuristics)
            for _, source in validation_rows.iterrows()
        )
        severe_censoring = any(
            _severe_censoring(source, heuristics)
            for _, source in validation_rows.iterrows()
        )
        core_risk_axis_count = sum(
            (
                drift,
                currency_instability,
                horizon_instability,
                weak_separation or strong_not_better,
            )
        )
        promising = bool(
            currency_evidence["supported_currency_count"] == len(CURRENCIES)
            and currency_evidence["positive_currency_count"] == len(CURRENCIES)
            and _at_least(
                positive_ratio,
                _h(heuristics, "minimum_positive_supported_cell_ratio"),
            )
            and not drift
            and not strong_not_better
        )
        weak = bool(
            not promising
            and (
                core_risk_axis_count
                >= int(_h(heuristics, "weak_core_risk_axis_count"))
                or (
                    validation["adequate_cell_count"] >= len(CURRENCIES)
                    and _finite(positive_ratio)
                    and positive_ratio
                    < _h(heuristics, "weak_positive_supported_cell_ratio")
                )
            )
        )
        group = "PROMISING" if promising else "WEAK" if weak else "MIXED"
        flags = _configuration_flags(
            very_low_sample,
            low_sample,
            low_coverage,
            severe_censoring,
            gap_available,
            shift,
            drift,
            currency_unavailable,
            currency_instability,
            horizon_unavailable,
            horizon_instability,
            separation_unavailable,
            weak_separation,
            strong_too_rare,
            strong_not_better,
        )
        rows.append(
            _configuration_record(
                manifest_row,
                calibration,
                validation,
                matched,
                currency_evidence,
                horizon_evidence,
                positive_cells,
                positive_ratio,
                strong_count,
                minimum_strong_count,
                favorable_gap,
                mean_gap,
                median_gap,
                shift,
                drift,
                currency_instability,
                horizon_instability,
                weak_separation,
                strong_not_better,
                currency_unavailable,
                horizon_unavailable,
                separation_unavailable,
                very_low_sample,
                low_sample,
                low_coverage,
                severe_censoring,
                strong_too_rare,
                core_risk_axis_count,
                group,
                flags,
            )
        )
    return pd.DataFrame.from_records(rows, columns=CONFIGURATION_REVIEW_COLUMNS)


def _aggregate_good_strong_period(
    rows: pd.DataFrame, heuristics: object
) -> dict[str, object]:
    _require_group_size(rows, len(CURRENCIES) * len(HORIZONS), "period")
    supported = [
        source for _, source in rows.iterrows() if _supported(source, heuristics)
    ]
    adequate = [
        source for _, source in rows.iterrows() if _adequate(source, heuristics)
    ]
    outcome = _aggregate_outcome_metrics(rows)
    first_horizon = rows.loc[rows["horizon"].eq(HORIZONS[0])]
    _require_group_size(first_horizon, len(CURRENCIES), "period sample")
    # Occurrences are horizon invariant, so each currency is counted once.
    sample_count = int(first_horizon["good_strong_occurrence_count"].sum())
    sample_denominator = int(first_horizon["total_date_count"].sum())
    supported_evaluable = int(sum(
        _number(source["good_strong_evaluable_count"]) for source in supported
    ))
    return {
        "sample_count": sample_count,
        "occurrence_ratio": _safe_ratio(sample_count, sample_denominator),
        # Horizon labels overlap and are not a claim of independent samples.
        "total_evaluable_count": outcome["total_evaluable_count"],
        "supported_evaluable_count": supported_evaluable,
        "supported_cell_count": len(supported),
        "adequate_cell_count": len(adequate),
        "pooled_favorable_ratio": outcome["pooled_favorable_ratio"],
        "weighted_mean_advantage_pct": outcome["weighted_mean_advantage_pct"],
        # Raw outcomes are unavailable, so this is explicitly a median of
        # every evaluable cell median, not a pooled raw-outcome median.
        "median_of_cell_medians_pct": outcome["median_of_cell_medians_pct"],
        "minimum_coverage_ratio": _nan_min(
            source["good_strong_evaluation_coverage_ratio"]
            for _, source in rows.iterrows()
        ),
        "minimum_evaluable_count": min(
            _number(source["good_strong_evaluable_count"])
            for _, source in rows.iterrows()
        ),
    }


def _aggregate_outcome_metrics(rows: pd.DataFrame) -> dict[str, float]:
    evaluable_rows = [
        source
        for _, source in rows.iterrows()
        if _number(source["good_strong_evaluable_count"]) > 0
    ]
    total_evaluable = int(sum(
        _number(source["good_strong_evaluable_count"])
        for source in evaluable_rows
    ))
    favorable = sum(
        _number(source["good_strong_favorable_count"])
        for source in evaluable_rows
    )
    return {
        "total_evaluable_count": total_evaluable,
        "pooled_favorable_ratio": _safe_ratio(favorable, total_evaluable),
        "weighted_mean_advantage_pct": _weighted_mean(
            (
                _number(source["good_strong_mean_advantage_pct"]),
                _number(source["good_strong_evaluable_count"]),
            )
            for source in evaluable_rows
        ),
        "median_of_cell_medians_pct": _median(
            source["good_strong_median_advantage_pct"]
            for source in evaluable_rows
        ),
    }


def _matched_supported_outcomes(
    calibration_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    heuristics: object,
) -> dict[str, object]:
    key_columns = ("currency", "horizon")

    def supported_keys(rows: pd.DataFrame) -> set[tuple[str, int]]:
        return {
            (str(source["currency"]), int(source["horizon"]))
            for _, source in rows.iterrows()
            if _supported(source, heuristics)
        }

    common_keys = supported_keys(calibration_rows).intersection(
        supported_keys(validation_rows)
    )

    def matched_rows(rows: pd.DataFrame) -> pd.DataFrame:
        mask = rows.apply(
            lambda source: (
                str(source[key_columns[0]]),
                int(source[key_columns[1]]),
            )
            in common_keys,
            axis=1,
        )
        return rows.loc[mask].copy(deep=True)

    return {
        "cell_count": len(common_keys),
        "calibration": _aggregate_outcome_metrics(
            matched_rows(calibration_rows)
        ),
        "validation": _aggregate_outcome_metrics(
            matched_rows(validation_rows)
        ),
    }


def _configuration_currency_evidence(
    rows: pd.DataFrame, heuristics: object
) -> dict[str, object]:
    supported_currencies = 0
    adequate_currencies = 0
    positive_currencies = 0
    pooled_favorable_ratios: list[float] = []
    pooled_mean_advantages: list[float] = []
    currency_median_advantages: list[float] = []
    sample_counts: list[int] = []
    for currency in CURRENCIES:
        group = rows.loc[rows["currency"].eq(currency)]
        _require_group_size(group, len(HORIZONS), currency)
        supported = [
            source for _, source in group.iterrows() if _supported(source, heuristics)
        ]
        adequate = [
            source for _, source in group.iterrows() if _adequate(source, heuristics)
        ]
        if supported:
            supported_currencies += 1
            positive = sum(
                _positive_performance_cell(source, heuristics)
                for source in supported
            )
            if positive / len(supported) >= _h(
                heuristics, "minimum_positive_currency_cell_ratio"
            ):
                positive_currencies += 1
        if adequate:
            adequate_currencies += 1
        evaluable_rows = [
            source
            for _, source in group.iterrows()
            if _number(source["good_strong_evaluable_count"]) > 0
        ]
        evaluable_count = sum(
            _number(source["good_strong_evaluable_count"])
            for source in evaluable_rows
        )
        favorable_count = sum(
            _number(source["good_strong_favorable_count"])
            for source in evaluable_rows
        )
        pooled_favorable_ratios.append(
            _safe_ratio(favorable_count, evaluable_count)
        )
        pooled_mean_advantages.append(
            _weighted_mean(
                (
                    source["good_strong_mean_advantage_pct"],
                    source["good_strong_evaluable_count"],
                )
                for source in evaluable_rows
            )
        )
        currency_median_advantages.append(
            _median(
                source["good_strong_median_advantage_pct"]
                for source in evaluable_rows
            )
        )
        first_horizon = group.loc[group["horizon"].eq(HORIZONS[0])]
        _require_group_size(first_horizon, 1, currency)
        sample_counts.append(
            int(_number(first_horizon.iloc[0]["good_strong_occurrence_count"]))
        )
    favorable_range = _range(pooled_favorable_ratios)
    mean_conflict = _direction_conflict(pooled_mean_advantages)
    median_conflict = _direction_conflict(currency_median_advantages)
    currency_unavailable = supported_currencies < len(CURRENCIES)
    currency_instability = bool(
        not currency_unavailable
        and _at_least(
            favorable_range,
            _h(heuristics, "material_currency_favorable_range"),
        )
        and (mean_conflict or median_conflict)
    )
    return {
        "supported_currency_count": supported_currencies,
        "adequate_currency_count": adequate_currencies,
        "positive_currency_count": positive_currencies,
        "minimum_currency_pooled_favorable_ratio": _nan_min(
            pooled_favorable_ratios
        ),
        "currency_pooled_favorable_stddev": _population_std(
            pooled_favorable_ratios
        ),
        "currency_pooled_favorable_range": favorable_range,
        "minimum_currency_sample_count": min(sample_counts),
        "mean_direction_conflict": mean_conflict,
        "median_direction_conflict": median_conflict,
        "common_currency_evidence_unavailable": currency_unavailable,
        "currency_instability": currency_instability,
    }


def _configuration_horizon_evidence(
    rows: pd.DataFrame,
    heuristics: object,
) -> dict[str, object]:
    favorable_drops: list[float] = []
    comparable_currency_count = 0
    instability = False
    for currency in CURRENCIES:
        group = rows.loc[rows["currency"].eq(currency)]
        _require_group_size(group, len(HORIZONS), currency)
        supported = {
            int(source["horizon"]): source
            for _, source in group.iterrows()
            if _supported(source, heuristics)
        }
        short = [supported[horizon] for horizon in (5, 10) if horizon in supported]
        medium = [supported[horizon] for horizon in (20, 60) if horizon in supported]
        if not short or not medium:
            continue
        comparable_currency_count += 1
        short_favorable = _median(
            source["good_strong_favorable_ratio"] for source in short
        )
        medium_favorable = _median(
            source["good_strong_favorable_ratio"] for source in medium
        )
        favorable_drop = _difference(short_favorable, medium_favorable)
        favorable_drops.append(favorable_drop)
        short_mean = _median(
            source["good_strong_mean_advantage_pct"] for source in short
        )
        medium_mean = _median(
            source["good_strong_mean_advantage_pct"] for source in medium
        )
        short_median = _median(
            source["good_strong_median_advantage_pct"] for source in short
        )
        medium_median = _median(
            source["good_strong_median_advantage_pct"] for source in medium
        )
        instability = bool(
            instability
            or _at_least(
                favorable_drop,
                _h(heuristics, "material_horizon_favorable_drop"),
            )
            or _direction_reversal(short_mean, medium_mean)
            or _direction_reversal(short_median, medium_median)
        )
    return {
        "comparable_currency_count": comparable_currency_count,
        "maximum_favorable_drop": _nan_max(favorable_drops),
        "horizon_evidence_unavailable": (
            comparable_currency_count < len(CURRENCIES)
        ),
        "horizon_instability": instability,
    }


def _configuration_separation(
    rows: pd.DataFrame, heuristics: object
) -> dict[str, object]:
    minimum = int(_h(heuristics, "minimum_comparable_separation_cells"))
    failure_fraction = _h(heuristics, "comparison_failure_fraction")

    def result(prefix: str) -> dict[str, bool]:
        comparable = rows.loc[rows[f"{prefix}_comparable"].map(_bool_value)]
        count = len(comparable)
        failures = int(
            comparable[f"{prefix}_status"].isin(("WORSE", "NOT_SEPARATED")).sum()
        )
        worse = int(comparable[f"{prefix}_status"].eq("WORSE").sum())
        return {
            "failed": bool(count >= minimum and failures / count >= failure_fraction),
            "worse": bool(count >= minimum and worse / count >= failure_fraction),
            "unavailable": count < minimum,
        }

    watch_good = result("watch_to_good")
    good_strong = result("good_to_strong")
    return {
        "watch_good_failed": watch_good["failed"],
        "good_strong_failed": good_strong["failed"],
        "strong_not_better": good_strong["worse"],
        "unavailable": bool(
            watch_good["unavailable"] or good_strong["unavailable"]
        ),
    }
