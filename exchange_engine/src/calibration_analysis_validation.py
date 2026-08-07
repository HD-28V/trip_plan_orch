"""Strict offline validation for completed calibration report tables.

The validator accepts the six report frames through a duck-typed object,
validates their complete relational contract, and returns deep copies.  It
does not analyze candidate performance, access providers, or select a winner.
"""

from __future__ import annotations

from itertools import product
from numbers import Integral
import re

import numpy as np
import pandas as pd

from src.calibration import (
    CALIBRATION_CURRENCIES,
    CONFIGURATION_SUMMARY_COLUMNS,
    CROSS_CURRENCY_SUMMARY_COLUMNS,
    PERIOD_DATA_SUMMARY_COLUMNS,
    REPORT_METADATA_COLUMNS,
    SIGNAL_HORIZON_REPORT_COLUMNS,
    PeriodType,
    build_cross_currency_summary,
)
from src.calibration_candidates import CANDIDATE_MANIFEST_COLUMNS
from src.signal_engine import Signal, SignalDecisionPolicy, SignalThresholds


PERIODS = tuple(period.value for period in PeriodType)
CURRENCIES = CALIBRATION_CURRENCIES
HORIZONS = (5, 10, 20, 60)
SIGNALS = tuple(signal.value for signal in Signal)

TABLE_SCHEMAS = {
    "report_metadata": REPORT_METADATA_COLUMNS,
    "candidate_configurations": CANDIDATE_MANIFEST_COLUMNS,
    "period_data_summary": PERIOD_DATA_SUMMARY_COLUMNS,
    "configuration_summary": CONFIGURATION_SUMMARY_COLUMNS,
    "signal_horizon_summary": SIGNAL_HORIZON_REPORT_COLUMNS,
    "cross_currency_summary": CROSS_CURRENCY_SUMMARY_COLUMNS,
}

METADATA_FIELDS = (
    "report_purpose",
    "advantage_pct",
    "favorable",
    "neutral",
    "unfavorable",
    "horizon_unit",
    "calibration_period",
    "validation_period",
    "occurrence_ratio_denominator",
    "coverage_denominator",
    "direction_ratio_denominator",
    "cross_currency_weighting",
    "cross_currency_stddev",
    "currencies",
    "horizons",
    "candidate_configuration_count",
    "total_runtime_seconds",
)


class CalibrationReportValidationError(ValueError):
    """Raised when one of the six source reports violates its contract."""


def validate_calibration_results_impl(
    tables: object,
    *,
    expected_count: int = 12,
) -> dict[str, pd.DataFrame]:
    """Validate all six reports and return independent deep copies.

    ``expected_count`` exists for deterministic offline fixtures.  Production
    calibration analysis uses the default initial manifest size of twelve.
    """
    count = _positive_integer(expected_count, "expected_count")
    copied: dict[str, pd.DataFrame] = {}
    for name, schema in TABLE_SCHEMAS.items():
        try:
            frame = getattr(tables, name)
        except AttributeError:
            raise CalibrationReportValidationError(
                f"tables is missing required frame: {name}"
            ) from None
        _exact_schema(frame, schema, name)
        copied[name] = frame.copy(deep=True)

    manifest = copied["candidate_configurations"]
    metadata = copied["report_metadata"]
    period_data = copied["period_data_summary"]
    configuration = copied["configuration_summary"]
    signal = copied["signal_horizon_summary"]
    cross = copied["cross_currency_summary"]

    _candidate_manifest(manifest, count)
    split_date = _metadata(metadata, count)
    _period_data(period_data, count, split_date=split_date)
    configuration_ids = manifest["configuration_id"].tolist()
    _dimensions(
        period_data,
        configuration,
        signal,
        cross,
        configuration_ids,
    )
    _signal_summary(signal)
    _configuration_summary(configuration)
    _configuration_signal_consistency(configuration, signal)
    _period_count_consistency(period_data, configuration, signal)
    _cross_consistency(cross, signal, configuration_ids)
    return copied


def _exact_schema(
    frame: object,
    expected_columns: tuple[str, ...],
    name: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise CalibrationReportValidationError(
            f"{name} must not contain duplicate columns"
        )
    if frame.columns.tolist() != list(expected_columns):
        raise CalibrationReportValidationError(
            f"{name} columns must exactly match the report schema"
        )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _numeric(
    values: pd.Series,
    name: str,
    *,
    missing: bool,
) -> pd.Series:
    try:
        result = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError):
        raise CalibrationReportValidationError(
            f"{name} must contain numeric values"
        ) from None
    if not missing and result.isna().any():
        raise CalibrationReportValidationError(f"{name} must not be missing")
    if not np.isfinite(result.dropna()).all():
        raise CalibrationReportValidationError(f"{name} must be finite")
    return result


def _integers(
    values: pd.Series,
    name: str,
    *,
    minimum: int = 0,
) -> pd.Series:
    result = _numeric(values, name, missing=False)
    if not np.equal(result, np.floor(result)).all():
        raise CalibrationReportValidationError(f"{name} must contain integers")
    if result.lt(minimum).any():
        raise CalibrationReportValidationError(
            f"{name} must be at least {minimum}"
        )
    return result.astype(int)


def _candidate_manifest(manifest: pd.DataFrame, expected_count: int) -> None:
    if len(manifest) != expected_count:
        raise CalibrationReportValidationError(
            f"candidate manifest must contain {expected_count} rows"
        )
    for name in ("configuration_id", "threshold_id", "policy_id"):
        if manifest[name].isna().any() or not manifest[name].map(
            lambda value: isinstance(value, str) and bool(value)
        ).all():
            raise CalibrationReportValidationError(
                f"{name} must contain non-empty strings"
            )
    if not manifest["configuration_id"].is_unique:
        raise CalibrationReportValidationError(
            "configuration_id values must be unique"
        )
    if manifest[["threshold_id", "policy_id"]].duplicated().any():
        raise CalibrationReportValidationError(
            "threshold_id/policy_id pairs must be unique"
        )

    threshold_fields = (
        "sma60_good",
        "sma60_strong",
        "sma120_good",
        "sma120_strong",
        "percentile_good",
        "percentile_strong",
        "bollinger_near_lower_pct",
    )
    policy_fields = (
        "minimum_available_conditions",
        "watch_min_satisfied_conditions",
        "good_min_satisfied_conditions",
        "strong_min_satisfied_conditions",
        "strong_min_strong_conditions",
    )
    for name in threshold_fields:
        _numeric(manifest[name], name, missing=False)
    for name in policy_fields:
        _integers(manifest[name], name, minimum=1)

    for _, row in manifest.iterrows():
        try:
            SignalThresholds(
                **{name: float(row[name]) for name in threshold_fields}
            )
            SignalDecisionPolicy(
                **{name: int(row[name]) for name in policy_fields}
            )
        except (TypeError, ValueError) as error:
            raise CalibrationReportValidationError(
                "candidate manifest contains an invalid threshold or policy"
            ) from error
    for identifier, fields in (
        ("threshold_id", threshold_fields),
        ("policy_id", policy_fields),
    ):
        for _, group in manifest.groupby(identifier, sort=False):
            if len(group.loc[:, fields].drop_duplicates()) != 1:
                raise CalibrationReportValidationError(
                    f"one {identifier} must map to one profile"
                )


def _metadata(metadata: pd.DataFrame, expected_count: int) -> pd.Timestamp:
    if len(metadata) != len(METADATA_FIELDS):
        raise CalibrationReportValidationError(
            "report_metadata has an unexpected row count"
        )
    if metadata["field"].tolist() != list(METADATA_FIELDS):
        raise CalibrationReportValidationError(
            "report_metadata fields must exactly match the report contract"
        )
    if metadata["field"].duplicated().any() or metadata["value"].isna().any():
        raise CalibrationReportValidationError(
            "report_metadata fields must be unique and defined"
        )
    values = metadata.set_index("field")["value"].map(str)
    exact = {
        "advantage_pct": "(forward_mean_rate - entry_rate) / entry_rate * 100",
        "favorable": "advantage_pct > 0",
        "neutral": "advantage_pct == 0",
        "unfavorable": "advantage_pct < 0",
        "horizon_unit": "subsequent trading observations; entry excluded",
        "occurrence_ratio_denominator": "all evaluation dates in period",
        "coverage_denominator": "signal occurrence_count",
        "direction_ratio_denominator": "non-NaN evaluable_count",
        "cross_currency_weighting": "equal weight per defined currency metric",
        "cross_currency_stddev": "population standard deviation (ddof=0)",
        "currencies": ",".join(CURRENCIES),
        "horizons": ",".join(map(str, HORIZONS)),
        "candidate_configuration_count": str(expected_count),
    }
    for field, expected in exact.items():
        if values[field] != expected:
            raise CalibrationReportValidationError(
                f"report_metadata has an invalid {field} value"
            )
    calibration_match = re.fullmatch(
        r"date < (\d{4}-\d{2}-\d{2})", values["calibration_period"]
    )
    validation_match = re.fullmatch(
        r"date >= (\d{4}-\d{2}-\d{2})", values["validation_period"]
    )
    if (
        calibration_match is None
        or validation_match is None
        or calibration_match.group(1) != validation_match.group(1)
    ):
        raise CalibrationReportValidationError(
            "calibration and validation metadata must share one split date"
        )
    try:
        split = pd.Timestamp(calibration_match.group(1))
        runtime = float(values["total_runtime_seconds"])
    except (TypeError, ValueError):
        raise CalibrationReportValidationError(
            "report_metadata contains an invalid date or runtime"
        ) from None
    if pd.isna(split) or not np.isfinite(runtime) or runtime < 0:
        raise CalibrationReportValidationError(
            "report_metadata contains an invalid date or runtime"
        )
    return split


def _period_data(
    period_data: pd.DataFrame,
    expected_count: int,
    *,
    split_date: pd.Timestamp,
) -> None:
    expected_keys = set(product(PERIODS, CURRENCIES))
    _key_product(period_data, ("period_type", "currency"), expected_keys)
    date_columns = (
        "source_start_date",
        "source_end_date",
        "evaluation_start_date",
        "evaluation_end_date",
    )
    dates: dict[str, pd.Series] = {}
    for name in date_columns:
        try:
            dates[name] = pd.to_datetime(period_data[name], errors="raise")
        except (TypeError, ValueError):
            raise CalibrationReportValidationError(
                f"period_data_summary contains an invalid {name}"
            ) from None
        if dates[name].isna().any():
            raise CalibrationReportValidationError(
                f"period_data_summary contains an invalid {name}"
            )
    count_columns = (
        "source_row_count",
        "replay_row_count",
        "warmup_row_count",
        "evaluation_row_count",
        "candidate_configuration_count",
    )
    counts = {
        name: _integers(period_data[name], name) for name in count_columns
    }
    if not counts["candidate_configuration_count"].eq(expected_count).all():
        raise CalibrationReportValidationError(
            "period_data_summary candidate count does not match the manifest"
        )
    if (
        counts["source_row_count"].le(0).any()
        or counts["replay_row_count"].le(0).any()
        or counts["evaluation_row_count"].le(0).any()
    ):
        raise CalibrationReportValidationError(
            "period_data_summary row counts must be positive"
        )
    elapsed = _numeric(period_data["elapsed_seconds"], "elapsed_seconds", missing=False)
    if elapsed.lt(0).any():
        raise CalibrationReportValidationError(
            "period_data_summary elapsed_seconds must be non-negative"
        )
    if not (
        (dates["source_start_date"] <= dates["source_end_date"]).all()
        and (dates["evaluation_start_date"] <= dates["evaluation_end_date"]).all()
        and (dates["evaluation_start_date"] >= dates["source_start_date"]).all()
        and (dates["evaluation_end_date"] <= dates["source_end_date"]).all()
    ):
        raise CalibrationReportValidationError(
            "period_data_summary contains inconsistent date boundaries"
        )

    for currency in CURRENCIES:
        group = period_data.loc[period_data["currency"].eq(currency)].set_index(
            "period_type"
        )
        calibration = group.loc[PeriodType.CALIBRATION.value]
        validation = group.loc[PeriodType.VALIDATION.value]
        if (
            calibration["source_start_date"] != validation["source_start_date"]
            or calibration["source_end_date"] != validation["source_end_date"]
            or calibration["source_row_count"] != validation["source_row_count"]
        ):
            raise CalibrationReportValidationError(
                "period rows for one currency must describe the same source"
            )
        if (
            int(calibration["warmup_row_count"]) != 0
            or int(calibration["replay_row_count"])
            != int(calibration["evaluation_row_count"])
            or int(validation["replay_row_count"])
            != int(validation["source_row_count"])
            or int(validation["warmup_row_count"])
            != int(calibration["evaluation_row_count"])
            or int(calibration["evaluation_row_count"])
            + int(validation["evaluation_row_count"])
            != int(validation["source_row_count"])
        ):
            raise CalibrationReportValidationError(
                "period_data_summary row accounting is inconsistent"
            )
        if pd.Timestamp(calibration["evaluation_end_date"]) >= pd.Timestamp(
            validation["evaluation_start_date"]
        ):
            raise CalibrationReportValidationError(
                "calibration and validation date ranges must not overlap"
            )
        if not (
            pd.Timestamp(calibration["evaluation_end_date"]) < split_date
            <= pd.Timestamp(validation["evaluation_start_date"])
        ):
            raise CalibrationReportValidationError(
                "period date ranges must agree with the metadata split date"
            )


def _key_product(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    expected: set[tuple[object, ...]],
) -> None:
    if frame.duplicated(list(columns)).any():
        raise CalibrationReportValidationError(
            "report keys must be unique: " + ", ".join(columns)
        )
    received = set(frame.loc[:, columns].itertuples(index=False, name=None))
    if received != expected or len(frame) != len(expected):
        raise CalibrationReportValidationError(
            "report dimensions are incomplete: " + ", ".join(columns)
        )


def _dimensions(
    period_data: pd.DataFrame,
    configuration: pd.DataFrame,
    signal: pd.DataFrame,
    cross: pd.DataFrame,
    configuration_ids: list[str],
) -> None:
    _key_product(
        period_data,
        ("period_type", "currency"),
        set(product(PERIODS, CURRENCIES)),
    )
    _key_product(
        configuration,
        ("period_type", "currency", "configuration_id", "horizon"),
        set(product(PERIODS, CURRENCIES, configuration_ids, HORIZONS)),
    )
    _key_product(
        signal,
        ("period_type", "currency", "configuration_id", "signal", "horizon"),
        set(
            product(
                PERIODS,
                CURRENCIES,
                configuration_ids,
                SIGNALS,
                HORIZONS,
            )
        ),
    )
    _key_product(
        cross,
        ("period_type", "configuration_id", "signal", "horizon"),
        set(product(PERIODS, configuration_ids, SIGNALS, HORIZONS)),
    )


def _close(actual: object, expected: object, name: str) -> None:
    actual_values = np.asarray(actual, dtype=float)
    expected_values = np.asarray(expected, dtype=float)
    if not np.all(
        np.isclose(
            actual_values,
            expected_values,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
    ):
        raise CalibrationReportValidationError(f"{name} arithmetic is inconsistent")


def _summary_contract(
    frame: pd.DataFrame,
    *,
    prefix: str,
) -> None:
    total_name = "total_date_count"
    occurrence_name = f"{prefix}occurrence_count"
    evaluable_name = f"{prefix}evaluable_count"
    unavailable_name = f"{prefix}unavailable_count"
    coverage_name = f"{prefix}evaluation_coverage_ratio"
    mean_name = f"{prefix}mean_advantage_pct"
    median_name = f"{prefix}median_advantage_pct"
    occurrence_ratio_name = f"{prefix}occurrence_ratio"
    directions = ("favorable", "neutral", "unfavorable")
    integer_columns = (
        total_name,
        occurrence_name,
        evaluable_name,
        unavailable_name,
        *(f"{prefix}{direction}_count" for direction in directions),
    )
    values = {name: _integers(frame[name], name) for name in integer_columns}
    if values[total_name].le(0).any():
        raise CalibrationReportValidationError("total_date_count must be positive")
    if (
        (values[evaluable_name] + values[unavailable_name] != values[occurrence_name]).any()
        or (values[occurrence_name] > values[total_name]).any()
    ):
        raise CalibrationReportValidationError(
            "evaluable and unavailable counts must partition occurrence_count"
        )
    direction_total = sum(values[f"{prefix}{name}_count"] for name in directions)
    if (direction_total != values[evaluable_name]).any():
        raise CalibrationReportValidationError(
            "direction counts must partition evaluable_count"
        )

    occurrence_ratio = _numeric(frame[occurrence_ratio_name], occurrence_ratio_name, missing=False)
    _close(
        occurrence_ratio,
        values[occurrence_name] / values[total_name],
        occurrence_ratio_name,
    )
    expected_coverage = np.where(
        values[occurrence_name].gt(0),
        values[evaluable_name] / values[occurrence_name].replace(0, np.nan),
        np.nan,
    )
    coverage = _numeric(frame[coverage_name], coverage_name, missing=True)
    _close(coverage, expected_coverage, coverage_name)

    mean = _numeric(frame[mean_name], mean_name, missing=True)
    median = _numeric(frame[median_name], median_name, missing=True)
    expected_missing = values[evaluable_name].eq(0)
    if not mean.isna().eq(expected_missing).all() or not median.isna().eq(
        expected_missing
    ).all():
        raise CalibrationReportValidationError(
            "advantage metrics must be missing exactly when evaluable_count is zero"
        )
    for direction in directions:
        ratio_name = f"{prefix}{direction}_ratio"
        ratio = _numeric(frame[ratio_name], ratio_name, missing=True)
        expected_ratio = np.where(
            values[evaluable_name].gt(0),
            values[f"{prefix}{direction}_count"]
            / values[evaluable_name].replace(0, np.nan),
            np.nan,
        )
        _close(ratio, expected_ratio, ratio_name)


def _signal_summary(signal: pd.DataFrame) -> None:
    _summary_contract(signal, prefix="")
    stable = signal.groupby(
        ["period_type", "currency", "configuration_id", "signal"],
        sort=False,
    )[["total_date_count", "occurrence_count"]].nunique(dropna=False)
    if stable.gt(1).any().any():
        raise CalibrationReportValidationError(
            "signal occurrence counts must not change across horizons"
        )
    partition_source = signal.assign(
        _occurrence_count=_integers(
            signal["occurrence_count"], "occurrence_count"
        ),
        _total_date_count=_integers(
            signal["total_date_count"], "total_date_count"
        ),
    )
    partition = partition_source.groupby(
        ["period_type", "currency", "configuration_id", "horizon"],
        sort=False,
    ).agg(
        occurrence_sum=("_occurrence_count", "sum"),
        total_min=("_total_date_count", "min"),
        total_max=("_total_date_count", "max"),
    )
    if (
        partition["total_min"].ne(partition["total_max"]).any()
        or partition["occurrence_sum"].ne(partition["total_min"]).any()
    ):
        raise CalibrationReportValidationError(
            "WAIT/WATCH/GOOD/STRONG occurrences must partition total_date_count"
        )


def _configuration_summary(configuration: pd.DataFrame) -> None:
    _summary_contract(configuration, prefix="good_strong_")
    strong_occurrence = _integers(
        configuration["strong_occurrence_count"],
        "strong_occurrence_count",
    )
    strong_evaluable = _integers(
        configuration["strong_evaluable_count"],
        "strong_evaluable_count",
    )
    if (
        strong_evaluable.gt(strong_occurrence).any()
        or strong_occurrence.gt(configuration["good_strong_occurrence_count"]).any()
        or strong_evaluable.gt(configuration["good_strong_evaluable_count"]).any()
    ):
        raise CalibrationReportValidationError(
            "STRONG counts must be a subset of GOOD+STRONG counts"
        )
    stable = configuration.groupby(
        ["period_type", "currency", "configuration_id"],
        sort=False,
    )[
        [
            "total_date_count",
            "good_strong_occurrence_count",
            "strong_occurrence_count",
        ]
    ].nunique(dropna=False)
    if stable.gt(1).any().any():
        raise CalibrationReportValidationError(
            "configuration occurrence counts must not change across horizons"
        )


def _configuration_signal_consistency(
    configuration: pd.DataFrame,
    signal: pd.DataFrame,
) -> None:
    key_columns = ("period_type", "currency", "configuration_id", "horizon")
    review = signal.loc[signal["signal"].isin(("GOOD", "STRONG"))]
    for row in configuration.itertuples(index=False):
        group = review.loc[
            review["period_type"].eq(row.period_type)
            & review["currency"].eq(row.currency)
            & review["configuration_id"].eq(row.configuration_id)
            & review["horizon"].eq(row.horizon)
        ]
        if len(group) != 2 or set(group["signal"]) != {"GOOD", "STRONG"}:
            raise CalibrationReportValidationError(
                "configuration summary is missing GOOD or STRONG input"
            )
        expected = {
            "total_date_count": int(group["total_date_count"].iloc[0]),
            "good_strong_occurrence_count": int(group["occurrence_count"].sum()),
            "good_strong_evaluable_count": int(group["evaluable_count"].sum()),
            "good_strong_unavailable_count": int(group["unavailable_count"].sum()),
            "good_strong_favorable_count": int(group["favorable_count"].sum()),
            "good_strong_neutral_count": int(group["neutral_count"].sum()),
            "good_strong_unfavorable_count": int(group["unfavorable_count"].sum()),
        }
        strong = group.loc[group["signal"].eq("STRONG")].iloc[0]
        expected["strong_occurrence_count"] = int(strong["occurrence_count"])
        expected["strong_evaluable_count"] = int(strong["evaluable_count"])
        for name, value in expected.items():
            if int(getattr(row, name)) != value:
                raise CalibrationReportValidationError(
                    f"configuration and signal summaries disagree on {name}"
                )
        evaluable = expected["good_strong_evaluable_count"]
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
        _close(row.good_strong_mean_advantage_pct, weighted_mean, "combined mean")


def _period_count_consistency(
    period_data: pd.DataFrame,
    configuration: pd.DataFrame,
    signal: pd.DataFrame,
) -> None:
    counts = period_data.set_index(["period_type", "currency"])[
        "evaluation_row_count"
    ]
    for frame, name in (
        (configuration, "configuration_summary"),
        (signal, "signal_horizon_summary"),
    ):
        expected = np.asarray(
            [counts.loc[(row.period_type, row.currency)] for row in frame.itertuples()],
            dtype=float,
        )
        _close(frame["total_date_count"], expected, f"{name} total_date_count")


def _cross_consistency(
    cross: pd.DataFrame,
    signal: pd.DataFrame,
    configuration_ids: list[str],
) -> None:
    try:
        expected = build_cross_currency_summary(
            signal,
            currencies=CURRENCIES,
            configuration_ids=configuration_ids,
            horizons=HORIZONS,
        )
    except (TypeError, ValueError) as error:
        raise CalibrationReportValidationError(
            "cross-currency summary could not be recomputed"
        ) from error
    keys = ["period_type", "configuration_id", "signal", "horizon"]
    actual_sorted = cross.sort_values(keys).reset_index(drop=True)
    expected_sorted = expected.sort_values(keys).reset_index(drop=True)
    for name in keys:
        if actual_sorted[name].tolist() != expected_sorted[name].tolist():
            raise CalibrationReportValidationError(
                "cross-currency report keys do not match the signal summary"
            )
    for name in CROSS_CURRENCY_SUMMARY_COLUMNS:
        if name in keys:
            continue
        _close(
            _numeric(actual_sorted[name], name, missing=True),
            _numeric(expected_sorted[name], name, missing=True),
            f"cross-currency {name}",
        )
