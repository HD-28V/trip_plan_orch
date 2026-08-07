"""Threshold-candidate calibration reports built on the Backtest Engine.

This module compares caller-supplied configurations and produces descriptive
historical reports.  It does not rank candidates, select a winner, or define a
production threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.backtest import (
    SUMMARY_COLUMNS,
    IndicatorCalculator,
    advantage_column,
    compare_configurations,
    summarize_by_signal,
)
from src.calibration_candidates import CandidatePlan
from src.indicators import calculate_indicators, prepare_exchange_data
from src.signal_engine import Signal


CALIBRATION_CURRENCIES = ("USD", "JPY", "EUR")


class PeriodType(str, Enum):
    CALIBRATION = "CALIBRATION"
    VALIDATION = "VALIDATION"


SIGNAL_HORIZON_REPORT_COLUMNS = (
    "period_type",
    "currency",
    "configuration_id",
    *SUMMARY_COLUMNS,
)

CONFIGURATION_SUMMARY_COLUMNS = (
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
    "good_strong_neutral_count",
    "good_strong_neutral_ratio",
    "good_strong_unfavorable_count",
    "good_strong_unfavorable_ratio",
    "strong_occurrence_count",
    "strong_evaluable_count",
)

CROSS_CURRENCY_METRICS = (
    "occurrence_count",
    "evaluable_count",
    "evaluation_coverage_ratio",
    "mean_advantage_pct",
    "favorable_ratio",
)

CROSS_CURRENCY_SUMMARY_COLUMNS = (
    "period_type",
    "configuration_id",
    "signal",
    "horizon",
    *tuple(
        f"{currency}_{metric}"
        for currency in CALIBRATION_CURRENCIES
        for metric in CROSS_CURRENCY_METRICS
    ),
    "reported_currency_count",
    "minimum_occurrence_count",
    "minimum_evaluable_count",
    "mean_advantage_defined_currency_count",
    "mean_advantage_across_currencies_pct",
    "minimum_mean_advantage_pct",
    "maximum_mean_advantage_pct",
    "mean_advantage_currency_stddev_pct",
    "favorable_defined_currency_count",
    "mean_favorable_ratio_across_currencies",
    "minimum_favorable_ratio",
    "maximum_favorable_ratio",
    "favorable_ratio_currency_stddev",
)

PERIOD_DATA_SUMMARY_COLUMNS = (
    "period_type",
    "currency",
    "source_start_date",
    "source_end_date",
    "source_row_count",
    "replay_row_count",
    "warmup_row_count",
    "evaluation_start_date",
    "evaluation_end_date",
    "evaluation_row_count",
    "candidate_configuration_count",
    "elapsed_seconds",
)

REPORT_METADATA_COLUMNS = ("field", "value")

REPORT_FILES = (
    ("report_metadata.csv", "report_metadata"),
    ("candidate_configurations.csv", "candidate_configurations"),
    ("period_data_summary.csv", "period_data_summary"),
    ("configuration_summary.csv", "configuration_summary"),
    ("signal_horizon_summary.csv", "signal_horizon_summary"),
    ("cross_currency_summary.csv", "cross_currency_summary"),
)


class CalibrationDataError(ValueError):
    """Raised when historical data cannot form both requested periods."""


@dataclass(frozen=True, eq=False)
class CalibrationReport:
    """In-memory replay details and small human-reviewable report tables."""

    daily_results: pd.DataFrame = field(repr=False)
    signal_horizon_summary: pd.DataFrame = field(repr=False)
    configuration_summary: pd.DataFrame = field(repr=False)
    cross_currency_summary: pd.DataFrame = field(repr=False)
    report_metadata: pd.DataFrame = field(repr=False)
    candidate_configurations: pd.DataFrame = field(repr=False)
    period_data_summary: pd.DataFrame = field(repr=False)
    currencies: tuple[str, ...]
    horizons: tuple[int, ...]
    validation_start: pd.Timestamp
    elapsed_seconds: float


def run_calibration(
    historical_by_currency: Mapping[str, pd.DataFrame],
    *,
    validation_start: object,
    candidate_plan: CandidatePlan,
    horizons: Iterable[int],
    indicator_calculator: IndicatorCalculator = calculate_indicators,
    timer: Callable[[], float] = perf_counter,
) -> CalibrationReport:
    """Run separate calibration and validation replays for every currency.

    Calibration receives only rows before ``validation_start`` so its forward
    labels cannot cross the split. Validation replays the complete history for
    causal indicator warm-up, then keeps only rows on or after the split.
    """
    if not isinstance(candidate_plan, CandidatePlan):
        raise TypeError("candidate_plan must be a CandidatePlan instance")
    normalized_horizons = _normalize_horizons(horizons)
    split_date = _normalize_date(validation_start, "validation_start")
    histories = _normalize_historical_mapping(historical_by_currency)
    if not callable(indicator_calculator):
        raise TypeError("indicator_calculator must be callable")
    if not callable(timer):
        raise TypeError("timer must be callable")

    overall_started = timer()
    calibration_daily_frames: list[pd.DataFrame] = []
    validation_daily_frames: list[pd.DataFrame] = []
    calibration_summary_frames: list[pd.DataFrame] = []
    validation_summary_frames: list[pd.DataFrame] = []
    calibration_period_rows: list[dict[str, object]] = []
    validation_period_rows: list[dict[str, object]] = []

    for currency, historical in histories.items():
        calibration_history = historical.loc[
            historical["date"].lt(split_date)
        ].reset_index(drop=True)
        validation_history = historical.loc[
            historical["date"].ge(split_date)
        ].reset_index(drop=True)
        if calibration_history.empty:
            raise CalibrationDataError(
                f"calibration period has no observations for {currency}"
            )
        if validation_history.empty:
            raise CalibrationDataError(
                f"validation period has no observations for {currency}"
            )

        calibration_started = timer()
        calibration_comparison = compare_configurations(
            calibration_history,
            candidate_plan.configurations,
            horizons=normalized_horizons,
            indicator_calculator=indicator_calculator,
        )
        calibration_elapsed = timer() - calibration_started

        validation_started = timer()
        full_comparison = compare_configurations(
            historical,
            candidate_plan.configurations,
            horizons=normalized_horizons,
            indicator_calculator=indicator_calculator,
        )
        full_daily = full_comparison.daily_results.copy(deep=True)
        validation_daily = full_daily.loc[
            pd.to_datetime(full_daily["date"]).ge(split_date)
        ].reset_index(drop=True)
        validation_elapsed = timer() - validation_started

        calibration_daily = _tag_daily_results(
            calibration_comparison.daily_results,
            period_type=PeriodType.CALIBRATION,
            currency=currency,
        )
        validation_daily = _tag_daily_results(
            validation_daily,
            period_type=PeriodType.VALIDATION,
            currency=currency,
        )
        calibration_daily_frames.append(calibration_daily)
        validation_daily_frames.append(validation_daily)

        calibration_summary_frames.append(
            _tag_signal_summary(
                calibration_comparison.summary,
                period_type=PeriodType.CALIBRATION,
                currency=currency,
            )
        )
        validation_summary_frames.append(
            _summarize_filtered_period(
                validation_daily,
                candidate_plan=candidate_plan,
                horizons=normalized_horizons,
                period_type=PeriodType.VALIDATION,
                currency=currency,
            )
        )

        source_start = historical["date"].iloc[0]
        source_end = historical["date"].iloc[-1]
        calibration_period_rows.append(
            _period_data_row(
                period_type=PeriodType.CALIBRATION,
                currency=currency,
                source_start=source_start,
                source_end=source_end,
                source_count=len(historical),
                replay_count=len(calibration_history),
                warmup_count=0,
                evaluation_dates=calibration_history["date"],
                candidate_count=candidate_plan.candidate_count,
                elapsed_seconds=calibration_elapsed,
            )
        )
        validation_period_rows.append(
            _period_data_row(
                period_type=PeriodType.VALIDATION,
                currency=currency,
                source_start=source_start,
                source_end=source_end,
                source_count=len(historical),
                replay_count=len(historical),
                warmup_count=len(calibration_history),
                evaluation_dates=validation_history["date"],
                candidate_count=candidate_plan.candidate_count,
                elapsed_seconds=validation_elapsed,
            )
        )

    daily_results = pd.concat(
        [*calibration_daily_frames, *validation_daily_frames],
        ignore_index=True,
    )
    signal_horizon_summary = pd.concat(
        [*calibration_summary_frames, *validation_summary_frames],
        ignore_index=True,
    ).loc[:, SIGNAL_HORIZON_REPORT_COLUMNS]
    period_data_summary = pd.DataFrame.from_records(
        [*calibration_period_rows, *validation_period_rows],
        columns=PERIOD_DATA_SUMMARY_COLUMNS,
    )
    configuration_ids = tuple(
        configuration.configuration_id
        for configuration in candidate_plan.configurations
    )
    currencies = tuple(histories)
    configuration_summary = build_configuration_summary(
        daily_results,
        currencies=currencies,
        configuration_ids=configuration_ids,
        horizons=normalized_horizons,
    )
    cross_currency_summary = build_cross_currency_summary(
        signal_horizon_summary,
        currencies=currencies,
        configuration_ids=configuration_ids,
        horizons=normalized_horizons,
    )
    elapsed_seconds = timer() - overall_started
    report_metadata = _build_report_metadata(
        currencies=currencies,
        horizons=normalized_horizons,
        validation_start=split_date,
        candidate_count=candidate_plan.candidate_count,
        elapsed_seconds=elapsed_seconds,
    )

    return CalibrationReport(
        daily_results=daily_results,
        signal_horizon_summary=signal_horizon_summary,
        configuration_summary=configuration_summary,
        cross_currency_summary=cross_currency_summary,
        report_metadata=report_metadata,
        candidate_configurations=candidate_plan.to_manifest(),
        period_data_summary=period_data_summary,
        currencies=currencies,
        horizons=normalized_horizons,
        validation_start=split_date,
        elapsed_seconds=float(elapsed_seconds),
    )


def build_configuration_summary(
    daily_results: pd.DataFrame,
    *,
    currencies: Iterable[str],
    configuration_ids: Iterable[str],
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Aggregate GOOD+STRONG directly from daily labels for each horizon."""
    source = _copy_required_frame(
        daily_results,
        {
            "period_type",
            "currency",
            "configuration_id",
            "signal",
        },
        "daily_results",
    )
    normalized_currencies = _normalize_currency_sequence(currencies)
    identifiers = _normalize_configuration_ids(configuration_ids)
    normalized_horizons = _normalize_horizons(horizons)
    for horizon in normalized_horizons:
        column = advantage_column(horizon)
        if column not in source.columns:
            raise ValueError(f"daily_results is missing required column: {column}")

    rows: list[dict[str, object]] = []
    for period_type in PeriodType:
        for currency in normalized_currencies:
            for configuration_id in identifiers:
                group = source.loc[
                    source["period_type"].eq(period_type.value)
                    & source["currency"].eq(currency)
                    & source["configuration_id"].eq(configuration_id)
                ]
                if group.empty:
                    raise ValueError(
                        "daily_results is missing a requested period/currency/"
                        f"configuration group: {period_type.value}/{currency}/"
                        f"{configuration_id}"
                    )
                total_count = len(group)
                good_strong_mask = group["signal"].isin(
                    (Signal.GOOD.value, Signal.STRONG.value)
                )
                strong_mask = group["signal"].eq(Signal.STRONG.value)

                for horizon in normalized_horizons:
                    advantages = _numeric_series(
                        group.loc[good_strong_mask, advantage_column(horizon)],
                        advantage_column(horizon),
                    )
                    evaluable = advantages.dropna()
                    occurrence_count = int(good_strong_mask.sum())
                    evaluable_count = len(evaluable)
                    favorable_count = int(evaluable.gt(0).sum())
                    neutral_count = int(evaluable.eq(0).sum())
                    unfavorable_count = int(evaluable.lt(0).sum())
                    strong_evaluable_count = int(
                        _numeric_series(
                            group.loc[
                                strong_mask,
                                advantage_column(horizon),
                            ],
                            advantage_column(horizon),
                        )
                        .notna()
                        .sum()
                    )
                    rows.append(
                        {
                            "period_type": period_type.value,
                            "currency": currency,
                            "configuration_id": configuration_id,
                            "horizon": horizon,
                            "total_date_count": total_count,
                            "good_strong_occurrence_count": occurrence_count,
                            "good_strong_occurrence_ratio": (
                                occurrence_count / total_count
                            ),
                            "good_strong_evaluable_count": evaluable_count,
                            "good_strong_unavailable_count": (
                                occurrence_count - evaluable_count
                            ),
                            "good_strong_evaluation_coverage_ratio": (
                                evaluable_count / occurrence_count
                                if occurrence_count
                                else np.nan
                            ),
                            "good_strong_mean_advantage_pct": (
                                float(evaluable.mean())
                                if evaluable_count
                                else np.nan
                            ),
                            "good_strong_median_advantage_pct": (
                                float(evaluable.median())
                                if evaluable_count
                                else np.nan
                            ),
                            "good_strong_favorable_count": favorable_count,
                            "good_strong_favorable_ratio": _safe_ratio(
                                favorable_count,
                                evaluable_count,
                            ),
                            "good_strong_neutral_count": neutral_count,
                            "good_strong_neutral_ratio": _safe_ratio(
                                neutral_count,
                                evaluable_count,
                            ),
                            "good_strong_unfavorable_count": unfavorable_count,
                            "good_strong_unfavorable_ratio": _safe_ratio(
                                unfavorable_count,
                                evaluable_count,
                            ),
                            "strong_occurrence_count": int(strong_mask.sum()),
                            "strong_evaluable_count": strong_evaluable_count,
                        }
                    )

    return pd.DataFrame.from_records(rows, columns=CONFIGURATION_SUMMARY_COLUMNS)


def build_cross_currency_summary(
    signal_horizon_summary: pd.DataFrame,
    *,
    currencies: Iterable[str],
    configuration_ids: Iterable[str],
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Place currency results side by side without ranking configurations."""
    source = _copy_required_frame(
        signal_horizon_summary,
        set(SIGNAL_HORIZON_REPORT_COLUMNS),
        "signal_horizon_summary",
    )
    normalized_currencies = _normalize_currency_sequence(currencies)
    identifiers = _normalize_configuration_ids(configuration_ids)
    normalized_horizons = _normalize_horizons(horizons)
    rows: list[dict[str, object]] = []

    for period_type in PeriodType:
        for configuration_id in identifiers:
            for signal in Signal:
                for horizon in normalized_horizons:
                    group = source.loc[
                        source["period_type"].eq(period_type.value)
                        & source["configuration_id"].eq(configuration_id)
                        & source["signal"].eq(signal.value)
                        & source["horizon"].eq(horizon)
                    ]
                    expected = set(normalized_currencies)
                    received = set(group["currency"])
                    if received != expected or len(group) != len(expected):
                        raise ValueError(
                            "signal_horizon_summary must contain exactly one row "
                            "per requested currency and report key"
                        )

                    row: dict[str, object] = {
                        "period_type": period_type.value,
                        "configuration_id": configuration_id,
                        "signal": signal.value,
                        "horizon": horizon,
                    }
                    for supported_currency in CALIBRATION_CURRENCIES:
                        currency_row = group.loc[
                            group["currency"].eq(supported_currency)
                        ]
                        for metric in CROSS_CURRENCY_METRICS:
                            row[f"{supported_currency}_{metric}"] = (
                                currency_row.iloc[0][metric]
                                if not currency_row.empty
                                else np.nan
                            )

                    occurrence_counts = _numeric_series(
                        group["occurrence_count"],
                        "occurrence_count",
                    ).dropna()
                    evaluable_counts = _numeric_series(
                        group["evaluable_count"],
                        "evaluable_count",
                    ).dropna()
                    mean_advantages = _numeric_series(
                        group["mean_advantage_pct"],
                        "mean_advantage_pct",
                    ).dropna()
                    favorable_ratios = _numeric_series(
                        group["favorable_ratio"],
                        "favorable_ratio",
                    ).dropna()

                    row.update(
                        {
                            "reported_currency_count": len(group),
                            "minimum_occurrence_count": (
                                int(occurrence_counts.min())
                                if len(occurrence_counts)
                                else np.nan
                            ),
                            "minimum_evaluable_count": (
                                int(evaluable_counts.min())
                                if len(evaluable_counts)
                                else np.nan
                            ),
                            "mean_advantage_defined_currency_count": len(
                                mean_advantages
                            ),
                            "mean_advantage_across_currencies_pct": (
                                float(mean_advantages.mean())
                                if len(mean_advantages)
                                else np.nan
                            ),
                            "minimum_mean_advantage_pct": (
                                float(mean_advantages.min())
                                if len(mean_advantages)
                                else np.nan
                            ),
                            "maximum_mean_advantage_pct": (
                                float(mean_advantages.max())
                                if len(mean_advantages)
                                else np.nan
                            ),
                            "mean_advantage_currency_stddev_pct": _population_std(
                                mean_advantages
                            ),
                            "favorable_defined_currency_count": len(
                                favorable_ratios
                            ),
                            "mean_favorable_ratio_across_currencies": (
                                float(favorable_ratios.mean())
                                if len(favorable_ratios)
                                else np.nan
                            ),
                            "minimum_favorable_ratio": (
                                float(favorable_ratios.min())
                                if len(favorable_ratios)
                                else np.nan
                            ),
                            "maximum_favorable_ratio": (
                                float(favorable_ratios.max())
                                if len(favorable_ratios)
                                else np.nan
                            ),
                            "favorable_ratio_currency_stddev": _population_std(
                                favorable_ratios
                            ),
                        }
                    )
                    rows.append(row)

    return pd.DataFrame.from_records(rows, columns=CROSS_CURRENCY_SUMMARY_COLUMNS)


def write_calibration_reports(
    report: CalibrationReport,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Write only compact report tables and refuse implicit overwrites."""
    if not isinstance(report, CalibrationReport):
        raise TypeError("report must be a CalibrationReport instance")
    destination = Path(output_dir)
    targets = tuple(destination / filename for filename, _ in REPORT_FILES)
    existing = [target for target in targets if target.exists()]
    if existing:
        raise FileExistsError(
            "calibration report file already exists: " + existing[0].name
        )
    destination.mkdir(parents=True, exist_ok=True)

    for target, (_, attribute_name) in zip(targets, REPORT_FILES, strict=True):
        frame = getattr(report, attribute_name)
        frame.to_csv(target, index=False, encoding="utf-8")
    return targets


def _normalize_historical_mapping(
    historical_by_currency: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if not isinstance(historical_by_currency, Mapping):
        raise TypeError("historical_by_currency must be a mapping")
    if not historical_by_currency:
        raise ValueError("historical_by_currency must not be empty")

    normalized: dict[str, pd.DataFrame] = {}
    for currency, data in historical_by_currency.items():
        normalized_currency = _normalize_currency(currency)
        if normalized_currency in normalized:
            raise ValueError("historical currency keys must be unique")
        normalized[normalized_currency] = _prepare_historical(data)
    return normalized


def _prepare_historical(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("each historical value must be a pandas DataFrame")
    if not data.columns.is_unique:
        raise ValueError("historical data must not contain duplicate columns")
    missing = {"date", "rate"}.difference(data.columns)
    if missing:
        raise ValueError(
            "historical data is missing required columns: "
            + ", ".join(sorted(missing))
        )
    result = prepare_exchange_data(data.loc[:, ["date", "rate"]].copy(deep=True))
    if result.empty:
        raise CalibrationDataError("historical data must not be empty")
    result["date"] = result["date"].dt.normalize()
    if result["date"].duplicated().any():
        raise CalibrationDataError(
            "historical data must contain one observation per date"
        )
    nonpositive = result["rate"].notna() & result["rate"].le(0)
    if nonpositive.any():
        raise CalibrationDataError(
            "historical rate must be greater than zero when available"
        )
    return result.reset_index(drop=True)


def _tag_daily_results(
    daily_results: pd.DataFrame,
    *,
    period_type: PeriodType,
    currency: str,
) -> pd.DataFrame:
    result = daily_results.copy(deep=True)
    result.insert(0, "currency", currency)
    result.insert(0, "period_type", period_type.value)
    return result


def _tag_signal_summary(
    summary: pd.DataFrame,
    *,
    period_type: PeriodType,
    currency: str,
) -> pd.DataFrame:
    result = summary.copy(deep=True)
    result.insert(0, "currency", currency)
    result.insert(0, "period_type", period_type.value)
    return result.loc[:, SIGNAL_HORIZON_REPORT_COLUMNS]


def _summarize_filtered_period(
    daily_results: pd.DataFrame,
    *,
    candidate_plan: CandidatePlan,
    horizons: tuple[int, ...],
    period_type: PeriodType,
    currency: str,
) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for configuration in candidate_plan.configurations:
        configuration_daily = daily_results.loc[
            daily_results["configuration_id"].eq(configuration.configuration_id)
        ]
        summary = summarize_by_signal(
            configuration_daily,
            horizons=horizons,
        )
        summary.insert(0, "configuration_id", configuration.configuration_id)
        summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True)
    return _tag_signal_summary(
        combined,
        period_type=period_type,
        currency=currency,
    )


def _period_data_row(
    *,
    period_type: PeriodType,
    currency: str,
    source_start: pd.Timestamp,
    source_end: pd.Timestamp,
    source_count: int,
    replay_count: int,
    warmup_count: int,
    evaluation_dates: pd.Series,
    candidate_count: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "period_type": period_type.value,
        "currency": currency,
        "source_start_date": source_start,
        "source_end_date": source_end,
        "source_row_count": source_count,
        "replay_row_count": replay_count,
        "warmup_row_count": warmup_count,
        "evaluation_start_date": evaluation_dates.iloc[0],
        "evaluation_end_date": evaluation_dates.iloc[-1],
        "evaluation_row_count": len(evaluation_dates),
        "candidate_configuration_count": candidate_count,
        "elapsed_seconds": float(elapsed_seconds),
    }


def _copy_required_frame(
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


def _normalize_currency_sequence(currencies: Iterable[str]) -> tuple[str, ...]:
    if currencies is None or isinstance(currencies, (str, bytes)):
        raise TypeError("currencies must be an iterable of currency codes")
    try:
        normalized = tuple(_normalize_currency(value) for value in currencies)
    except TypeError as error:
        raise TypeError("currencies must be an iterable of currency codes") from error
    if not normalized:
        raise ValueError("currencies must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("currencies must not contain duplicates")
    return normalized


def _normalize_currency(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("currency must be a string")
    normalized = value.strip().upper()
    if normalized not in CALIBRATION_CURRENCIES:
        raise ValueError(
            "calibration currency must be one of: "
            + ", ".join(CALIBRATION_CURRENCIES)
        )
    return normalized


def _normalize_configuration_ids(values: Iterable[str]) -> tuple[str, ...]:
    if values is None or isinstance(values, (str, bytes)):
        raise TypeError("configuration_ids must be an iterable")
    try:
        identifiers = tuple(values)
    except TypeError as error:
        raise TypeError("configuration_ids must be an iterable") from error
    if not identifiers:
        raise ValueError("configuration_ids must not be empty")
    if not all(isinstance(value, str) and value for value in identifiers):
        raise ValueError("configuration_ids must contain non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("configuration_ids must be unique")
    return identifiers


def _normalize_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    if horizons is None or isinstance(horizons, (str, bytes)):
        raise TypeError("horizons must be an iterable of positive integers")
    try:
        normalized = tuple(horizons)
    except TypeError as error:
        raise TypeError(
            "horizons must be an iterable of positive integers"
        ) from error
    if not normalized:
        raise ValueError("horizons must not be empty")
    for horizon in normalized:
        if (
            isinstance(horizon, (bool, np.bool_))
            or not isinstance(horizon, Integral)
        ):
            raise TypeError("each horizon must be a positive integer")
        if horizon <= 0:
            raise ValueError("each horizon must be a positive integer")
    integer_horizons = tuple(int(horizon) for horizon in normalized)
    if len(set(integer_horizons)) != len(integer_horizons):
        raise ValueError("horizons must not contain duplicates")
    return integer_horizons


def _normalize_date(value: object, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a valid date") from error
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must be a valid date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else np.nan


def _population_std(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) >= 2 else np.nan


def _numeric_series(values: pd.Series, field_name: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must contain numeric or missing values"
        ) from error
    return numeric.replace([np.inf, -np.inf], np.nan)


def _build_report_metadata(
    *,
    currencies: tuple[str, ...],
    horizons: tuple[int, ...],
    validation_start: pd.Timestamp,
    candidate_count: int,
    elapsed_seconds: float,
) -> pd.DataFrame:
    rows = (
        ("report_purpose", "historical candidate comparison; no winner selected"),
        (
            "advantage_pct",
            "(forward_mean_rate - entry_rate) / entry_rate * 100",
        ),
        ("favorable", "advantage_pct > 0"),
        ("neutral", "advantage_pct == 0"),
        ("unfavorable", "advantage_pct < 0"),
        ("horizon_unit", "subsequent trading observations; entry excluded"),
        ("calibration_period", f"date < {validation_start.date().isoformat()}"),
        ("validation_period", f"date >= {validation_start.date().isoformat()}"),
        ("occurrence_ratio_denominator", "all evaluation dates in period"),
        ("coverage_denominator", "signal occurrence_count"),
        ("direction_ratio_denominator", "non-NaN evaluable_count"),
        ("cross_currency_weighting", "equal weight per defined currency metric"),
        ("cross_currency_stddev", "population standard deviation (ddof=0)"),
        ("currencies", ",".join(currencies)),
        ("horizons", ",".join(str(horizon) for horizon in horizons)),
        ("candidate_configuration_count", str(candidate_count)),
        ("total_runtime_seconds", f"{float(elapsed_seconds):.6f}"),
    )
    return pd.DataFrame(rows, columns=REPORT_METADATA_COLUMNS)
