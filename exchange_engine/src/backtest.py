"""Look-ahead-safe historical replay for the exchange Signal Engine.

Signals are evaluated from a prefix ending at each historical date.  Forward
rates are attached only after every signal has been produced, and are labels
for historical evaluation rather than inputs to the Signal Engine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
import pandas as pd

from src.indicators import (
    INDICATOR_COLUMNS,
    calculate_indicators,
    prepare_exchange_data,
)
from src.signal_engine import (
    Signal,
    SignalDecisionPolicy,
    SignalThresholds,
    evaluate_signal,
)


# This is a public candidate set, not a hidden production default.  Callers
# must still pass their chosen horizons explicitly to every public operation.
STANDARD_EVALUATION_HORIZONS = (5, 10, 20, 60)

IndicatorCalculator = Callable[[pd.DataFrame], pd.DataFrame]

DAILY_RESULT_COLUMNS = (
    "date",
    "rate",
    "signal",
    *tuple(INDICATOR_COLUMNS),
    "available_condition_count",
    "satisfied_condition_count",
    "strong_condition_count",
)

SUMMARY_COLUMNS = (
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
)


@dataclass(frozen=True)
class BacktestConfiguration:
    """One explicitly named threshold and aggregation-policy candidate."""

    configuration_id: str
    thresholds: SignalThresholds
    policy: SignalDecisionPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.configuration_id, str):
            raise TypeError("configuration_id must be a string")
        if (
            not self.configuration_id
            or not self.configuration_id.strip()
            or self.configuration_id != self.configuration_id.strip()
        ):
            raise ValueError(
                "configuration_id must be non-empty without surrounding whitespace"
            )
        _validate_signal_configuration(self.thresholds, self.policy)


@dataclass(frozen=True, eq=False)
class BacktestResult:
    """Daily evaluations and per-Signal historical outcome summaries.

    The dataclass fields cannot be rebound, but the contained DataFrames remain
    ordinary mutable pandas objects for caller-side reporting.
    """

    daily_results: pd.DataFrame = field(repr=False)
    summary: pd.DataFrame = field(repr=False)
    thresholds: SignalThresholds
    policy: SignalDecisionPolicy
    horizons: tuple[int, ...]
    indicator_calculator_name: str


@dataclass(frozen=True, eq=False)
class BacktestComparison:
    """Results for caller-supplied candidates, without ranking a winner.

    The dataclass fields cannot be rebound, but the contained DataFrames remain
    ordinary mutable pandas objects for caller-side reporting.
    """

    daily_results: pd.DataFrame = field(repr=False)
    summary: pd.DataFrame = field(repr=False)
    configurations: tuple[BacktestConfiguration, ...]
    horizons: tuple[int, ...]
    indicator_calculator_name: str


def forward_mean_column(horizon: int) -> str:
    """Return the stable daily-result column name for a forward mean."""
    return f"forward_{horizon}d_mean_rate"


def advantage_column(horizon: int) -> str:
    """Return the stable daily-result column name for an advantage label."""
    return f"advantage_{horizon}d_pct"


def run_backtest(
    historical_rates: pd.DataFrame,
    *,
    thresholds: SignalThresholds,
    policy: SignalDecisionPolicy,
    horizons: Iterable[int],
    indicator_calculator: IndicatorCalculator = calculate_indicators,
) -> BacktestResult:
    """Replay one configuration over historical observations.

    ``horizons`` counts subsequent observations, not calendar days.  The
    current row is excluded from each forward window.
    """
    _validate_signal_configuration(thresholds, policy)
    normalized_horizons = _normalize_horizons(horizons)
    calculator = _validate_indicator_calculator(indicator_calculator)
    historical = _prepare_historical_rates(historical_rates)

    # Phase 1 is causal: each calculator call receives only data through t.
    causal_indicators = _calculate_causal_indicators(historical, calculator)
    signal_rows = _evaluate_signal_rows(
        causal_indicators,
        thresholds=thresholds,
        policy=policy,
    )

    # Phase 2 deliberately starts after Signal evaluation and only adds labels.
    forward_labels = _calculate_forward_labels(historical, normalized_horizons)
    daily_results = pd.concat(
        [signal_rows.reset_index(drop=True), forward_labels],
        axis=1,
    )
    summary = summarize_by_signal(
        daily_results,
        horizons=normalized_horizons,
    )

    return BacktestResult(
        daily_results=daily_results,
        summary=summary,
        thresholds=thresholds,
        policy=policy,
        horizons=normalized_horizons,
        indicator_calculator_name=_calculator_name(calculator),
    )


def compare_configurations(
    historical_rates: pd.DataFrame,
    configurations: Iterable[BacktestConfiguration],
    *,
    horizons: Iterable[int],
    indicator_calculator: IndicatorCalculator = calculate_indicators,
) -> BacktestComparison:
    """Evaluate explicitly supplied configurations on one shared replay.

    This function performs no grid generation, scoring, ranking, optimization,
    or production-threshold selection.
    """
    candidates = _normalize_configurations(configurations)
    normalized_horizons = _normalize_horizons(horizons)
    calculator = _validate_indicator_calculator(indicator_calculator)
    historical = _prepare_historical_rates(historical_rates)

    # All candidates share the exact same causal indicator rows.  Signal rows
    # are completed for every candidate before any future label is calculated.
    causal_indicators = _calculate_causal_indicators(historical, calculator)
    candidate_signal_rows = [
        _evaluate_signal_rows(
            causal_indicators,
            thresholds=candidate.thresholds,
            policy=candidate.policy,
        )
        for candidate in candidates
    ]
    forward_labels = _calculate_forward_labels(historical, normalized_horizons)

    daily_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for candidate, signal_rows in zip(
        candidates,
        candidate_signal_rows,
        strict=True,
    ):
        daily = pd.concat(
            [signal_rows.reset_index(drop=True), forward_labels.copy(deep=True)],
            axis=1,
        )
        daily.insert(0, "configuration_id", candidate.configuration_id)
        daily_frames.append(daily)

        summary = summarize_by_signal(daily, horizons=normalized_horizons)
        summary.insert(0, "configuration_id", candidate.configuration_id)
        summary_frames.append(summary)

    return BacktestComparison(
        daily_results=pd.concat(daily_frames, ignore_index=True),
        summary=pd.concat(summary_frames, ignore_index=True),
        configurations=candidates,
        horizons=normalized_horizons,
        indicator_calculator_name=_calculator_name(calculator),
    )


def summarize_by_signal(
    daily_results: pd.DataFrame,
    *,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Summarize historical labels for every Signal and horizon.

    Favorable means ``advantage_pct > 0``; unfavorable means ``< 0`` and an
    exact zero is neutral.  Direction ratios use only rows whose advantage is
    available.  All four Signal values remain present even when their count is
    zero.
    """
    normalized_horizons = _normalize_horizons(horizons)
    if not isinstance(daily_results, pd.DataFrame):
        raise TypeError("daily_results must be a pandas DataFrame")
    if not daily_results.columns.is_unique:
        raise ValueError("daily_results must not contain duplicate columns")
    if daily_results.empty:
        raise ValueError("daily_results must not be empty")

    required_columns = {
        "signal",
        *(advantage_column(horizon) for horizon in normalized_horizons),
    }
    missing_columns = required_columns.difference(daily_results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"daily_results is missing required columns: {missing}")

    source = daily_results.copy(deep=True)
    source["signal"] = source["signal"].map(_normalize_signal_value)
    total_count = len(source)
    records: list[dict[str, object]] = []

    for signal in Signal:
        signal_mask = source["signal"].eq(signal.value)
        occurrence_count = int(signal_mask.sum())
        occurrence_ratio = occurrence_count / total_count

        for horizon in normalized_horizons:
            column = advantage_column(horizon)
            advantages = _numeric_advantages(source.loc[signal_mask, column], column)
            evaluable = advantages.dropna()
            evaluable_count = len(evaluable)
            unavailable_count = occurrence_count - evaluable_count
            favorable_count = int(evaluable.gt(0).sum())
            neutral_count = int(evaluable.eq(0).sum())
            unfavorable_count = int(evaluable.lt(0).sum())

            records.append(
                {
                    "signal": signal.value,
                    "horizon": horizon,
                    "total_date_count": total_count,
                    "occurrence_count": occurrence_count,
                    "occurrence_ratio": occurrence_ratio,
                    "evaluable_count": evaluable_count,
                    "unavailable_count": unavailable_count,
                    "evaluation_coverage_ratio": (
                        evaluable_count / occurrence_count
                        if occurrence_count
                        else np.nan
                    ),
                    "mean_advantage_pct": (
                        float(evaluable.mean()) if evaluable_count else np.nan
                    ),
                    "median_advantage_pct": (
                        float(evaluable.median()) if evaluable_count else np.nan
                    ),
                    "favorable_count": favorable_count,
                    "favorable_ratio": (
                        favorable_count / evaluable_count
                        if evaluable_count
                        else np.nan
                    ),
                    "neutral_count": neutral_count,
                    "neutral_ratio": (
                        neutral_count / evaluable_count
                        if evaluable_count
                        else np.nan
                    ),
                    "unfavorable_count": unfavorable_count,
                    "unfavorable_ratio": (
                        unfavorable_count / evaluable_count
                        if evaluable_count
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def _prepare_historical_rates(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("historical_rates must be a pandas DataFrame")
    if not data.columns.is_unique:
        raise ValueError("historical_rates must not contain duplicate columns")

    # Restrict the replay contract to the internal date/rate standard and keep
    # all normalization work off the caller's object.
    missing_columns = {"date", "rate"}.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"missing required columns: {missing}")
    historical = prepare_exchange_data(
        data.loc[:, ["date", "rate"]].copy(deep=True)
    )
    if historical.empty:
        raise ValueError("historical_rates must not be empty")
    historical["date"] = historical["date"].dt.normalize()
    if historical["date"].duplicated().any():
        raise ValueError("historical_rates must contain one observation per date")

    nonpositive = historical["rate"].notna() & historical["rate"].le(0)
    if nonpositive.any():
        raise ValueError(
            "historical rate must be greater than zero when available"
        )
    return historical.reset_index(drop=True)


def _calculate_causal_indicators(
    historical: pd.DataFrame,
    indicator_calculator: IndicatorCalculator,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    required_columns = {"date", "rate", *tuple(INDICATOR_COLUMNS)}

    for position in range(len(historical)):
        prefix = historical.iloc[: position + 1].copy(deep=True)
        calculated = indicator_calculator(prefix.copy(deep=True))
        if not isinstance(calculated, pd.DataFrame):
            raise TypeError("indicator_calculator must return a pandas DataFrame")
        if not calculated.columns.is_unique:
            raise ValueError(
                "indicator_calculator result must not contain duplicate columns"
            )

        missing_columns = required_columns.difference(calculated.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                "indicator_calculator result is missing required columns: "
                + missing
            )
        if len(calculated) != len(prefix):
            raise ValueError(
                "indicator_calculator must return one row per input observation"
            )

        output = calculated.copy(deep=True).reset_index(drop=True)
        try:
            output_dates = pd.to_datetime(output["date"], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "indicator_calculator result contains an invalid date"
            ) from error
        if output_dates.isna().any() or not output_dates.equals(prefix["date"]):
            raise ValueError(
                "indicator_calculator must preserve input dates and order"
            )

        output_rates = pd.to_numeric(output["rate"], errors="coerce").astype(float)
        output_rates = output_rates.replace([np.inf, -np.inf], np.nan)
        if not np.array_equal(
            output_rates.to_numpy(),
            prefix["rate"].to_numpy(),
            equal_nan=True,
        ):
            raise ValueError("indicator_calculator must preserve input rates")

        latest = output.iloc[-1]
        record = {"date": prefix.iloc[-1]["date"], "rate": prefix.iloc[-1]["rate"]}
        record.update({column: latest[column] for column in INDICATOR_COLUMNS})
        records.append(record)

    return pd.DataFrame.from_records(
        records,
        columns=("date", "rate", *tuple(INDICATOR_COLUMNS)),
    )


def _evaluate_signal_rows(
    causal_indicators: pd.DataFrame,
    *,
    thresholds: SignalThresholds,
    policy: SignalDecisionPolicy,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, indicator_row in causal_indicators.iterrows():
        decision = evaluate_signal(
            indicator_row,
            thresholds=thresholds,
            policy=policy,
        )
        record = {
            "date": indicator_row["date"],
            "rate": indicator_row["rate"],
            "signal": decision.signal.value,
        }
        record.update(
            {column: indicator_row[column] for column in INDICATOR_COLUMNS}
        )
        record.update(
            {
                "available_condition_count": decision.available_condition_count,
                "satisfied_condition_count": decision.satisfied_condition_count,
                "strong_condition_count": decision.strong_condition_count,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records, columns=DAILY_RESULT_COLUMNS)


def _calculate_forward_labels(
    historical: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rates = historical["rate"]
    labels: dict[str, pd.Series] = {}

    for horizon in horizons:
        # At row t, the rolling value at t+h covers exactly t+1 ... t+h.
        forward_mean = rates.rolling(
            window=horizon,
            min_periods=horizon,
        ).mean().shift(-horizon)
        advantage = ((forward_mean - rates) / rates) * 100
        advantage = advantage.replace([np.inf, -np.inf], np.nan)
        labels[forward_mean_column(horizon)] = forward_mean
        labels[advantage_column(horizon)] = advantage

    return pd.DataFrame(labels).reset_index(drop=True)


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


def _normalize_configurations(
    configurations: Iterable[BacktestConfiguration],
) -> tuple[BacktestConfiguration, ...]:
    if configurations is None or isinstance(configurations, (str, bytes)):
        raise TypeError(
            "configurations must be an iterable of BacktestConfiguration"
        )
    try:
        candidates = tuple(configurations)
    except TypeError as error:
        raise TypeError(
            "configurations must be an iterable of BacktestConfiguration"
        ) from error
    if not candidates:
        raise ValueError("configurations must not be empty")
    if not all(isinstance(candidate, BacktestConfiguration) for candidate in candidates):
        raise TypeError(
            "each configuration must be a BacktestConfiguration instance"
        )

    identifiers = [candidate.configuration_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("configuration_id values must be unique")
    return candidates


def _validate_signal_configuration(
    thresholds: object,
    policy: object,
) -> None:
    if not isinstance(thresholds, SignalThresholds):
        raise TypeError("thresholds must be a SignalThresholds instance")
    if not isinstance(policy, SignalDecisionPolicy):
        raise TypeError("policy must be a SignalDecisionPolicy instance")


def _validate_indicator_calculator(calculator: object) -> IndicatorCalculator:
    if not callable(calculator):
        raise TypeError("indicator_calculator must be callable")
    return calculator


def _calculator_name(calculator: IndicatorCalculator) -> str:
    return getattr(calculator, "__name__", calculator.__class__.__name__)


def _normalize_signal_value(value: object) -> str:
    if isinstance(value, Signal):
        return value.value
    if isinstance(value, str) and value in {signal.value for signal in Signal}:
        return value
    raise ValueError(f"unknown signal value: {value!r}")


def _numeric_advantages(values: pd.Series, column_name: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column_name} must contain numeric or missing values") from error
    return numeric.replace([np.inf, -np.inf], np.nan)
