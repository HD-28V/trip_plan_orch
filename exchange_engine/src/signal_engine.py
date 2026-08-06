"""Deterministic interpretation of one standard exchange indicator row.

This module describes whether the current observed rate is relatively low
against recent statistics. It does not predict future exchange rates. Numeric
indicator thresholds and condition aggregation rules are both supplied by the
caller so a later backtest can select them without changing this engine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real

import numpy as np
import pandas as pd

from src.indicators import INDICATOR_COLUMNS


SIGNAL_INPUT_COLUMNS = ("rate", *tuple(INDICATOR_COLUMNS))
CONDITION_NAMES = (
    "sma60_condition",
    "sma120_condition",
    "percentile_condition",
    "bollinger_condition",
)
CONDITION_COUNT = len(CONDITION_NAMES)


class Signal(str, Enum):
    """Supported descriptive signal levels, ordered from weakest to strongest."""

    WAIT = "WAIT"
    WATCH = "WATCH"
    GOOD = "GOOD"
    STRONG = "STRONG"


class ConditionStatus(str, Enum):
    """Outcome of evaluating one independent indicator condition."""

    UNAVAILABLE = "UNAVAILABLE"
    NOT_MET = "NOT_MET"
    GOOD = "GOOD"
    STRONG = "STRONG"


@dataclass(frozen=True)
class SignalThresholds:
    """Caller-supplied indicator boundaries; no production defaults exist."""

    sma60_good: float
    sma60_strong: float
    sma120_good: float
    sma120_strong: float
    percentile_good: float
    percentile_strong: float
    bollinger_near_lower_pct: float

    def __post_init__(self) -> None:
        for field_name in (
            "sma60_good",
            "sma60_strong",
            "sma120_good",
            "sma120_strong",
            "percentile_good",
            "percentile_strong",
            "bollinger_near_lower_pct",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_number(getattr(self, field_name), field_name),
            )

        if self.sma60_strong > self.sma60_good:
            raise ValueError("sma60_strong must be less than or equal to sma60_good")
        if self.sma120_strong > self.sma120_good:
            raise ValueError(
                "sma120_strong must be less than or equal to sma120_good"
            )
        if not 0 <= self.percentile_strong <= self.percentile_good <= 100:
            raise ValueError(
                "percentile thresholds must satisfy "
                "0 <= strong <= good <= 100"
            )
        if self.bollinger_near_lower_pct < 0:
            raise ValueError("bollinger_near_lower_pct must be zero or greater")


@dataclass(frozen=True)
class SignalDecisionPolicy:
    """Caller-supplied condition counts used to aggregate the final signal.

    Regardless of these counts, STRONG requires all core conditions to be
    available. This is a data-quality safeguard, not a production threshold.
    """

    minimum_available_conditions: int
    watch_min_satisfied_conditions: int
    good_min_satisfied_conditions: int
    strong_min_satisfied_conditions: int
    strong_min_strong_conditions: int

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_available_conditions",
            "watch_min_satisfied_conditions",
            "good_min_satisfied_conditions",
            "strong_min_satisfied_conditions",
            "strong_min_strong_conditions",
        ):
            object.__setattr__(
                self,
                field_name,
                _condition_count(getattr(self, field_name), field_name),
            )

        if not (
            self.watch_min_satisfied_conditions
            <= self.good_min_satisfied_conditions
            <= self.strong_min_satisfied_conditions
        ):
            raise ValueError(
                "satisfied-condition thresholds must be ordered "
                "watch <= good <= strong"
            )
        if (
            self.strong_min_strong_conditions
            > self.strong_min_satisfied_conditions
        ):
            raise ValueError(
                "strong_min_strong_conditions cannot exceed "
                "strong_min_satisfied_conditions"
            )


@dataclass(frozen=True)
class ConditionResult:
    """Auditable evaluation details for one independent condition."""

    name: str
    status: ConditionStatus
    observed_value: float | None
    good_threshold: float
    strong_threshold: float

    @property
    def available(self) -> bool:
        return self.status is not ConditionStatus.UNAVAILABLE

    @property
    def satisfied(self) -> bool:
        return self.status in {ConditionStatus.GOOD, ConditionStatus.STRONG}

    @property
    def strong(self) -> bool:
        return self.status is ConditionStatus.STRONG


@dataclass(frozen=True)
class SignalResult:
    """Structured signal plus every condition and configuration used."""

    signal: Signal
    current_rate: float | None
    sma60_condition: ConditionResult
    sma120_condition: ConditionResult
    percentile_condition: ConditionResult
    bollinger_condition: ConditionResult
    satisfied_conditions: tuple[str, ...]
    unavailable_indicators: tuple[str, ...]
    thresholds: SignalThresholds
    policy: SignalDecisionPolicy
    available_condition_count: int
    satisfied_condition_count: int
    strong_condition_count: int

    @property
    def conditions(self) -> tuple[ConditionResult, ...]:
        """Return conditions in a stable order for reporting and auditing."""
        return (
            self.sma60_condition,
            self.sma120_condition,
            self.percentile_condition,
            self.bollinger_condition,
        )


def evaluate_signal(
    indicator_row: pd.Series | Mapping[str, object],
    *,
    thresholds: SignalThresholds,
    policy: SignalDecisionPolicy,
) -> SignalResult:
    """Evaluate one indicator row without modifying it or predicting the future."""
    if not isinstance(thresholds, SignalThresholds):
        raise TypeError("thresholds must be a SignalThresholds instance")
    if not isinstance(policy, SignalDecisionPolicy):
        raise TypeError("policy must be a SignalDecisionPolicy instance")

    values = _copy_and_validate_indicator_row(indicator_row)
    unavailable_indicators = tuple(
        column for column in SIGNAL_INPUT_COLUMNS if values[column] is None
    )

    sma60_condition = _evaluate_lower_is_better(
        name="sma60_condition",
        value=(
            values["SMA60_distance_pct"]
            if values["SMA60"] is not None
            else None
        ),
        good_threshold=thresholds.sma60_good,
        strong_threshold=thresholds.sma60_strong,
    )
    sma120_condition = _evaluate_lower_is_better(
        name="sma120_condition",
        value=(
            values["SMA120_distance_pct"]
            if values["SMA120"] is not None
            else None
        ),
        good_threshold=thresholds.sma120_good,
        strong_threshold=thresholds.sma120_strong,
    )
    percentile_condition = _evaluate_lower_is_better(
        name="percentile_condition",
        value=values["percentile_rank_180"],
        good_threshold=thresholds.percentile_good,
        strong_threshold=thresholds.percentile_strong,
    )
    bollinger_condition = _evaluate_bollinger(
        current_rate=values["rate"],
        lower_band=values["BB_lower"],
        near_lower_pct=thresholds.bollinger_near_lower_pct,
    )

    conditions = (
        sma60_condition,
        sma120_condition,
        percentile_condition,
        bollinger_condition,
    )
    available_count = sum(condition.available for condition in conditions)
    satisfied_count = sum(condition.satisfied for condition in conditions)
    strong_count = sum(condition.strong for condition in conditions)
    signal = _aggregate_signal(
        available_count=available_count,
        satisfied_count=satisfied_count,
        strong_count=strong_count,
        policy=policy,
    )

    return SignalResult(
        signal=signal,
        current_rate=values["rate"],
        sma60_condition=sma60_condition,
        sma120_condition=sma120_condition,
        percentile_condition=percentile_condition,
        bollinger_condition=bollinger_condition,
        satisfied_conditions=tuple(
            condition.name for condition in conditions if condition.satisfied
        ),
        unavailable_indicators=unavailable_indicators,
        thresholds=thresholds,
        policy=policy,
        available_condition_count=available_count,
        satisfied_condition_count=satisfied_count,
        strong_condition_count=strong_count,
    )


def _copy_and_validate_indicator_row(
    indicator_row: pd.Series | Mapping[str, object],
) -> dict[str, float | None]:
    if isinstance(indicator_row, pd.Series):
        source = indicator_row.copy(deep=True).to_dict()
    elif isinstance(indicator_row, Mapping):
        source = dict(indicator_row)
    else:
        raise TypeError("indicator_row must be a pandas Series or mapping")

    missing_columns = [
        column for column in SIGNAL_INPUT_COLUMNS if column not in source
    ]
    if missing_columns:
        raise ValueError(
            "indicator_row is missing required columns: "
            + ", ".join(missing_columns)
        )

    values = {
        column: _optional_number(source[column], column)
        for column in SIGNAL_INPUT_COLUMNS
    }
    for column in ("rate", "SMA60", "SMA120", "BB_middle", "BB_upper"):
        value = values[column]
        if value is not None and value <= 0:
            raise ValueError(f"{column} must be greater than zero when available")

    percentile = values["percentile_rank_180"]
    if percentile is not None and not 0 <= percentile <= 100:
        raise ValueError("percentile_rank_180 must be between 0 and 100")

    lower_band = values["BB_lower"]
    if lower_band is not None and lower_band <= 0:
        values["BB_lower"] = None
        lower_band = None
    middle_band = values["BB_middle"]
    upper_band = values["BB_upper"]
    if (
        lower_band is not None
        and middle_band is not None
        and upper_band is not None
        and not lower_band <= middle_band <= upper_band
    ):
        raise ValueError("Bollinger Bands must satisfy lower <= middle <= upper")

    return values


def _evaluate_lower_is_better(
    *,
    name: str,
    value: float | None,
    good_threshold: float,
    strong_threshold: float,
) -> ConditionResult:
    if value is None:
        status = ConditionStatus.UNAVAILABLE
    elif value <= strong_threshold:
        status = ConditionStatus.STRONG
    elif value <= good_threshold:
        status = ConditionStatus.GOOD
    else:
        status = ConditionStatus.NOT_MET
    return ConditionResult(
        name=name,
        status=status,
        observed_value=value,
        good_threshold=good_threshold,
        strong_threshold=strong_threshold,
    )


def _evaluate_bollinger(
    *,
    current_rate: float | None,
    lower_band: float | None,
    near_lower_pct: float,
) -> ConditionResult:
    if current_rate is None or lower_band is None:
        return ConditionResult(
            name="bollinger_condition",
            status=ConditionStatus.UNAVAILABLE,
            observed_value=None,
            good_threshold=near_lower_pct,
            strong_threshold=0.0,
        )

    gap_pct = ((current_rate - lower_band) / lower_band) * 100
    if gap_pct <= 0:
        status = ConditionStatus.STRONG
    elif gap_pct <= near_lower_pct:
        status = ConditionStatus.GOOD
    else:
        status = ConditionStatus.NOT_MET
    return ConditionResult(
        name="bollinger_condition",
        status=status,
        observed_value=gap_pct,
        good_threshold=near_lower_pct,
        strong_threshold=0.0,
    )


def _aggregate_signal(
    *,
    available_count: int,
    satisfied_count: int,
    strong_count: int,
    policy: SignalDecisionPolicy,
) -> Signal:
    if available_count < policy.minimum_available_conditions:
        return Signal.WAIT
    if (
        available_count == CONDITION_COUNT
        and satisfied_count >= policy.strong_min_satisfied_conditions
        and strong_count >= policy.strong_min_strong_conditions
    ):
        return Signal.STRONG
    if satisfied_count >= policy.good_min_satisfied_conditions:
        return Signal.GOOD
    if satisfied_count >= policy.watch_min_satisfied_conditions:
        return Signal.WATCH
    return Signal.WAIT


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be a finite number")
    return numeric_value


def _condition_count(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    numeric_value = int(value)
    if not 1 <= numeric_value <= CONDITION_COUNT:
        raise ValueError(
            f"{field_name} must be between 1 and {CONDITION_COUNT}"
        )
    return numeric_value


def _optional_number(value: object, field_name: str) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric or missing")

    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)):
        if bool(missing):
            return None
    else:
        raise ValueError(f"{field_name} must be a scalar value")

    if not isinstance(value, Real):
        raise ValueError(f"{field_name} must be numeric or missing")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value
