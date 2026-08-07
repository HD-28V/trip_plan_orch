"""One production-v1 recommendation assembled from the existing engines.

This module is deliberately an orchestration layer.  It introduces no FX
threshold, D-Day, or split-exchange rule; each stage delegates to the existing
production entry point and preserves its immutable result for auditability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real

import pandas as pd

from src.dday_policy import (
    DdaySignalResult,
    get_production_dday_policy,
    evaluate_production_dday_signal,
)
from src.production_config import (
    evaluate_production_signal,
    get_production_signal_configuration,
)
from src.signal_engine import ConditionResult, Signal, SignalResult
from src.split_exchange import (
    SplitExchangeResult,
    evaluate_production_split_exchange,
    get_production_split_exchange_policy,
)


class RecommendationValidationError(ValueError):
    """Raised when production-stage metadata cannot form one coherent result."""


@dataclass(frozen=True)
class RecommendationTrace:
    """Structured stage evidence for presentation layers and audits."""

    market_conditions: tuple[ConditionResult, ...]
    satisfied_conditions: tuple[str, ...]
    unavailable_indicators: tuple[str, ...]
    dday_reason: str
    split_reason: str


@dataclass(frozen=True)
class ExchangeRecommendationResult:
    """Immutable aggregate of the three existing production-v1 results."""

    market_result: SignalResult
    dday_result: DdaySignalResult
    split_result: SplitExchangeResult
    trace: RecommendationTrace

    @property
    def base_signal(self) -> Signal:
        return self.market_result.signal

    @property
    def current_rate(self) -> float | None:
        return self.market_result.current_rate

    @property
    def conditions(self) -> tuple[ConditionResult, ...]:
        return self.market_result.conditions

    @property
    def satisfied_conditions(self) -> tuple[str, ...]:
        return self.market_result.satisfied_conditions

    @property
    def unavailable_indicators(self) -> tuple[str, ...]:
        return self.market_result.unavailable_indicators

    @property
    def adjusted_signal(self) -> Signal:
        return self.dday_result.adjusted_signal

    @property
    def days_until_departure(self) -> int:
        return self.dday_result.days_until_departure

    @property
    def dday_band(self):
        return self.dday_result.dday_band

    @property
    def urgency(self):
        return self.dday_result.urgency

    @property
    def applied_promotion_steps(self) -> int:
        return self.dday_result.applied_promotion_steps

    @property
    def total_target_krw(self) -> Decimal:
        return self.split_result.total_target_krw

    @property
    def already_exchanged_krw(self) -> Decimal:
        return self.split_result.already_exchanged_krw

    @property
    def signal_target_ratio(self) -> Decimal:
        return self.split_result.signal_target_ratio

    @property
    def urgency_minimum_ratio(self) -> Decimal:
        return self.split_result.urgency_minimum_ratio

    @property
    def target_cumulative_ratio(self) -> Decimal:
        return self.split_result.target_cumulative_ratio

    @property
    def target_cumulative_amount_krw(self) -> Decimal:
        return self.split_result.target_cumulative_amount_krw

    @property
    def recommended_additional_krw(self) -> Decimal:
        return self.split_result.recommended_additional_krw

    @property
    def remaining_after_recommendation_krw(self) -> Decimal:
        return self.split_result.remaining_after_recommendation_krw

    @property
    def production_configuration_id(self) -> str | None:
        return self.split_result.production_configuration_id

    @property
    def production_configuration_version(self) -> str | None:
        return self.split_result.production_configuration_version

    @property
    def dday_policy_id(self) -> str:
        return self.split_result.dday_policy_id

    @property
    def dday_policy_version(self) -> str:
        return self.split_result.dday_policy_version

    @property
    def split_policy_id(self) -> str:
        return self.split_result.split_policy_id

    @property
    def split_policy_version(self) -> str:
        return self.split_result.split_policy_version


def evaluate_exchange_recommendation(
    indicator_row: pd.Series | Mapping[str, object],
    days_until_departure: int,
    total_target_krw: Decimal | Real,
    already_exchanged_krw: Decimal | Real,
) -> ExchangeRecommendationResult:
    """Evaluate the established production-v1 stages in their required order."""
    market_result = evaluate_production_signal(indicator_row)
    dday_result = evaluate_production_dday_signal(
        market_result,
        days_until_departure,
    )
    split_result = evaluate_production_split_exchange(
        dday_result,
        total_target_krw,
        already_exchanged_krw,
    )
    _validate_metadata_consistency(market_result, dday_result, split_result)
    return ExchangeRecommendationResult(
        market_result=market_result,
        dday_result=dday_result,
        split_result=split_result,
        trace=RecommendationTrace(
            market_conditions=market_result.conditions,
            satisfied_conditions=market_result.satisfied_conditions,
            unavailable_indicators=market_result.unavailable_indicators,
            dday_reason=dday_result.reason,
            split_reason=split_result.reason,
        ),
    )


def _validate_metadata_consistency(
    market_result: SignalResult,
    dday_result: DdaySignalResult,
    split_result: SplitExchangeResult,
) -> None:
    """Reject results that do not refer to the same frozen production contract."""
    configuration = get_production_signal_configuration()
    dday_policy = get_production_dday_policy()
    split_policy = get_production_split_exchange_policy()
    if (
        market_result.thresholds != configuration.thresholds
        or market_result.policy != configuration.policy
    ):
        raise RecommendationValidationError(
            "market result does not use the production FX configuration"
        )
    if (
        dday_result.production_configuration_id != configuration.configuration_id
        or dday_result.production_configuration_version != configuration.version
        or dday_result.policy_id != dday_policy.policy_id
        or dday_result.policy_version != dday_policy.version
    ):
        raise RecommendationValidationError(
            "D-Day result metadata does not match the production contract"
        )
    if (
        split_result.production_configuration_id != configuration.configuration_id
        or split_result.production_configuration_version != configuration.version
        or split_result.dday_policy_id != dday_policy.policy_id
        or split_result.dday_policy_version != dday_policy.version
        or split_result.split_policy_id != split_policy.policy_id
        or split_result.split_policy_version != split_policy.version
    ):
        raise RecommendationValidationError(
            "split result metadata does not match the production contract"
        )
    if (
        dday_result.base_signal is not market_result.signal
        or split_result.base_signal is not market_result.signal
        or split_result.adjusted_signal is not dday_result.adjusted_signal
        or split_result.days_until_departure != dday_result.days_until_departure
        or split_result.dday_band is not dday_result.dday_band
        or split_result.urgency is not dday_result.urgency
    ):
        raise RecommendationValidationError(
            "production stage results are internally inconsistent"
        )
