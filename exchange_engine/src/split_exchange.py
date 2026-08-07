"""Cumulative KRW exchange recommendations from D-Day-adjusted Signals.

This module is a product-policy layer.  It neither changes market Signals nor
D-Day policy, and it does not forecast exchange rates or calculate foreign
currency quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Real

from src.dday_policy import (
    DdayBand,
    DdaySignalResult,
    DdayUrgency,
    get_production_dday_policy,
)
from src.production_config import get_production_signal_configuration
from src.signal_engine import Signal


class SplitExchangeValidationError(ValueError):
    """Raised when split-exchange inputs cannot form a safe recommendation."""


@dataclass(frozen=True)
class SignalTargetRatioRule:
    """One adjusted-market-Signal target cumulative exchange ratio."""

    signal: Signal
    target_ratio: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.signal, Signal):
            raise TypeError("signal must be a Signal")
        object.__setattr__(
            self,
            "target_ratio",
            _validate_ratio(self.target_ratio, "target_ratio"),
        )


@dataclass(frozen=True)
class UrgencyMinimumRatioRule:
    """One D-Day urgency minimum cumulative exchange ratio."""

    urgency: DdayUrgency
    minimum_ratio: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.urgency, DdayUrgency):
            raise TypeError("urgency must be a DdayUrgency")
        object.__setattr__(
            self,
            "minimum_ratio",
            _validate_ratio(self.minimum_ratio, "minimum_ratio"),
        )


@dataclass(frozen=True)
class SplitExchangePolicy:
    """Immutable signal/urgency ratio policy for cumulative KRW exchange."""

    policy_id: str
    version: str
    signal_target_rules: tuple[SignalTargetRatioRule, ...]
    urgency_minimum_rules: tuple[UrgencyMinimumRatioRule, ...]

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.policy_id, "policy_id")
        _validate_nonempty_string(self.version, "version")
        _validate_complete_rule_order(
            self.signal_target_rules,
            SignalTargetRatioRule,
            tuple(Signal),
            "signal_target_rules",
            "signal",
        )
        _validate_complete_rule_order(
            self.urgency_minimum_rules,
            UrgencyMinimumRatioRule,
            tuple(DdayUrgency),
            "urgency_minimum_rules",
            "urgency",
        )

    def signal_target_ratio_for(self, signal: Signal) -> Decimal:
        """Return the configured cumulative target for one adjusted Signal."""
        if not isinstance(signal, Signal):
            raise TypeError("signal must be a Signal")
        return self.signal_target_rules[tuple(Signal).index(signal)].target_ratio

    def urgency_minimum_ratio_for(self, urgency: DdayUrgency) -> Decimal:
        """Return the configured minimum cumulative ratio for one urgency."""
        if not isinstance(urgency, DdayUrgency):
            raise TypeError("urgency must be a DdayUrgency")
        return self.urgency_minimum_rules[
            tuple(DdayUrgency).index(urgency)
        ].minimum_ratio


@dataclass(frozen=True)
class SplitExchangeResult:
    """Auditable cumulative target and additional KRW recommendation."""

    base_signal: Signal
    adjusted_signal: Signal
    urgency: DdayUrgency
    days_until_departure: int
    dday_band: DdayBand
    total_target_krw: Decimal
    already_exchanged_krw: Decimal
    signal_target_ratio: Decimal
    urgency_minimum_ratio: Decimal
    target_cumulative_ratio: Decimal
    target_cumulative_amount_krw: Decimal
    recommended_additional_krw: Decimal
    remaining_after_recommendation_krw: Decimal
    split_policy_id: str
    split_policy_version: str
    production_configuration_id: str | None
    production_configuration_version: str | None
    dday_policy_id: str
    dday_policy_version: str
    reason: str


_ZERO = Decimal("0")
_ONE = Decimal("1")


def _validate_complete_rule_order(
    rules: object,
    rule_type: type,
    expected_values: tuple,
    field_name: str,
    value_field_name: str,
) -> None:
    if not isinstance(rules, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if len(rules) != len(expected_values):
        raise ValueError(f"{field_name} must define every supported value once")
    for rule, expected_value in zip(rules, expected_values, strict=True):
        if not isinstance(rule, rule_type):
            raise TypeError(
                f"each {field_name} item must be a {rule_type.__name__}"
            )
        if getattr(rule, value_field_name) is not expected_value:
            raise ValueError(
                f"{field_name} must use the stable supported-value order"
            )


def _validate_ratio(value: object, field_name: str) -> Decimal:
    ratio = _normalize_decimal(value, field_name)
    if not _ZERO <= ratio <= _ONE:
        raise ValueError(f"{field_name} must be between zero and one")
    return ratio


def _normalize_amount(value: object, field_name: str) -> Decimal:
    return _normalize_decimal(value, field_name)


def _normalize_decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, Real)):
        raise TypeError(f"{field_name} must be a finite numeric value")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite numeric value") from error
    if not decimal_value.is_finite():
        raise ValueError(f"{field_name} must be a finite numeric value")
    return decimal_value


def _validate_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

_PRODUCTION_SPLIT_EXCHANGE_POLICY = SplitExchangePolicy(
    policy_id="split_exchange_policy_v1",
    version="v1",
    signal_target_rules=(
        SignalTargetRatioRule(Signal.WAIT, Decimal("0.00")),
        SignalTargetRatioRule(Signal.WATCH, Decimal("0.25")),
        SignalTargetRatioRule(Signal.GOOD, Decimal("0.50")),
        SignalTargetRatioRule(Signal.STRONG, Decimal("0.75")),
    ),
    urgency_minimum_rules=(
        UrgencyMinimumRatioRule(DdayUrgency.LOW, Decimal("0.00")),
        UrgencyMinimumRatioRule(DdayUrgency.NORMAL, Decimal("0.00")),
        UrgencyMinimumRatioRule(DdayUrgency.HIGH, Decimal("0.25")),
        UrgencyMinimumRatioRule(DdayUrgency.CRITICAL, Decimal("0.75")),
        UrgencyMinimumRatioRule(DdayUrgency.DEADLINE, Decimal("1.00")),
    ),
)


def get_production_split_exchange_policy() -> SplitExchangePolicy:
    """Return the immutable production-v1 split-exchange product policy."""
    return _PRODUCTION_SPLIT_EXCHANGE_POLICY


def evaluate_split_exchange(
    dday_result: DdaySignalResult,
    total_target_krw: Decimal | Real,
    already_exchanged_krw: Decimal | Real,
    *,
    policy: SplitExchangePolicy,
) -> SplitExchangeResult:
    """Calculate a cumulative KRW target and safe additional recommendation."""
    _validate_dday_result(dday_result)
    if not isinstance(policy, SplitExchangePolicy):
        raise TypeError("policy must be a SplitExchangePolicy")
    total_amount = _normalize_amount(total_target_krw, "total_target_krw")
    exchanged_amount = _normalize_amount(
        already_exchanged_krw,
        "already_exchanged_krw",
    )
    if total_amount <= _ZERO:
        raise SplitExchangeValidationError("total_target_krw must be greater than zero")
    if exchanged_amount < _ZERO:
        raise SplitExchangeValidationError(
            "already_exchanged_krw must be zero or greater"
        )
    if exchanged_amount > total_amount:
        raise SplitExchangeValidationError(
            "already_exchanged_krw must not exceed total_target_krw"
        )

    signal_target_ratio = policy.signal_target_ratio_for(
        dday_result.adjusted_signal
    )
    urgency_minimum_ratio = policy.urgency_minimum_ratio_for(
        dday_result.urgency
    )
    target_ratio = max(signal_target_ratio, urgency_minimum_ratio)
    target_amount = total_amount * target_ratio
    remaining_before_recommendation = total_amount - exchanged_amount
    recommended_additional = min(
        max(_ZERO, target_amount - exchanged_amount),
        remaining_before_recommendation,
    )
    remaining_after_recommendation = max(
        _ZERO,
        remaining_before_recommendation - recommended_additional,
    )

    return SplitExchangeResult(
        base_signal=dday_result.base_signal,
        adjusted_signal=dday_result.adjusted_signal,
        urgency=dday_result.urgency,
        days_until_departure=dday_result.days_until_departure,
        dday_band=dday_result.dday_band,
        total_target_krw=total_amount,
        already_exchanged_krw=exchanged_amount,
        signal_target_ratio=signal_target_ratio,
        urgency_minimum_ratio=urgency_minimum_ratio,
        target_cumulative_ratio=target_ratio,
        target_cumulative_amount_krw=target_amount,
        recommended_additional_krw=recommended_additional,
        remaining_after_recommendation_krw=remaining_after_recommendation,
        split_policy_id=policy.policy_id,
        split_policy_version=policy.version,
        production_configuration_id=dday_result.production_configuration_id,
        production_configuration_version=(
            dday_result.production_configuration_version
        ),
        dday_policy_id=dday_result.policy_id,
        dday_policy_version=dday_result.policy_version,
        reason=_build_reason(
            dday_result=dday_result,
            signal_target_ratio=signal_target_ratio,
            urgency_minimum_ratio=urgency_minimum_ratio,
            target_ratio=target_ratio,
            target_amount=target_amount,
            exchanged_amount=exchanged_amount,
            recommended_additional=recommended_additional,
        ),
    )


def evaluate_production_split_exchange(
    dday_result: DdaySignalResult,
    total_target_krw: Decimal | Real,
    already_exchanged_krw: Decimal | Real,
) -> SplitExchangeResult:
    """Evaluate split-exchange v1 for a production-v1 D-Day result."""
    _validate_dday_result(dday_result)
    market_configuration = get_production_signal_configuration()
    dday_policy = get_production_dday_policy()
    if (
        dday_result.production_configuration_id
        != market_configuration.configuration_id
        or dday_result.production_configuration_version
        != market_configuration.version
    ):
        raise ValueError(
            "dday_result must reference the production market configuration"
        )
    if (
        dday_result.policy_id != dday_policy.policy_id
        or dday_result.policy_version != dday_policy.version
    ):
        raise ValueError("dday_result must reference the production D-Day policy")
    return evaluate_split_exchange(
        dday_result,
        total_target_krw,
        already_exchanged_krw,
        policy=get_production_split_exchange_policy(),
    )


def _validate_dday_result(dday_result: object) -> None:
    if not isinstance(dday_result, DdaySignalResult):
        raise TypeError("dday_result must be a DdaySignalResult")
    if not isinstance(dday_result.base_signal, Signal):
        raise ValueError("dday_result must contain a valid base_signal")
    if not isinstance(dday_result.adjusted_signal, Signal):
        raise ValueError("dday_result must contain a valid adjusted_signal")
    if not isinstance(dday_result.urgency, DdayUrgency):
        raise ValueError("dday_result must contain a valid urgency")
    if not isinstance(dday_result.dday_band, DdayBand):
        raise ValueError("dday_result must contain a valid dday_band")
    if isinstance(dday_result.days_until_departure, bool) or not isinstance(
        dday_result.days_until_departure,
        int,
    ):
        raise ValueError("dday_result must contain an integer days_until_departure")
    if dday_result.days_until_departure < 0:
        raise ValueError("dday_result must contain a nonnegative days_until_departure")


def _build_reason(
    *,
    dday_result: DdaySignalResult,
    signal_target_ratio: Decimal,
    urgency_minimum_ratio: Decimal,
    target_ratio: Decimal,
    target_amount: Decimal,
    exchanged_amount: Decimal,
    recommended_additional: Decimal,
) -> str:
    return (
        f"adjusted_signal={dday_result.adjusted_signal.value} sets "
        f"signal_target_ratio={signal_target_ratio}; "
        f"urgency={dday_result.urgency.value} sets "
        f"urgency_minimum_ratio={urgency_minimum_ratio}; "
        f"target_cumulative_ratio={target_ratio}; "
        f"target_cumulative_amount_krw={target_amount}; "
        f"already_exchanged_krw={exchanged_amount}; "
        f"recommended_additional_krw={recommended_additional}"
    )
