"""Deadline-aware product policy layered on an immutable market SignalResult.

This module does not evaluate exchange-rate indicators, alter market
thresholds, or predict exchange rates.  It only combines an existing market
signal with an explicitly supplied number of days until departure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral

from src.production_config import get_production_signal_configuration
from src.signal_engine import Signal, SignalResult


class DdayBand(str, Enum):
    """Stable D-Day ranges for the production-v1 product policy."""

    D0 = "D0"
    D1_TO_D6 = "D1_TO_D6"
    D7_TO_D14 = "D7_TO_D14"
    D15_TO_D30 = "D15_TO_D30"
    D31_TO_D60 = "D31_TO_D60"
    D61_PLUS = "D61_PLUS"


class DdayUrgency(str, Enum):
    """Time pressure only; it does not describe exchange-rate quality."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    DEADLINE = "DEADLINE"


class DdayValidationError(ValueError):
    """Raised when a departure countdown cannot form a valid recommendation."""


_SIGNAL_LEVELS = (Signal.WAIT, Signal.WATCH, Signal.GOOD, Signal.STRONG)
_SIGNAL_RANK = {signal: rank for rank, signal in enumerate(_SIGNAL_LEVELS)}


def _validate_nonnegative_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be a nonnegative integer")
    if int(value) < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


@dataclass(frozen=True)
class DdayBandRule:
    """One validated D-Day range and its deadline-aware adjustment policy."""

    band: DdayBand
    minimum_days: int
    maximum_days: int | None
    promotion_steps: int
    urgency: DdayUrgency
    minimum_adjusted_signal: Signal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.band, DdayBand):
            raise TypeError("band must be a DdayBand")
        _validate_nonnegative_integer(self.minimum_days, "minimum_days")
        if self.maximum_days is not None:
            _validate_nonnegative_integer(self.maximum_days, "maximum_days")
            if self.maximum_days < self.minimum_days:
                raise ValueError("maximum_days must not be less than minimum_days")
        _validate_nonnegative_integer(self.promotion_steps, "promotion_steps")
        if self.promotion_steps >= len(_SIGNAL_LEVELS):
            raise ValueError("promotion_steps exceeds the Signal range")
        if not isinstance(self.urgency, DdayUrgency):
            raise TypeError("urgency must be a DdayUrgency")
        if (
            self.minimum_adjusted_signal is not None
            and not isinstance(self.minimum_adjusted_signal, Signal)
        ):
            raise TypeError("minimum_adjusted_signal must be a Signal or None")

    def contains(self, days_until_departure: int) -> bool:
        """Return whether a validated D-Day belongs to this rule."""
        return (
            days_until_departure >= self.minimum_days
            and (
                self.maximum_days is None
                or days_until_departure <= self.maximum_days
            )
        )


@dataclass(frozen=True)
class DdayPolicy:
    """Immutable, contiguous D-Day product-policy configuration."""

    policy_id: str
    version: str
    band_rules: tuple[DdayBandRule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        if not isinstance(self.band_rules, tuple):
            raise TypeError("band_rules must be a tuple of DdayBandRule")
        if len(self.band_rules) != len(DdayBand):
            raise ValueError("band_rules must define every D-Day band exactly once")

        next_minimum = 0
        for position, rule in enumerate(self.band_rules):
            if not isinstance(rule, DdayBandRule):
                raise TypeError("each band_rules item must be a DdayBandRule")
            expected_band = tuple(DdayBand)[position]
            if rule.band is not expected_band:
                raise ValueError("band_rules must use the stable D-Day band order")
            if rule.minimum_days != next_minimum:
                raise ValueError("band_rules must cover D-Day ranges contiguously")
            if position == len(self.band_rules) - 1:
                if rule.maximum_days is not None:
                    raise ValueError("the final D-Day band must have no maximum")
            elif rule.maximum_days is None:
                raise ValueError("only the final D-Day band may have no maximum")
            else:
                next_minimum = rule.maximum_days + 1

    def rule_for(self, days_until_departure: int) -> DdayBandRule:
        """Return the unique rule for a validated nonnegative D-Day."""
        for rule in self.band_rules:
            if rule.contains(days_until_departure):
                return rule
        raise RuntimeError("D-Day policy has no rule for the supplied value")


@dataclass(frozen=True)
class DdaySignalResult:
    """Auditable market signal plus separate deadline-adjusted recommendation."""

    base_signal: Signal
    adjusted_signal: Signal
    days_until_departure: int
    dday_band: DdayBand
    promotion_steps: int
    applied_promotion_steps: int
    urgency: DdayUrgency
    reason: str
    policy_id: str
    policy_version: str
    production_configuration_id: str | None
    production_configuration_version: str | None


_PRODUCTION_DDAY_POLICY = DdayPolicy(
    policy_id="dday_policy_v1",
    version="v1",
    band_rules=(
        DdayBandRule(DdayBand.D0, 0, 0, 0, DdayUrgency.DEADLINE),
        DdayBandRule(DdayBand.D1_TO_D6, 1, 6, 2, DdayUrgency.CRITICAL),
        DdayBandRule(
            DdayBand.D7_TO_D14,
            7,
            14,
            1,
            DdayUrgency.HIGH,
            minimum_adjusted_signal=Signal.WATCH,
        ),
        DdayBandRule(DdayBand.D15_TO_D30, 15, 30, 1, DdayUrgency.HIGH),
        DdayBandRule(DdayBand.D31_TO_D60, 31, 60, 0, DdayUrgency.NORMAL),
        DdayBandRule(DdayBand.D61_PLUS, 61, None, 0, DdayUrgency.LOW),
    ),
)


def get_production_dday_policy() -> DdayPolicy:
    """Return the immutable production-v1 D-Day product policy."""
    return _PRODUCTION_DDAY_POLICY


def evaluate_dday_signal(
    signal_result: SignalResult,
    days_until_departure: int,
    *,
    policy: DdayPolicy,
    production_configuration_id: str | None = None,
    production_configuration_version: str | None = None,
) -> DdaySignalResult:
    """Adjust an existing market Signal by explicit D-Day policy only.

    The supplied ``SignalResult`` is read without modification.  The optional
    production configuration references let a production wrapper record its
    market-policy provenance without coupling generic policy evaluation to a
    particular threshold configuration.
    """
    _validate_signal_result(signal_result)
    days = _normalize_days_until_departure(days_until_departure)
    if not isinstance(policy, DdayPolicy):
        raise TypeError("policy must be a DdayPolicy")
    _validate_configuration_reference(
        production_configuration_id,
        production_configuration_version,
    )

    rule = policy.rule_for(days)
    base_signal = signal_result.signal
    adjusted_signal = _promote_signal(
        base_signal,
        promotion_steps=rule.promotion_steps,
        minimum_signal=rule.minimum_adjusted_signal,
    )
    applied_steps = _SIGNAL_RANK[adjusted_signal] - _SIGNAL_RANK[base_signal]

    return DdaySignalResult(
        base_signal=base_signal,
        adjusted_signal=adjusted_signal,
        days_until_departure=days,
        dday_band=rule.band,
        promotion_steps=rule.promotion_steps,
        applied_promotion_steps=applied_steps,
        urgency=rule.urgency,
        reason=_build_reason(
            rule=rule,
            base_signal=base_signal,
            adjusted_signal=adjusted_signal,
            applied_steps=applied_steps,
        ),
        policy_id=policy.policy_id,
        policy_version=policy.version,
        production_configuration_id=production_configuration_id,
        production_configuration_version=production_configuration_version,
    )


def evaluate_production_dday_signal(
    signal_result: SignalResult,
    days_until_departure: int,
) -> DdaySignalResult:
    """Evaluate production-v1 D-Day policy for a production market Signal."""
    _validate_signal_result(signal_result)
    market_configuration = get_production_signal_configuration()
    if (
        signal_result.thresholds != market_configuration.thresholds
        or signal_result.policy != market_configuration.policy
    ):
        raise ValueError(
            "signal_result must use the production market configuration"
        )
    return evaluate_dday_signal(
        signal_result,
        days_until_departure,
        policy=get_production_dday_policy(),
        production_configuration_id=market_configuration.configuration_id,
        production_configuration_version=market_configuration.version,
    )


def _promote_signal(
    base_signal: Signal,
    *,
    promotion_steps: int,
    minimum_signal: Signal | None,
) -> Signal:
    promoted_rank = min(
        _SIGNAL_RANK[base_signal] + promotion_steps,
        len(_SIGNAL_LEVELS) - 1,
    )
    if minimum_signal is not None:
        promoted_rank = max(promoted_rank, _SIGNAL_RANK[minimum_signal])
    return _SIGNAL_LEVELS[promoted_rank]


def _build_reason(
    *,
    rule: DdayBandRule,
    base_signal: Signal,
    adjusted_signal: Signal,
    applied_steps: int,
) -> str:
    if rule.urgency is DdayUrgency.DEADLINE:
        return (
            "departure-day deadline: market signal is preserved and deadline "
            "urgency is reported separately"
        )
    if applied_steps:
        return (
            f"{rule.band.value} permits {rule.promotion_steps} promotion "
            f"step(s): {base_signal.value} adjusted to {adjusted_signal.value}"
        )
    if rule.promotion_steps:
        return (
            f"{rule.band.value} permits {rule.promotion_steps} promotion "
            f"step(s), but {base_signal.value} is already clamped at "
            f"{adjusted_signal.value}"
        )
    return (
        f"{rule.band.value} preserves the market signal "
        f"{base_signal.value}"
    )


def _validate_signal_result(signal_result: object) -> None:
    if not isinstance(signal_result, SignalResult):
        raise TypeError("signal_result must be a SignalResult")
    if not isinstance(signal_result.signal, Signal):
        raise ValueError("signal_result must contain a valid Signal")


def _normalize_days_until_departure(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("days_until_departure must be a nonnegative integer")
    days = int(value)
    if days < 0:
        raise DdayValidationError(
            "days_until_departure cannot be negative; departure date has passed"
        )
    return days


def _validate_configuration_reference(
    configuration_id: object,
    configuration_version: object,
) -> None:
    if (configuration_id is None) != (configuration_version is None):
        raise ValueError(
            "production configuration id and version must be supplied together"
        )
    if configuration_id is None:
        return
    if not isinstance(configuration_id, str) or not configuration_id.strip():
        raise ValueError("production configuration id must be a non-empty string")
    if (
        not isinstance(configuration_version, str)
        or not configuration_version.strip()
    ):
        raise ValueError(
            "production configuration version must be a non-empty string"
        )
