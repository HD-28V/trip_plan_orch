"""Explicit exploratory candidates for threshold calibration.

The profiles in this module are research candidates, not production defaults.
Configurations are listed explicitly so this module never creates an
unbounded Cartesian product or selects a winning configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from src.backtest import BacktestConfiguration
from src.signal_engine import SignalDecisionPolicy, SignalThresholds


INITIAL_MAX_CANDIDATE_COUNT = 24

CANDIDATE_MANIFEST_COLUMNS = (
    "configuration_id",
    "threshold_id",
    "policy_id",
    "sma60_good",
    "sma60_strong",
    "sma120_good",
    "sma120_strong",
    "percentile_good",
    "percentile_strong",
    "bollinger_near_lower_pct",
    "minimum_available_conditions",
    "watch_min_satisfied_conditions",
    "good_min_satisfied_conditions",
    "strong_min_satisfied_conditions",
    "strong_min_strong_conditions",
)

_CANDIDATE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class CandidateLimitError(ValueError):
    """Raised before data collection when an explicit plan exceeds its cap."""


@dataclass(frozen=True)
class ThresholdCandidate:
    """A named, human-reviewable exploratory threshold profile."""

    threshold_id: str
    thresholds: SignalThresholds

    def __post_init__(self) -> None:
        _validate_candidate_id(self.threshold_id, "threshold_id")
        if not isinstance(self.thresholds, SignalThresholds):
            raise TypeError("thresholds must be a SignalThresholds instance")


@dataclass(frozen=True)
class PolicyCandidate:
    """A named, human-reviewable exploratory decision-policy profile."""

    policy_id: str
    policy: SignalDecisionPolicy

    def __post_init__(self) -> None:
        _validate_candidate_id(self.policy_id, "policy_id")
        if not isinstance(self.policy, SignalDecisionPolicy):
            raise TypeError("policy must be a SignalDecisionPolicy instance")


@dataclass(frozen=True)
class CandidateConfigurationSpec:
    """One explicit threshold/policy pairing to evaluate."""

    configuration_id: str
    threshold_id: str
    policy_id: str

    def __post_init__(self) -> None:
        _validate_candidate_id(self.configuration_id, "configuration_id")
        _validate_candidate_id(self.threshold_id, "threshold_id")
        _validate_candidate_id(self.policy_id, "policy_id")


@dataclass(frozen=True)
class CandidatePlan:
    """Validated configurations and the explicit profiles that produced them."""

    threshold_candidates: tuple[ThresholdCandidate, ...]
    policy_candidates: tuple[PolicyCandidate, ...]
    specifications: tuple[CandidateConfigurationSpec, ...]
    configurations: tuple[BacktestConfiguration, ...]
    max_candidate_count: int

    def __post_init__(self) -> None:
        thresholds = _validate_candidate_collection(
            self.threshold_candidates,
            ThresholdCandidate,
            "threshold_candidates",
        )
        policies = _validate_candidate_collection(
            self.policy_candidates,
            PolicyCandidate,
            "policy_candidates",
        )
        specs = _validate_candidate_collection(
            self.specifications,
            CandidateConfigurationSpec,
            "specifications",
        )
        configurations = _validate_candidate_collection(
            self.configurations,
            BacktestConfiguration,
            "configurations",
        )
        maximum = _validate_max_candidate_count(self.max_candidate_count)
        _require_unique_ids(
            (candidate.threshold_id for candidate in thresholds),
            "threshold_id",
        )
        _require_unique_ids(
            (candidate.policy_id for candidate in policies),
            "policy_id",
        )
        _require_unique_ids(
            (specification.configuration_id for specification in specs),
            "configuration_id",
        )
        if len(specs) != len(configurations):
            raise ValueError(
                "specifications and configurations must have the same count"
            )
        if len(configurations) > maximum:
            raise CandidateLimitError(
                f"candidate configuration count {len(configurations)} exceeds "
                f"max_candidate_count {maximum}"
            )

        thresholds_by_id = {
            candidate.threshold_id: candidate.thresholds
            for candidate in thresholds
        }
        policies_by_id = {
            candidate.policy_id: candidate.policy for candidate in policies
        }
        seen_pairs: set[tuple[str, str]] = set()
        for specification, configuration in zip(
            specs,
            configurations,
            strict=True,
        ):
            if specification.threshold_id not in thresholds_by_id:
                raise ValueError(
                    "configuration references unknown threshold_id: "
                    f"{specification.threshold_id}"
                )
            if specification.policy_id not in policies_by_id:
                raise ValueError(
                    "configuration references unknown policy_id: "
                    f"{specification.policy_id}"
                )
            pair = (specification.threshold_id, specification.policy_id)
            if pair in seen_pairs:
                raise ValueError(
                    "threshold_id/policy_id pairs must be unique: "
                    f"{specification.threshold_id}/{specification.policy_id}"
                )
            seen_pairs.add(pair)
            if (
                configuration.configuration_id
                != specification.configuration_id
                or configuration.thresholds
                != thresholds_by_id[specification.threshold_id]
                or configuration.policy != policies_by_id[specification.policy_id]
            ):
                raise ValueError(
                    "configuration must match its explicit specification"
                )

        object.__setattr__(self, "threshold_candidates", thresholds)
        object.__setattr__(self, "policy_candidates", policies)
        object.__setattr__(self, "specifications", specs)
        object.__setattr__(self, "configurations", configurations)
        object.__setattr__(self, "max_candidate_count", maximum)

    @property
    def candidate_count(self) -> int:
        return len(self.configurations)

    def to_manifest(self) -> pd.DataFrame:
        """Return every supplied field without scores, ranks, or selection."""
        thresholds_by_id = {
            candidate.threshold_id: candidate.thresholds
            for candidate in self.threshold_candidates
        }
        policies_by_id = {
            candidate.policy_id: candidate.policy
            for candidate in self.policy_candidates
        }
        rows: list[dict[str, object]] = []
        for specification in self.specifications:
            thresholds = thresholds_by_id[specification.threshold_id]
            policy = policies_by_id[specification.policy_id]
            rows.append(
                {
                    "configuration_id": specification.configuration_id,
                    "threshold_id": specification.threshold_id,
                    "policy_id": specification.policy_id,
                    "sma60_good": thresholds.sma60_good,
                    "sma60_strong": thresholds.sma60_strong,
                    "sma120_good": thresholds.sma120_good,
                    "sma120_strong": thresholds.sma120_strong,
                    "percentile_good": thresholds.percentile_good,
                    "percentile_strong": thresholds.percentile_strong,
                    "bollinger_near_lower_pct": (
                        thresholds.bollinger_near_lower_pct
                    ),
                    "minimum_available_conditions": (
                        policy.minimum_available_conditions
                    ),
                    "watch_min_satisfied_conditions": (
                        policy.watch_min_satisfied_conditions
                    ),
                    "good_min_satisfied_conditions": (
                        policy.good_min_satisfied_conditions
                    ),
                    "strong_min_satisfied_conditions": (
                        policy.strong_min_satisfied_conditions
                    ),
                    "strong_min_strong_conditions": (
                        policy.strong_min_strong_conditions
                    ),
                }
            )
        return pd.DataFrame.from_records(rows, columns=CANDIDATE_MANIFEST_COLUMNS)


def build_candidate_plan(
    threshold_candidates: tuple[ThresholdCandidate, ...],
    policy_candidates: tuple[PolicyCandidate, ...],
    specifications: tuple[CandidateConfigurationSpec, ...],
    *,
    max_candidate_count: int,
) -> CandidatePlan:
    """Resolve only explicitly listed pairings into Backtest configurations."""
    maximum = _validate_max_candidate_count(max_candidate_count)
    thresholds = _validate_candidate_collection(
        threshold_candidates,
        ThresholdCandidate,
        "threshold_candidates",
    )
    policies = _validate_candidate_collection(
        policy_candidates,
        PolicyCandidate,
        "policy_candidates",
    )
    specs = _validate_candidate_collection(
        specifications,
        CandidateConfigurationSpec,
        "specifications",
    )

    _require_unique_ids(
        (candidate.threshold_id for candidate in thresholds),
        "threshold_id",
    )
    _require_unique_ids(
        (candidate.policy_id for candidate in policies),
        "policy_id",
    )
    _require_unique_ids(
        (specification.configuration_id for specification in specs),
        "configuration_id",
    )

    if len(specs) > maximum:
        raise CandidateLimitError(
            f"candidate configuration count {len(specs)} exceeds "
            f"max_candidate_count {maximum}"
        )

    thresholds_by_id = {
        candidate.threshold_id: candidate.thresholds for candidate in thresholds
    }
    policies_by_id = {
        candidate.policy_id: candidate.policy for candidate in policies
    }
    seen_pairs: set[tuple[str, str]] = set()
    configurations: list[BacktestConfiguration] = []

    for specification in specs:
        if specification.threshold_id not in thresholds_by_id:
            raise ValueError(
                "configuration references unknown threshold_id: "
                f"{specification.threshold_id}"
            )
        if specification.policy_id not in policies_by_id:
            raise ValueError(
                "configuration references unknown policy_id: "
                f"{specification.policy_id}"
            )

        pair = (specification.threshold_id, specification.policy_id)
        if pair in seen_pairs:
            raise ValueError(
                "threshold_id/policy_id pairs must be unique: "
                f"{specification.threshold_id}/{specification.policy_id}"
            )
        seen_pairs.add(pair)
        configurations.append(
            BacktestConfiguration(
                configuration_id=specification.configuration_id,
                thresholds=thresholds_by_id[specification.threshold_id],
                policy=policies_by_id[specification.policy_id],
            )
        )

    return CandidatePlan(
        threshold_candidates=thresholds,
        policy_candidates=policies,
        specifications=specs,
        configurations=tuple(configurations),
        max_candidate_count=maximum,
    )


def build_initial_candidate_plan(
    *,
    max_candidate_count: int = INITIAL_MAX_CANDIDATE_COUNT,
) -> CandidatePlan:
    """Build the bounded initial research plan; no profile is production-ready."""
    return build_candidate_plan(
        initial_threshold_candidates(),
        initial_policy_candidates(),
        initial_configuration_specs(),
        max_candidate_count=max_candidate_count,
    )


def initial_threshold_candidates() -> tuple[ThresholdCandidate, ...]:
    """Return one-factor-at-a-time exploratory threshold profiles."""
    baseline = {
        "sma60_good": -2.0,
        "sma60_strong": -4.0,
        "sma120_good": -2.0,
        "sma120_strong": -4.0,
        "percentile_good": 25.0,
        "percentile_strong": 10.0,
        "bollinger_near_lower_pct": 1.0,
    }

    def candidate(
        threshold_id: str,
        **updates: float,
    ) -> ThresholdCandidate:
        values = {**baseline, **updates}
        return ThresholdCandidate(
            threshold_id,
            SignalThresholds(**values),
        )

    return (
        candidate("baseline"),
        candidate("sma60_sensitive", sma60_good=-1.0, sma60_strong=-2.0),
        candidate("sma60_strict", sma60_good=-3.0, sma60_strong=-5.0),
        candidate("sma120_sensitive", sma120_good=-1.0, sma120_strong=-2.0),
        candidate("sma120_strict", sma120_good=-3.0, sma120_strong=-5.0),
        candidate(
            "percentile_sensitive",
            percentile_good=30.0,
            percentile_strong=15.0,
        ),
        candidate(
            "percentile_strict",
            percentile_good=20.0,
            percentile_strong=5.0,
        ),
        candidate("bollinger_strict", bollinger_near_lower_pct=0.0),
        candidate("bollinger_tight", bollinger_near_lower_pct=0.5),
        candidate("bollinger_sensitive", bollinger_near_lower_pct=2.0),
    )


def initial_policy_candidates() -> tuple[PolicyCandidate, ...]:
    """Return three meaningfully different, valid exploratory policies."""
    return (
        PolicyCandidate(
            "conservative",
            SignalDecisionPolicy(
                minimum_available_conditions=4,
                watch_min_satisfied_conditions=2,
                good_min_satisfied_conditions=4,
                strong_min_satisfied_conditions=4,
                strong_min_strong_conditions=4,
            ),
        ),
        PolicyCandidate(
            "balanced",
            SignalDecisionPolicy(
                minimum_available_conditions=3,
                watch_min_satisfied_conditions=1,
                good_min_satisfied_conditions=3,
                strong_min_satisfied_conditions=4,
                strong_min_strong_conditions=3,
            ),
        ),
        PolicyCandidate(
            "sensitive",
            SignalDecisionPolicy(
                minimum_available_conditions=2,
                watch_min_satisfied_conditions=1,
                good_min_satisfied_conditions=2,
                strong_min_satisfied_conditions=3,
                strong_min_strong_conditions=2,
            ),
        ),
    )


def initial_configuration_specs() -> tuple[CandidateConfigurationSpec, ...]:
    """List the bounded comparisons instead of generating every combination."""
    threshold_variants = (
        "sma60_sensitive",
        "sma60_strict",
        "sma120_sensitive",
        "sma120_strict",
        "percentile_sensitive",
        "percentile_strict",
        "bollinger_strict",
        "bollinger_tight",
        "bollinger_sensitive",
    )
    baseline_policy_specs = (
        CandidateConfigurationSpec(
            f"baseline__{policy_id}",
            "baseline",
            policy_id,
        )
        for policy_id in ("conservative", "balanced", "sensitive")
    )
    threshold_specs = (
        CandidateConfigurationSpec(
            f"{threshold_id}__balanced",
            threshold_id,
            "balanced",
        )
        for threshold_id in threshold_variants
    )
    return (*baseline_policy_specs, *threshold_specs)


def _validate_candidate_id(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not _CANDIDATE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_CANDIDATE_ID_PATTERN.pattern}"
        )


def _validate_candidate_collection(
    values: object,
    expected_type: type,
    field_name: str,
) -> tuple:
    if values is None or isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable")
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError(f"{field_name} must be an iterable") from error
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if not all(isinstance(value, expected_type) for value in normalized):
        raise TypeError(
            f"each {field_name} item must be a {expected_type.__name__}"
        )
    return normalized


def _require_unique_ids(values: object, field_name: str) -> None:
    identifiers = tuple(values)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{field_name} values must be unique")


def _validate_max_candidate_count(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("max_candidate_count must be a positive integer")
    maximum = int(value)
    if maximum <= 0:
        raise ValueError("max_candidate_count must be a positive integer")
    return maximum
