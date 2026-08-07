"""Immutable production-v1 configuration for the rule-based Signal Engine.

The selected threshold and decision-policy objects are resolved from the
existing calibration candidate plan.  Numeric thresholds therefore remain in
``calibration_candidates`` as their single source of truth, while this module
records the explicit human decision to use one candidate in production.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.calibration_candidates import build_initial_candidate_plan
from src.signal_engine import (
    SignalDecisionPolicy,
    SignalResult,
    SignalThresholds,
    evaluate_signal,
)


PRODUCTION_CONFIGURATION_ID = "sma60_sensitive__balanced"
PRODUCTION_THRESHOLD_PROFILE_ID = "sma60_sensitive"
PRODUCTION_DECISION_POLICY_ID = "balanced"
PRODUCTION_VERSION = "v1"

# These dates describe the observations actually evaluated by the completed
# calibration report, not its longer source/warm-up range.
PRODUCTION_CALIBRATION_START_DATE = date(2018, 1, 1)
PRODUCTION_CALIBRATION_END_DATE = date(2023, 12, 29)
PRODUCTION_VALIDATION_START_DATE = date(2024, 1, 1)


@dataclass(frozen=True)
class ProductionSignalConfiguration:
    """Auditable metadata and immutable objects used by production v1."""

    configuration_id: str
    threshold_profile_id: str
    decision_policy_id: str
    version: str
    calibration_data_start_date: date
    calibration_data_end_date: date
    validation_start_date: date
    thresholds: SignalThresholds
    policy: SignalDecisionPolicy

    @property
    def calibration_data_range(self) -> tuple[date, date]:
        """Return the inclusive observed calibration range."""
        return (
            self.calibration_data_start_date,
            self.calibration_data_end_date,
        )


def _resolve_production_signal_configuration() -> ProductionSignalConfiguration:
    """Resolve and cross-check the explicitly adopted calibration candidate."""
    candidate_plan = build_initial_candidate_plan()
    matches = [
        (specification, configuration)
        for specification, configuration in zip(
            candidate_plan.specifications,
            candidate_plan.configurations,
            strict=True,
        )
        if specification.configuration_id == PRODUCTION_CONFIGURATION_ID
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "production signal configuration must resolve to exactly one "
            "calibration candidate"
        )

    specification, configuration = matches[0]
    if (
        specification.threshold_id != PRODUCTION_THRESHOLD_PROFILE_ID
        or specification.policy_id != PRODUCTION_DECISION_POLICY_ID
    ):
        raise RuntimeError(
            "production signal configuration does not match its adopted "
            "threshold profile and decision policy"
        )

    return ProductionSignalConfiguration(
        configuration_id=specification.configuration_id,
        threshold_profile_id=specification.threshold_id,
        decision_policy_id=specification.policy_id,
        version=PRODUCTION_VERSION,
        calibration_data_start_date=PRODUCTION_CALIBRATION_START_DATE,
        calibration_data_end_date=PRODUCTION_CALIBRATION_END_DATE,
        validation_start_date=PRODUCTION_VALIDATION_START_DATE,
        thresholds=configuration.thresholds,
        policy=configuration.policy,
    )


_PRODUCTION_SIGNAL_CONFIGURATION = _resolve_production_signal_configuration()


def get_production_signal_configuration() -> ProductionSignalConfiguration:
    """Return the immutable, explicitly adopted production-v1 configuration."""
    return _PRODUCTION_SIGNAL_CONFIGURATION


def get_production_signal_thresholds() -> SignalThresholds:
    """Return production v1 thresholds from the existing candidate definition."""
    return _PRODUCTION_SIGNAL_CONFIGURATION.thresholds


def get_production_signal_policy() -> SignalDecisionPolicy:
    """Return production v1 decision policy from the existing candidate definition."""
    return _PRODUCTION_SIGNAL_CONFIGURATION.policy


def evaluate_production_signal(
    indicator_row: pd.Series | Mapping[str, object],
) -> SignalResult:
    """Evaluate one latest indicator row with the production-v1 configuration."""
    return evaluate_signal(
        indicator_row,
        thresholds=get_production_signal_thresholds(),
        policy=get_production_signal_policy(),
    )
