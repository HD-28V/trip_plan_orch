"""Human-review analysis for existing threshold-calibration CSV reports.

This module validates and analyzes an already-completed calibration run.  It
does not create threshold candidates, run providers/backtests, rank candidates,
or select a production configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import REPORT_FILES
from src.calibration_analysis_configuration import build_configuration_review
from src.calibration_analysis_output import (
    OUTPUT_FILES,
    _shortlist_status,
    build_analysis_metadata,
    build_candidate_shortlist,
    render_analysis_notes,
    write_analysis_reports,
)
from src.calibration_analysis_reviews import (
    build_currency_stability_review,
    build_horizon_stability_review,
    build_signal_separation_review,
    build_stability_review,
)
from src.calibration_analysis_validation import (
    CalibrationReportValidationError,
    validate_calibration_results_impl,
)
EXPECTED_CONFIGURATION_COUNT = 12

ANALYSIS_REPORT_FILES = (*OUTPUT_FILES, ("analysis_notes.md", "analysis_notes"))


class CalibrationAnalysisError(ValueError):
    """Raised when calibration inputs or analysis outputs break the contract."""


@dataclass(frozen=True)
class AnalysisHeuristics:
    """Explicit analysis-only heuristics; none are production thresholds."""

    low_sample_count: int = 25
    very_low_sample_count: int = 10
    low_coverage_ratio: float = 0.80
    severe_coverage_ratio: float = 0.50
    material_favorable_gap: float = 0.10
    material_mean_gap_pct: float = 0.50
    material_currency_favorable_range: float = 0.15
    material_horizon_favorable_drop: float = 0.15
    signal_comparison_min_evaluable_count: int = 10
    favorable_tie_band: float = 0.02
    advantage_tie_band_pct: float = 0.10
    minimum_comparable_separation_cells: int = 4
    comparison_failure_fraction: float = 0.50
    positive_metric_vote_count: int = 2
    positive_favorable_ratio_floor: float = 0.50
    positive_advantage_floor_pct: float = 0.0
    minimum_positive_currency_cell_ratio: float = 0.50
    minimum_positive_supported_cell_ratio: float = 0.75
    weak_positive_supported_cell_ratio: float = 0.50
    weak_core_risk_axis_count: int = 2
    shortlist_min_count: int = 2
    shortlist_max_count: int = 5

    def __post_init__(self) -> None:
        integer_fields = (
            "low_sample_count",
            "very_low_sample_count",
            "signal_comparison_min_evaluable_count",
            "minimum_comparable_separation_cells",
            "positive_metric_vote_count",
            "weak_core_risk_axis_count",
            "shortlist_min_count",
            "shortlist_max_count",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"{field_name} must be positive")

        ratio_fields = (
            "low_coverage_ratio",
            "severe_coverage_ratio",
            "material_favorable_gap",
            "material_currency_favorable_range",
            "material_horizon_favorable_drop",
            "favorable_tie_band",
            "comparison_failure_fraction",
            "positive_favorable_ratio_floor",
            "minimum_positive_currency_cell_ratio",
            "minimum_positive_supported_cell_ratio",
            "weak_positive_supported_cell_ratio",
        )
        for field_name in ratio_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be numeric")
            if not np.isfinite(value) or not 0 <= float(value) <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")

        for field_name in (
            "material_mean_gap_pct",
            "advantage_tie_band_pct",
            "positive_advantage_floor_pct",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be numeric")
            if not np.isfinite(value) or float(value) < 0:
                raise ValueError(f"{field_name} must be non-negative")

        if self.very_low_sample_count > self.low_sample_count:
            raise ValueError(
                "very_low_sample_count must not exceed low_sample_count"
            )
        if self.severe_coverage_ratio > self.low_coverage_ratio:
            raise ValueError(
                "severe_coverage_ratio must not exceed low_coverage_ratio"
            )
        if self.shortlist_min_count > self.shortlist_max_count:
            raise ValueError(
                "shortlist_min_count must not exceed shortlist_max_count"
            )
        if self.positive_metric_vote_count > 3:
            raise ValueError("positive_metric_vote_count must not exceed 3")
        if self.weak_core_risk_axis_count > 4:
            raise ValueError("weak_core_risk_axis_count must not exceed 4")


@dataclass(frozen=True, eq=False)
class CalibrationInputTables:
    """Deep-copyable representations of the six immutable source reports."""

    report_metadata: pd.DataFrame = field(repr=False)
    candidate_configurations: pd.DataFrame = field(repr=False)
    period_data_summary: pd.DataFrame = field(repr=False)
    configuration_summary: pd.DataFrame = field(repr=False)
    signal_horizon_summary: pd.DataFrame = field(repr=False)
    cross_currency_summary: pd.DataFrame = field(repr=False)
    source_dir: Path | None = None


@dataclass(frozen=True, eq=False)
class CalibrationAnalysisReport:
    """Human-review tables with no score, rank, winner, or production choice."""

    analysis_metadata: pd.DataFrame = field(repr=False)
    configuration_review: pd.DataFrame = field(repr=False)
    stability_review: pd.DataFrame = field(repr=False)
    currency_stability_review: pd.DataFrame = field(repr=False)
    horizon_stability_review: pd.DataFrame = field(repr=False)
    signal_separation_review: pd.DataFrame = field(repr=False)
    candidate_shortlist: pd.DataFrame = field(repr=False)
    analysis_notes: str
    shortlist_status: str
    heuristics: AnalysisHeuristics
    source_dir: Path | None = None


def load_calibration_results(input_dir: str | Path) -> CalibrationInputTables:
    """Read the six source CSV files without modifying them."""

    source = Path(input_dir)
    if not source.is_dir():
        raise CalibrationAnalysisError(
            "calibration input directory does not exist or is not a directory"
        )

    loaded: dict[str, pd.DataFrame] = {}
    for filename, attribute_name in REPORT_FILES:
        path = source / filename
        if not path.is_file():
            raise CalibrationAnalysisError(
                f"required calibration report is missing: {filename}"
            )
        try:
            loaded[attribute_name] = pd.read_csv(path)
        except Exception:
            raise CalibrationAnalysisError(
                f"calibration report could not be read: {filename}"
            ) from None

    tables = CalibrationInputTables(
        source_dir=source.resolve(),
        **loaded,
    )
    return validate_calibration_results(tables)


def validate_calibration_results(
    tables: CalibrationInputTables,
) -> CalibrationInputTables:
    """Validate exact schemas, dimensions, arithmetic, and NaN contracts."""

    if not isinstance(tables, CalibrationInputTables):
        raise TypeError("tables must be a CalibrationInputTables instance")
    try:
        validated = validate_calibration_results_impl(
            tables,
            expected_count=EXPECTED_CONFIGURATION_COUNT,
        )
    except CalibrationReportValidationError as exc:
        raise CalibrationAnalysisError(str(exc)) from None

    return CalibrationInputTables(
        report_metadata=validated["report_metadata"],
        candidate_configurations=validated["candidate_configurations"],
        period_data_summary=validated["period_data_summary"],
        configuration_summary=validated["configuration_summary"],
        signal_horizon_summary=validated["signal_horizon_summary"],
        cross_currency_summary=validated["cross_currency_summary"],
        source_dir=(
            Path(tables.source_dir) if tables.source_dir is not None else None
        ),
    )


def analyze_calibration_results(
    tables: CalibrationInputTables,
    *,
    heuristics: AnalysisHeuristics | None = None,
) -> CalibrationAnalysisReport:
    """Build descriptive review groups and an unranked review shortlist."""

    policy = heuristics or AnalysisHeuristics()
    if not isinstance(policy, AnalysisHeuristics):
        raise TypeError("heuristics must be an AnalysisHeuristics instance")
    validated = validate_calibration_results(tables)

    stability = build_stability_review(
        validated.signal_horizon_summary,
        heuristics=policy,
        configuration_ids=validated.candidate_configurations[
            "configuration_id"
        ].tolist(),
    )
    currency_stability = build_currency_stability_review(
        validated.signal_horizon_summary,
        heuristics=policy,
        configuration_ids=validated.candidate_configurations[
            "configuration_id"
        ].tolist(),
    )
    horizon_stability = build_horizon_stability_review(
        validated.signal_horizon_summary,
        heuristics=policy,
        configuration_ids=validated.candidate_configurations[
            "configuration_id"
        ].tolist(),
    )
    separation = build_signal_separation_review(
        validated.signal_horizon_summary,
        heuristics=policy,
        configuration_ids=validated.candidate_configurations[
            "configuration_id"
        ].tolist(),
    )
    configuration_review = build_configuration_review(
        validated,
        stability_review=stability,
        currency_stability_review=currency_stability,
        horizon_stability_review=horizon_stability,
        signal_separation_review=separation,
        heuristics=policy,
    )
    shortlist = build_candidate_shortlist(configuration_review)
    shortlist_status = _shortlist_status(len(shortlist), policy)
    metadata = build_analysis_metadata(
        heuristics=policy,
        shortlist_status=shortlist_status,
        shortlist_count=len(shortlist),
    )
    notes = render_analysis_notes(
        configuration_review,
        shortlist,
        heuristics=policy,
        shortlist_status=shortlist_status,
    )

    return CalibrationAnalysisReport(
        analysis_metadata=metadata,
        configuration_review=configuration_review,
        stability_review=stability,
        currency_stability_review=currency_stability,
        horizon_stability_review=horizon_stability,
        signal_separation_review=separation,
        candidate_shortlist=shortlist,
        analysis_notes=notes,
        shortlist_status=shortlist_status,
        heuristics=policy,
        source_dir=validated.source_dir,
    )
