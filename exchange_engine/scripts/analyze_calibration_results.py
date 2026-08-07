"""Validate and analyze existing threshold-calibration CSV reports.

This command is deliberately offline.  It reads the six compact calibration
reports, delegates descriptive analysis to ``src.calibration_analysis``, and
writes separate review artifacts.  It does not fetch exchange rates, read
``.env``, rank a winner, or change a production threshold.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration_analysis import (  # noqa: E402
    AnalysisHeuristics,
    CalibrationAnalysisError,
    analyze_calibration_results,
    load_calibration_results,
    write_analysis_reports,
)


AnalysisLoader = Callable[[str | Path], object]
AnalysisRunner = Callable[..., object]
AnalysisWriter = Callable[[object, str | Path], tuple[Path, ...]]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the offline CLI parser without reading any report files."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze existing calibration and validation reports for human "
            "review. No winner or production threshold is selected."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        metavar="PATH",
        help="Directory containing the six original calibration CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="PATH",
        help="Analysis directory (default: INPUT_DIR/analysis).",
    )
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse paths and keep analysis artifacts separate from source CSVs."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    if arguments.output_dir is None:
        arguments.output_dir = arguments.input_dir / "analysis"

    if _resolved_path(arguments.input_dir) == _resolved_path(
        arguments.output_dir
    ):
        parser.error("--output-dir must differ from --input-dir")
    return arguments


def format_analysis_report(
    report: object,
    written_paths: Sequence[Path],
) -> str:
    """Format factual counts and an explicitly unranked review shortlist."""
    configuration_review = getattr(report, "configuration_review")
    candidate_shortlist = getattr(report, "candidate_shortlist")
    shortlist_status = _display_value(getattr(report, "shortlist_status"))

    lines = [
        f"configuration review count: {len(configuration_review)}",
    ]
    group_column = _review_group_column(configuration_review)
    if group_column is not None:
        group_counts = configuration_review[group_column].value_counts(
            dropna=False,
            sort=False,
        )
        lines.append("review groups:")
        lines.extend(
            f"- {_display_value(group)}: {int(count)}"
            for group, count in group_counts.items()
        )

    lines.extend(
        [
            f"review shortlist count: {len(candidate_shortlist)}",
            f"shortlist status: {shortlist_status}",
            "review shortlist (unranked):",
        ]
    )
    identifiers = _shortlist_identifiers(candidate_shortlist)
    lines.extend(f"- {identifier}" for identifier in identifiers)
    if not identifiers:
        lines.append("- none")

    lines.append("analysis reports:")
    lines.extend(f"- {path}" for path in written_paths)
    lines.append("production configuration selection: none (human review required)")
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    loader: AnalysisLoader = load_calibration_results,
    analyzer: AnalysisRunner = analyze_calibration_results,
    writer: AnalysisWriter = write_analysis_reports,
    heuristics: AnalysisHeuristics | None = None,
) -> int:
    """Run the offline workflow and translate failures into safe messages."""
    arguments = parse_arguments(argv)
    selected_heuristics = heuristics or AnalysisHeuristics()

    try:
        input_tables = loader(arguments.input_dir)
        report = analyzer(
            input_tables,
            heuristics=selected_heuristics,
        )
        written_paths = writer(report, arguments.output_dir)
        output = format_analysis_report(report, written_paths)
    except FileNotFoundError:
        return _report_error(
            "A required calibration result CSV is missing from the input "
            "directory."
        )
    except FileExistsError:
        return _report_error(
            "An analysis report already exists in the output directory; "
            "choose a new directory."
        )
    except CalibrationAnalysisError:
        return _report_error(
            "Calibration result files failed schema, dimension, or metric "
            "consistency validation."
        )
    except (TypeError, ValueError):
        return _report_error(
            "Calibration analysis inputs or review heuristics violated their "
            "contract."
        )
    except OSError:
        return _report_error("Analysis reports could not be read or written.")

    print(output)
    return 0


def _resolved_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _review_group_column(configuration_review: object) -> str | None:
    columns = getattr(configuration_review, "columns", ())
    for candidate in ("review_group", "analysis_group"):
        if candidate in columns:
            return candidate
    return None


def _shortlist_identifiers(candidate_shortlist: object) -> tuple[str, ...]:
    columns = getattr(candidate_shortlist, "columns", ())
    if "configuration_id" not in columns:
        raise ValueError(
            "candidate_shortlist must contain configuration_id"
        )
    return tuple(
        str(value)
        for value in candidate_shortlist["configuration_id"].tolist()
    )


def _display_value(value: object) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value)


def _report_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
