"""Fetch long-term FX history and write bounded calibration reports.

This command evaluates explicit research candidates only. It never reads
``.env``, selects a winner, or writes a production threshold.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import STANDARD_EVALUATION_HORIZONS  # noqa: E402
from src.calibration import (  # noqa: E402
    CALIBRATION_CURRENCIES,
    CalibrationDataError,
    CalibrationReport,
    run_calibration,
    write_calibration_reports,
)
from src.calibration_candidates import (  # noqa: E402
    INITIAL_MAX_CANDIDATE_COUNT,
    CandidateLimitError,
    CandidatePlan,
    build_initial_candidate_plan,
)
from src.providers import (  # noqa: E402
    ExchangeRateProvider,
    ExchangeRateProviderError,
    InvalidExchangeRateDataError,
    UnsupportedCurrencyError,
    YFinanceExchangeRateProvider,
)


CalibrationRunner = Callable[..., CalibrationReport]
ReportWriter = Callable[[CalibrationReport, str | Path], tuple[Path, ...]]
CandidatePlanBuilder = Callable[..., CandidatePlan]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without importing yfinance or using the network."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare bounded FX threshold candidates across separate "
            "calibration and validation periods. No winner is selected."
        )
    )
    parser.add_argument(
        "--currencies",
        nargs="+",
        required=True,
        type=_parse_currency_argument,
        metavar="CODE",
        help="Currencies to evaluate (USD, JPY, EUR).",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=_parse_date_argument,
        metavar="YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=_parse_date_argument,
        metavar="YYYY-MM-DD",
    )
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--validation-start",
        type=_parse_date_argument,
        metavar="YYYY-MM-DD",
        help="First date assigned to VALIDATION.",
    )
    split_group.add_argument(
        "--calibration-end",
        type=_parse_date_argument,
        metavar="YYYY-MM-DD",
        help="Last calendar date assigned to CALIBRATION.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="PATH",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=_parse_positive_integer,
        default=list(STANDARD_EVALUATION_HORIZONS),
        metavar="N",
        help="Forward observation horizons (default: 5 10 20 60).",
    )
    parser.add_argument(
        "--max-candidates",
        type=_parse_positive_integer,
        default=INITIAL_MAX_CANDIDATE_COUNT,
        metavar="N",
        help=(
            "Fail before download when the explicit candidate plan exceeds "
            f"this count (default: {INITIAL_MAX_CANDIDATE_COUNT})."
        ),
    )
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI values and validate cross-argument date boundaries."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    if len(set(arguments.currencies)) != len(arguments.currencies):
        parser.error("--currencies must not contain duplicates")
    if len(set(arguments.horizons)) != len(arguments.horizons):
        parser.error("--horizons must not contain duplicates")
    if arguments.start_date > arguments.end_date:
        parser.error("--start-date must be on or before --end-date")

    validation_start = arguments.validation_start
    if validation_start is None:
        validation_start = arguments.calibration_end + pd.Timedelta(days=1)
    if not arguments.start_date < validation_start <= arguments.end_date:
        parser.error(
            "the period split must leave non-empty calibration and validation "
            "date ranges"
        )
    arguments.validation_start = validation_start
    arguments.horizons = tuple(arguments.horizons)
    arguments.currencies = tuple(arguments.currencies)
    return arguments


def run_calibration_workflow(
    *,
    currencies: Sequence[str],
    start_date: object,
    end_date: object,
    validation_start: object,
    candidate_plan: CandidatePlan,
    horizons: Sequence[int],
    historical_provider: ExchangeRateProvider | None = None,
) -> CalibrationReport:
    """Fetch each currency once, then delegate all analysis to core engines."""
    provider = historical_provider or YFinanceExchangeRateProvider()
    historical_by_currency: dict[str, pd.DataFrame] = {}
    for currency in currencies:
        historical = provider.fetch_daily_rates(
            currency,
            start_date,
            end_date,
        )
        if not isinstance(historical, pd.DataFrame):
            raise CalibrationDataError(
                f"historical provider returned invalid data for {currency}"
            )
        if historical.empty:
            raise CalibrationDataError(
                f"historical provider returned no rows for {currency}"
            )
        historical_by_currency[currency] = historical.copy(deep=True)

    return run_calibration(
        historical_by_currency,
        validation_start=validation_start,
        candidate_plan=candidate_plan,
        horizons=horizons,
    )


def format_run_report(
    report: CalibrationReport,
    written_paths: Sequence[Path],
) -> str:
    """Format factual run metadata without scoring any configuration."""
    lines = [
        f"validation start: {report.validation_start.date().isoformat()}",
        f"total runtime seconds: {report.elapsed_seconds:.3f}",
    ]
    calibration_rows = report.period_data_summary.loc[
        report.period_data_summary["period_type"].eq("CALIBRATION")
    ]
    for currency in report.currencies:
        row = calibration_rows.loc[
            calibration_rows["currency"].eq(currency)
        ].iloc[0]
        lines.append(
            f"{currency} historical: "
            f"{pd.Timestamp(row['source_start_date']).date().isoformat()} to "
            f"{pd.Timestamp(row['source_end_date']).date().isoformat()}, "
            f"{int(row['source_row_count'])} rows"
        )
    for _, row in report.period_data_summary.iterrows():
        lines.append(
            f"{row['period_type']} {row['currency']} runtime seconds: "
            f"{float(row['elapsed_seconds']):.3f}"
        )
    lines.append("reports:")
    lines.extend(f"- {path}" for path in written_paths)
    lines.append("configuration selection: none (human review required)")
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: CalibrationRunner = run_calibration_workflow,
    writer: ReportWriter = write_calibration_reports,
    candidate_plan_builder: CandidatePlanBuilder = build_initial_candidate_plan,
) -> int:
    """Run the bounded workflow and translate failures into safe messages."""
    arguments = parse_arguments(argv)
    try:
        candidate_plan = candidate_plan_builder(
            max_candidate_count=arguments.max_candidates
        )
        print(
            "candidate configuration count: "
            f"{candidate_plan.candidate_count} "
            f"(limit: {candidate_plan.max_candidate_count})"
        )
        report = runner(
            currencies=arguments.currencies,
            start_date=arguments.start_date,
            end_date=arguments.end_date,
            validation_start=arguments.validation_start,
            candidate_plan=candidate_plan,
            horizons=arguments.horizons,
        )
        written_paths = writer(report, arguments.output_dir)
    except CandidateLimitError as error:
        return _report_error(str(error))
    except UnsupportedCurrencyError:
        return _report_error("Yahoo Finance does not support a requested currency.")
    except InvalidExchangeRateDataError:
        return _report_error("Yahoo Finance returned malformed historical data.")
    except ExchangeRateProviderError:
        return _report_error(
            "Yahoo Finance historical download failed "
            "(network failure, timeout, or unavailable dependency)."
        )
    except CalibrationDataError:
        return _report_error(
            "Historical data cannot form non-empty calibration and validation "
            "periods for every requested currency."
        )
    except FileExistsError:
        return _report_error(
            "A calibration report file already exists in the output directory; "
            "choose a new directory."
        )
    except OSError:
        return _report_error("Calibration reports could not be written.")
    except (TypeError, ValueError):
        return _report_error(
            "Calibration configuration or historical data violated its contract."
        )

    print(format_run_report(report, written_paths))
    return 0


def _parse_currency_argument(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in CALIBRATION_CURRENCIES:
        raise argparse.ArgumentTypeError(
            "unsupported currency; choose from: "
            + ", ".join(CALIBRATION_CURRENCIES)
        )
    return normalized


def _parse_date_argument(value: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from None
    if pd.isna(timestamp) or timestamp.tzinfo is not None:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD")
    normalized = timestamp.normalize()
    if value != normalized.date().isoformat():
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD")
    return normalized


def _parse_positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("value must be a positive integer") from None
    if number <= 0 or str(number) != value:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _report_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
