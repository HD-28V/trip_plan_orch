"""Run the live FX pipeline and emit one production-v1 exchange recommendation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import check_exchange_live as live  # noqa: E402
from src.config import MissingConfigurationError  # noqa: E402
from src.dday_policy import DdayValidationError  # noqa: E402
from src.recommendation_engine import (  # noqa: E402
    ExchangeRecommendationResult,
    RecommendationValidationError,
    evaluate_exchange_recommendation,
)
from src.split_exchange import SplitExchangeValidationError  # noqa: E402


PRODUCTION_CURRENCIES = ("USD", "JPY", "EUR")


@dataclass(frozen=True)
class LiveRecommendationCheckResult:
    """The reused live data result and its immutable recommendation."""

    currency: str
    live_result: live.LiveCheckResult = field(repr=False)
    recommendation: ExchangeRecommendationResult


def build_argument_parser() -> argparse.ArgumentParser:
    """Build a validated, side-effect-free CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the live FX pipeline through the production-v1 recommendation engine.",
    )
    parser.add_argument("--currency", required=True, type=_parse_currency, metavar="CODE")
    parser.add_argument(
        "--days-until-departure",
        required=True,
        type=_parse_days,
        metavar="DAYS",
    )
    parser.add_argument(
        "--total-target-krw",
        required=True,
        type=_parse_positive_amount,
        metavar="KRW",
    )
    parser.add_argument(
        "--already-exchanged-krw",
        required=True,
        type=_parse_amount,
        metavar="KRW",
    )
    return parser


def run_live_recommendation_check(
    currency: str,
    days_until_departure: int,
    total_target_krw: Decimal,
    already_exchanged_krw: Decimal,
    *,
    live_runner: Callable[[str], live.LiveCheckResult] | None = None,
) -> LiveRecommendationCheckResult:
    """Reuse live fetch/merge/indicator work and evaluate its latest row once."""
    normalized_currency = _normalize_currency(currency)
    selected_live_runner = live_runner or live.run_live_check
    live_result = selected_live_runner(normalized_currency)
    if live_result.indicator_rows.empty:
        raise live.LiveCheckDataError("live pipeline produced no indicator rows")
    recommendation = evaluate_exchange_recommendation(
        live_result.indicator_rows.iloc[-1].copy(deep=True),
        days_until_departure,
        total_target_krw,
        already_exchanged_krw,
    )
    return LiveRecommendationCheckResult(
        currency=normalized_currency,
        live_result=live_result,
        recommendation=recommendation,
    )


def format_recommendation_report(result: LiveRecommendationCheckResult) -> str:
    """Render the required live fields without exposing configuration secrets."""
    row = result.live_result.indicator_rows.iloc[-1]
    recommendation = result.recommendation
    lines = [
        f"currency: {result.currency}",
        f"latest date: {live._format_date(row.get('date'))}",
        f"current rate: {live._format_number(recommendation.current_rate)}",
        f"SMA60: {live._format_number(row.get('SMA60'))}",
        f"SMA120: {live._format_number(row.get('SMA120'))}",
        f"SMA60 distance: {live._format_number(row.get('SMA60_distance_pct'))}",
        f"SMA120 distance: {live._format_number(row.get('SMA120_distance_pct'))}",
        f"percentile: {live._format_number(row.get('percentile_rank_180'))}",
        f"BB lower: {live._format_number(row.get('BB_lower'))}",
        f"base market signal: {recommendation.base_signal.value}",
        f"adjusted signal: {recommendation.adjusted_signal.value}",
        f"days until departure: {recommendation.days_until_departure}",
        f"D-Day band: {recommendation.dday_band.value}",
        f"urgency: {recommendation.urgency.value}",
        f"total target KRW: {_format_decimal(recommendation.total_target_krw)}",
        f"already exchanged KRW: {_format_decimal(recommendation.already_exchanged_krw)}",
        f"target cumulative ratio: {_format_decimal(recommendation.target_cumulative_ratio)}",
        f"target cumulative amount KRW: {_format_decimal(recommendation.target_cumulative_amount_krw)}",
        f"recommended additional KRW: {_format_decimal(recommendation.recommended_additional_krw)}",
        f"remaining after recommendation KRW: {_format_decimal(recommendation.remaining_after_recommendation_krw)}",
        f"FX production config: {recommendation.production_configuration_id} {recommendation.production_configuration_version}",
        f"D-Day policy: {recommendation.dday_policy_id} {recommendation.dday_policy_version}",
        f"Split policy: {recommendation.split_policy_id} {recommendation.split_policy_version}",
        "unavailable indicators: " + _format_names(recommendation.unavailable_indicators),
    ]
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[str, int, Decimal, Decimal], LiveRecommendationCheckResult] | None = None,
) -> int:
    """Run the secret-safe production recommendation smoke CLI."""
    arguments = build_argument_parser().parse_args(argv)
    if arguments.already_exchanged_krw > arguments.total_target_krw:
        return _report_error(
            "already exchanged KRW must not exceed total target KRW."
        )
    live.load_environment(PROJECT_ROOT / ".env")
    selected_runner = runner or run_live_recommendation_check
    try:
        result = selected_runner(
            arguments.currency,
            arguments.days_until_departure,
            arguments.total_target_krw,
            arguments.already_exchanged_krw,
        )
        report = format_recommendation_report(result)
    except MissingConfigurationError:
        return _report_error("EXCHANGE_RATE_API_KEY is missing. Add it to exchange_engine/.env and retry.")
    except live.EmptyHistoricalDataError as error:
        return _report_error(f"Yahoo Finance returned empty historical data for {error.currency}.")
    except live.InsufficientHistoricalDataError as error:
        return _report_error(f"At least {error.required} valid historical observations are required for {error.currency}; received {error.received}.")
    except live.InvalidExchangeRateDataError:
        return _report_error("Yahoo Finance returned malformed historical exchange-rate data.")
    except live.ExchangeRateProviderError:
        return _report_error("Yahoo Finance historical request failed (network failure or timeout).")
    except live.LatestExchangeRateNetworkError:
        return _report_error("ExchangeRate-API latest request failed (HTTP/network failure or timeout).")
    except live.LatestExchangeRateResponseError:
        return _report_error("ExchangeRate-API returned a malformed or unsuccessful response.")
    except live.LatestExchangeRateProviderError:
        return _report_error("ExchangeRate-API client is unavailable; verify the requests dependency.")
    except live.LiveCheckDataError:
        return _report_error("The live exchange-rate pipeline produced no analyzable data.")
    except (DdayValidationError, SplitExchangeValidationError, RecommendationValidationError, TypeError, ValueError):
        return _report_error("Recommendation inputs or production results violate the established contract.")
    print(report)
    return 0


def _normalize_currency(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("currency must be a string")
    currency = value.strip().upper()
    if currency not in PRODUCTION_CURRENCIES:
        raise ValueError("unsupported production currency")
    return currency


def _parse_currency(value: str) -> str:
    try:
        return _normalize_currency(value)
    except ValueError:
        raise argparse.ArgumentTypeError("unsupported currency; choose USD, JPY, or EUR") from None


def _parse_days(value: str) -> int:
    try:
        days = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("days until departure must be a nonnegative integer") from None
    if days < 0:
        raise argparse.ArgumentTypeError("days until departure must be a nonnegative integer")
    return days


def _parse_amount(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError):
        raise argparse.ArgumentTypeError("KRW amount must be a finite number") from None
    if not amount.is_finite() or amount < 0:
        raise argparse.ArgumentTypeError("KRW amount must be a finite nonnegative number")
    return amount


def _parse_positive_amount(value: str) -> Decimal:
    amount = _parse_amount(value)
    if amount <= 0:
        raise argparse.ArgumentTypeError("KRW amount must be greater than zero")
    return amount


def _format_decimal(value: Decimal) -> str:
    return format(value, "f")


def _format_names(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _report_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
