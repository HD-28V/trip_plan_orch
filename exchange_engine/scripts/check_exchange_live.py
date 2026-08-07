"""Run a live historical/latest FX smoke check through existing engines.

The script deliberately owns only orchestration and user-facing reporting.
Provider normalization, historical/latest merging, and indicator calculations
remain in their existing modules. Exchange rates are always KRW required for
one unit of the selected foreign currency.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import SupportsFloat, SupportsIndex, TypeAlias, cast

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MissingConfigurationError, load_environment  # noqa: E402
from src.exchange_series import merge_historical_and_latest  # noqa: E402
from src.indicators import (  # noqa: E402
    INDICATOR_COLUMNS,
    PERCENTILE_WINDOW,
    calculate_indicators,
)
from src.providers import (  # noqa: E402
    ExchangeRateApiProvider,
    ExchangeRateProvider,
    ExchangeRateProviderError,
    InvalidExchangeRateDataError,
    LatestExchangeRateNetworkError,
    LatestExchangeRateProvider,
    LatestExchangeRateProviderError,
    LatestExchangeRateResponseError,
    SUPPORTED_FOREIGN_CURRENCIES,
    UnsupportedCurrencyError,
    YFinanceExchangeRateProvider,
)


HISTORICAL_LOOKBACK_DAYS = 400
MINIMUM_HISTORICAL_OBSERVATIONS = PERCENTILE_WINDOW
UNAVAILABLE_TEXT = "unavailable"

TimestampInput: TypeAlias = (
    pd.Timestamp | datetime | date | str | float | np.datetime64
)
PandasScalar: TypeAlias = (
    pd.Timestamp | datetime | date | str | bool | int | float | None
)
FloatInput: TypeAlias = str | bytes | bytearray | SupportsFloat | SupportsIndex


class LiveCheckDataError(RuntimeError):
    """Base exception for unusable data discovered by the live workflow."""


class EmptyHistoricalDataError(LiveCheckDataError):
    """Raised when Yahoo Finance provides no historical observations."""

    def __init__(self, currency: str) -> None:
        super().__init__("historical exchange-rate data is empty")
        self.currency = currency


class InsufficientHistoricalDataError(LiveCheckDataError):
    """Raised when the historical result cannot fill the longest window."""

    def __init__(self, currency: str, received: int) -> None:
        super().__init__("historical exchange-rate data is insufficient")
        self.currency = currency
        self.received = received
        self.required = MINIMUM_HISTORICAL_OBSERVATIONS


@dataclass(frozen=True)
class LiveCheckResult:
    """Hold each stage so the smoke check remains inspectable and testable."""

    currency: str
    historical_rates: pd.DataFrame = field(repr=False)
    latest_rates: pd.DataFrame = field(repr=False)
    merged_rates: pd.DataFrame = field(repr=False)
    indicator_rows: pd.DataFrame = field(repr=False)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without reading configuration or APIs."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate live historical and latest FX data through the existing "
            "merge and indicator engines."
        ),
    )
    parser.add_argument(
        "--currency",
        required=True,
        type=_parse_currency_argument,
        metavar="CODE",
        help=(
            "Foreign currency code (supported: "
            f"{', '.join(sorted(SUPPORTED_FOREIGN_CURRENCIES))})."
        ),
    )
    return parser


def run_live_check(
    currency: str,
    *,
    historical_provider: ExchangeRateProvider | None = None,
    latest_provider: LatestExchangeRateProvider | None = None,
    as_of_date: date | datetime | str | pd.Timestamp | None = None,
) -> LiveCheckResult:
    """Fetch, merge, and calculate indicators without duplicating their logic."""
    normalized_currency = _normalize_supported_currency(currency)
    historical_end = _normalize_as_of_date(as_of_date)
    historical_start = historical_end - timedelta(
        days=HISTORICAL_LOOKBACK_DAYS
    )

    historical_source = (
        historical_provider
        if historical_provider is not None
        else YFinanceExchangeRateProvider()
    )
    historical_rates = historical_source.fetch_daily_rates(
        normalized_currency,
        historical_start,
        historical_end,
    )
    if historical_rates.empty:
        raise EmptyHistoricalDataError(normalized_currency)
    if len(historical_rates) < MINIMUM_HISTORICAL_OBSERVATIONS:
        raise InsufficientHistoricalDataError(
            normalized_currency,
            len(historical_rates),
        )

    latest_source = (
        latest_provider
        if latest_provider is not None
        else ExchangeRateApiProvider()
    )
    latest_rates = latest_source.fetch_latest_rates([normalized_currency])

    merged_rates = merge_historical_and_latest(
        historical_rates,
        latest_rates,
        normalized_currency,
    )
    indicator_rows = calculate_indicators(merged_rates)
    if indicator_rows.empty:
        raise LiveCheckDataError("merged exchange-rate data is empty")

    return LiveCheckResult(
        currency=normalized_currency,
        historical_rates=historical_rates,
        latest_rates=latest_rates,
        merged_rates=merged_rates,
        indicator_rows=indicator_rows,
    )


def format_report(result: LiveCheckResult) -> str:
    """Format required metadata and the newest indicator row safely."""
    historical_dates = pd.to_datetime(
        result.historical_rates["date"],
        errors="coerce",
        utc=True,
    )
    latest_for_currency = result.latest_rates.loc[
        result.latest_rates["currency"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq(result.currency)
    ].copy()
    if latest_for_currency.empty:
        raise ValueError("latest data does not contain the selected currency")

    latest_for_currency["_normalized_date"] = pd.to_datetime(
        latest_for_currency["date"],
        errors="coerce",
        utc=True,
    )
    latest_row = latest_for_currency.sort_values(
        "_normalized_date",
        kind="stable",
    ).iloc[-1]
    current_row = result.indicator_rows.iloc[-1]

    lines = [
        f"currency: {result.currency}",
        f"historical row count: {len(result.historical_rates)}",
        f"historical start date: {_format_date(historical_dates.min())}",
        f"historical end date: {_format_date(historical_dates.max())}",
        f"latest data date: {_format_date(latest_row['date'])}",
        f"latest rate: {_format_number(latest_row['rate'])}",
        f"merged row count: {len(result.merged_rates)}",
        "",
        f"current rate: {_format_number(current_row['rate'])}",
    ]

    unavailable_indicators: list[str] = []
    for indicator_name in INDICATOR_COLUMNS:
        indicator_value = current_row[indicator_name]
        formatted_value = _format_number(indicator_value)
        lines.append(f"{indicator_name}: {formatted_value}")
        if formatted_value == UNAVAILABLE_TEXT:
            unavailable_indicators.append(indicator_name)

    unavailable_summary = (
        ", ".join(unavailable_indicators)
        if unavailable_indicators
        else "none"
    )
    lines.append(f"unavailable indicators: {unavailable_summary}")
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[str], LiveCheckResult] | None = None,
) -> int:
    """Run the CLI and translate provider failures into secret-safe messages."""
    arguments = build_argument_parser().parse_args(argv)
    load_environment(PROJECT_ROOT / ".env")
    selected_runner = runner or run_live_check

    try:
        report = format_report(selected_runner(arguments.currency))
    except MissingConfigurationError:
        return _report_error(
            "EXCHANGE_RATE_API_KEY is missing. Add it to "
            "exchange_engine/.env and retry."
        )
    except EmptyHistoricalDataError as error:
        return _report_error(
            "Yahoo Finance returned empty historical data for "
            f"{error.currency}."
        )
    except InsufficientHistoricalDataError as error:
        return _report_error(
            f"At least {error.required} valid historical observations are "
            f"required for {error.currency}; received {error.received}."
        )
    except UnsupportedCurrencyError:
        return _report_error(
            "Unsupported currency. Supported currencies: "
            f"{', '.join(sorted(SUPPORTED_FOREIGN_CURRENCIES))}."
        )
    except InvalidExchangeRateDataError:
        return _report_error(
            "Yahoo Finance returned malformed historical exchange-rate data."
        )
    except ExchangeRateProviderError:
        return _report_error(
            "Yahoo Finance historical request failed "
            "(network failure or timeout)."
        )
    except LatestExchangeRateNetworkError:
        return _report_error(
            "ExchangeRate-API latest request failed "
            "(HTTP/network failure or timeout)."
        )
    except LatestExchangeRateResponseError:
        return _report_error(
            "ExchangeRate-API returned a malformed or unsuccessful response."
        )
    except LatestExchangeRateProviderError:
        return _report_error(
            "ExchangeRate-API client is unavailable; verify the requests "
            "dependency."
        )
    except LiveCheckDataError:
        return _report_error(
            "The live exchange-rate pipeline produced no analyzable data."
        )
    except (TypeError, ValueError):
        return _report_error(
            "Exchange-rate data could not be merged or analyzed because it "
            "violated the existing data contract."
        )

    print(report)
    return 0


def _normalize_supported_currency(currency: object) -> str:
    if not isinstance(currency, str):
        raise UnsupportedCurrencyError("currency must be a string")
    normalized = currency.strip().upper()
    if normalized not in SUPPORTED_FOREIGN_CURRENCIES:
        raise UnsupportedCurrencyError("unsupported currency")
    return normalized


def _parse_currency_argument(value: str) -> str:
    try:
        return _normalize_supported_currency(value)
    except UnsupportedCurrencyError:
        supported = ", ".join(sorted(SUPPORTED_FOREIGN_CURRENCIES))
        raise argparse.ArgumentTypeError(
            f"unsupported currency; choose one of: {supported}"
        ) from None


def _normalize_as_of_date(
    value: date | datetime | str | pd.Timestamp | None,
) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("as_of_date must be a valid date") from error
    if _is_missing_scalar(timestamp):
        raise ValueError("as_of_date must be a valid date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.date()


def _format_date(value: object) -> str:
    try:
        timestamp = pd.Timestamp(cast(TimestampInput, value))
    except (TypeError, ValueError):
        return UNAVAILABLE_TEXT
    if _is_missing_scalar(timestamp):
        return UNAVAILABLE_TEXT
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.date().isoformat()


def _format_number(value: object) -> str:
    if _is_missing_scalar(value):
        return UNAVAILABLE_TEXT
    try:
        numeric_value = float(cast(FloatInput, value))
    except (TypeError, ValueError):
        return UNAVAILABLE_TEXT
    if not math.isfinite(numeric_value):
        return UNAVAILABLE_TEXT
    return f"{numeric_value:.6f}"


def _is_missing_scalar(value: object) -> bool:
    """Apply pandas' scalar missing-value check at an explicit type boundary."""
    return bool(pd.isna(cast(PandasScalar, value)))


def _report_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
