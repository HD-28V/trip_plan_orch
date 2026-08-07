"""Run the existing live FX pipeline and evaluate its production-v1 Signal."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import check_exchange_live as live  # noqa: E402
from src.config import MissingConfigurationError  # noqa: E402
from src.production_config import (  # noqa: E402
    ProductionSignalConfiguration,
    evaluate_production_signal,
    get_production_signal_configuration,
)
from src.providers import (  # noqa: E402
    ExchangeRateProviderError,
    InvalidExchangeRateDataError,
    LatestExchangeRateNetworkError,
    LatestExchangeRateProviderError,
    LatestExchangeRateResponseError,
    UnsupportedCurrencyError,
)
from src.signal_engine import SignalResult  # noqa: E402


PRODUCTION_CURRENCIES = ("USD", "JPY", "EUR")

LiveRunner = Callable[[str], live.LiveCheckResult]
SignalEvaluator = Callable[[pd.Series], SignalResult]


@dataclass(frozen=True)
class ProductionSignalCheckResult:
    """Hold the reused live pipeline output and its production Signal result."""

    live_result: live.LiveCheckResult = field(repr=False)
    signal_result: SignalResult
    configuration: ProductionSignalConfiguration

    def __post_init__(self) -> None:
        if not isinstance(self.live_result, live.LiveCheckResult):
            raise TypeError("live_result must be a LiveCheckResult")
        if not isinstance(self.signal_result, SignalResult):
            raise TypeError("signal_result must be a SignalResult")
        if not isinstance(self.configuration, ProductionSignalConfiguration):
            raise TypeError(
                "configuration must be a ProductionSignalConfiguration"
            )
        if (
            self.signal_result.thresholds != self.configuration.thresholds
            or self.signal_result.policy != self.configuration.policy
        ):
            raise ValueError(
                "signal_result must use the reported production configuration"
            )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build a CLI limited to the currencies validated for production v1."""
    parser = argparse.ArgumentParser(
        description=(
            "Run live historical/latest FX data through the existing merge and "
            "indicator pipeline, then evaluate the production-v1 Signal."
        ),
    )
    parser.add_argument(
        "--currency",
        required=True,
        type=_parse_currency_argument,
        metavar="CODE",
        help=(
            "Foreign currency code validated for production v1 "
            f"({', '.join(PRODUCTION_CURRENCIES)})."
        ),
    )
    return parser


def run_production_signal_check(
    currency: str,
    *,
    live_runner: LiveRunner | None = None,
    evaluator: SignalEvaluator | None = None,
) -> ProductionSignalCheckResult:
    """Reuse the live data pipeline and evaluate only its newest indicator row."""
    normalized_currency = _normalize_production_currency(currency)
    selected_live_runner = live_runner or live.run_live_check
    selected_evaluator = evaluator or evaluate_production_signal
    live_result = selected_live_runner(normalized_currency)
    if live_result.indicator_rows.empty:
        raise live.LiveCheckDataError(
            "the live exchange-rate pipeline produced no indicator rows"
        )

    latest_indicator_row = live_result.indicator_rows.iloc[-1].copy(deep=True)
    signal_result = selected_evaluator(latest_indicator_row)
    if not isinstance(signal_result, SignalResult):
        raise TypeError("evaluator must return a SignalResult")

    return ProductionSignalCheckResult(
        live_result=live_result,
        signal_result=signal_result,
        configuration=get_production_signal_configuration(),
    )


def format_production_signal_report(result: ProductionSignalCheckResult) -> str:
    """Append auditable production Signal details to the existing live report."""
    signal_result = result.signal_result
    lines = [
        live.format_report(result.live_result),
        "",
        f"production configuration id: {result.configuration.configuration_id}",
        f"production version: {result.configuration.version}",
        f"Signal: {signal_result.signal.value}",
    ]
    for condition in signal_result.conditions:
        lines.append(
            f"condition {condition.name}: {condition.status.value} "
            f"(observed={live._format_number(condition.observed_value)}, "
            f"good_threshold={live._format_number(condition.good_threshold)}, "
            f"strong_threshold={live._format_number(condition.strong_threshold)})"
        )

    lines.append(
        "satisfied conditions: "
        + _format_names(signal_result.satisfied_conditions)
    )
    lines.append(
        "signal unavailable indicators: "
        + _format_names(signal_result.unavailable_indicators)
    )
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[str], ProductionSignalCheckResult] | None = None,
) -> int:
    """Run the production smoke CLI with fixed, secret-safe error messages."""
    arguments = build_argument_parser().parse_args(argv)
    live.load_environment(PROJECT_ROOT / ".env")
    selected_runner = runner or run_production_signal_check

    try:
        report = format_production_signal_report(
            selected_runner(arguments.currency)
        )
    except MissingConfigurationError:
        return _report_error(
            "EXCHANGE_RATE_API_KEY is missing. Add it to "
            "exchange_engine/.env and retry."
        )
    except live.EmptyHistoricalDataError as error:
        return _report_error(
            "Yahoo Finance returned empty historical data for "
            f"{error.currency}."
        )
    except live.InsufficientHistoricalDataError as error:
        return _report_error(
            f"At least {error.required} valid historical observations are "
            f"required for {error.currency}; received {error.received}."
        )
    except UnsupportedCurrencyError:
        return _report_error(
            "Unsupported currency. Production v1 currencies: "
            f"{', '.join(PRODUCTION_CURRENCIES)}."
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
    except live.LiveCheckDataError:
        return _report_error(
            "The live exchange-rate pipeline produced no analyzable data."
        )
    except (TypeError, ValueError):
        return _report_error(
            "Exchange-rate indicators could not be evaluated because they "
            "violated the existing production Signal contract."
        )

    print(report)
    return 0


def _parse_currency_argument(value: str) -> str:
    try:
        return _normalize_production_currency(value)
    except UnsupportedCurrencyError:
        supported = ", ".join(PRODUCTION_CURRENCIES)
        raise argparse.ArgumentTypeError(
            f"unsupported currency; choose one of: {supported}"
        ) from None


def _normalize_production_currency(value: object) -> str:
    if not isinstance(value, str):
        raise UnsupportedCurrencyError("currency must be a string")
    normalized = value.strip().upper()
    if normalized not in PRODUCTION_CURRENCIES:
        raise UnsupportedCurrencyError("unsupported production currency")
    return normalized


def _format_names(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _report_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
