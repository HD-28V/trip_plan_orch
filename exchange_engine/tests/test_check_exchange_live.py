import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scripts import check_exchange_live as live
from src.config import MissingConfigurationError
from src.indicators import INDICATOR_COLUMNS
from src.providers import (
    ExchangeRateProviderError,
    InvalidExchangeRateDataError,
    LatestExchangeRateNetworkError,
    LatestExchangeRateResponseError,
)


AS_OF_DATE = pd.Timestamp("2026-08-06")
REPORT_LABELS = (
    "currency",
    "historical row count",
    "historical start date",
    "historical end date",
    "latest data date",
    "latest rate",
    "merged row count",
    "current rate",
    *INDICATOR_COLUMNS,
)


def make_historical_rates(
    row_count: int = 180,
    *,
    end_date: pd.Timestamp = AS_OF_DATE,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range(end=end_date, periods=row_count),
            "rate": np.arange(1200.0, 1200.0 + row_count),
        }
    )


def make_latest_rates(
    *,
    date: pd.Timestamp = AS_OF_DATE,
    currency: str = "USD",
    rate: float = 1400.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date],
            "currency": [currency],
            "rate": [rate],
        }
    )


class FakeHistoricalProvider:
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        error: Exception | None = None,
    ) -> None:
        self.data = data
        self.error = error
        self.calls: list[tuple[str, object, object]] = []

    def fetch_daily_rates(
        self,
        currency: str,
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        self.calls.append((currency, start_date, end_date))
        if self.error is not None:
            raise self.error
        return self.data


class FakeLatestProvider:
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        error: Exception | None = None,
    ) -> None:
        self.data = data
        self.error = error
        self.calls: list[list[str]] = []

    def fetch_latest_rates(self, currencies: object) -> pd.DataFrame:
        requested = list(currencies)
        self.calls.append(requested)
        if self.error is not None:
            raise self.error
        return self.data


class LiveExchangeCliTests(unittest.TestCase):
    def test_argument_parser_normalizes_currency_and_rejects_unsupported(self) -> None:
        parser = live.build_argument_parser()

        self.assertEqual(
            parser.parse_args(["--currency", "USD"]).currency,
            "USD",
        )
        self.assertEqual(
            parser.parse_args(["--currency", "jpy"]).currency,
            "JPY",
        )

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                parser.parse_args(["--currency", "CHF"])

        self.assertEqual(context.exception.code, 2)

    def test_main_uses_injected_runner_without_loading_real_environment(self) -> None:
        historical = FakeHistoricalProvider(make_historical_rates())
        latest = FakeLatestProvider(make_latest_rates())
        original_run_live_check = live.run_live_check

        def fake_runner(currency: str) -> live.LiveCheckResult:
            return original_run_live_check(
                currency,
                historical_provider=historical,
                latest_provider=latest,
                as_of_date=AS_OF_DATE,
            )

        stdout = io.StringIO()
        with (
            patch.object(live, "load_environment", return_value=False) as loader,
            patch.object(
                live,
                "run_live_check",
                side_effect=AssertionError("default runner must not be used"),
            ),
            redirect_stdout(stdout),
        ):
            exit_code = live.main(["--currency", "USD"], runner=fake_runner)

        self.assertEqual(exit_code, 0)
        loader.assert_called_once()
        self.assertIn("currency: USD", stdout.getvalue())
        self.assertEqual(historical.calls[0][0], "USD")
        self.assertEqual(latest.calls, [["USD"]])


class LiveExchangePipelineTests(unittest.TestCase):
    def test_requests_a_400_day_historical_range(self) -> None:
        historical = FakeHistoricalProvider(make_historical_rates())
        latest = FakeLatestProvider(make_latest_rates())

        live.run_live_check(
            "USD",
            historical_provider=historical,
            latest_provider=latest,
            as_of_date=AS_OF_DATE,
        )

        self.assertEqual(live.HISTORICAL_LOOKBACK_DAYS, 400)
        currency, start_date, end_date = historical.calls[0]
        self.assertEqual(currency, "USD")
        self.assertEqual(pd.Timestamp(end_date), AS_OF_DATE)
        self.assertEqual(
            pd.Timestamp(start_date),
            AS_OF_DATE - pd.Timedelta(days=400),
        )

    def test_same_day_latest_rate_wins_without_adding_a_merged_row(self) -> None:
        historical_data = make_historical_rates()
        replacement_rate = 2222.0
        latest_data = make_latest_rates(rate=replacement_rate)

        result = live.run_live_check(
            "USD",
            historical_provider=FakeHistoricalProvider(historical_data),
            latest_provider=FakeLatestProvider(latest_data),
            as_of_date=AS_OF_DATE,
        )
        report = live.format_report(result)

        self.assertRegex(report, r"(?m)^historical row count:\s*180$")
        self.assertRegex(report, r"(?m)^merged row count:\s*180$")
        self.assertRegex(report, r"(?m)^latest rate:\s*2222(?:\.0+)?$")
        self.assertRegex(report, r"(?m)^current rate:\s*2222(?:\.0+)?$")

    def test_report_uses_the_latest_indicator_row_and_has_every_label(self) -> None:
        historical_data = make_historical_rates(
            end_date=AS_OF_DATE - pd.Timedelta(days=1)
        )
        latest_data = make_latest_rates(rate=1555.0)
        expected_values = {
            column: float(3000 + position)
            for position, column in enumerate(INDICATOR_COLUMNS)
        }

        def fake_calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
            calculated = data.copy(deep=True)
            for column, latest_value in expected_values.items():
                calculated[column] = -1.0
                calculated.loc[calculated.index[-1], column] = latest_value
            return calculated

        with patch.object(
            live,
            "calculate_indicators",
            side_effect=fake_calculate_indicators,
        ):
            result = live.run_live_check(
                "USD",
                historical_provider=FakeHistoricalProvider(historical_data),
                latest_provider=FakeLatestProvider(latest_data),
                as_of_date=AS_OF_DATE,
            )

        report = live.format_report(result)
        for label in REPORT_LABELS:
            with self.subTest(label=label):
                self.assertRegex(report, rf"(?m)^{label}:")
        for label, expected_value in expected_values.items():
            with self.subTest(indicator=label):
                self.assertRegex(
                    report,
                    rf"(?m)^{label}:\s*{expected_value:g}(?:\.0+)?$",
                )

    def test_nan_indicators_are_reported_as_unavailable(self) -> None:
        def nan_indicators(data: pd.DataFrame) -> pd.DataFrame:
            calculated = data.copy(deep=True)
            for column in INDICATOR_COLUMNS:
                calculated[column] = np.nan
            return calculated

        with patch.object(
            live,
            "calculate_indicators",
            side_effect=nan_indicators,
        ):
            result = live.run_live_check(
                "USD",
                historical_provider=FakeHistoricalProvider(
                    make_historical_rates()
                ),
                latest_provider=FakeLatestProvider(make_latest_rates()),
                as_of_date=AS_OF_DATE,
            )

        report = live.format_report(result)
        for indicator in INDICATOR_COLUMNS:
            with self.subTest(indicator=indicator):
                self.assertRegex(
                    report,
                    rf"(?m)^{indicator}:\s*unavailable$",
                )

    def test_rejects_empty_and_insufficient_historical_data(self) -> None:
        empty = pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "rate": pd.Series(dtype=float),
            }
        )
        latest = FakeLatestProvider(make_latest_rates())

        with self.assertRaises(live.EmptyHistoricalDataError):
            live.run_live_check(
                "USD",
                historical_provider=FakeHistoricalProvider(empty),
                latest_provider=latest,
                as_of_date=AS_OF_DATE,
            )

        with self.assertRaises(live.InsufficientHistoricalDataError):
            live.run_live_check(
                "USD",
                historical_provider=FakeHistoricalProvider(
                    make_historical_rates(179)
                ),
                latest_provider=latest,
                as_of_date=AS_OF_DATE,
            )

    def test_does_not_modify_provider_dataframes(self) -> None:
        historical_data = make_historical_rates()
        latest_data = make_latest_rates(rate=1444.0)
        historical_before = historical_data.copy(deep=True)
        latest_before = latest_data.copy(deep=True)

        live.run_live_check(
            "USD",
            historical_provider=FakeHistoricalProvider(historical_data),
            latest_provider=FakeLatestProvider(latest_data),
            as_of_date=AS_OF_DATE,
        )

        assert_frame_equal(historical_data, historical_before)
        assert_frame_equal(latest_data, latest_before)


class LiveExchangeErrorHandlingTests(unittest.TestCase):
    def test_main_reports_provider_failures_without_exposing_secret(self) -> None:
        secret = "dummy-secret-that-must-not-appear"
        cases = (
            (
                MissingConfigurationError(f"missing key: {secret}"),
                "EXCHANGE_RATE_API_KEY",
            ),
            (
                ExchangeRateProviderError(f"download failed: {secret}"),
                "Yahoo Finance",
            ),
            (
                InvalidExchangeRateDataError(f"bad history: {secret}"),
                "malformed historical",
            ),
            (
                LatestExchangeRateNetworkError(f"timeout: {secret}"),
                "ExchangeRate-API",
            ),
            (
                LatestExchangeRateResponseError(f"bad response: {secret}"),
                "response",
            ),
        )

        for error, expected_text in cases:
            with self.subTest(error=type(error).__name__):
                def failing_runner(
                    currency: str,
                    *,
                    failure: Exception = error,
                ) -> live.LiveCheckResult:
                    raise failure

                stderr = io.StringIO()
                with (
                    patch.object(live, "load_environment", return_value=False),
                    patch.object(
                        live,
                        "run_live_check",
                        side_effect=AssertionError(
                            "default runner must not be used"
                        ),
                    ),
                    redirect_stderr(stderr),
                ):
                    exit_code = live.main(
                        ["--currency", "USD"],
                        runner=failing_runner,
                    )

                error_output = stderr.getvalue()
                self.assertEqual(exit_code, 1)
                self.assertIn(expected_text, error_output)
                self.assertNotIn(secret, error_output)


if __name__ == "__main__":
    unittest.main()
