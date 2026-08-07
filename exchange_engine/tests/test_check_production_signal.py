import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from scripts import check_exchange_live as live
from scripts import check_production_signal as production_live
from src.config import MissingConfigurationError
from src.production_config import evaluate_production_signal
from src.providers import (
    ExchangeRateProviderError,
    InvalidExchangeRateDataError,
    LatestExchangeRateNetworkError,
    LatestExchangeRateProviderError,
    LatestExchangeRateResponseError,
    UnsupportedCurrencyError,
)


LATEST_DATE = pd.Timestamp("2026-08-06")


def make_indicator_row(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "date": LATEST_DATE,
        "rate": 100.5,
        "SMA60": 102.0,
        "SMA120": 104.0,
        "SMA60_distance_pct": -1.5,
        "SMA120_distance_pct": -3.0,
        "percentile_rank_180": 20.0,
        "BB_middle": 103.0,
        "BB_upper": 106.0,
        "BB_lower": 100.0,
    }
    values.update(overrides)
    return values


def make_live_result(
    currency: str = "USD",
    *,
    latest_overrides: dict[str, object] | None = None,
) -> live.LiveCheckResult:
    historical = pd.DataFrame(
        {
            "date": pd.bdate_range(end=LATEST_DATE, periods=180),
            "rate": np.linspace(90.0, 100.0, 180),
        }
    )
    latest_values = make_indicator_row(**(latest_overrides or {}))
    latest = pd.DataFrame(
        {
            "date": [latest_values["date"]],
            "currency": [currency],
            "rate": [latest_values["rate"]],
        }
    )
    merged = historical.copy(deep=True)
    merged.loc[merged.index[-1], "rate"] = latest_values["rate"]
    previous_values = make_indicator_row(
        date=LATEST_DATE - pd.Timedelta(days=1),
        rate=999.0,
        SMA60_distance_pct=9.0,
    )
    indicators = pd.DataFrame([previous_values, latest_values])
    return live.LiveCheckResult(
        currency=currency,
        historical_rates=historical,
        latest_rates=latest,
        merged_rates=merged,
        indicator_rows=indicators,
    )


class ProductionSignalCliTests(unittest.TestCase):
    def test_argument_parser_normalizes_and_limits_production_currencies(self) -> None:
        parser = production_live.build_argument_parser()

        self.assertEqual(
            parser.parse_args(["--currency", "jpy"]).currency,
            "JPY",
        )
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                parser.parse_args(["--currency", "GBP"])

        self.assertEqual(context.exception.code, 2)

    def test_main_uses_injected_runner_without_a_network_call(self) -> None:
        result = production_live.run_production_signal_check(
            "USD",
            live_runner=lambda currency: make_live_result(currency),
        )
        stdout = io.StringIO()

        with (
            patch.object(
                production_live.live,
                "load_environment",
                return_value=False,
            ) as loader,
            patch.object(
                production_live,
                "run_production_signal_check",
                side_effect=AssertionError("default live runner must not be used"),
            ),
            redirect_stdout(stdout),
        ):
            exit_code = production_live.main(
                ["--currency", "USD"],
                runner=lambda currency: result,
            )

        self.assertEqual(exit_code, 0)
        loader.assert_called_once()
        self.assertIn("production configuration id", stdout.getvalue())


class ProductionSignalPipelineTests(unittest.TestCase):
    def test_direct_runner_normalizes_and_rejects_unvalidated_currency(self) -> None:
        requested: list[str] = []

        production_live.run_production_signal_check(
            " eur ",
            live_runner=lambda currency: (
                requested.append(currency) or make_live_result(currency)
            ),
        )

        self.assertEqual(requested, ["EUR"])
        with self.assertRaisesRegex(UnsupportedCurrencyError, "unsupported"):
            production_live.run_production_signal_check(
                "GBP",
                live_runner=lambda currency: make_live_result(currency),
            )

    def test_reuses_live_result_and_evaluates_only_the_latest_indicator_row(self) -> None:
        live_result = make_live_result()
        received_rows: list[pd.Series] = []

        def evaluator(row: pd.Series):
            received_rows.append(row.copy(deep=True))
            return evaluate_production_signal(row)

        result = production_live.run_production_signal_check(
            "USD",
            live_runner=lambda currency: live_result,
            evaluator=evaluator,
        )

        self.assertIs(result.live_result, live_result)
        self.assertEqual(len(received_rows), 1)
        assert_series_equal(
            received_rows[0],
            live_result.indicator_rows.iloc[-1],
        )
        self.assertEqual(
            result.configuration.configuration_id,
            "sma60_sensitive__balanced",
        )

    def test_default_path_delegates_to_existing_live_pipeline(self) -> None:
        live_result = make_live_result("EUR")

        with (
            patch.object(
                production_live.live,
                "run_live_check",
                return_value=live_result,
            ) as live_runner,
            patch.object(
                production_live,
                "evaluate_production_signal",
                wraps=evaluate_production_signal,
            ) as evaluator,
        ):
            result = production_live.run_production_signal_check("EUR")

        live_runner.assert_called_once_with("EUR")
        evaluator.assert_called_once()
        self.assertEqual(result.live_result.currency, "EUR")

    def test_rejects_a_signal_result_from_a_different_configuration(self) -> None:
        def mismatched_evaluator(row: pd.Series):
            signal_result = evaluate_production_signal(row)
            other_thresholds = replace(
                signal_result.thresholds,
                sma60_good=-0.5,
            )
            return replace(signal_result, thresholds=other_thresholds)

        with self.assertRaisesRegex(ValueError, "production configuration"):
            production_live.run_production_signal_check(
                "USD",
                live_runner=lambda currency: make_live_result(currency),
                evaluator=mismatched_evaluator,
            )

    def test_report_contains_live_indicators_and_explainable_signal_details(self) -> None:
        result = production_live.run_production_signal_check(
            "USD",
            live_runner=lambda currency: make_live_result(currency),
        )

        report = production_live.format_production_signal_report(result)

        for label in (
            "currency",
            "latest data date",
            "current rate",
            "SMA60",
            "SMA120",
            "SMA60_distance_pct",
            "SMA120_distance_pct",
            "percentile_rank_180",
            "BB_lower",
            "production configuration id",
            "Signal",
            "satisfied conditions",
            "signal unavailable indicators",
        ):
            with self.subTest(label=label):
                self.assertRegex(report, rf"(?m)^{label}:")
        for condition_name in (
            "sma60_condition",
            "sma120_condition",
            "percentile_condition",
            "bollinger_condition",
        ):
            with self.subTest(condition=condition_name):
                self.assertRegex(
                    report,
                    rf"(?m)^condition {condition_name}: GOOD ",
                )
        self.assertIn(
            "production configuration id: sma60_sensitive__balanced",
            report,
        )
        self.assertIn("Signal: GOOD", report)

    def test_nan_indicators_are_reported_without_crashing(self) -> None:
        result = production_live.run_production_signal_check(
            "JPY",
            live_runner=lambda currency: make_live_result(
                currency,
                latest_overrides={
                    "SMA60": np.nan,
                    "SMA120": np.nan,
                    "SMA60_distance_pct": np.nan,
                    "SMA120_distance_pct": np.nan,
                    "percentile_rank_180": np.nan,
                    "BB_middle": np.nan,
                    "BB_upper": np.nan,
                    "BB_lower": np.nan,
                },
            ),
        )

        report = production_live.format_production_signal_report(result)

        self.assertIn("SMA60: unavailable", report)
        self.assertIn("condition sma60_condition: UNAVAILABLE", report)
        self.assertIn("condition percentile_condition: UNAVAILABLE", report)
        self.assertIn("Signal: WAIT", report)
        self.assertIn("signal unavailable indicators:", report)

    def test_live_result_frames_and_latest_series_are_not_modified(self) -> None:
        live_result = make_live_result()
        frame_snapshots = {
            name: getattr(live_result, name).copy(deep=True)
            for name in (
                "historical_rates",
                "latest_rates",
                "merged_rates",
                "indicator_rows",
            )
        }
        latest_before = live_result.indicator_rows.iloc[-1].copy(deep=True)

        production_live.run_production_signal_check(
            "USD",
            live_runner=lambda currency: live_result,
        )

        for name, expected in frame_snapshots.items():
            with self.subTest(frame=name):
                assert_frame_equal(getattr(live_result, name), expected)
        assert_series_equal(live_result.indicator_rows.iloc[-1], latest_before)


class ProductionSignalErrorHandlingTests(unittest.TestCase):
    def test_provider_and_contract_failures_never_echo_exception_secrets(self) -> None:
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
            (
                LatestExchangeRateProviderError(f"client failed: {secret}"),
                "client",
            ),
            (
                UnsupportedCurrencyError(f"unsupported: {secret}"),
                "Production v1 currencies",
            ),
            (
                live.LiveCheckDataError(f"empty pipeline: {secret}"),
                "no analyzable data",
            ),
            (ValueError(f"bad indicator: {secret}"), "Signal contract"),
            (TypeError(f"bad result: {secret}"), "Signal contract"),
        )

        for error, expected_text in cases:
            with self.subTest(error=type(error).__name__):
                def failing_runner(
                    currency: str,
                    *,
                    failure: Exception = error,
                ) -> production_live.ProductionSignalCheckResult:
                    raise failure

                stderr = io.StringIO()
                with (
                    patch.object(
                        production_live.live,
                        "load_environment",
                        return_value=False,
                    ),
                    redirect_stderr(stderr),
                ):
                    exit_code = production_live.main(
                        ["--currency", "USD"],
                        runner=failing_runner,
                    )

                error_output = stderr.getvalue()
                self.assertEqual(exit_code, 1)
                self.assertIn(expected_text, error_output)
                self.assertNotIn(secret, error_output)

    def test_empty_and_insufficient_history_have_actionable_messages(self) -> None:
        cases = (
            (live.EmptyHistoricalDataError("USD"), "empty historical data"),
            (
                live.InsufficientHistoricalDataError("JPY", 179),
                "received 179",
            ),
        )

        for error, expected_text in cases:
            with self.subTest(error=type(error).__name__):
                def failing_runner(
                    currency: str,
                    *,
                    failure: Exception = error,
                ) -> production_live.ProductionSignalCheckResult:
                    raise failure

                stderr = io.StringIO()
                with (
                    patch.object(
                        production_live.live,
                        "load_environment",
                        return_value=False,
                    ),
                    redirect_stderr(stderr),
                ):
                    exit_code = production_live.main(
                        ["--currency", "USD"],
                        runner=failing_runner,
                    )

                self.assertEqual(exit_code, 1)
                self.assertIn(expected_text, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
