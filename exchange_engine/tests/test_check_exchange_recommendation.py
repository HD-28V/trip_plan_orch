import contextlib
import io
import unittest
from decimal import Decimal
from unittest.mock import patch

import pandas as pd

from scripts import check_exchange_live as live
from scripts import check_exchange_recommendation as recommendation_cli
from src.recommendation_engine import RecommendationValidationError


def make_live_result(currency: str = "USD") -> live.LiveCheckResult:
    row = {
        "date": pd.Timestamp("2026-08-07"),
        "rate": 100.0,
        "SMA60": 101.0,
        "SMA120": 102.0,
        "SMA60_distance_pct": -1.5,
        "SMA120_distance_pct": -3.0,
        "percentile_rank_180": 20.0,
        "BB_middle": 105.0,
        "BB_upper": 110.0,
        "BB_lower": 100.0,
    }
    frame = pd.DataFrame([row])
    return live.LiveCheckResult(
        currency=currency,
        historical_rates=frame[["date", "rate"]].copy(),
        latest_rates=pd.DataFrame({"date": [row["date"]], "currency": [currency], "rate": [100.0]}),
        merged_rates=frame[["date", "rate"]].copy(),
        indicator_rows=frame,
    )


class RecommendationCliTests(unittest.TestCase):
    def test_parser_normalizes_and_validates_all_arguments(self) -> None:
        parser = recommendation_cli.build_argument_parser()
        parsed = parser.parse_args([
            "--currency", " jpy ", "--days-until-departure", "30",
            "--total-target-krw", "1000000", "--already-exchanged-krw", "300000",
        ])
        self.assertEqual(parsed.currency, "JPY")
        self.assertEqual(parsed.total_target_krw, Decimal("1000000"))
        for argv in (
            ["--currency", "GBP", "--days-until-departure", "30", "--total-target-krw", "1", "--already-exchanged-krw", "0"],
            ["--currency", "USD", "--days-until-departure", "-1", "--total-target-krw", "1", "--already-exchanged-krw", "0"],
            ["--currency", "USD", "--days-until-departure", "1", "--total-target-krw", "NaN", "--already-exchanged-krw", "0"],
            ["--currency", "USD", "--days-until-departure", "1", "--total-target-krw", "0", "--already-exchanged-krw", "0"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    parser.parse_args(argv)

    def test_stubbed_live_pipeline_runs_without_network_and_formats_required_fields(self) -> None:
        calls: list[str] = []

        def fake_live_runner(currency: str) -> live.LiveCheckResult:
            calls.append(currency)
            return make_live_result(currency)

        result = recommendation_cli.run_live_recommendation_check(
            "eur", 30, Decimal("1000000"), Decimal("300000"), live_runner=fake_live_runner
        )
        report = recommendation_cli.format_recommendation_report(result)
        self.assertEqual(calls, ["EUR"])
        self.assertIn("currency: EUR", report)
        self.assertIn("base market signal: GOOD", report)
        self.assertIn("adjusted signal: STRONG", report)
        self.assertIn("recommended additional KRW: 450000", report)
        self.assertIn("unavailable indicators: none", report)

    def test_main_rejects_already_exchanged_above_total_before_live_runner(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = recommendation_cli.main(
                ["--currency", "USD", "--days-until-departure", "30", "--total-target-krw", "100", "--already-exchanged-krw", "101"],
                runner=lambda *_: self.fail("live runner must not run"),
            )
        self.assertEqual(code, 1)
        self.assertIn("must not exceed", stderr.getvalue())

    def test_main_hides_secret_like_exception_text(self) -> None:
        secret = "dummy-secret-that-must-not-appear"
        stderr = io.StringIO()
        with patch.object(recommendation_cli.live, "load_environment", return_value=None):
            with contextlib.redirect_stderr(stderr):
                code = recommendation_cli.main(
                    ["--currency", "USD", "--days-until-departure", "30", "--total-target-krw", "100", "--already-exchanged-krw", "0"],
                    runner=lambda *_: (_ for _ in ()).throw(RecommendationValidationError(secret)),
                )
        self.assertEqual(code, 1)
        self.assertNotIn(secret, stderr.getvalue())
        self.assertIn("established contract", stderr.getvalue())

    def test_main_prioritizes_provider_errors_over_generic_value_error(self) -> None:
        cases = (
            (
                live.InvalidExchangeRateDataError("malformed historical data"),
                "Yahoo Finance returned malformed historical exchange-rate data.",
            ),
            (
                live.LatestExchangeRateResponseError("malformed latest data"),
                "ExchangeRate-API returned a malformed or unsuccessful response.",
            ),
            (
                ValueError("generic contract failure"),
                "Recommendation inputs or production results violate the established contract.",
            ),
        )
        argv = [
            "--currency", "USD", "--days-until-departure", "30",
            "--total-target-krw", "100", "--already-exchanged-krw", "0",
        ]

        for error, expected_message in cases:
            with self.subTest(error_type=type(error).__name__):
                stderr = io.StringIO()
                with patch.object(
                    recommendation_cli.live,
                    "load_environment",
                    return_value=None,
                ):
                    with contextlib.redirect_stderr(stderr):
                        code = recommendation_cli.main(
                            argv,
                            runner=lambda *_args, _error=error: (
                                (_ for _ in ()).throw(_error)
                            ),
                        )
                self.assertEqual(code, 1)
                self.assertIn(expected_message, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
