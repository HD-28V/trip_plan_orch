import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scripts import run_threshold_calibration as cli
from src.calibration import run_calibration
from src.calibration_candidates import (
    CandidateConfigurationSpec,
    PolicyCandidate,
    ThresholdCandidate,
    build_candidate_plan,
)
from src.providers import ExchangeRateProviderError
from src.signal_engine import SignalDecisionPolicy, SignalThresholds


def make_rates(scale: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=6),
            "rate": np.array([100, 101, 99, 102, 98, 103], dtype=float) * scale,
        }
    )


def fixed_indicator_calculator(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy(deep=True)
    rate = pd.to_numeric(result["rate"], errors="coerce").astype(float)
    result["rate"] = rate
    result["SMA60"] = rate
    result["SMA120"] = rate
    result["SMA60_distance_pct"] = 0.0
    result["SMA120_distance_pct"] = 0.0
    result["percentile_rank_180"] = 50.0
    result["BB_middle"] = rate
    result["BB_upper"] = rate
    result["BB_lower"] = rate
    return result


def make_plan():
    thresholds = ThresholdCandidate(
        "threshold",
        SignalThresholds(-2, -4, -2, -4, 25, 10, 1),
    )
    policy = PolicyCandidate(
        "policy",
        SignalDecisionPolicy(3, 1, 3, 4, 3),
    )
    specification = CandidateConfigurationSpec(
        "threshold__policy",
        "threshold",
        "policy",
    )
    return build_candidate_plan(
        (thresholds,),
        (policy,),
        (specification,),
        max_candidate_count=1,
    )


def make_report():
    source = make_rates()
    return run_calibration(
        {"USD": source},
        validation_start=source.loc[3, "date"],
        candidate_plan=make_plan(),
        horizons=(1,),
        indicator_calculator=fixed_indicator_calculator,
    )


def valid_arguments(*extra: str) -> list[str]:
    return [
        "--currencies",
        "USD",
        "JPY",
        "EUR",
        "--start-date",
        "2018-01-01",
        "--end-date",
        "2026-08-06",
        "--validation-start",
        "2024-01-01",
        "--output-dir",
        "results/calibration",
        *extra,
    ]


class CalibrationCliArgumentTests(unittest.TestCase):
    def test_parses_and_normalizes_required_arguments(self) -> None:
        arguments = cli.parse_arguments(
            [
                "--currencies",
                "usd",
                "jPy",
                "--start-date",
                "2018-01-01",
                "--end-date",
                "2026-08-06",
                "--validation-start",
                "2024-01-01",
                "--output-dir",
                "reports",
            ]
        )

        self.assertEqual(arguments.currencies, ("USD", "JPY"))
        self.assertEqual(arguments.start_date, pd.Timestamp("2018-01-01"))
        self.assertEqual(arguments.end_date, pd.Timestamp("2026-08-06"))
        self.assertEqual(arguments.validation_start, pd.Timestamp("2024-01-01"))
        self.assertEqual(arguments.output_dir, Path("reports"))
        self.assertEqual(arguments.horizons, (5, 10, 20, 60))

    def test_calibration_end_becomes_next_validation_calendar_date(self) -> None:
        arguments = cli.parse_arguments(
            [
                "--currencies",
                "USD",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2025-12-31",
                "--calibration-end",
                "2023-12-31",
                "--output-dir",
                "reports",
                "--horizons",
                "5",
                "20",
            ]
        )

        self.assertEqual(arguments.validation_start, pd.Timestamp("2024-01-01"))
        self.assertEqual(arguments.horizons, (5, 20))

    def test_rejects_unsupported_duplicate_and_malformed_cli_values(self) -> None:
        invalid_argv = (
            [
                "--currencies",
                "CHF",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2025-01-01",
                "--validation-start",
                "2024-01-01",
                "--output-dir",
                "reports",
            ],
            [
                "--currencies",
                "USD",
                "usd",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2025-01-01",
                "--validation-start",
                "2024-01-01",
                "--output-dir",
                "reports",
            ],
            [
                "--currencies",
                "USD",
                "--start-date",
                "2025-01-01",
                "--end-date",
                "2020-01-01",
                "--validation-start",
                "2024-01-01",
                "--output-dir",
                "reports",
            ],
            [
                "--currencies",
                "USD",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2025-01-01",
                "--validation-start",
                "2020-01-01",
                "--output-dir",
                "reports",
            ],
            [
                "--currencies",
                "USD",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2025-01-01",
                "--validation-start",
                "2024-01-01",
                "--calibration-end",
                "2023-12-31",
                "--output-dir",
                "reports",
            ],
            [
                "--currencies",
                "USD",
                "--start-date",
                "2020/01/01",
                "--end-date",
                "2025-01-01",
                "--validation-start",
                "2024-01-01",
                "--output-dir",
                "reports",
            ],
            [
                "--currencies",
                "USD",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2025-01-01",
                "--validation-start",
                "2024-01-01",
                "--output-dir",
                "reports",
                "--horizons",
                "5",
                "5",
            ],
        )
        for argv in invalid_argv:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as context:
                    cli.parse_arguments(argv)
                self.assertEqual(context.exception.code, 2)


class FakeHistoricalProvider:
    def __init__(self, data_by_currency: dict[str, pd.DataFrame]) -> None:
        self.data_by_currency = data_by_currency
        self.calls: list[tuple[str, object, object]] = []

    def fetch_daily_rates(
        self,
        currency: str,
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        self.calls.append((currency, start_date, end_date))
        return self.data_by_currency[currency]


class CalibrationCliWorkflowTests(unittest.TestCase):
    def test_workflow_fetches_each_currency_once_without_mutating_provider_data(self) -> None:
        data = {
            "USD": make_rates(13.0),
            "JPY": make_rates(0.09),
            "EUR": make_rates(15.0),
        }
        before = {currency: frame.copy(deep=True) for currency, frame in data.items()}
        provider = FakeHistoricalProvider(data)

        report = cli.run_calibration_workflow(
            currencies=("USD", "JPY", "EUR"),
            start_date=pd.Timestamp("2025-01-01"),
            end_date=pd.Timestamp("2025-01-31"),
            validation_start=data["USD"].loc[3, "date"],
            candidate_plan=make_plan(),
            horizons=(1,),
            historical_provider=provider,
        )

        self.assertEqual(
            [call[0] for call in provider.calls],
            ["USD", "JPY", "EUR"],
        )
        self.assertEqual(report.currencies, ("USD", "JPY", "EUR"))
        for currency in data:
            assert_frame_equal(data[currency], before[currency])

    def test_main_prints_candidate_count_before_injected_runner_and_writer(self) -> None:
        report = make_report()
        events: list[str] = []
        captured: dict[str, object] = {}

        def fake_runner(**kwargs: object):
            events.append("runner")
            captured.update(kwargs)
            return report

        def fake_writer(received_report: object, output_dir: object):
            events.append("writer")
            self.assertIs(received_report, report)
            self.assertEqual(output_dir, Path("results/calibration"))
            return (Path("results/calibration/configuration_summary.csv"),)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main(
                valid_arguments(),
                runner=fake_runner,
                writer=fake_writer,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(events, ["runner", "writer"])
        self.assertEqual(captured["currencies"], ("USD", "JPY", "EUR"))
        self.assertEqual(captured["horizons"], (5, 10, 20, 60))
        output = stdout.getvalue()
        self.assertLess(
            output.index("candidate configuration count: 12"),
            output.index("validation start:"),
        )
        self.assertIn("configuration selection: none", output)

    def test_candidate_limit_failure_occurs_before_runner(self) -> None:
        runner = Mock(side_effect=AssertionError("runner must not be called"))
        stderr = io.StringIO()

        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            exit_code = cli.main(
                valid_arguments("--max-candidates", "11"),
                runner=runner,
            )

        self.assertEqual(exit_code, 1)
        runner.assert_not_called()
        self.assertIn(
            "candidate configuration count 12 exceeds max_candidate_count 11",
            stderr.getvalue(),
        )

    def test_main_hides_provider_error_details(self) -> None:
        secret = "network-detail-that-must-not-appear"

        def failing_runner(**kwargs: object):
            raise ExchangeRateProviderError(secret)

        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            exit_code = cli.main(valid_arguments(), runner=failing_runner)

        self.assertEqual(exit_code, 1)
        self.assertIn("Yahoo Finance", stderr.getvalue())
        self.assertNotIn(secret, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
