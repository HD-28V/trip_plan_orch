import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts import analyze_calibration_results as cli
from src.calibration_analysis import CalibrationAnalysisError


def make_report() -> SimpleNamespace:
    return SimpleNamespace(
        configuration_review=pd.DataFrame(
            {
                "configuration_id": ("config_a", "config_b", "config_c"),
                "review_group": ("PROMISING", "MIXED", "MIXED"),
            }
        ),
        candidate_shortlist=pd.DataFrame(
            {
                "configuration_id": ("config_a", "config_c"),
            }
        ),
        shortlist_status="READY_FOR_HUMAN_REVIEW",
    )


class CalibrationAnalysisCliArgumentTests(unittest.TestCase):
    def test_requires_input_and_defaults_output_to_analysis_subdirectory(self) -> None:
        arguments = cli.parse_arguments(
            ["--input-dir", "results/calibration_run"]
        )

        self.assertEqual(arguments.input_dir, Path("results/calibration_run"))
        self.assertEqual(
            arguments.output_dir,
            Path("results/calibration_run/analysis"),
        )

    def test_accepts_an_explicit_separate_output_directory(self) -> None:
        arguments = cli.parse_arguments(
            [
                "--input-dir",
                "results/calibration_run",
                "--output-dir",
                "results/review_run",
            ]
        )

        self.assertEqual(arguments.output_dir, Path("results/review_run"))

    def test_rejects_missing_input_and_output_equal_to_input(self) -> None:
        invalid_arguments = (
            [],
            [
                "--input-dir",
                "results/calibration_run",
                "--output-dir",
                "results/calibration_run",
            ],
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as context:
                    cli.parse_arguments(arguments)
                self.assertEqual(context.exception.code, 2)


class CalibrationAnalysisCliWorkflowTests(unittest.TestCase):
    def test_main_uses_injected_offline_workflow_and_prints_unranked_summary(
        self,
    ) -> None:
        report = make_report()
        tables = object()
        heuristics = object()
        events: list[str] = []

        def fake_loader(input_dir: object) -> object:
            events.append("loader")
            self.assertEqual(input_dir, Path("results/calibration_run"))
            return tables

        def fake_analyzer(
            received_tables: object,
            *,
            heuristics: object,
        ) -> object:
            events.append("analyzer")
            self.assertIs(received_tables, tables)
            self.assertIs(heuristics, test_heuristics)
            return report

        def fake_writer(
            received_report: object,
            output_dir: object,
        ) -> tuple[Path, ...]:
            events.append("writer")
            self.assertIs(received_report, report)
            self.assertEqual(
                output_dir,
                Path("results/calibration_run/analysis"),
            )
            return (
                Path("results/calibration_run/analysis/configuration_review.csv"),
                Path("results/calibration_run/analysis/analysis_notes.md"),
            )

        test_heuristics = heuristics
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main(
                ["--input-dir", "results/calibration_run"],
                loader=fake_loader,
                analyzer=fake_analyzer,
                writer=fake_writer,
                heuristics=test_heuristics,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(events, ["loader", "analyzer", "writer"])
        output = stdout.getvalue()
        self.assertIn("configuration review count: 3", output)
        self.assertIn("PROMISING: 1", output)
        self.assertIn("MIXED: 2", output)
        self.assertIn("review shortlist count: 2", output)
        self.assertIn("shortlist status: READY_FOR_HUMAN_REVIEW", output)
        self.assertIn("- config_a", output)
        self.assertIn("- config_c", output)
        self.assertIn("review shortlist (unranked)", output)
        self.assertIn("production configuration selection: none", output)

    def test_format_report_accepts_analysis_group_and_enum_like_status(self) -> None:
        report = make_report()
        report.configuration_review = report.configuration_review.rename(
            columns={"review_group": "analysis_group"}
        )
        report.shortlist_status = SimpleNamespace(value="HUMAN_REVIEW_REQUIRED")

        output = cli.format_analysis_report(report, ())

        self.assertIn("analysis reports:", output)
        self.assertIn("shortlist status: HUMAN_REVIEW_REQUIRED", output)

    def test_main_hides_internal_error_details(self) -> None:
        secret = "internal-analysis-detail-that-must-not-appear"

        def analysis_failure(input_dir: object) -> object:
            raise CalibrationAnalysisError(secret)

        def missing_file(input_dir: object) -> object:
            raise FileNotFoundError(secret)

        def invalid_contract(input_dir: object) -> object:
            raise ValueError(secret)

        failures = (
            (analysis_failure, "failed schema"),
            (missing_file, "required calibration result CSV"),
            (invalid_contract, "violated their contract"),
        )
        for loader, expected_message in failures:
            with self.subTest(loader=loader.__name__):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    exit_code = cli.main(
                        ["--input-dir", "results/calibration_run"],
                        loader=loader,
                        heuristics=object(),
                    )

                error_output = stderr.getvalue()
                self.assertEqual(exit_code, 1)
                self.assertIn(expected_message, error_output)
                self.assertNotIn(secret, error_output)

    def test_main_reports_existing_or_unwritable_outputs_safely(self) -> None:
        report = make_report()
        secret = "private-output-detail"

        def fake_loader(input_dir: object) -> object:
            return object()

        def fake_analyzer(tables: object, *, heuristics: object) -> object:
            return report

        failures = (
            (FileExistsError(secret), "already exists"),
            (OSError(secret), "could not be read or written"),
        )
        for failure, expected_message in failures:
            with self.subTest(failure=type(failure).__name__):
                def failing_writer(
                    received_report: object,
                    output_dir: object,
                    *,
                    error: Exception = failure,
                ) -> tuple[Path, ...]:
                    raise error

                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    exit_code = cli.main(
                        ["--input-dir", "results/calibration_run"],
                        loader=fake_loader,
                        analyzer=fake_analyzer,
                        writer=failing_writer,
                        heuristics=object(),
                    )

                error_output = stderr.getvalue()
                self.assertEqual(exit_code, 1)
                self.assertIn(expected_message, error_output)
                self.assertNotIn(secret, error_output)

    def test_cli_has_no_provider_network_or_environment_dependencies(self) -> None:
        forbidden_globals = (
            "YFinanceExchangeRateProvider",
            "ExchangeRateProvider",
            "load_environment",
            "ExchangeRateApiProvider",
        )
        for name in forbidden_globals:
            with self.subTest(name=name):
                self.assertNotIn(name, vars(cli))


if __name__ == "__main__":
    unittest.main()
