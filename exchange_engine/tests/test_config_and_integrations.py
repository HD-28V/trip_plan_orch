import os
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pandas as pd

from scripts.check_integrations import collect_integration_status
from src.config import (
    MissingConfigurationError,
    get_environment_flag,
    require_environment_variable,
)
from src.indicators import calculate_indicators
from src.integrations.google_places import GooglePlacesClient
from src.integrations.kakao_message import KakaoMessageClient
from src.integrations.langsmith_config import get_langsmith_settings
from src.integrations.maps_url import build_google_maps_directions_url


class ConfigurationTests(unittest.TestCase):
    def test_missing_secret_error_contains_name_but_no_value(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingConfigurationError) as context:
                require_environment_variable("GEMINI_API_KEY")

        self.assertIn("GEMINI_API_KEY", str(context.exception))
        self.assertNotIn("dummy-secret-value", str(context.exception))

    def test_analysis_still_works_without_environment_variables(self) -> None:
        data = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=180, freq="D"),
                "rate": range(1300, 1480),
            }
        )
        with patch.dict(os.environ, {}, clear=True):
            result = calculate_indicators(data)

        self.assertFalse(pd.isna(result.iloc[-1]["percentile_rank_180"]))

    def test_langsmith_is_disabled_without_key_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = get_langsmith_settings()

        self.assertFalse(settings.tracing_enabled)
        self.assertEqual(settings.project, "budgettrip-ai")

    def test_boolean_environment_parser_rejects_unknown_value(self) -> None:
        with patch.dict(os.environ, {"FLAG": "sometimes"}, clear=True):
            with self.assertRaisesRegex(ValueError, "FLAG"):
                get_environment_flag("FLAG")

    def test_status_report_never_contains_secret_values(self) -> None:
        secret = "dummy-secret-that-must-not-be-printed"
        configured = {
            "EXCHANGE_RATE_API_KEY": secret,
            "NVIDIA_API_KEY": secret,
            "NVIDIA_BASE_URL": "https://example.invalid/v1",
            "NVIDIA_MODEL": "dummy-model",
            "GEMINI_API_KEY": secret,
            "GEMINI_MODEL": "dummy-model",
            "LANGSMITH_API_KEY": secret,
            "LANGSMITH_TRACING": "true",
            "GOOGLE_PLACES_API_KEY": secret,
            "KAKAO_REST_API_KEY": secret,
            "KAKAO_ACCESS_TOKEN": secret,
        }

        with patch.dict(os.environ, configured, clear=True):
            report = "\n".join(collect_integration_status())

        self.assertNotIn(secret, report)
        self.assertIn("[READY] EXCHANGE_RATE_API_KEY", report)
        self.assertIn("[READY] LangSmith tracing", report)


class IntegrationSkeletonTests(unittest.TestCase):
    def test_builds_google_maps_url_without_api_key(self) -> None:
        url = build_google_maps_directions_url(
            "Seoul Station",
            "Incheon Airport",
            travel_mode="transit",
            waypoints=["Hongdae"],
        )
        parsed = urlparse(url)
        query = parse_qs(parsed.query)

        self.assertEqual(parsed.netloc, "www.google.com")
        self.assertEqual(query["api"], ["1"])
        self.assertEqual(query["origin"], ["Seoul Station"])
        self.assertEqual(query["destination"], ["Incheon Airport"])
        self.assertEqual(query["waypoints"], ["Hongdae"])
        self.assertNotIn("key", query)

    def test_paid_and_message_skeletons_do_not_call_network(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingConfigurationError):
                GooglePlacesClient().search_places("museum")
            with self.assertRaises(MissingConfigurationError):
                KakaoMessageClient().send_message("exchange alert")


if __name__ == "__main__":
    unittest.main()
