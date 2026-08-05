"""Report integration readiness without revealing secrets or calling APIs."""

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    get_environment_flag,
    get_optional_environment_variable,
    load_environment,
)


def collect_integration_status() -> list[str]:
    """Return readiness lines based only on local packages and configuration."""
    statuses: list[str] = []
    if importlib.util.find_spec("yfinance") is None:
        statuses.append("[MISSING] yfinance package")
    else:
        statuses.append("[READY] yfinance")

    for variable_name in (
        "EXCHANGE_RATE_API_KEY",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "NVIDIA_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GOOGLE_PLACES_API_KEY",
        "KAKAO_REST_API_KEY",
        "KAKAO_ACCESS_TOKEN",
    ):
        if get_optional_environment_variable(variable_name) is None:
            statuses.append(f"[MISSING] {variable_name}")
        else:
            statuses.append(f"[READY] {variable_name}")

    try:
        tracing_enabled = get_environment_flag(
            "LANGSMITH_TRACING",
            default=False,
        )
    except ValueError:
        statuses.append("[INVALID] LANGSMITH_TRACING")
    else:
        if not tracing_enabled:
            statuses.append("[DISABLED] LangSmith tracing")
        elif get_optional_environment_variable("LANGSMITH_API_KEY") is None:
            statuses.append("[MISSING] LANGSMITH_API_KEY")
        else:
            statuses.append("[READY] LangSmith tracing")

    statuses.append("[READY] Google Maps URL")
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check integration configuration without exposing secrets.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Reserved for future opt-in live connectivity checks.",
    )
    arguments = parser.parse_args()

    load_environment()
    for status in collect_integration_status():
        print(status)
    if arguments.live:
        print("[DISABLED] live connectivity checks are not implemented")


if __name__ == "__main__":
    main()
