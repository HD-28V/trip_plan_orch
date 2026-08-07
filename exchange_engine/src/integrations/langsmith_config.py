"""Optional LangSmith tracing settings without mandatory connectivity."""

from dataclasses import dataclass

from src.config import (
    get_environment_flag,
    get_optional_environment_variable,
    require_environment_variable,
)


@dataclass(frozen=True)
class LangSmithSettings:
    tracing_enabled: bool
    project: str


def get_langsmith_settings() -> LangSmithSettings:
    """Return tracing settings and require a key only when tracing is enabled."""
    tracing_enabled = get_environment_flag("LANGSMITH_TRACING", default=False)
    project = get_optional_environment_variable(
        "LANGSMITH_PROJECT",
        "budgettrip-ai",
    )
    if tracing_enabled:
        require_environment_variable("LANGSMITH_API_KEY")
    return LangSmithSettings(
        tracing_enabled=tracing_enabled,
        project=project or "budgettrip-ai",
    )
