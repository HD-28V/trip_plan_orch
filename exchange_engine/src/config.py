"""Environment configuration helpers with lazy secret validation.

Importing this module never requires an API key and never prints secret values.
Applications may explicitly call load_environment to load a local environment
file through python-dotenv. Tests and analysis-only workflows do not need it.
"""

import os
from pathlib import Path


class MissingConfigurationError(RuntimeError):
    """Raised only when an operation needs a missing environment variable."""


def load_environment(
    dotenv_path: str | Path | None = None,
    *,
    override: bool = False,
) -> bool:
    """Load environment variables through python-dotenv when it is installed.

    Returning False when python-dotenv is unavailable keeps local analysis and
    unit tests usable before optional integrations are installed.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    return bool(load_dotenv(dotenv_path=dotenv_path, override=override))


def get_optional_environment_variable(
    name: str,
    default: str | None = None,
) -> str | None:
    """Return a stripped non-empty environment value without exposing it."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def require_environment_variable(name: str) -> str:
    """Return a required value or raise an error containing only its name."""
    value = get_optional_environment_variable(name)
    if value is None:
        raise MissingConfigurationError(
            f"required environment variable is missing: {name}"
        )
    return value


def get_environment_flag(name: str, *, default: bool = False) -> bool:
    """Parse a conventional boolean environment variable."""
    value = get_optional_environment_variable(name)
    if value is None:
        return default

    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"environment variable must be a boolean: {name}")
