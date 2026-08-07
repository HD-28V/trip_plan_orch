"""Lazy OpenAI-compatible NVIDIA client construction."""

from dataclasses import dataclass, field
from typing import Any

from src.config import require_environment_variable


@dataclass(frozen=True)
class NvidiaClientBundle:
    """Hold a configured client and centrally selected model name."""

    client: Any = field(repr=False)
    model: str


def create_nvidia_client() -> NvidiaClientBundle:
    """Create a client only after every required NVIDIA setting is present."""
    api_key = require_environment_variable("NVIDIA_API_KEY")
    base_url = require_environment_variable("NVIDIA_BASE_URL")
    model = require_environment_variable("NVIDIA_MODEL")

    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package is required for NVIDIA integration") from None

    client = OpenAI(api_key=api_key, base_url=base_url)
    return NvidiaClientBundle(client=client, model=model)
