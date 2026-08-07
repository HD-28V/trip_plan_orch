"""Lazy Google Gemini client construction."""

from dataclasses import dataclass, field
from typing import Any

from src.config import require_environment_variable


@dataclass(frozen=True)
class GeminiClientBundle:
    """Hold a configured Gemini client and centrally selected model name."""

    client: Any = field(repr=False)
    model: str


def create_gemini_client() -> GeminiClientBundle:
    """Create a Gemini client only when its key and model are configured."""
    api_key = require_environment_variable("GEMINI_API_KEY")
    model = require_environment_variable("GEMINI_MODEL")

    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai package is required for Gemini integration") from None

    client = genai.Client(api_key=api_key)
    return GeminiClientBundle(client=client, model=model)
