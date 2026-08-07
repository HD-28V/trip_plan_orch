"""Interfaces for a future Google Places integration.

No paid Google Places request is implemented at this stage.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.config import require_environment_variable


class PlaceSearchClient(ABC):
    @abstractmethod
    def search_places(self, query: str) -> Sequence[dict[str, object]]:
        """Search places without prescribing an API response model."""


class GooglePlacesClient(PlaceSearchClient):
    """Configuration-aware placeholder for a future paid API adapter."""

    def search_places(self, query: str) -> Sequence[dict[str, object]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        require_environment_variable("GOOGLE_PLACES_API_KEY")
        raise NotImplementedError(
            "Google Places network calls are not implemented yet"
        )
