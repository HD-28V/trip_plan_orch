"""Pure Google Maps direction URL generation without an API key."""

from collections.abc import Iterable
from urllib.parse import urlencode


GOOGLE_MAPS_DIRECTIONS_URL = "https://www.google.com/maps/dir/"
SUPPORTED_TRAVEL_MODES = frozenset(
    {"driving", "walking", "bicycling", "transit"}
)


def build_google_maps_directions_url(
    origin: str,
    destination: str,
    *,
    travel_mode: str = "transit",
    waypoints: Iterable[str] | None = None,
) -> str:
    """Build a browser directions URL without calling a Google API."""
    normalized_origin = origin.strip()
    normalized_destination = destination.strip()
    normalized_mode = travel_mode.strip().lower()

    if not normalized_origin:
        raise ValueError("origin must not be empty")
    if not normalized_destination:
        raise ValueError("destination must not be empty")
    if normalized_mode not in SUPPORTED_TRAVEL_MODES:
        raise ValueError(f"unsupported travel_mode: {travel_mode}")

    parameters: dict[str, str] = {
        "api": "1",
        "origin": normalized_origin,
        "destination": normalized_destination,
        "travelmode": normalized_mode,
    }
    if waypoints is not None:
        normalized_waypoints = [
            waypoint.strip()
            for waypoint in waypoints
            if waypoint.strip()
        ]
        if normalized_waypoints:
            parameters["waypoints"] = "|".join(normalized_waypoints)

    return f"{GOOGLE_MAPS_DIRECTIONS_URL}?{urlencode(parameters)}"
