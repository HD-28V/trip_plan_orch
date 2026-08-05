"""Reusable statistical indicators for historical exchange-rate data.

This module only describes the current rate's position in historical data. It
does not predict future exchange rates or decide when a user should exchange.
Each row is expected to represent one daily observation with ``date`` and
``rate`` columns.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


SMA_SHORT_WINDOW = 60
SMA_LONG_WINDOW = 120
PERCENTILE_WINDOW = 180
BOLLINGER_WINDOW = 20
BOLLINGER_STD_MULTIPLIER = 2.0
BOLLINGER_STD_DDOF = 0

REQUIRED_COLUMNS = frozenset({"date", "rate"})
INDICATOR_COLUMNS: Sequence[str] = (
    "SMA60",
    "SMA120",
    "SMA60_distance_pct",
    "SMA120_distance_pct",
    "percentile_rank_180",
    "BB_middle",
    "BB_upper",
    "BB_lower",
)


def prepare_exchange_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, copy, normalize, and sort a ``date``/``rate`` DataFrame.

    Invalid dates are rejected because chronological order would be ambiguous.
    Missing, non-numeric, and infinite rates become ``NaN`` so indicators can
    remain unavailable safely until a complete valid window exists.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"missing required columns: {missing}")

    result = data.copy()
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("date column contains an invalid date") from error

    if result["date"].isna().any():
        raise ValueError("date column contains a missing date")

    result["rate"] = pd.to_numeric(result["rate"], errors="coerce").astype(float)
    result["rate"] = result["rate"].replace([np.inf, -np.inf], np.nan)
    return result.sort_values("date", kind="stable").reset_index(drop=True)


def calculate_sma(rate: pd.Series, window: int) -> pd.Series:
    """Return a simple moving average after ``window`` valid observations."""
    _validate_window(window)
    return rate.rolling(window=window, min_periods=window).mean()


def calculate_distance_percent(rate: pd.Series, average: pd.Series) -> pd.Series:
    """Return how far the rate is above or below an average, in percent.

    A negative value means the current rate is below the moving average. A zero
    or missing average produces ``NaN`` instead of an infinite result.
    """
    safe_average = average.where(average.ne(0))
    return ((rate - safe_average) / safe_average) * 100


def calculate_percentile_rank(
    rate: pd.Series,
    window: int = PERCENTILE_WINDOW,
) -> pd.Series:
    """Rank each rate within its latest rolling window on a 0-to-100 scale.

    The current observation is included in the window. Ties use their average
    rank, matching pandas' default rank behavior. A complete valid window is
    required; otherwise the result is ``NaN``.
    """
    _validate_window(window)
    return rate.rolling(window=window, min_periods=window).apply(
        _latest_value_percentile_rank,
        raw=True,
    )


def calculate_bollinger_bands(
    rate: pd.Series,
    window: int = BOLLINGER_WINDOW,
    std_multiplier: float = BOLLINGER_STD_MULTIPLIER,
) -> pd.DataFrame:
    """Return middle, upper, and lower Bollinger Bands for a rate series.

    The middle band is the SMA for ``window`` observations. The upper and lower
    bands add or subtract ``std_multiplier`` population standard deviations.
    """
    _validate_window(window)
    if std_multiplier < 0:
        raise ValueError("std_multiplier must be zero or greater")

    middle = calculate_sma(rate, window)
    standard_deviation = rate.rolling(
        window=window,
        min_periods=window,
    ).std(ddof=BOLLINGER_STD_DDOF)

    return pd.DataFrame(
        {
            "BB_middle": middle,
            "BB_upper": middle + std_multiplier * standard_deviation,
            "BB_lower": middle - std_multiplier * standard_deviation,
        },
        index=rate.index,
    )


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Return normalized exchange-rate data with every MVP indicator added."""
    result = prepare_exchange_data(data)

    result["SMA60"] = calculate_sma(result["rate"], SMA_SHORT_WINDOW)
    result["SMA120"] = calculate_sma(result["rate"], SMA_LONG_WINDOW)
    result["SMA60_distance_pct"] = calculate_distance_percent(
        result["rate"],
        result["SMA60"],
    )
    result["SMA120_distance_pct"] = calculate_distance_percent(
        result["rate"],
        result["SMA120"],
    )
    result["percentile_rank_180"] = calculate_percentile_rank(result["rate"])

    bollinger_bands = calculate_bollinger_bands(result["rate"])
    for column in bollinger_bands.columns:
        result[column] = bollinger_bands[column]

    return result


def _latest_value_percentile_rank(values: np.ndarray) -> float:
    """Return the average rank of the final value as a percentage."""
    current = values[-1]
    count_below = np.count_nonzero(values < current)
    count_equal = np.count_nonzero(values == current)
    average_rank = count_below + (count_equal + 1) / 2
    return float(average_rank / len(values) * 100)


def _validate_window(window: int) -> None:
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
