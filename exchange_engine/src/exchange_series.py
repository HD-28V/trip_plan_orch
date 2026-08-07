"""Pure helpers for joining historical and latest exchange-rate data."""

import numpy as np
import pandas as pd


def merge_historical_and_latest(
    historical_rates: pd.DataFrame,
    latest_rates: pd.DataFrame,
    foreign_currency: str,
) -> pd.DataFrame:
    """Return one date/rate series where a same-day latest value wins.

    Historical input follows the date/rate contract. Latest input follows the
    date/currency/rate contract and may contain several currencies; only the
    explicitly requested currency is selected. Neither input is modified.
    """
    currency = _normalize_currency_code(foreign_currency)
    _require_columns(historical_rates, {"date", "rate"}, "historical_rates")
    _require_columns(
        latest_rates,
        {"date", "currency", "rate"},
        "latest_rates",
    )

    historical = historical_rates.copy(deep=True)
    latest = latest_rates.copy(deep=True)

    if "currency" in historical.columns:
        historical_currencies = historical["currency"].map(
            _normalize_currency_code
        )
        if not historical_currencies.eq(currency).all():
            raise ValueError(
                "historical_rates contains a different currency"
            )

    latest["currency"] = latest["currency"].map(_normalize_currency_code)
    latest = latest.loc[latest["currency"].eq(currency)].copy()
    if latest.empty:
        raise ValueError(
            f"latest_rates does not contain requested currency: {currency}"
        )

    historical = _normalize_rate_frame(
        historical.loc[:, ["date", "rate"]],
        "historical_rates",
    )
    latest_for_currency = _normalize_rate_frame(
        latest.loc[:, ["date", "rate"]],
        "latest_rates",
    )

    # Historical rows come first, so keep=last gives a duplicate day to latest.
    combined = pd.concat(
        [historical, latest_for_currency],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset="date", keep="last")
    return combined.sort_values("date", kind="stable").reset_index(drop=True)


def _require_columns(
    data: pd.DataFrame,
    required: set[str],
    frame_name: str,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{frame_name} must be a pandas DataFrame")
    missing = required.difference(data.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{frame_name} is missing columns: {missing_text}")


def _normalize_rate_frame(data: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    result = data.copy(deep=True)
    parsed_dates = pd.to_datetime(
        result["date"],
        errors="coerce",
        format="mixed",
        utc=True,
    )
    if parsed_dates.isna().any():
        raise ValueError(f"{frame_name} contains an invalid date")
    result["date"] = parsed_dates.dt.tz_convert(None).dt.normalize()

    numeric_rates = pd.to_numeric(result["rate"], errors="coerce")
    invalid_rates = (
        numeric_rates.isna()
        | ~np.isfinite(numeric_rates)
        | numeric_rates.le(0)
    )
    if invalid_rates.any():
        raise ValueError(f"{frame_name} contains an invalid rate")
    result["rate"] = numeric_rates.astype(float)
    return result


def _normalize_currency_code(currency: object) -> str:
    if not isinstance(currency, str):
        raise ValueError("currency must be a string")
    normalized = currency.strip().upper()
    if len(normalized) != 3 or not normalized.isascii() or not normalized.isalpha():
        raise ValueError("currency must be a three-letter code")
    return normalized
