"""Exchange-rate data providers that are independent from analysis logic.

Every provider returns one normalized daily series with exactly two columns:
``date`` and ``rate``. ``rate`` always means the KRW required to buy one unit
of the requested foreign currency.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    MissingConfigurationError,
    get_optional_environment_variable,
)


STANDARD_COLUMNS = ("date", "rate")
LATEST_STANDARD_COLUMNS = ("date", "currency", "rate")
SUPPORTED_FOREIGN_CURRENCIES = frozenset(
    {"USD", "JPY", "EUR", "GBP", "CNY", "AUD", "CAD"}
)

DateInput = date | datetime | str | pd.Timestamp


@dataclass(frozen=True)
class YFinanceTickerConfig:
    """Describe how a Yahoo Finance quote becomes KRW per foreign unit."""

    ticker: str
    invert: bool = False
    foreign_units_per_rate: float = 1.0


YFINANCE_TICKERS: dict[str, YFinanceTickerConfig] = {
    "USD": YFinanceTickerConfig("KRW=X"),
    "JPY": YFinanceTickerConfig("JPYKRW=X"),
    "EUR": YFinanceTickerConfig("EURKRW=X"),
    "GBP": YFinanceTickerConfig("GBPKRW=X"),
    "CNY": YFinanceTickerConfig("CNYKRW=X"),
    "AUD": YFinanceTickerConfig("AUDKRW=X"),
    "CAD": YFinanceTickerConfig("CADKRW=X"),
}


class ExchangeRateProviderError(Exception):
    """Base exception for provider-layer failures."""


class UnsupportedCurrencyError(ExchangeRateProviderError, ValueError):
    """Raised when a provider cannot supply the requested currency."""


class InvalidExchangeRateDataError(ExchangeRateProviderError, ValueError):
    """Raised when source data cannot satisfy the standard data contract."""


class LatestExchangeRateProviderError(Exception):
    """Base exception for latest-rate provider failures."""


class LatestExchangeRateNetworkError(LatestExchangeRateProviderError):
    """Raised when the latest-rate HTTP request cannot be completed."""


class LatestExchangeRateResponseError(LatestExchangeRateProviderError, ValueError):
    """Raised when a latest-rate response violates the expected contract."""


class ExchangeRateProvider(ABC):
    """Abstract source of normalized daily exchange rates."""

    @abstractmethod
    def fetch_daily_rates(
        self,
        foreign_currency: str,
        start_date: DateInput,
        end_date: DateInput,
    ) -> pd.DataFrame:
        """Return inclusive daily rates as sorted ``date``/``rate`` columns.

        ``rate`` must be normalized to KRW per one unit of ``foreign_currency``.
        Implementations own collection, validation, normalization, duplicate
        removal, and date filtering. They must not calculate indicators or make
        exchange recommendations.
        """


class LatestExchangeRateProvider(ABC):
    """Abstract source of one latest normalized rate per foreign currency."""

    @abstractmethod
    def fetch_latest_rates(
        self,
        foreign_currencies: Iterable[str],
    ) -> pd.DataFrame:
        """Return date, currency, and KRW-per-one-foreign-unit rate columns."""


class CSVExchangeRateProvider(ExchangeRateProvider):
    """Read normalized exchange-rate input from a local CSV file.

    ``foreign_units_per_rate`` describes the source quotation unit. For
    example, set it to 100 when a JPY CSV rate means KRW per JPY 100. Returned
    rates are divided by this value so they always mean KRW per JPY 1.
    """

    def __init__(
        self,
        csv_path: str | Path,
        foreign_currency: str,
        foreign_units_per_rate: float = 1.0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.foreign_currency = _normalize_currency_code(foreign_currency)
        self.foreign_units_per_rate = _validate_foreign_units_per_rate(
            foreign_units_per_rate
        )

    def fetch_daily_rates(
        self,
        foreign_currency: str,
        start_date: DateInput,
        end_date: DateInput,
    ) -> pd.DataFrame:
        """Read, normalize, de-duplicate, filter, and return CSV rate data."""
        requested_currency = _normalize_currency_code(foreign_currency)
        if requested_currency != self.foreign_currency:
            raise UnsupportedCurrencyError(
                f"CSV provider supports {self.foreign_currency}, "
                f"not {requested_currency}"
            )

        start = _normalize_date_input(start_date, "start_date")
        end = _normalize_date_input(end_date, "end_date")
        if start > end:
            raise ValueError("start_date must be on or before end_date")

        try:
            source_data = pd.read_csv(self.csv_path)
        except (OSError, UnicodeError, pd.errors.ParserError) as error:
            raise ExchangeRateProviderError(
                f"failed to read exchange-rate CSV: {self.csv_path}"
            ) from error

        standardized = _standardize_daily_rates(
            source_data,
            self.foreign_units_per_rate,
        )
        in_range = standardized["date"].between(start, end, inclusive="both")
        return standardized.loc[in_range, list(STANDARD_COLUMNS)].reset_index(
            drop=True
        )


class YFinanceExchangeRateProvider(ExchangeRateProvider):
    """Fetch historical daily rates from Yahoo Finance through yfinance.

    yfinance is imported only when a real download is requested. Tests can
    inject a download function and therefore never access the network.
    """

    def __init__(
        self,
        download_function: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        self._download_function = download_function

    def fetch_daily_rates(
        self,
        foreign_currency: str,
        start_date: DateInput,
        end_date: DateInput,
    ) -> pd.DataFrame:
        """Download and normalize an inclusive historical daily-rate range."""
        currency = _normalize_currency_code(foreign_currency)
        ticker_config = YFINANCE_TICKERS.get(currency)
        if ticker_config is None:
            raise UnsupportedCurrencyError(
                f"yfinance provider does not support {currency}"
            )

        start = _normalize_date_input(start_date, "start_date")
        end = _normalize_date_input(end_date, "end_date")
        if start > end:
            raise ValueError("start_date must be on or before end_date")

        download = self._download_function or _load_yfinance_download()
        try:
            source_data = download(
                ticker_config.ticker,
                start=start.date().isoformat(),
                end=(end + timedelta(days=1)).date().isoformat(),
                progress=False,
                auto_adjust=False,
            )
        except Exception:
            raise ExchangeRateProviderError(
                "failed to download historical exchange-rate data"
            ) from None

        if not isinstance(source_data, pd.DataFrame):
            raise InvalidExchangeRateDataError(
                "yfinance response must be a pandas DataFrame"
            )
        if source_data.empty:
            return _empty_daily_rate_frame()

        close_rates = _extract_yfinance_close(source_data, ticker_config.ticker)
        parsed_index = pd.to_datetime(close_rates.index, errors="coerce", utc=True)
        if parsed_index.isna().any():
            raise InvalidExchangeRateDataError(
                "yfinance response contains an invalid date"
            )

        rate_values = pd.to_numeric(close_rates, errors="coerce")
        if ticker_config.invert:
            invalid_for_inversion = (
                rate_values.isna()
                | ~np.isfinite(rate_values)
                | rate_values.le(0)
            )
            if invalid_for_inversion.any():
                raise InvalidExchangeRateDataError(
                    "yfinance response contains a rate that cannot be inverted"
                )
            rate_values = 1.0 / rate_values

        normalized_source = pd.DataFrame(
            {
                "date": parsed_index.tz_convert(None),
                "rate": rate_values.to_numpy(copy=True),
            }
        )
        standardized = _standardize_daily_rates(
            normalized_source,
            ticker_config.foreign_units_per_rate,
        )
        in_range = standardized["date"].between(start, end, inclusive="both")
        return standardized.loc[in_range, list(STANDARD_COLUMNS)].reset_index(
            drop=True
        )


class ExchangeRateApiProvider(LatestExchangeRateProvider):
    """Fetch current rates from ExchangeRate-API with KRW as the base."""

    API_URL_TEMPLATE = "https://v6.exchangerate-api.com/v6/{api_key}/latest/KRW"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_get: Callable[..., Any] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive number")
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._http_get = http_get
        self.timeout_seconds = float(timeout_seconds)

    def fetch_latest_rates(
        self,
        foreign_currencies: Iterable[str],
    ) -> pd.DataFrame:
        """Fetch requested currencies in one request and invert KRW-base rates."""
        currencies = _normalize_requested_currencies(foreign_currencies)
        if not currencies:
            return _empty_latest_rate_frame()

        unsupported = [
            currency
            for currency in currencies
            if currency not in SUPPORTED_FOREIGN_CURRENCIES
        ]
        if unsupported:
            raise UnsupportedCurrencyError(
                f"latest-rate provider does not support {unsupported[0]}"
            )

        api_key = self._api_key or get_optional_environment_variable(
            "EXCHANGE_RATE_API_KEY"
        )
        if api_key is None:
            raise MissingConfigurationError(
                "required environment variable is missing: EXCHANGE_RATE_API_KEY"
            )

        http_get = self._http_get or _load_requests_get()
        request_url = self.API_URL_TEMPLATE.format(api_key=api_key)
        try:
            response = http_get(request_url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except Exception:
            raise LatestExchangeRateNetworkError(
                "ExchangeRate-API request failed or timed out"
            ) from None

        try:
            payload = response.json()
        except Exception:
            raise LatestExchangeRateResponseError(
                "ExchangeRate-API returned invalid JSON"
            ) from None

        if not isinstance(payload, Mapping):
            raise LatestExchangeRateResponseError(
                "ExchangeRate-API JSON must be an object"
            )
        if payload.get("result") != "success":
            error_type = payload.get("error-type", "unknown-error")
            raise LatestExchangeRateResponseError(
                f"ExchangeRate-API returned an error: {error_type}"
            )
        if payload.get("base_code") != "KRW":
            raise LatestExchangeRateResponseError(
                "ExchangeRate-API response base_code must be KRW"
            )

        observation_date = _parse_latest_rate_date(payload)
        conversion_rates = payload.get("conversion_rates")
        if not isinstance(conversion_rates, Mapping):
            raise LatestExchangeRateResponseError(
                "ExchangeRate-API response is missing conversion_rates"
            )

        rows: list[dict[str, object]] = []
        for currency in currencies:
            if currency not in conversion_rates:
                raise LatestExchangeRateResponseError(
                    f"ExchangeRate-API response is missing currency: {currency}"
                )

            conversion_rate = _validate_conversion_rate(
                conversion_rates[currency],
                currency,
            )
            rows.append(
                {
                    "date": observation_date,
                    "currency": currency,
                    "rate": 1.0 / conversion_rate,
                }
            )

        return pd.DataFrame(rows, columns=list(LATEST_STANDARD_COLUMNS))


def _standardize_daily_rates(
    source_data: pd.DataFrame,
    foreign_units_per_rate: float,
) -> pd.DataFrame:
    """Convert provider data to the strict internal daily-rate contract."""
    missing_columns = set(STANDARD_COLUMNS).difference(source_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise InvalidExchangeRateDataError(
            f"source data is missing required columns: {missing}"
        )

    standardized = source_data.loc[:, list(STANDARD_COLUMNS)].copy()
    parsed_dates = pd.to_datetime(
        standardized["date"],
        errors="coerce",
        format="mixed",
    )
    if parsed_dates.isna().any():
        raise InvalidExchangeRateDataError("source data contains an invalid date")
    standardized["date"] = parsed_dates.dt.normalize()

    numeric_rates = pd.to_numeric(standardized["rate"], errors="coerce")
    invalid_rates = (
        numeric_rates.isna()
        | ~np.isfinite(numeric_rates)
        | numeric_rates.le(0)
    )
    if invalid_rates.any():
        invalid_rows = ", ".join(
            str(position + 2)
            for position in np.flatnonzero(invalid_rates.to_numpy())
        )
        raise InvalidExchangeRateDataError(
            f"source data contains an invalid rate at CSV row(s): {invalid_rows}"
        )

    standardized["rate"] = (
        numeric_rates.astype(float) / foreign_units_per_rate
    )

    # If a source contains multiple values for one date, its last value wins.
    standardized = standardized.drop_duplicates(subset="date", keep="last")
    return standardized.sort_values("date", kind="stable").reset_index(drop=True)


def _normalize_currency_code(currency: str) -> str:
    """Return a validated uppercase three-letter currency code."""
    if not isinstance(currency, str):
        raise UnsupportedCurrencyError("foreign_currency must be a string")

    normalized = currency.strip().upper()
    if len(normalized) != 3 or not normalized.isascii() or not normalized.isalpha():
        raise UnsupportedCurrencyError(
            "foreign_currency must be a three-letter currency code"
        )
    return normalized


def _normalize_date_input(value: DateInput, field_name: str) -> pd.Timestamp:
    """Convert a date boundary to a timezone-naive midnight Timestamp."""
    try:
        normalized = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a valid date") from error

    if pd.isna(normalized):
        raise ValueError(f"{field_name} must be a valid date")
    if normalized.tzinfo is not None:
        normalized = normalized.tz_localize(None)
    return normalized.normalize()


def _validate_foreign_units_per_rate(value: float) -> float:
    """Validate the source quotation unit used for rate normalization."""
    if isinstance(value, bool):
        raise ValueError("foreign_units_per_rate must be a positive number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "foreign_units_per_rate must be a positive number"
        ) from error

    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError("foreign_units_per_rate must be a positive number")
    return numeric_value


def _load_yfinance_download() -> Callable[..., pd.DataFrame]:
    """Load yfinance only when a real historical request is made."""
    try:
        import yfinance as yf
    except ImportError:
        raise ExchangeRateProviderError(
            "yfinance is required for historical exchange-rate downloads"
        ) from None
    return yf.download


def _load_requests_get() -> Callable[..., Any]:
    """Load requests only when a real latest-rate request is made."""
    try:
        import requests
    except ImportError:
        raise LatestExchangeRateProviderError(
            "requests is required for latest exchange-rate requests"
        ) from None
    return requests.get


def _extract_yfinance_close(
    source_data: pd.DataFrame,
    ticker: str,
) -> pd.Series:
    """Extract one Close series from simple or MultiIndex yfinance columns."""
    if isinstance(source_data.columns, pd.MultiIndex):
        if "Close" not in source_data.columns.get_level_values(0):
            raise InvalidExchangeRateDataError(
                "yfinance response is missing the Close column"
            )
        close_data = source_data["Close"]
        if isinstance(close_data, pd.Series):
            return close_data.copy()
        if ticker in close_data.columns:
            return close_data[ticker].copy()
        if close_data.shape[1] == 1:
            return close_data.iloc[:, 0].copy()
        raise InvalidExchangeRateDataError(
            "yfinance response contains ambiguous Close columns"
        )

    if "Close" not in source_data.columns:
        raise InvalidExchangeRateDataError(
            "yfinance response is missing the Close column"
        )
    close_data = source_data["Close"]
    if not isinstance(close_data, pd.Series):
        raise InvalidExchangeRateDataError(
            "yfinance Close data must be a single series"
        )
    return close_data.copy()


def _normalize_requested_currencies(
    foreign_currencies: Iterable[str],
) -> list[str]:
    """Validate and de-duplicate requested currencies while preserving order."""
    if isinstance(foreign_currencies, str):
        raise UnsupportedCurrencyError(
            "foreign_currencies must be an iterable of currency codes"
        )
    try:
        normalized = [
            _normalize_currency_code(currency)
            for currency in foreign_currencies
        ]
    except TypeError as error:
        raise UnsupportedCurrencyError(
            "foreign_currencies must be an iterable of currency codes"
        ) from error
    return list(dict.fromkeys(normalized))


def _parse_latest_rate_date(payload: Mapping[str, Any]) -> pd.Timestamp:
    """Parse the provider update timestamp into a timezone-naive UTC date."""
    unix_timestamp = payload.get("time_last_update_unix")
    utc_timestamp = payload.get("time_last_update_utc")

    try:
        if unix_timestamp is not None and not isinstance(unix_timestamp, bool):
            numeric_timestamp = float(unix_timestamp)
            if not np.isfinite(numeric_timestamp):
                raise ValueError
            parsed = pd.to_datetime(numeric_timestamp, unit="s", utc=True)
        elif utc_timestamp:
            parsed = pd.to_datetime(utc_timestamp, errors="raise", utc=True)
        else:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise LatestExchangeRateResponseError(
            "ExchangeRate-API response has no valid update date"
        ) from None

    if pd.isna(parsed):
        raise LatestExchangeRateResponseError(
            "ExchangeRate-API response has no valid update date"
        )
    return pd.Timestamp(parsed).tz_convert(None).normalize()


def _validate_conversion_rate(value: object, currency: str) -> float:
    """Validate a KRW-base conversion value before taking its reciprocal."""
    if isinstance(value, bool):
        raise LatestExchangeRateResponseError(
            f"ExchangeRate-API returned an invalid rate for {currency}"
        )
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise LatestExchangeRateResponseError(
            f"ExchangeRate-API returned an invalid rate for {currency}"
        ) from None

    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise LatestExchangeRateResponseError(
            f"ExchangeRate-API returned an invalid rate for {currency}"
        )
    return numeric_value


def _empty_daily_rate_frame() -> pd.DataFrame:
    """Return an empty DataFrame that still follows the historical contract."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "rate": pd.Series(dtype=float),
        }
    )


def _empty_latest_rate_frame() -> pd.DataFrame:
    """Return an empty DataFrame that still follows the latest-rate contract."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "currency": pd.Series(dtype=str),
            "rate": pd.Series(dtype=float),
        }
    )
