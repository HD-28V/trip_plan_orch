import os
import unittest
from unittest.mock import patch

import pandas as pd

from src.config import MissingConfigurationError
from src.providers import (
    ExchangeRateApiProvider,
    InvalidExchangeRateDataError,
    LatestExchangeRateNetworkError,
    LatestExchangeRateResponseError,
    SUPPORTED_FOREIGN_CURRENCIES,
    UnsupportedCurrencyError,
    YFINANCE_TICKERS,
    YFinanceExchangeRateProvider,
    YFinanceTickerConfig,
)


class FakeResponse:
    def __init__(
        self,
        payload: object = None,
        *,
        json_error: Exception | None = None,
        http_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.json_error = json_error
        self.http_error = http_error

    def raise_for_status(self) -> None:
        if self.http_error is not None:
            raise self.http_error

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


def successful_latest_payload(**conversion_rates: object) -> dict[str, object]:
    return {
        "result": "success",
        "base_code": "KRW",
        "time_last_update_unix": int(
            pd.Timestamp("2026-08-05", tz="UTC").timestamp()
        ),
        "time_last_update_utc": "Wed, 05 Aug 2026 00:00:00 +0000",
        "conversion_rates": conversion_rates,
    }


class ExchangeRateApiProviderTests(unittest.TestCase):
    def test_inverts_usd_and_jpy_from_one_krw_base_response(self) -> None:
        calls: list[tuple[str, float]] = []
        response = FakeResponse(
            successful_latest_payload(USD=1 / 1380.5, JPY=1 / 9.21)
        )

        def fake_get(url: str, *, timeout: float) -> FakeResponse:
            calls.append((url, timeout))
            return response

        provider = ExchangeRateApiProvider(
            api_key="dummy-exchange-key",
            http_get=fake_get,
            timeout_seconds=3,
        )
        result = provider.fetch_latest_rates(["USD", "JPY", "USD"])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 3)
        self.assertEqual(result.columns.tolist(), ["date", "currency", "rate"])
        self.assertEqual(result["currency"].tolist(), ["USD", "JPY"])
        self.assertAlmostEqual(result.iloc[0]["rate"], 1380.5)
        self.assertAlmostEqual(result.iloc[1]["rate"], 9.21)
        self.assertEqual(result.iloc[0]["date"], pd.Timestamp("2026-08-05"))

    def test_rejects_missing_currency_and_invalid_conversion_rates(self) -> None:
        provider = ExchangeRateApiProvider(
            api_key="dummy-key",
            http_get=lambda *args, **kwargs: FakeResponse(
                successful_latest_payload(USD=1 / 1380)
            ),
        )
        with self.assertRaisesRegex(
            LatestExchangeRateResponseError,
            "missing currency: EUR",
        ):
            provider.fetch_latest_rates(["EUR"])

        for value in (0, -1, "not-a-number"):
            with self.subTest(value=value):
                invalid_provider = ExchangeRateApiProvider(
                    api_key="dummy-key",
                    http_get=lambda *args, _value=value, **kwargs: FakeResponse(
                        successful_latest_payload(USD=_value)
                    ),
                )
                with self.assertRaises(LatestExchangeRateResponseError):
                    invalid_provider.fetch_latest_rates(["USD"])

    def test_rejects_api_error_responses(self) -> None:
        for error_type in ("invalid-key", "quota-reached"):
            with self.subTest(error_type=error_type):
                provider = ExchangeRateApiProvider(
                    api_key="dummy-key",
                    http_get=lambda *args, _error=error_type, **kwargs: FakeResponse(
                        {"result": "error", "error-type": _error}
                    ),
                )
                with self.assertRaisesRegex(
                    LatestExchangeRateResponseError,
                    error_type,
                ):
                    provider.fetch_latest_rates(["USD"])

    def test_requires_api_key_only_when_fetching(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            provider = ExchangeRateApiProvider(
                http_get=lambda *args, **kwargs: None
            )
            with self.assertRaisesRegex(
                MissingConfigurationError,
                "EXCHANGE_RATE_API_KEY",
            ):
                provider.fetch_latest_rates(["USD"])

    def test_handles_http_json_and_date_errors_without_exposing_key(self) -> None:
        secret = "dummy-secret-that-must-not-appear"

        def failing_get(url: str, *, timeout: float) -> None:
            raise TimeoutError(f"timeout while requesting {url}")

        provider = ExchangeRateApiProvider(
            api_key=secret,
            http_get=failing_get,
        )
        with self.assertRaises(LatestExchangeRateNetworkError) as context:
            provider.fetch_latest_rates(["USD"])
        self.assertNotIn(secret, str(context.exception))

        invalid_json_provider = ExchangeRateApiProvider(
            api_key=secret,
            http_get=lambda *args, **kwargs: FakeResponse(
                json_error=ValueError("bad json")
            ),
        )
        with self.assertRaises(LatestExchangeRateResponseError):
            invalid_json_provider.fetch_latest_rates(["USD"])

        missing_date_payload = successful_latest_payload(USD=1 / 1380)
        del missing_date_payload["time_last_update_unix"]
        del missing_date_payload["time_last_update_utc"]
        missing_date_provider = ExchangeRateApiProvider(
            api_key=secret,
            http_get=lambda *args, **kwargs: FakeResponse(missing_date_payload),
        )
        with self.assertRaisesRegex(
            LatestExchangeRateResponseError,
            "update date",
        ):
            missing_date_provider.fetch_latest_rates(["USD"])


class YFinanceExchangeRateProviderTests(unittest.TestCase):
    def test_ticker_mapping_covers_supported_currencies(self) -> None:
        self.assertEqual(set(YFINANCE_TICKERS), set(SUPPORTED_FOREIGN_CURRENCIES))
        self.assertEqual(YFINANCE_TICKERS["USD"].ticker, "KRW=X")
        self.assertEqual(YFINANCE_TICKERS["JPY"].ticker, "JPYKRW=X")

    def test_returns_standard_sorted_daily_rates_without_network(self) -> None:
        calls: list[dict[str, object]] = []
        source = pd.DataFrame(
            {"Close": [1350.0, 1340.0]},
            index=pd.to_datetime(["2025-01-02", "2025-01-01"]),
        )

        def fake_download(ticker: str, **kwargs: object) -> pd.DataFrame:
            calls.append({"ticker": ticker, **kwargs})
            return source

        provider = YFinanceExchangeRateProvider(fake_download)
        result = provider.fetch_daily_rates(
            "usd",
            "2025-01-01",
            "2025-01-02",
        )

        self.assertEqual(calls[0]["ticker"], "KRW=X")
        self.assertEqual(calls[0]["end"], "2025-01-03")
        self.assertEqual(result.columns.tolist(), ["date", "rate"])
        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2025-01-01", "2025-01-02"],
        )
        self.assertEqual(result["rate"].tolist(), [1340.0, 1350.0])
        self.assertEqual(source["Close"].tolist(), [1350.0, 1340.0])

    def test_applies_inverse_ticker_configuration(self) -> None:
        source = pd.DataFrame(
            {"Close": [0.00075]},
            index=pd.to_datetime(["2025-01-01"]),
        )
        provider = YFinanceExchangeRateProvider(
            lambda *args, **kwargs: source
        )

        with patch.dict(
            YFINANCE_TICKERS,
            {"USD": YFinanceTickerConfig("KRWUSD=X", invert=True)},
        ):
            result = provider.fetch_daily_rates(
                "USD",
                "2025-01-01",
                "2025-01-01",
            )

        self.assertAlmostEqual(result.iloc[0]["rate"], 1 / 0.00075)

    def test_rejects_unsupported_currency_and_invalid_close_data(self) -> None:
        provider = YFinanceExchangeRateProvider(
            lambda *args, **kwargs: pd.DataFrame()
        )
        with self.assertRaises(UnsupportedCurrencyError):
            provider.fetch_daily_rates(
                "CHF",
                "2025-01-01",
                "2025-01-02",
            )

        missing_close_provider = YFinanceExchangeRateProvider(
            lambda *args, **kwargs: pd.DataFrame(
                {"Open": [1300]},
                index=pd.to_datetime(["2025-01-01"]),
            )
        )
        with self.assertRaises(InvalidExchangeRateDataError):
            missing_close_provider.fetch_daily_rates(
                "USD",
                "2025-01-01",
                "2025-01-01",
            )


if __name__ == "__main__":
    unittest.main()
