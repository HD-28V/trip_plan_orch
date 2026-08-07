"""Calculate and plot moving averages from synthetic USD/KRW test data.

The bundled CSV contains generated values for development only. It does not
contain actual exchange rates and must not be used for financial decisions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_exchange.csv"


def load_exchange_data(csv_path: Path = SAMPLE_DATA_PATH) -> pd.DataFrame:
    """Read exchange-rate data, parse its dates, and return it in date order."""
    data = pd.read_csv(csv_path)

    # 날짜 문자열을 pandas가 날짜로 계산할 수 있는 datetime 형식으로 바꿉니다.
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["rate"] = pd.to_numeric(data["rate"], errors="raise")

    # 이동평균은 행의 순서에 영향을 받으므로 과거 날짜부터 정렬합니다.
    return data.sort_values("date").reset_index(drop=True)


def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """Add 60-day and 120-day simple moving averages to a copy of the data."""
    result = data.copy()

    # SMA60은 현재 날짜를 포함한 최근 60개 환율의 산술평균입니다.
    result["SMA60"] = result["rate"].rolling(window=60, min_periods=60).mean()

    # SMA120은 더 긴 흐름을 보기 위한 최근 120개 환율의 산술평균입니다.
    result["SMA120"] = result["rate"].rolling(window=120, min_periods=120).mean()
    return result


def plot_exchange_rates(data: pd.DataFrame) -> None:
    """Display the synthetic rate and both moving averages on one chart."""
    plt.figure(figsize=(12, 6))
    plt.plot(data["date"], data["rate"], label="Rate (synthetic USD/KRW)")
    plt.plot(data["date"], data["SMA60"], label="SMA60")
    plt.plot(data["date"], data["SMA120"], label="SMA120")
    plt.title("Exchange Rate with SMA60 and SMA120")
    plt.xlabel("Date")
    plt.ylabel("KRW per USD (synthetic)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the Phase 1 CSV, calculation, output, and plotting workflow."""
    exchange_data = load_exchange_data()
    result = calculate_moving_averages(exchange_data)

    print("Synthetic USD/KRW test data only; these are not actual exchange rates.")
    print(result.tail(10).to_string(index=False))
    plot_exchange_rates(result)


if __name__ == "__main__":
    main()
