import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.moving_average import calculate_moving_averages, load_exchange_data


class LoadExchangeDataTests(unittest.TestCase):
    def test_loads_dates_as_datetime_and_sorts_oldest_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            csv_path = Path(temp_directory) / "sample.csv"
            csv_path.write_text(
                "date,rate\n2025-01-02,1350\n2025-01-01,1340\n",
                encoding="utf-8",
            )

            result = load_exchange_data(csv_path)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["date"]))
        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2025-01-01", "2025-01-02"],
        )
        self.assertEqual(result["rate"].tolist(), [1340, 1350])


class MovingAverageTests(unittest.TestCase):
    def test_calculates_sma60_and_sma120_after_full_windows(self) -> None:
        data = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=120, freq="D"),
                "rate": range(1, 121),
            }
        )

        result = calculate_moving_averages(data)

        self.assertTrue(result["SMA60"].iloc[:59].isna().all())
        self.assertAlmostEqual(result["SMA60"].iloc[59], 30.5)
        self.assertAlmostEqual(result["SMA60"].iloc[119], 90.5)
        self.assertTrue(result["SMA120"].iloc[:119].isna().all())
        self.assertAlmostEqual(result["SMA120"].iloc[119], 60.5)

    def test_does_not_modify_the_input_dataframe(self) -> None:
        data = pd.DataFrame({"rate": range(1, 121)})

        calculate_moving_averages(data)

        self.assertNotIn("SMA60", data.columns)
        self.assertNotIn("SMA120", data.columns)


if __name__ == "__main__":
    unittest.main()
