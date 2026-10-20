"""Historical closures, missing-session errors, and five-session decisions."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from src.portfolio_rl import data
from src.portfolio_rl.calendar import exchange_sessions, select_exchange_sessions
from src.portfolio_rl.environment import SectorAllocationEnv


class TestExchangeCalendar(unittest.TestCase):
    def test_1999_mlk_holiday_and_exclusive_end_date(self):
        sessions = exchange_sessions("1999-01-15", "1999-01-20")
        assert_array_equal(sessions, pd.to_datetime(["1999-01-15", "1999-01-19"]))

    def test_exceptional_closures_are_excluded(self):
        for start, end, expected in (
            ("2001-09-10", "2001-09-18", ["2001-09-10", "2001-09-17"]),
            ("2012-10-26", "2012-11-01", ["2012-10-26", "2012-10-31"]),
            ("2025-01-08", "2025-01-11", ["2025-01-08", "2025-01-10"]),
        ):
            with self.subTest(start=start):
                assert_array_equal(exchange_sessions(start, end), pd.to_datetime(expected))

    def test_early_close_still_counts_as_one_session(self):
        sessions = exchange_sessions("2024-11-27", "2024-12-03")
        assert_array_equal(sessions, pd.to_datetime(["2024-11-27", "2024-11-29", "2024-12-02"]))

    def test_invalid_ranges_and_calendars_raise_value_errors(self):
        for start, end, name in (
            ("2024-01-02", "2024-01-01", "XNYS"),
            ("2024-01-01", "2024-01-01", "XNYS"),
            ("2024-01-13", "2024-01-15", "XNYS"),
            ("2024-01-01 10:00", "2024-02-01", "XNYS"),
            (None, "2024-02-01", "XNYS"),
            ("2024-01-01", "2024-02-01", "INVALID"),
            ("2024-01-01", "2024-02-01", None),
        ):
            with self.subTest(start=start, end=end, name=name):
                with self.assertRaises(ValueError):
                    exchange_sessions(start, end, name)

    def test_selection_uses_dates_not_price_validity_and_does_not_mutate(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["1999-01-15", "1999-01-18", "1999-01-19"]),
                              "symbol": "XLB", "close": [100., np.nan, np.nan]})
        original = frame.copy(deep=True)
        with self.assertLogs("src.portfolio_rl.calendar", level="INFO") as messages:
            result = select_exchange_sessions(frame, exchange_sessions("1999-01-15", "1999-01-20"))
        assert_array_equal(result.index, [0, 2])
        self.assertTrue(np.isnan(result.loc[2, "close"]))
        self.assertIn("Excluded 1 rows on 1 non-session dates", messages.output[0])
        assert_frame_equal(frame, original)

    def test_invalid_dates_cannot_be_hidden_by_session_selection(self):
        frame = pd.DataFrame({"date": [pd.NaT], "symbol": ["XLB"]})
        with self.assertRaisesRegex(ValueError, "Missing or intraday dates"):
            select_exchange_sessions(frame, exchange_sessions("2024-01-02", "2024-01-04"))

    def test_environment_counts_five_sessions_across_a_holiday(self):
        dates = exchange_sessions("2024-01-11", "2024-01-26")
        dataset = {
            "dates": dates.to_numpy(), "asset_states": np.zeros((len(dates), 2)),
            "spy_states": np.zeros((len(dates), 1)), "market_states": np.zeros((len(dates), 1)),
            "asset_returns": np.full((len(dates), 2), .001),
        }
        environment = SectorAllocationEnv(dataset, decision_frequency_days=5,
                                           long_exposure=1., short_exposure=0.)
        environment.reset(seed=42)
        _, _, _, truncated, info = environment.step(np.zeros(2))
        self.assertFalse(truncated)
        assert_array_equal(pd.DatetimeIndex([row["date"] for row in info["daily_records"]]),
                           pd.to_datetime(["2024-01-12", "2024-01-16", "2024-01-17",
                                           "2024-01-18", "2024-01-19"]))
        _, _, _, truncated, info = environment.step(np.zeros(2))
        self.assertTrue(truncated)
        self.assertEqual(len(info["daily_records"]), 4)
        self.assertEqual(info["daily_records"][0]["date"], pd.Timestamp("2024-01-22"))

    def test_training_rejects_missing_sessions_and_extra_holiday_observations(self):
        sessions = exchange_sessions("2020-01-02", "2023-01-01")
        frame = pd.DataFrame([
            {"date": date, "symbol": symbol, "signal": float(i), "return_1d": .01}
            for i, date in enumerate(sessions) for symbol in ("XLB", "SPY")
        ])
        missing = frame.loc[~frame["date"].eq(sessions[100])]
        holiday = frame.iloc[:2].copy()
        holiday["date"] = pd.Timestamp("2020-01-20")
        extra = pd.concat([frame, holiday], ignore_index=True)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            for invalid, message in ((missing, "Missing trading dates"),
                                      (extra, "Unexpected trading dates")):
                with self.subTest(message=message):
                    invalid.to_csv(path, index=False)
                    with self.assertRaisesRegex(ValueError, message):
                        data.prepare_datasets(path, ["XLB"], ["signal"], ["signal"], [],
                                              final_holdout_years=1, scaling_window_years=3 / 252)


class TestHolidayDownloadValidation(unittest.TestCase):
    def setUp(self):
        dates = pd.to_datetime(["1999-01-15", "1999-01-18", "1999-01-19", "1999-01-20"])
        self.wide = pd.concat({symbol: pd.DataFrame({"Close": [100., np.nan, 101., 102.]}, index=dates)
                               for symbol in ("XLB", "SPY", "^VIX")}, axis=1)
        self.wide.loc[dates[1], ("^VIX", "Close")] = 30.
        self.metadata = pd.DataFrame({"symbol": ["XLB", "SPY", "^VIX"],
                                      "tradable": [True, False, False]})

    def download(self):
        with patch.object(data, "download_daily_data", return_value=self.wide):
            return data.download_market_data(["XLB"], self.metadata, "1999-01-15", "1999-01-21")

    def test_holiday_placeholders_do_not_raise_errors(self):
        result = self.download()
        self.assertEqual(len(result), 9)
        self.assertFalse(result["date"].eq("1999-01-18").any())

    def test_missing_price_on_an_open_session_still_raises(self):
        self.wide.loc[pd.Timestamp("1999-01-19"), ("XLB", "Close")] = np.nan
        with self.assertRaisesRegex(ValueError, "close contains nonfinite"):
            self.download()

    def test_session_missing_from_every_symbol_still_raises(self):
        self.wide = self.wide.drop(index=pd.Timestamp("1999-01-19"))
        with self.assertRaisesRegex(ValueError, "Missing trading dates for XLB"):
            self.download()


if __name__ == "__main__":
    unittest.main()
