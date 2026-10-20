"""Validation accepts usable data and raises useful errors for unsafe inputs."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.portfolio_rl import validation


def market_frame():
    dates = pd.bdate_range("2024-01-04", periods=5)
    return pd.DataFrame([
        {"date": date, "symbol": symbol, "tradable": symbol == "XLB",
         "close": 100. + i, "signal": i / 10, "return_1d": .01 if i else np.nan}
        for i, date in enumerate(dates)
        for symbol in ("XLB", "SPY", "^VIX")
    ])


class TestMarketValidation(unittest.TestCase):
    def setUp(self):
        self.frame = market_frame()
        self.dates = pd.DatetimeIndex(self.frame["date"].unique())

    def validate(self, frame=None, **settings):
        validation.validate_market_data(self.frame if frame is None else frame,
                                        ["XLB"], **settings)

    def test_valid_data_is_not_modified_and_optional_metadata_is_not_required(self):
        self.frame["volume"] = 0
        self.frame["sector"] = np.nan
        original = self.frame.copy(deep=True)
        self.assertIsNone(self.validate(expected_dates=self.dates))
        assert_frame_equal(self.frame, original)

    def test_empty_missing_and_duplicate_columns_raise_value_errors(self):
        for frame in (self.frame.iloc[:0], self.frame.drop(columns="close"),
                      pd.concat([self.frame, self.frame[["close"]]], axis=1)):
            with self.subTest(columns=frame.columns.tolist()):
                with self.assertRaises(ValueError):
                    self.validate(frame)

    def test_unparsed_missing_and_intraday_dates_are_rejected(self):
        for dates in (self.frame["date"].astype(str),
                      self.frame["date"].mask(self.frame.index == 0),
                      self.frame["date"] + pd.Timedelta(hours=1)):
            with self.subTest(dtype=dates.dtype):
                frame = self.frame.copy(deep=True)
                frame["date"] = dates
                with self.assertRaisesRegex(ValueError, "date"):
                    self.validate(frame)

    def test_unsorted_symbol_history_and_duplicate_observations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "chronological"):
            self.validate(self.frame.iloc[::-1])
        frame = pd.concat([self.frame.iloc[[0]], self.frame], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate date/symbol"):
            self.validate(frame)

    def test_invalid_symbol_configuration_and_identifiers_are_rejected(self):
        for assets, context in (([], ["SPY"]), (["XLB", "XLB"], ["SPY"]),
                                (["XLB"], ["XLB"]), (["MISSING"], ["SPY"]),
                                ([""], ["SPY"]), ("XLB", ["SPY"])):
            with self.subTest(assets=assets, context=context):
                with self.assertRaises(ValueError):
                    validation.validate_symbols(self.frame, assets, context)
        for value in (None, "", " "):
            frame = self.frame.copy(deep=True)
            frame.loc[0, "symbol"] = value
            with self.subTest(identifier=value):
                with self.assertRaises(ValueError):
                    self.validate(frame)

    def test_missing_string_and_incorrect_tradable_flags_are_rejected(self):
        for value in (None, "False", 1, False):
            frame = self.frame.copy(deep=True)
            frame["tradable"] = frame["tradable"].astype(object)
            frame.loc[0, "tradable"] = value
            with self.subTest(flag=value):
                with self.assertRaisesRegex(ValueError, "tradable"):
                    self.validate(frame)

    def test_numeric_strings_booleans_complex_and_invalid_prices_are_rejected(self):
        for value in ("100", True, 1 + 2j, np.nan, np.inf, -np.inf, 0, -1):
            frame = self.frame.copy(deep=True)
            frame["close"] = frame["close"].astype(object)
            frame.loc[0, "close"] = value
            with self.subTest(price=value):
                with self.assertRaisesRegex(ValueError, "close"):
                    self.validate(frame)

    def test_missing_symbol_date_is_not_hidden_by_calendar_intersection(self):
        other = self.frame.loc[self.frame["symbol"].eq("XLB")].copy()
        other["symbol"] = "XLE"
        frame = pd.concat([self.frame.drop(index=3), other]).sort_values(["date", "symbol"])
        with self.assertRaisesRegex(ValueError, "Missing trading dates for XLB"):
            validation.validate_market_data(frame, ["XLB", "XLE"])

    def test_independent_calendar_catches_a_date_missing_from_every_symbol(self):
        frame = self.frame.loc[~self.frame["date"].eq(self.dates[2])]
        self.validate(frame)  # Relative coverage alone cannot detect this gap.
        with self.assertRaisesRegex(ValueError, "Missing trading dates"):
            self.validate(frame, expected_dates=self.dates)

    def test_extra_context_dates_are_allowed_but_extra_asset_dates_are_rejected(self):
        extra = self.frame.loc[self.frame["symbol"].eq("^VIX")].iloc[[-1]].copy()
        extra["date"] += pd.Timedelta(days=1)
        extra["close"] = np.nan  # Outside the required calendar, so not consumed.
        self.validate(pd.concat([self.frame, extra], ignore_index=True),
                      expected_dates=self.dates)
        extra["symbol"] = "XLB"
        extra["tradable"] = True
        with self.assertRaisesRegex(ValueError, "Unexpected trading dates"):
            self.validate(pd.concat([self.frame, extra], ignore_index=True),
                          expected_dates=self.dates)

    def test_invalid_calendars_are_rejected(self):
        calendars = (pd.DatetimeIndex([]), self.dates[::-1],
                     self.dates.insert(0, self.dates[0]),
                     self.dates.insert(0, pd.NaT), self.dates.tz_localize("UTC"),
                     self.dates + pd.Timedelta(hours=1), self.dates.tolist())
        for calendar in calendars:
            with self.subTest(calendar=calendar):
                with self.assertRaisesRegex(ValueError, "expected_dates"):
                    self.validate(expected_dates=calendar)


class TestFeatureValidation(unittest.TestCase):
    def setUp(self):
        self.frame = market_frame()
        self.dates = pd.DatetimeIndex(self.frame["date"].unique())
        self.settings = {"warmup_periods": {"signal": 0, "return_1d": 1}}

    def validate(self, frame=None, **settings):
        validation.validate_feature_data(
            self.frame if frame is None else frame, ["XLB"], ["signal"],
            ["signal"], [], **(self.settings | settings))

    def test_valid_shuffled_features_are_not_modified(self):
        frame = self.frame.sample(frac=1, random_state=42)
        original = frame.copy(deep=True)
        self.assertIsNone(self.validate(frame))
        assert_frame_equal(frame, original)

    def test_only_configured_features_on_their_applicable_symbols_are_checked(self):
        frame = self.frame.copy(deep=True)
        frame.loc[frame["symbol"].eq("^VIX"), "signal"] = np.nan
        frame.loc[~frame["symbol"].eq("XLB"), "return_1d"] = np.nan
        self.validate(frame)
        self.validate(frame.loc[~frame["symbol"].eq("^VIX")])

    def test_missing_or_invalid_warmup_configuration_is_rejected(self):
        for warmups in ({}, {"signal": 0},
                        {"signal": -1, "return_1d": 1},
                        {"signal": True, "return_1d": 1},
                        {"signal": 1.5, "return_1d": 1},
                        {"signal": 5, "return_1d": 1}, None):
            with self.subTest(warmups=warmups):
                with self.assertRaises(ValueError):
                    self.validate(warmup_periods=warmups)

    def test_leading_missing_features_need_an_explicit_allowance(self):
        frame = self.frame.copy(deep=True)
        frame.loc[frame["date"].eq(self.dates[0]), "signal"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing signal beyond"):
            self.validate(frame)
        self.validate(frame, warmup_periods={"signal": 1, "return_1d": 1})

    def test_interior_and_trailing_missing_values_are_rejected(self):
        for column, position in (("signal", 2), ("signal", 4), ("return_1d", 2)):
            frame = self.frame.copy(deep=True)
            frame.loc[frame["symbol"].eq("XLB") &
                      frame["date"].eq(self.dates[position]), column] = np.nan
            with self.subTest(column=column, position=position):
                with self.assertRaisesRegex(ValueError, column):
                    self.validate(frame)

    def test_infinity_is_rejected_even_during_permitted_warmup(self):
        for value in (np.inf, -np.inf):
            frame = self.frame.copy(deep=True)
            frame.loc[0, "return_1d"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.validate(frame)

    def test_non_numeric_features_are_rejected(self):
        frame = self.frame.copy(deep=True)
        frame["signal"] = frame["signal"].astype(object)
        frame.loc[0, "signal"] = "0.1"
        with self.assertRaisesRegex(ValueError, "signal must contain numeric"):
            self.validate(frame)

    def test_nullable_numeric_features_and_timezone_consistency_are_supported(self):
        frame = self.frame.copy(deep=True)
        frame["signal"] = frame["signal"].astype("Float64")
        frame["return_1d"] = frame["return_1d"].astype("Float64")
        frame["date"] = frame["date"].dt.tz_localize("UTC")
        self.validate(frame, expected_dates=self.dates.tz_localize("UTC"))

    def test_simple_returns_reject_total_loss_but_allow_large_positive_gains(self):
        for value in (-1., -1.1, 2.):
            frame = self.frame.copy(deep=True)
            frame.loc[3, "return_1d"] = value
            with self.subTest(value=value):
                if value <= -1:
                    with self.assertRaisesRegex(ValueError, "return_1d must exceed -1"):
                        self.validate(frame)
                else:
                    self.validate(frame)

    def test_explicit_bounds_reject_values_outside_the_range(self):
        self.validate(feature_bounds={"signal": (0, 1)})
        with self.assertRaisesRegex(ValueError, "signal is outside bounds"):
            self.validate(feature_bounds={"signal": (0, .2)})
        with self.assertRaisesRegex(ValueError, "unrequested"):
            self.validate(feature_bounds={"typo": (0, 1)})

    def test_bound_tolerance_and_unbounded_endpoints(self):
        frame = self.frame.copy(deep=True)
        frame.loc[0, "signal"] = -1e-11
        self.validate(frame, feature_bounds={"signal": (0, None)})
        with self.assertRaisesRegex(ValueError, "outside bounds"):
            validation.validate_feature_bounds(frame, {"signal": (0, None)},
                                                symbols=["XLB"], tolerance=0)
        validation.validate_feature_bounds(frame, {"signal": (None, None)}, symbols=["XLB"])

    def test_invalid_bounds_and_tolerances_are_rejected(self):
        for limits in ((1, 0), (np.nan, 1), (-np.inf, 1), (True, 1), (0,), "01"):
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    self.validate(feature_bounds={"signal": limits})
        for tolerance in (-1, np.inf, np.nan, True):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    validation.validate_feature_bounds(self.frame, {}, symbols=["XLB"],
                                                        tolerance=tolerance)


if __name__ == "__main__":
    unittest.main()
