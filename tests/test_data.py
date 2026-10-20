"""Offline tests for data preparation, scaling, and chronological splits.

Run from the project root: python -m unittest discover -s tests -v
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from src.portfolio_rl import data
from src.portfolio_rl.environment import build_feature_panel, scale_rolling, slice_dataset


def feature_frame():
    """Three years of varying features and distinguishable per-asset returns."""
    dates = pd.bdate_range("2020-01-02", periods=800)
    rows = [
        {"date": date, "symbol": symbol,
         "signal": float(sign * (i + 1) ** 2),
         "trend": float((i + offset) % 7),
         "market_signal": float(i % 11),
         "return_1d": offset / 100 + i / 100000}
        for i, date in enumerate(dates)
        for symbol, sign, offset in (("XLB", 1, 1), ("XLE", -1, 2), ("SPY", 2, 3))
    ]
    return pd.DataFrame(rows)


def assert_dataset_equal(test, actual, expected):
    test.assertEqual(set(actual), set(expected))
    for name in expected:
        assert_array_equal(actual[name], expected[name], err_msg=name)


class TestFeaturePanel(unittest.TestCase):
    def test_panel_sorts_dates_and_preserves_requested_symbol_and_feature_order(self):
        frame = feature_frame().iloc[:9].sample(frac=1, random_state=42)
        original = frame.copy(deep=True)
        panel = build_feature_panel(frame, ["XLE", "XLB"], ["trend", "signal"])
        self.assertEqual(list(panel.columns),
                         [("XLE", "trend"), ("XLE", "signal"),
                          ("XLB", "trend"), ("XLB", "signal")])
        self.assertTrue(panel.index.is_monotonic_increasing)
        assert_allclose(panel.iloc[0], [2, -1, 1, 1])
        assert_allclose(panel.iloc[2], [4, -9, 3, 9])
        assert_frame_equal(frame, original)

    def test_missing_symbol_date_is_missing_not_forward_filled(self):
        frame = feature_frame().iloc[:9]
        missing_date = frame["date"].unique()[1]
        frame = frame.loc[~(frame["symbol"].eq("XLE") & frame["date"].eq(missing_date))]
        panel = build_feature_panel(frame, ["XLB", "XLE"], ["signal"])
        self.assertTrue(np.isnan(panel.loc[missing_date, ("XLE", "signal")]))
        self.assertEqual(panel.loc[missing_date, ("XLB", "signal")], 4)


class TestRollingScaling(unittest.TestCase):
    def test_scaling_uses_preceding_values_and_population_standard_deviation(self):
        panel = pd.DataFrame({"signal": [1., 3., 5., 7.]})
        result = scale_rolling(panel, 2, 10)
        assert_allclose(result["signal"], [np.nan, np.nan, 3, 3])

    def test_clipping_bounds_positive_and_negative_observations(self):
        for values, expected in (([1., 3., 100.], 2), ([1., 3., -100.], -2)):
            with self.subTest(values=values):
                panel = pd.DataFrame({"signal": values})
                self.assertEqual(scale_rolling(panel, 2, 2).iloc[-1, 0], expected)

    def test_scaling_is_independent_for_each_column_and_preserves_input(self):
        panel = pd.DataFrame({"up": [1., 3., 5.], "down": [5., 3., 1.]},
                             index=pd.bdate_range("2024-01-04", periods=3))
        original = panel.copy(deep=True)
        result = scale_rolling(panel, 2, 10)
        assert_allclose(result.iloc[-1], [3, -3])
        self.assertTrue(result.index.equals(panel.index))
        self.assertTrue(result.columns.equals(panel.columns))
        assert_frame_equal(panel, original)

    def test_missing_values_require_a_new_complete_history_window(self):
        panel = pd.DataFrame({"signal": [1., 3., np.nan, 7., 9., 11.]})
        assert_allclose(scale_rolling(panel, 2, 10)["signal"],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, 3])

    def test_constant_history_is_undefined_until_the_observation_changes(self):
        # Current behavior: 0/0 stays NaN; a nonzero change / 0 clips to a bound.
        panel = pd.DataFrame({"signal": [5., 5., 5., 7.]})
        assert_allclose(scale_rolling(panel, 2, 10)["signal"],
                        [np.nan, np.nan, np.nan, 10])

    def test_short_history_has_no_scaled_observations(self):
        panel = pd.DataFrame({"signal": [1., 3.]})
        self.assertTrue(scale_rolling(panel, 3, 10).isna().all().all())

    def test_future_values_cannot_change_earlier_scaled_observations(self):
        panel = pd.DataFrame({"signal": [1., 3., 5., 7., 9.]})
        changed = panel.copy(deep=True)
        changed.iloc[4, 0] = 999
        original = scale_rolling(panel, 2, 10).iloc[:4]
        assert_frame_equal(original, scale_rolling(changed, 2, 10).iloc[:4])
        assert_frame_equal(original, scale_rolling(panel.iloc[:4], 2, 10))


class TestDatasetSlicing(unittest.TestCase):
    def test_slice_applies_the_same_mask_to_every_array_without_mutating_input(self):
        dataset = {"dates": np.arange(4), "states": np.arange(8).reshape(4, 2)}
        original = {name: values.copy() for name, values in dataset.items()}
        result = slice_dataset(dataset, np.array([False, True, False, True]))
        assert_array_equal(result["dates"], [1, 3])
        assert_array_equal(result["states"], [[2, 3], [6, 7]])
        assert_dataset_equal(self, dataset, original)

    def test_empty_selection_preserves_array_dimensions_and_types(self):
        dataset = {"dates": np.arange(2), "states": np.ones((2, 3), dtype=np.float32)}
        result = slice_dataset(dataset, np.array([False, False]))
        self.assertEqual(result["dates"].shape, (0,))
        self.assertEqual(result["states"].shape, (0, 3))
        self.assertEqual(result["states"].dtype, np.float32)


class TestPrepareDatasets(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.frame = feature_frame()
        self.dates = pd.DatetimeIndex(self.frame["date"].unique())
        self.path = Path(self.directory.name) / "features.csv"
        self.settings = {
            "tradable_symbols": ["XLE", "XLB"],
            "asset_features": ["trend", "signal"],
            "spy_features": ["signal"],
            "market_features": ["market_signal"],
            "development_train_fraction": .5,
            "final_holdout_years": 1,
            "scaling_window_years": 3 / 252,
            "observation_clip": 10.,
        }

    def prepare(self, frame=None, **overrides):
        frame = self.frame if frame is None else frame
        frame.to_csv(self.path, index=False)
        return data.prepare_datasets(self.path, **(self.settings | overrides))

    def test_arrays_have_expected_dimensions_types_and_finite_values(self):
        train, validation, _ = self.prepare()
        widths = {"asset_states": 4, "spy_states": 1,
                  "market_states": 1, "asset_returns": 2}
        for split in (train, validation):
            self.assertEqual(set(split), {"dates", *widths})
            self.assertTrue(np.issubdtype(split["dates"].dtype, np.datetime64))
            self.assertTrue((np.diff(split["dates"]) > np.timedelta64(0, "D")).all())
            for name, width in widths.items():
                with self.subTest(array=name, start=split["dates"][0]):
                    self.assertEqual(split[name].shape, (len(split["dates"]), width))
                    self.assertEqual(split[name].dtype, np.float32)
                    self.assertTrue(np.isfinite(split[name]).all())
                    if name != "asset_returns":
                        self.assertTrue((np.abs(split[name]) <= 10).all())

    def test_returns_keep_configured_asset_order_and_are_not_scaled_or_clipped(self):
        frame = self.frame.copy(deep=True)
        frame.loc[frame["symbol"].eq("XLE"), "return_1d"] = -.2
        frame.loc[frame["symbol"].eq("XLB"), "return_1d"] = 1.2
        train, validation, _ = self.prepare(frame, observation_clip=.5)
        for split in (train, validation):
            assert_allclose(split["asset_returns"],
                            np.tile([-.2, 1.2], (len(split["dates"]), 1)))
            self.assertTrue((np.abs(split["asset_states"]) <= .5).all())

    def test_states_match_their_dates_symbols_and_feature_order(self):
        train, validation, _ = self.prepare()
        sources = self.frame.set_index(["date", "symbol"])
        columns = {"asset_states": [("XLE", "trend"), ("XLE", "signal"),
                                    ("XLB", "trend"), ("XLB", "signal")],
                   "spy_states": [("SPY", "signal")],
                   "market_states": [("SPY", "market_signal")]}
        for split in (train, validation):
            # Check observations near both ends of each split against raw history.
            for row in (0, 1, -1):
                date = pd.Timestamp(split["dates"][row])
                position = self.dates.get_loc(date)
                history_dates = self.dates[position - 3:position]
                for name, fields in columns.items():
                    expected = []
                    for symbol, feature in fields:
                        history = [sources.loc[(day, symbol), feature] for day in history_dates]
                        value = sources.loc[(date, symbol), feature]
                        expected.append(np.clip((value - np.mean(history)) /
                                                np.std(history, ddof=0), -10, 10))
                    assert_allclose(split[name][row], expected, rtol=1e-6)

    def test_splits_are_chronological_disjoint_and_report_exact_boundaries(self):
        train, validation, summary = self.prepare()
        holdout_cutoff = self.dates[-1] - pd.DateOffset(years=1)
        development = self.dates[self.dates <= holdout_cutoff]
        train_end = len(development) // 2
        expected_dates = [self.dates[3:train_end], development[train_end:],
                          self.dates[self.dates > holdout_cutoff]]
        self.assertEqual(summary["split"].tolist(),
                         ["train", "development validation", "locked final holdout"])
        for row, dates in zip(summary.itertuples(), expected_dates):
            self.assertEqual(row.first_date, dates[0])
            self.assertEqual(row.last_date, dates[-1])
            self.assertEqual(row.dates, len(dates))
        assert_array_equal(train["dates"], expected_dates[0].to_numpy())
        assert_array_equal(validation["dates"], expected_dates[1].to_numpy())
        self.assertLess(train["dates"][-1], validation["dates"][0])
        self.assertTrue((validation["dates"] <= holdout_cutoff.to_datetime64()).all())

    def test_shuffled_csv_rows_produce_identical_outputs(self):
        expected = self.prepare()
        actual = self.prepare(self.frame.sample(frac=1, random_state=42))
        for actual_split, expected_split in zip(actual[:2], expected[:2]):
            assert_dataset_equal(self, actual_split, expected_split)
        assert_frame_equal(actual[2], expected[2])

    def test_leading_missing_features_are_removed_before_scaling_warmup(self):
        frame = self.frame.copy(deep=True)
        frame.loc[frame["date"].isin(self.dates[:2]), "signal"] = np.nan
        train, _, _ = self.prepare(frame, feature_warmup_periods={"signal": 2})
        self.assertEqual(train["dates"][0], self.dates[5].to_datetime64())

    def test_extra_context_dates_and_unused_columns_do_not_change_outputs(self):
        expected = self.prepare()
        extra = self.frame.loc[self.frame["symbol"].eq("SPY")].iloc[[-1]].copy()
        extra["date"] = self.dates[-1] + pd.Timedelta(days=1)
        frame = pd.concat([self.frame, extra], ignore_index=True)
        frame["unused"] = np.nan
        actual = self.prepare(frame)
        for actual_split, expected_split in zip(actual[:2], expected[:2]):
            assert_dataset_equal(self, actual_split, expected_split)
        assert_frame_equal(actual[2], expected[2])

    def test_compressed_csv_is_supported(self):
        expected = self.prepare()
        self.path = self.path.with_suffix(".csv.gz")
        actual = self.prepare()
        for actual_split, expected_split in zip(actual[:2], expected[:2]):
            assert_dataset_equal(self, actual_split, expected_split)
        assert_frame_equal(actual[2], expected[2])

    def test_holdout_values_cannot_change_training_or_validation(self):
        expected = self.prepare()
        frame = self.frame.copy(deep=True)
        cutoff = self.dates[-1] - pd.DateOffset(years=1)
        fields = ["signal", "trend", "market_signal", "return_1d"]
        mask = frame["date"] > cutoff
        frame.loc[mask, fields] = frame.loc[mask, fields] * -3 + 1000
        actual = self.prepare(frame)
        for actual_split, expected_split in zip(actual[:2], expected[:2]):
            assert_dataset_equal(self, actual_split, expected_split)
        assert_frame_equal(actual[2], expected[2])

    def test_validation_values_cannot_change_training(self):
        train, validation, _ = self.prepare()
        frame = self.frame.copy(deep=True)
        fields = ["signal", "trend", "market_signal", "return_1d"]
        mask = frame["date"] >= pd.Timestamp(validation["dates"][0])
        frame.loc[mask, fields] = frame.loc[mask, fields] * -3 + 1000
        actual_train, _, _ = self.prepare(frame)
        assert_dataset_equal(self, actual_train, train)

    def test_duplicate_date_symbol_rows_are_rejected(self):
        frame = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate date/symbol"):
            self.prepare(frame)

    def test_missing_required_columns_are_rejected(self):
        for column in ("date", "symbol", "signal", "trend", "market_signal", "return_1d"):
            with self.subTest(column=column):
                with self.assertRaises((KeyError, ValueError)):
                    self.prepare(self.frame.drop(columns=column))

    def test_missing_configured_symbols_are_rejected(self):
        for symbol in ("XLB", "XLE", "SPY"):
            with self.subTest(symbol=symbol):
                with self.assertRaises(ValueError):
                    self.prepare(self.frame.loc[~self.frame["symbol"].eq(symbol)])

    def test_invalid_split_and_scaling_settings_are_rejected(self):
        for settings in ({"development_train_fraction": 0},
                         {"development_train_fraction": 1},
                         {"development_train_fraction": -.1},
                         {"development_train_fraction": 1.1},
                         {"final_holdout_years": 0},
                         {"final_holdout_years": -1},
                         {"scaling_window_years": 0},
                         {"scaling_window_years": 1 / 252}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    self.prepare(**settings)

    def test_missing_file_and_empty_csv_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            data.prepare_datasets(self.path, **self.settings)
        self.path.write_text("")
        with self.assertRaises(pd.errors.EmptyDataError):
            data.prepare_datasets(self.path, **self.settings)
        with self.assertRaisesRegex(ValueError, "No dates"):
            self.prepare(self.frame.iloc[:0])

    def test_all_missing_features_are_rejected(self):
        frame = self.frame.copy(deep=True)
        frame["signal"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing signal beyond"):
            self.prepare(frame)

    def test_nonnumeric_feature_values_are_rejected(self):
        frame = self.frame.copy(deep=True)
        frame["signal"] = "invalid"
        with self.assertRaises((TypeError, ValueError, pd.errors.DataError)):
            self.prepare(frame)

    def test_nonfinite_returns_are_rejected(self):
        for value in (np.inf, -np.inf):
            with self.subTest(value=value):
                frame = self.frame.copy(deep=True)
                frame.loc[frame["date"].eq(self.dates[100]), "return_1d"] = value
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.prepare(frame)

    def test_insufficient_history_and_undersized_splits_are_rejected(self):
        for settings, message in (
            ({"final_holdout_years": 10}, "Not enough history"),
            ({"scaling_window_years": 10}, "at least two dates"),
            ({"development_train_fraction": .001}, "Not enough history"),
            ({"development_train_fraction": .999}, "at least two dates"),
        ):
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare(**settings)

    def test_interior_missing_returns_are_rejected_to_preserve_daily_continuity(self):
        frame = self.frame.copy(deep=True)
        frame.loc[frame["date"].eq(self.dates[100]), "return_1d"] = np.nan
        with self.assertRaises(ValueError):
            self.prepare(frame)

    def test_interior_missing_features_are_rejected_to_preserve_daily_continuity(self):
        frame = self.frame.copy(deep=True)
        frame.loc[frame["date"].eq(self.dates[100]), "signal"] = np.nan
        with self.assertRaises(ValueError):
            self.prepare(frame)

    def test_infinite_features_are_rejected_before_clipping(self):
        frame = self.frame.copy(deep=True)
        frame.loc[frame["date"].eq(self.dates[100]), "signal"] = np.inf
        with self.assertRaises(ValueError):
            self.prepare(frame)

    def test_missing_entire_session_is_rejected_with_an_independent_calendar(self):
        frame = self.frame.loc[~self.frame["date"].eq(self.dates[100])]
        with self.assertRaisesRegex(ValueError, "Missing trading dates"):
            self.prepare(frame, expected_dates=self.dates)

    def test_undefined_scaling_cannot_silently_remove_interior_dates(self):
        frame = self.frame.copy(deep=True)
        frame.loc[frame["date"].isin(self.dates[100:107]), "signal"] = 1.
        with self.assertRaisesRegex(ValueError, "Undefined scaled features"):
            self.prepare(frame)


if __name__ == "__main__":
    unittest.main()
