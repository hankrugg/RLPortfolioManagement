"""


"""

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal

from src.portfolio_rl import data as features


def market_frame(series, column="close", *, context_symbols=("SPY", "^VIX")):
    """Interleave symbols by trading date and retain a nondefault row index."""
    dates = pd.bdate_range("2024-01-04", periods=max(map(len, series.values())))
    rows = [
        {"date": dates[i], "symbol": symbol,
         "tradable": symbol not in context_symbols, column: value}
        for symbol, values in series.items()
        for i, value in enumerate(values)
    ]
    frame = pd.DataFrame(rows).sort_values(["date", "symbol"])
    frame.index = np.arange(len(frame)) * 3 + 100
    return frame


def for_symbol(frame, result, symbol):
    return result.loc[frame["symbol"].eq(symbol)].to_numpy(dtype=float)


class TestReturns(unittest.TestCase):
    def test_simple_returns_use_each_symbols_own_history(self):
        frame = market_frame({"XLB": [100, 110, 99, 198],
                              "XLE": [50, 50, 25, 50]})
        for window, expected in [
            (1, {"XLB": [np.nan, .1, -.1, 1],
                 "XLE": [np.nan, 0, -.5, 1]}),
            (2, {"XLB": [np.nan, np.nan, -.01, .8],
                 "XLE": [np.nan, np.nan, -.5, 0]}),
        ]:
            with self.subTest(window=window):
                result = features.calculate_simple_return(frame, window)
                for symbol, values in expected.items():
                    assert_allclose(for_symbol(frame, result, symbol), values)

    def test_simple_returns_do_not_fill_missing_prices(self):
        frame = market_frame({"XLB": [100, np.nan, 110, 121]})
        assert_allclose(features.calculate_simple_return(frame, 1),
                        [np.nan, np.nan, np.nan, .1])

    def test_log_returns_are_log_price_ratios(self):
        frame = market_frame({"XLB": [100, 200, 100], "XLE": [50, 50, 100]})
        result = features.calculate_log_return(frame)
        assert_allclose(for_symbol(frame, result, "XLB"),
                        [np.nan, np.log(2), -np.log(2)])
        assert_allclose(for_symbol(frame, result, "XLE"),
                        [np.nan, 0, np.log(2)])

    def test_log_returns_do_not_bridge_missing_prices(self):
        frame = market_frame({"XLB": [100, np.nan, 200, 100]})
        assert_allclose(features.calculate_log_return(frame),
                        [np.nan, np.nan, np.nan, -np.log(2)])

    def test_large_valid_price_moves_are_not_clipped(self):
        frame = market_frame({"XLE": [100, 80, 160]})
        assert_allclose(features.calculate_simple_return(frame, 1), [np.nan, -.2, 1])


class TestRollingPriceFeatures(unittest.TestCase):
    def test_volatility_uses_sample_log_return_std_and_252_days(self):
        frame = market_frame({"XLB": [100, 200, 100, 200],
                              "XLE": [10, 10, 10, 10]})
        result = features.calculate_annualized_volatility(frame, 2)
        expected = np.log(2) * np.sqrt(2) * np.sqrt(252)
        assert_allclose(for_symbol(frame, result, "XLB"),
                        [np.nan, np.nan, expected, expected])
        assert_allclose(for_symbol(frame, result, "XLE"), [np.nan, np.nan, 0, 0])

    def test_volatility_requires_a_complete_window_of_returns(self):
        frame = market_frame({"XLB": [100, 110, np.nan, 120, 100, 110]})
        result = features.calculate_annualized_volatility(frame, 2)
        self.assertTrue(result.iloc[:5].isna().all())
        expected = abs(np.log(100 / 120) - np.log(110 / 100)) * np.sqrt(126)
        self.assertAlmostEqual(result.iloc[5], expected)

    def test_drawdown_includes_today_and_forgets_old_peaks(self):
        frame = market_frame({"XLB": [100, 80, 90, 120, 60]})
        assert_allclose(features.calculate_rolling_drawdown(frame, 2),
                        [np.nan, -.2, 0, 0, -.5])

    def test_drawdown_is_isolated_by_symbol(self):
        frame = market_frame({"XLB": [100, 80, 60], "XLE": [10, 20, 30]})
        result = features.calculate_rolling_drawdown(frame, 2)
        assert_allclose(for_symbol(frame, result, "XLB"), [np.nan, -.2, -.25])
        assert_allclose(for_symbol(frame, result, "XLE"), [np.nan, 0, 0])

    def test_drawdown_recovers_only_after_a_complete_window(self):
        frame = market_frame({"XLB": [100, np.nan, 80, 60]})
        assert_allclose(features.calculate_rolling_drawdown(frame, 2),
                        [np.nan, np.nan, np.nan, -.25])

    def test_insufficient_history_stays_missing(self):
        frame = market_frame({"XLB": [100, 110]})
        for function in (features.calculate_simple_return,
                         features.calculate_annualized_volatility,
                         features.calculate_rolling_drawdown):
            with self.subTest(feature=function.__name__):
                self.assertTrue(function(frame, 3).isna().all())


class TestCrossSectionalFeatures(unittest.TestCase):
    def test_rank_uses_average_ties_and_only_available_tradable_values(self):
        frame = market_frame({"XLB": [.1, np.nan], "XLE": [.1, .2],
                              "XLF": [.3, .1], "SPY": [100, 100],
                              "^VIX": [-100, -100]}, "return_2d")
        result = features.calculate_cross_sectional_rank(frame, "return_2d")
        for symbol, expected in {"XLB": [.5, np.nan], "XLE": [.5, 1],
                                 "XLF": [1, .5], "SPY": [np.nan, np.nan],
                                 "^VIX": [np.nan, np.nan]}.items():
            assert_allclose(for_symbol(frame, result, symbol), expected)

    def test_breadth_counts_strictly_above_average_and_maps_to_context(self):
        frame = market_frame({"XLB": [100, 110, 110], "XLE": [100, 90, 90],
                              "SPY": [100, 200, 400]})
        result = features.calculate_sector_breadth(frame, 2)
        for symbol in ("XLB", "XLE", "SPY"):
            assert_allclose(for_symbol(frame, result, symbol), [np.nan, .5, 0])

    def test_breadth_excludes_sectors_without_a_complete_price_window(self):
        frame = market_frame({"XLB": [100, 110], "XLE": [np.nan, 90]})
        assert_allclose(features.calculate_sector_breadth(frame, 2).to_numpy(dtype=float),
                        [np.nan, np.nan, 1, 1])

    def test_dispersion_is_sample_std_across_sectors_on_each_date(self):
        frame = market_frame({"XLB": [.1, .2], "XLE": [.3, .2],
                              "SPY": [100, -100]}, "return_2d")
        result = features.calculate_sector_return_dispersion(frame, 2)
        for symbol in ("XLB", "XLE", "SPY"):
            assert_allclose(for_symbol(frame, result, symbol), [np.sqrt(.02), 0])

    def test_dispersion_needs_at_least_two_nonmissing_sectors(self):
        frame = market_frame({"XLB": [.1], "XLE": [np.nan], "SPY": [.2]},
                             "return_2d")
        self.assertTrue(features.calculate_sector_return_dispersion(frame, 2).isna().all())

    def test_no_tradable_observations_produce_no_sector_statistics(self):
        frame = market_frame({"SPY": [100, 110, 90], "^VIX": [20, 25, 15]})
        frame["return_1d"] = [.1, -.1, .2, -.2, .3, -.3]
        frame["return_2d"] = frame["return_1d"]
        calculations = (
            lambda d: features.calculate_cross_sectional_rank(d, "return_2d"),
            lambda d: features.calculate_sector_breadth(d, 2),
            lambda d: features.calculate_sector_return_dispersion(d, 2),
            lambda d: features.calculate_sector_average_correlation(d, 2),
        )
        for position, calculate in enumerate(calculations):
            with self.subTest(calculation=position):
                self.assertTrue(calculate(frame).isna().all())


class TestMarketContext(unittest.TestCase):
    def test_relative_returns_align_spy_by_date_not_row_position(self):
        frame = market_frame({"XLB": [.1, -.1, .2], "SPY": [.02, .03, -.04]},
                             "return_2d").iloc[::-1]
        result = features.calculate_market_relative_return(frame, 2)
        assert_allclose(for_symbol(frame, result, "XLB"), [.24, -.13, .08])
        assert_allclose(for_symbol(frame, result, "SPY"), [0, 0, 0])

    def test_missing_spy_date_does_not_use_previous_market_return(self):
        frame = market_frame({"XLB": [.1, .2], "SPY": [.02]}, "return_2d")
        result = features.calculate_market_relative_return(frame, 2)
        assert_allclose(for_symbol(frame, result, "XLB"), [.08, np.nan])

    def test_beta_has_known_positive_negative_and_market_values(self):
        frame = market_frame({"SPY": [.01, -.02, .03, -.04],
                              "XLB": [.02, -.04, .06, -.08],
                              "XLE": [-.01, .02, -.03, .04]}, "return_1d")
        result = features.calculate_rolling_market_beta(frame, 3)
        for symbol, beta in {"SPY": 1, "XLB": 2, "XLE": -1}.items():
            assert_allclose(for_symbol(frame, result, symbol),
                            [np.nan, np.nan, beta, beta])

    def test_beta_is_undefined_when_market_variance_is_zero(self):
        frame = market_frame({"SPY": [0, 0, 0], "XLB": [.1, -.1, .2]},
                             "return_1d")
        self.assertTrue(features.calculate_rolling_market_beta(frame, 3).isna().all())

    def test_context_maps_by_date_including_missing_and_extra_context_dates(self):
        frame = market_frame({"XLB": [100, 110, 120], "^VIX": [20, np.nan, 40, 50]})
        result = features.map_context(frame.iloc[::-1], "^VIX", "close")
        assert_allclose(for_symbol(frame.iloc[::-1], result, "XLB"), [40, np.nan, 20])
        self.assertEqual(result.iloc[0], 50)

    def test_absent_context_symbol_produces_missing_values(self):
        frame = market_frame({"XLB": [100, 110]})
        self.assertTrue(features.map_context(frame, "^VIX", "close").isna().all())


class TestSectorCorrelation(unittest.TestCase):
    def test_average_counts_each_distinct_pair_once_and_excludes_context(self):
        # Three pairs have correlations +1, -1, -1: average = -1/3.
        frame = market_frame({"XLB": [.01, -.02, .03, -.04],
                              "XLE": [.02, -.04, .06, -.08],
                              "XLF": [-.01, .02, -.03, .04],
                              "SPY": [.7, .2, -.1, .9]}, "return_1d")
        result = features.calculate_sector_average_correlation(frame, 3)
        for symbol in ("XLB", "XLE", "XLF", "SPY"):
            assert_allclose(for_symbol(frame, result, symbol),
                            [np.nan, np.nan, -1 / 3, -1 / 3])

    def test_correlation_window_advances_and_accepts_shuffled_rows(self):
        frame = market_frame({"XLB": [0, .1, .2, .3],
                              "XLE": [0, .1, .2, .1]}, "return_1d")
        frame = frame.sample(frac=1, random_state=42)
        result = features.calculate_sector_average_correlation(frame, 3)
        expected = frame["date"].map(dict(zip(
            sorted(frame["date"].unique()), [np.nan, np.nan, 1, 0])))
        assert_allclose(result, expected, atol=1e-12)

    def test_correlation_is_undefined_with_a_constant_or_missing_sector(self):
        for series in ({"XLB": [0, 0, 0], "XLE": [.1, .2, .3]},
                       {"XLB": [np.nan] * 3, "XLE": [.1, .2, .3]}):
            with self.subTest(series=series):
                frame = market_frame(series, "return_1d")
                self.assertTrue(features.calculate_sector_average_correlation(frame, 3).isna().all())

    def test_one_sector_has_no_pairwise_correlation(self):
        frame = market_frame({"XLB": [.1, .2, .3]}, "return_1d")
        # The current implementation warns when averaging an empty set of pairs.
        with self.assertWarns(RuntimeWarning):
            result = features.calculate_sector_average_correlation(frame, 3)
        self.assertTrue(result.isna().all())


class TestFeatureContracts(unittest.TestCase):
    def test_features_preserve_index_inputs_and_do_not_use_future_rows(self):
        frame = market_frame({"XLB": [100, 110, 90, 120, 80, 130],
                              "XLE": [50, 45, 60, 55, 70, 65],
                              "SPY": [100, 105, 95, 110, 100, 120],
                              "^VIX": [20, 25, 15, 30, 20, 35]})
        frame["return_1d"] = features.calculate_simple_return(frame, 1)
        frame["return_2d"] = features.calculate_simple_return(frame, 2)
        original = frame.copy(deep=True)
        cutoff = frame["date"].sort_values().unique()[3]
        prefix_mask = frame["date"] <= cutoff
        changed = frame.copy(deep=True)
        changed.loc[~prefix_mask, ["close", "return_1d", "return_2d"]] = 999
        calculations = {
            "simple return": lambda d: features.calculate_simple_return(d, 2),
            "log return": features.calculate_log_return,
            "volatility": lambda d: features.calculate_annualized_volatility(d, 2),
            "drawdown": lambda d: features.calculate_rolling_drawdown(d, 2),
            "rank": lambda d: features.calculate_cross_sectional_rank(d, "return_2d"),
            "relative return": lambda d: features.calculate_market_relative_return(d, 2),
            "beta": lambda d: features.calculate_rolling_market_beta(d, 2),
            "breadth": lambda d: features.calculate_sector_breadth(d, 2),
            "dispersion": lambda d: features.calculate_sector_return_dispersion(d, 2),
            "correlation": lambda d: features.calculate_sector_average_correlation(d, 2),
            "context": lambda d: features.map_context(d, "^VIX", "close"),
        }
        for name, calculate in calculations.items():
            with self.subTest(feature=name):
                result = calculate(frame)
                self.assertTrue(result.index.equals(frame.index))
                assert_frame_equal(frame, original)
                assert_series_equal(result.loc[prefix_mask],
                                    calculate(frame.loc[prefix_mask]), check_names=False)
                assert_series_equal(result.loc[prefix_mask],
                                    calculate(changed).loc[prefix_mask], check_names=False)


if __name__ == "__main__":
    unittest.main()
