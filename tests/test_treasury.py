"""Treasury data, sector-feature isolation, and long-only allocation checks."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from src.portfolio_rl import data
from src.portfolio_rl.environment import SectorAllocationEnv, action_to_weights


class TestTreasuryFeatures(unittest.TestCase):
    def setUp(self):
        self.dates = data.exchange_sessions("2024-01-04", "2024-07-01")[:100]
        self.frame = pd.DataFrame([
            {"date": date, "symbol": symbol, "tradable": symbol not in ("SPY", "^VIX"),
             "close": 100 * np.exp(.001 * i + .01 * np.sin(i / (position + 2)))}
            for i, date in enumerate(self.dates)
            for position, symbol in enumerate(("XLB", "XLE", "SHY", "SPY", "^VIX"))
        ])

    def test_shy_does_not_change_sector_features_or_sector_ranks(self):
        original = self.frame.copy(deep=True)
        baseline = data.build_market_features(self.frame.loc[self.frame["symbol"].ne("SHY")],
                                             ["XLB", "XLE"], expected_dates=self.dates)
        treatment = data.build_market_features(self.frame, ["XLB", "XLE", "SHY"],
                                              sector_symbols=["XLB", "XLE"], expected_dates=self.dates)
        common = treatment.loc[treatment["symbol"].ne("SHY")].reset_index(drop=True)
        assert_frame_equal(baseline, common)
        assert_frame_equal(self.frame, original)
        shy = treatment.loc[treatment["symbol"].eq("SHY")]
        self.assertTrue(shy["return_1d"].iloc[1:].notna().all())
        self.assertTrue(np.isfinite(shy[list(data.ASSET_FEATURES)].iloc[63:]).all().all())

    def test_treasury_rank_is_its_fraction_of_sector_peers_beaten(self):
        frame = self.frame.loc[self.frame["date"].eq(self.dates[0])].copy()
        frame["signal"] = frame["symbol"].map({"XLB": -.1, "XLE": .1, "SHY": 0., "SPY": .2, "^VIX": .3})
        ranks = data.calculate_cross_sectional_rank(frame, "signal", sector_symbols=["XLB", "XLE"])
        actual = dict(zip(frame["symbol"], ranks))
        self.assertEqual(actual["XLB"], .5)
        self.assertEqual(actual["XLE"], 1.)
        self.assertEqual(actual["SHY"], .5)
        self.assertTrue(np.isnan(actual["SPY"]))

    def test_missing_open_session_treasury_data_is_rejected(self):
        self.frame.loc[self.frame["symbol"].eq("SHY") & self.frame["date"].eq(self.dates[30]), "close"] = np.nan
        with self.assertRaisesRegex(ValueError, "close contains nonfinite"):
            data.build_market_features(self.frame, ["XLB", "XLE", "SHY"],
                                       sector_symbols=["XLB", "XLE"], expected_dates=self.dates)

    def test_treasury_requires_an_explicit_equity_sector_universe(self):
        for sectors in (None, ["XLB", "XLE", "SHY"]):
            with self.subTest(sectors=sectors), self.assertRaisesRegex(ValueError, "sector_symbols excluding SHY"):
                data.build_market_features(self.frame, ["XLB", "XLE", "SHY"],
                                           sector_symbols=sectors, expected_dates=self.dates)

    def test_pre_inception_missing_rows_are_rejected_instead_of_filled(self):
        frame = self.frame.loc[~(self.frame["symbol"].eq("SHY") & self.frame["date"].lt(self.dates[5]))]
        with self.assertRaisesRegex(ValueError, "Missing trading dates"):
            data.build_market_features(frame, ["XLB", "XLE", "SHY"],
                                       sector_symbols=["XLB", "XLE"], expected_dates=self.dates)

    def test_yahoo_download_requests_adjusted_prices_including_distributions(self):
        with patch.object(data.yf, "download", return_value=pd.DataFrame()) as download:
            data.download_daily_data(["SHY"], "2002-08-01", "2026-10-01")
        self.assertEqual(download.call_args.kwargs["tickers"], ["SHY"])
        self.assertTrue(download.call_args.kwargs["auto_adjust"])
        self.assertTrue(download.call_args.kwargs["actions"])
        self.assertTrue(download.call_args.kwargs["keepna"])


class TestTreasuryAllocation(unittest.TestCase):
    def weights(self, action, current=(.5, .5, 0.), **settings):
        defaults = {"action_scale": .1, "long_exposure": 1.3, "short_exposure": .3,
                    "treasury_index": 2}
        return action_to_weights(action, current, **(defaults | settings))

    def test_treasury_action_funds_it_by_reducing_equity_capital(self):
        assert_allclose(self.weights([0., 0., 1.]), [.45, .45, .1])
        assert_allclose(self.weights([0., 0., -1.]), [.5, .5, 0.])

    def test_full_defensive_allocation_and_return_to_equities(self):
        assert_allclose(self.weights([0., 0., 1.], current=(.025, .025, .95)), [0., 0., 1.])
        assert_allclose(self.weights([0., 0., -1.], current=(0., 0., 1.)), [.05, .05, .9])

    def test_treasury_scales_both_equity_longs_and_shorts(self):
        actual = self.weights([0., 0., 1.], current=(1.3, -.3, 0.))
        assert_allclose(actual, [1.17, -.27, .1], rtol=1e-6)
        self.assertAlmostEqual(actual.sum(), 1., places=6)
        self.assertLessEqual(np.abs(actual).sum(), 1.6)

    def test_treasury_position_can_be_anywhere_in_asset_order(self):
        actual = action_to_weights([1., 0., 0.], [0., .5, .5], .1, 1.3, .3, treasury_index=0)
        assert_allclose(actual, [.1, .45, .45])

    def test_zero_treasury_target_preserves_the_original_equity_mapping(self):
        action = [.8, -.4]
        expected = action_to_weights(action, [.5, .5], .1, 1.3, .3)
        actual = self.weights([*action, -1.])
        assert_allclose(actual[:2], expected)
        self.assertEqual(actual[2], 0.)

    def test_invalid_treasury_index_and_incompatible_budget_are_rejected(self):
        for options in ({"treasury_index": -1}, {"treasury_index": 3},
                        {"treasury_index": True}, {"long_exposure": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.weights([0., 0., 0.], **options)

    def test_environment_starts_in_equities_and_earns_treasury_return_after_action(self):
        returns = np.array([[0., 0., 0.], [.02, -.01, .001], [0., 0., 0.]], dtype=np.float32)
        dataset = {"dates": pd.bdate_range("2024-02-05", periods=3).to_numpy(),
                   "asset_states": np.zeros((3, 3), dtype=np.float32),
                   "spy_states": np.zeros((3, 1), dtype=np.float32),
                   "market_states": np.zeros((3, 1), dtype=np.float32),
                   "asset_returns": returns}
        env = SectorAllocationEnv(dataset, treasury_index=2, decision_frequency_days=1,
                                  transaction_cost_rate=.001, annual_short_borrow_rate=0.)
        observation, _ = env.reset(seed=42)
        assert_allclose(observation[-3:], [.5, .5, 0.])
        _, _, _, _, info = env.step([0., 0., 1.])
        record = info["daily_records"][0]
        assert_allclose(record["executed_weights"], [.45, .45, .1])
        self.assertAlmostEqual(record["turnover"], .2, places=6)
        self.assertAlmostEqual(record["net_return"], .45 * .02 - .45 * .01 + .1 * .001 - .001 * .2)
        self.assertEqual(record["short_borrow_cost"], 0.)



if __name__ == "__main__":
    unittest.main()
