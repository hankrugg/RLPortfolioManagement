"""Script-level checks using synthetic Yahoo responses and temporary outputs."""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import yaml

import prepare_data
from src.portfolio_rl import data


class TestPrepareDataScript(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.enterContext(patch.object(prepare_data, "PROJECT_ROOT", self.root))
        self.enterContext(redirect_stdout(StringIO()))
        self.config = {
            "start_date": "2024-01-04", "end_date": "2025-01-01",
            "tradable_symbols": ["XLB", "XLE"],
            "raw_path": "data/raw/market.csv.gz",
            "processed_path": "data/processed/features.csv.gz",
        }
        self.config_path = self.root / "data.yaml"
        dates = data.exchange_sessions("2024-01-04", "2024-07-01")[:100]
        self.config["end_date"] = (dates[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        self.wide = pd.concat({
            symbol: pd.DataFrame({"Close": 100 * np.exp(
                .001 * np.arange(100) + .01 * np.sin(np.arange(100) / (i + 2)))}, index=dates)
            for i, symbol in enumerate(("XLB", "XLE", "SPY", "^VIX"))
        }, axis=1)

    def run_script(self, *, raw_file=None):
        self.config_path.write_text(yaml.safe_dump(self.config))
        with patch.object(data, "download_daily_data", return_value=self.wide) as download:
            paths = prepare_data.prepare_data("data.yaml", raw_file=raw_file)
        return paths, download

    def test_download_validation_and_both_outputs_work_without_notebooks(self):
        (raw_path, processed_path), download = self.run_script()
        download.assert_called_once()
        self.assertEqual(raw_path, self.root / self.config["raw_path"])
        self.assertEqual(processed_path, self.root / self.config["processed_path"])
        raw = pd.read_csv(raw_path, parse_dates=["date"])
        processed = pd.read_csv(processed_path, parse_dates=["date"])
        self.assertEqual(len(raw), 400)
        self.assertEqual(len(processed), 400)
        self.assertNotIn("return_1d", raw.columns)
        self.assertTrue(set(data.FEATURE_WARMUP_PERIODS).issubset(processed.columns))
        expected = data.build_market_features(raw, self.config["tradable_symbols"])
        assert_frame_equal(processed, expected, check_dtype=False, rtol=1e-8, atol=1e-10)

    def test_invalid_download_does_not_replace_either_existing_output(self):
        self.wide.loc[self.wide.index[10], ("XLB", "Close")] = np.nan
        paths = [self.root / self.config[key] for key in ("raw_path", "processed_path")]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"previous output")
        with self.assertRaisesRegex(ValueError, "close contains nonfinite"):
            self.run_script()
        for path in paths:
            self.assertEqual(path.read_bytes(), b"previous output")

    def test_feature_failure_keeps_valid_raw_and_previous_processed_output(self):
        processed_path = self.root / self.config["processed_path"]
        processed_path.parent.mkdir(parents=True)
        processed_path.write_bytes(b"previous processed")
        with patch.object(prepare_data, "build_market_features", side_effect=ValueError("invalid feature")):
            with self.assertRaisesRegex(ValueError, "invalid feature"):
                self.run_script()
        raw_path = self.root / self.config["raw_path"]
        self.assertEqual(len(pd.read_csv(raw_path)), 400)
        self.assertEqual(processed_path.read_bytes(), b"previous processed")

    def test_local_raw_file_rebuilds_without_downloading(self):
        (raw_path, processed_path), _ = self.run_script()
        expected = pd.read_csv(processed_path)
        (_, actual_path), download = self.run_script(raw_file=raw_path)
        download.assert_not_called()
        assert_frame_equal(pd.read_csv(actual_path), expected)

    def test_independent_calendar_is_used_before_saving(self):
        calendar = pd.DataFrame({"date": self.wide.index})
        calendar.to_csv(self.root / "calendar.csv", index=False)
        self.config["calendar_path"] = "calendar.csv"
        self.wide = self.wide.drop(index=self.wide.index[10])
        with self.assertRaisesRegex(ValueError, "Missing trading dates"):
            self.run_script()
        self.assertFalse((self.root / self.config["raw_path"]).exists())

    def test_holiday_rows_are_excluded_from_raw_and_processed_outputs(self):
        holiday = pd.DataFrame(np.nan, index=pd.to_datetime(["2024-01-15"]), columns=self.wide.columns)
        holiday.loc[:, ("^VIX", "Close")] = 30.
        self.wide = pd.concat([self.wide, holiday]).sort_index()
        paths, _ = self.run_script()
        for path in paths:
            saved = pd.read_csv(path, parse_dates=["date"])
            self.assertEqual(len(saved), 400)
            self.assertFalse(saved["date"].eq("2024-01-15").any())

    def test_missing_entire_session_is_rejected_without_a_custom_calendar(self):
        self.wide = self.wide.drop(index=self.wide.index[10])
        with self.assertRaisesRegex(ValueError, "Missing trading dates"):
            self.run_script()

    def test_local_raw_holiday_rows_are_excluded_without_filling_session_prices(self):
        (raw_path, _), _ = self.run_script()
        raw = pd.read_csv(raw_path, parse_dates=["date"])
        holiday = raw.iloc[:4].copy()
        holiday["date"] = pd.Timestamp("2024-01-15")
        holiday["close"] = np.nan
        pd.concat([raw, holiday]).to_csv(raw_path, index=False)
        paths, download = self.run_script(raw_file=raw_path)
        download.assert_not_called()
        self.assertEqual(len(pd.read_csv(paths[0])), 400)

    def test_invalid_configuration_fails_before_download(self):
        for overrides in (
            {"end_date": "1990-01-01"}, {"start_date": None},
            {"tradable_symbols": []}, {"tradable_symbols": ["XLB", "XLB"]},
            {"context_symbols": ["SPY"]},
            {"processed_path": self.config["raw_path"]}, {"raw_path": "market.txt"},
        ):
            with self.subTest(overrides=overrides):
                self.config_path.write_text(yaml.safe_dump(self.config | overrides))
                with patch.object(data, "download_daily_data") as download:
                    with self.assertRaises(ValueError):
                        prepare_data.prepare_data(self.config_path)
                    download.assert_not_called()

    def test_failed_write_preserves_existing_file_and_cleans_temporary_output(self):
        path = self.root / "output.csv.gz"
        path.write_bytes(b"previous output")
        frame = pd.DataFrame({"close": [100]})
        with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                prepare_data.save_csv(frame, path)
        self.assertEqual(path.read_bytes(), b"previous output")
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_command_line_returns_nonzero_for_validation_errors(self):
        self.config_path.write_text(yaml.safe_dump(self.config | {"tradable_symbols": []}))
        errors = StringIO()
        with redirect_stderr(errors):
            with self.assertRaises(SystemExit) as raised:
                prepare_data.main([str(self.config_path)])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Data preparation failed", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
