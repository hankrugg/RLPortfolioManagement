"""Download validated market data and save the features consumed by run.py."""

import argparse
import logging
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import pandas as pd
import yaml

from src.portfolio_rl.data import download_market_data, build_market_features, sort_market_data
from src.portfolio_rl.calendar import exchange_sessions, select_exchange_sessions
from src.portfolio_rl.validation import validate_market_data, validate_symbols


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"


def save_csv(data, path):
    """Replace an output only after the complete CSV has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        data.to_csv(temporary_path, index=False,
                    compression="gzip" if path.suffix == ".gz" else None)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_data(config_path=DEFAULT_CONFIG, *, raw_file=None):
    """Validate and save raw data, then construct, validate, and save features.

    Resolve configured paths from the project root. A failed stage raises its
    error and preserves that stage's previous output; validated raw data remains
    available if feature construction fails. No training is performed here.
    """
    config_path = PROJECT_ROOT / Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    required = {"start_date", "end_date", "tradable_symbols", "raw_path", "processed_path"}
    if not isinstance(config, dict) or required - config.keys():
        raise ValueError(f"Data config must define {sorted(required)}.")
    assets = config["tradable_symbols"]
    context = config.get("context_symbols", ["SPY", "^VIX"])
    if not isinstance(assets, list) or not isinstance(context, list):
        raise ValueError("tradable_symbols and context_symbols must be lists.")
    metadata = pd.DataFrame([{"symbol": symbol, "tradable": symbol in assets}
                             for symbol in [*assets, *context]])
    validate_symbols(metadata, assets, context)
    sectors = config.get("sector_symbols")
    if sectors is not None:
        validate_symbols(metadata, sectors, context)
        if len(sectors) < 2 or not set(sectors).issubset(assets):
            raise ValueError("sector_symbols must contain at least two configured tradable symbols.")
    if "SHY" in assets and (sectors is None or "SHY" in sectors):
        raise ValueError("Specify sector_symbols excluding SHY so Treasury data cannot change sector features.")
    if set(context) != {"SPY", "^VIX"}:
        raise ValueError("The current feature builder requires SPY and ^VIX context.")
    start, end = pd.Timestamp(config["start_date"]), pd.Timestamp(config["end_date"])
    if (pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None
            or start != start.normalize() or end != end.normalize() or start >= end):
        raise ValueError("Use daily start_date < end_date without timezones; end_date is exclusive.")
    raw_path = (PROJECT_ROOT / config["raw_path"]).resolve()
    processed_path = (PROJECT_ROOT / config["processed_path"]).resolve()
    if raw_path == processed_path:
        raise ValueError("raw_path and processed_path must be different files.")
    for path in (raw_path, processed_path):
        if not (path.name.endswith(".csv") or path.name.endswith(".csv.gz")):
            raise ValueError("Output paths must end in .csv or .csv.gz.")
    calendar_path = config.get("calendar_path")
    calendar = (pd.DatetimeIndex(pd.read_csv(PROJECT_ROOT / calendar_path,
                                            parse_dates=["date"])["date"])
                if calendar_path else exchange_sessions(start, end, config.get("exchange_calendar", "XNYS")))
    print(f"Validating against {len(calendar):,} expected exchange sessions.", flush=True)

    if raw_file is None:
        print(f"Downloading {len(assets)} assets and {len(context)} context series...", flush=True)
        market = download_market_data(assets, metadata, start.strftime("%Y-%m-%d"),
                                      end.strftime("%Y-%m-%d"), context_symbols=context,
                                      expected_dates=calendar)
    else:
        market = sort_market_data(pd.read_csv(PROJECT_ROOT / Path(raw_file), parse_dates=["date"]))
        market = select_exchange_sessions(market, calendar).reset_index(drop=True)
        validate_market_data(market, assets, context_symbols=context, expected_dates=calendar)
    save_csv(market, raw_path)
    print(f"Saved {len(market):,} validated raw rows: {raw_path}", flush=True)

    featured = build_market_features(market, assets, expected_dates=calendar, sector_symbols=sectors)
    save_csv(featured, processed_path)
    print(f"Saved {len(featured):,} validated feature rows: {processed_path}", flush=True)
    print("Data preparation complete. Training can now load the processed file.", flush=True)
    return raw_path, processed_path


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG,
                        help="Data YAML; relative paths resolve from the project root.")
    parser.add_argument("--raw-file", help="Rebuild from a local raw CSV instead of downloading.")
    args = parser.parse_args(argv)
    try:
        prepare_data(args.config, raw_file=args.raw_file)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
