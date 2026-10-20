"""Download, validate, and construct features for portfolio training."""

import numpy as np
import pandas as pd
import yfinance as yf

from .calendar import exchange_sessions, select_exchange_sessions
from .validation import (
    FEATURE_BOUNDS, FEATURE_WARMUP_PERIODS,
    validate_dates, validate_feature_data, validate_market_data, validate_symbols,
)


ASSET_FEATURES = (
    "return_5d", "return_21d", "return_63d", "volatility_21d",
    "rolling_drawdown_63d", "cross_sectional_rank_return_21d",
    "market_relative_return_21d", "rolling_market_beta_63d",
)
SPY_FEATURES = ASSET_FEATURES[:5]
MARKET_FEATURES = (
    "sector_breadth_above_sma_63d", "sector_return_dispersion_21d",
    "sector_average_correlation_63d", "vix_level", "vix_return_5d",
)

def download_daily_data(tickers, start_date, end_date):
    """ Download the data from Yahoo finance for the given tickers, start, and end data"""

    raw_wide = yf.download(
    tickers=tickers,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=True,
    actions=True,
    group_by="ticker",
    threads=True,
    progress=True,
    keepna=True,
)
    return raw_wide

def calculate_simple_return(data, window):
    """Close-to-close percentage return over the selected window."""
    return data.groupby("symbol")["close"].pct_change(
        periods=window, fill_method=None
    )


def calculate_log_return(data):
    """One-day log return."""
    return np.log(data["close"]).groupby(data["symbol"]).diff()


def calculate_annualized_volatility(data, window):
    """Annualized rolling standard deviation of daily log returns."""
    log_return = calculate_log_return(data)
    return (
        log_return.groupby(data["symbol"])
        .transform(lambda returns: returns.rolling(window).std())
        * np.sqrt(252)
    )

def calculate_rolling_drawdown(data, window):
    """Current price relative to its rolling maximum."""
    rolling_high = data.groupby("symbol")["close"].transform(
        lambda prices: prices.rolling(window).max()
    )
    return data["close"] / rolling_high - 1

def calculate_cross_sectional_rank(data, feature, *, sector_symbols=None):
    """Rank sectors among themselves; score other assets against sector returns.

    A non-sector asset receives the fraction of available sector returns at or
    below its own return. It never changes sector ranks. Require complete peers
    for that comparison, preserving initial warm-up rather than inventing ranks.
    Without sector_symbols, retain the original tradable-universe ranks.
    """
    peers = data["tradable"] if sector_symbols is None else data["symbol"].isin(sector_symbols)
    ranks = (
        data[feature]
        .where(peers)
        .groupby(data["date"])
        .rank(pct=True)
    )
    if sector_symbols is not None:
        panel = data.loc[peers].pivot(index="date", columns="symbol", values=feature)
        other = data["tradable"] & ~peers
        for symbol in data.loc[other, "symbol"].unique():
            values = data.loc[data["symbol"].eq(symbol)].set_index("date")[feature]
            fractions = panel.le(values, axis=0).mean(axis=1)
            fractions = fractions.where(panel.notna().all(axis=1) & values.notna())
            mask = data["symbol"].eq(symbol)
            ranks.loc[mask] = data.loc[mask, "date"].map(fractions)
    return ranks


def calculate_market_relative_return(data, window):
    """Asset return minus the return on SPY over the same window."""
    return_column = f"return_{window}d"
    market_return = (
        data.loc[data["symbol"] == "SPY"]
        .set_index("date")[return_column]
    )
    return data[return_column] - data["date"].map(market_return)


def calculate_rolling_market_beta(data, window):
    """Rolling covariance with SPY divided by SPY variance."""
    spy_return = (
        data.loc[data["symbol"] == "SPY"]
        .set_index("date")["return_1d"]
    )
    market_return = data["date"].map(spy_return)
    covariance = data["return_1d"].groupby(data["symbol"]).transform(
        lambda asset_return: asset_return.rolling(window).cov(
            market_return.loc[asset_return.index]
        )
    )
    market_variance = market_return.groupby(data["symbol"]).transform(
        lambda returns: returns.rolling(window).var()
    )
    return covariance / market_variance

def calculate_sector_breadth(data, window):
    """Fraction of sector ETFs trading above their moving average."""
    moving_average = data.groupby("symbol")["close"].transform(
        lambda prices: prices.rolling(window).mean()
    )
    above_average = (
        data["close"].gt(moving_average)
        .where(data["tradable"] & moving_average.notna())
    )
    breadth = above_average.groupby(data["date"]).mean()
    return data["date"].map(breadth)


def calculate_sector_return_dispersion(data, window):
    """Cross-sectional standard deviation of sector returns."""
    return_column = f"return_{window}d"
    dispersion = (
        data[return_column]
        .where(data["tradable"])
        .groupby(data["date"])
        .std()
    )
    return data["date"].map(dispersion)


def calculate_sector_average_correlation(data, window):
    """Mean pairwise rolling correlation among sector ETF returns."""
    return_panel = data.loc[data["tradable"]].pivot(
        index="date", columns="symbol", values="return_1d"
    )
    average_correlation = pd.Series(index=return_panel.index, dtype=float)

    for position in range(window - 1, len(return_panel)):
        correlation = (
            return_panel.iloc[position - window + 1 : position + 1]
            .corr()
            .to_numpy()
        )
        upper_triangle = correlation[
            np.triu_indices(correlation.shape[0], k=1)
        ]
        average_correlation.iloc[position] = upper_triangle.mean()

    return data["date"].map(average_correlation)


def map_context(data, symbol, feature):
    """Map one context-series feature to every row on the same date."""
    context = data.loc[data["symbol"] == symbol].set_index("date")[feature]
    return data["date"].map(context)

def normalize_yahoo_data(tickers, yahoo_df, symbol_metadata):
    frames = []
    for symbol in tickers:
        frame = yahoo_df[symbol].copy()
        frame.columns.name = None
        frame.index.name = "date"
        frame = frame.reset_index()
        frame.insert(1, "symbol", symbol)
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
        frames.append(frame)

    market_data = pd.concat(frames, ignore_index=True)
    market_data = market_data.merge(
        symbol_metadata,
        on="symbol",
        how="left",
    )
    return market_data

def sort_market_data(market_data):
    """Sort by date and symbol without removing rows or columns."""
    return market_data.sort_values(["date", "symbol"]).reset_index(drop=True)


def download_market_data(
    tradable_symbols, symbol_metadata, start_date, end_date, *,
    context_symbols=("SPY", "^VIX"), expected_dates=None,
):
    """Download, normalize, and validate market data before it can be saved.

    Use NYSE sessions unless expected_dates supplies an explicit calendar. Only
    non-session rows are excluded; missing data on actual sessions raises errors.
    """
    symbols = [*tradable_symbols, *context_symbols]
    if expected_dates is None:
        expected_dates = exchange_sessions(start_date, end_date)
    raw_wide = download_daily_data(symbols, start_date, end_date)
    market_data = sort_market_data(normalize_yahoo_data(symbols, raw_wide, symbol_metadata))
    market_data = select_exchange_sessions(market_data, expected_dates).reset_index(drop=True)
    validate_market_data(market_data, tradable_symbols, context_symbols=context_symbols,
                         expected_dates=expected_dates)
    return market_data


def build_market_features(market_data, tradable_symbols, *, expected_dates=None, sector_symbols=None):
    """Validate market input, construct standard features, and validate the result.

    Use this entry point for downloaded data or a loaded raw CSV. The caller's
    table is not modified. Only return a feature table suitable for saving after
    every required raw and feature check has succeeded.
    """
    featured_data = sort_market_data(market_data)
    validate_dates(featured_data)
    validate_symbols(featured_data, tradable_symbols, ["SPY", "^VIX"])
    if expected_dates is None:
        dates = featured_data.loc[featured_data["symbol"].isin(tradable_symbols), "date"]
        expected_dates = exchange_sessions(dates.min(), dates.max() + pd.Timedelta(days=1))
    featured_data = select_exchange_sessions(featured_data, expected_dates).reset_index(drop=True)
    validate_market_data(featured_data, tradable_symbols, expected_dates=expected_dates)
    if "SHY" in tradable_symbols and (sector_symbols is None or "SHY" in sector_symbols):
        raise ValueError("Specify sector_symbols excluding SHY so Treasury data cannot change sector features.")
    if sector_symbols is not None:
        validate_symbols(featured_data, sector_symbols, ["SPY", "^VIX"])
        if len(sector_symbols) < 2 or not set(sector_symbols).issubset(tradable_symbols):
            raise ValueError("sector_symbols must contain at least two configured tradable symbols.")
    for window in (1, 5, 21, 63):
        featured_data[f"return_{window}d"] = calculate_simple_return(featured_data, window)
    featured_data["volatility_21d"] = calculate_annualized_volatility(featured_data, 21)
    featured_data["rolling_drawdown_63d"] = calculate_rolling_drawdown(featured_data, 63)
    featured_data["cross_sectional_rank_return_21d"] = calculate_cross_sectional_rank(
        featured_data, "return_21d", sector_symbols=sector_symbols)
    featured_data["market_relative_return_21d"] = calculate_market_relative_return(featured_data, 21)
    featured_data["rolling_market_beta_63d"] = calculate_rolling_market_beta(featured_data, 63)
    sector_data = featured_data.copy()
    if sector_symbols is not None:
        sector_data["tradable"] = sector_data["symbol"].isin(sector_symbols)
    featured_data["sector_breadth_above_sma_63d"] = calculate_sector_breadth(sector_data, 63)
    featured_data["sector_return_dispersion_21d"] = calculate_sector_return_dispersion(sector_data, 21)
    featured_data["sector_average_correlation_63d"] = calculate_sector_average_correlation(sector_data, 63)
    featured_data["vix_level"] = map_context(featured_data, "^VIX", "close")
    featured_data["vix_return_5d"] = map_context(featured_data, "^VIX", "return_5d")
    validate_feature_data(
        featured_data, tradable_symbols, ASSET_FEATURES, SPY_FEATURES, MARKET_FEATURES,
        warmup_periods=FEATURE_WARMUP_PERIODS, expected_dates=expected_dates,
        feature_bounds=FEATURE_BOUNDS,
    )
    return featured_data


def prepare_datasets(
    feature_path,
    tradable_symbols,
    asset_features,
    spy_features,
    market_features,
    development_train_fraction=0.80,
    final_holdout_years=2,
    scaling_window_years=5,
    observation_clip=10.0,
    feature_warmup_periods=None,
    expected_dates=None,
    exchange_calendar="XNYS",
):
    """Prepare the same chronological arrays and splits as notebook 03.

    Any prepared feature CSV with the same columns can be used, including a
    separately generated bootstrap path. The final holdout is never evaluated
    by the batch runner. Built-in features use their defined warm-up periods;
    custom features allow no missing values unless feature_warmup_periods declares
    an allowance. Validate dates against exchange_calendar unless expected_dates
    supplies an explicit session calendar. Use exchange_calendar=None only for
    synthetic timelines that intentionally do not represent exchange dates.
    """
    from .environment import build_feature_panel, scale_rolling, slice_dataset

    if not 0 < development_train_fraction < 1 or final_holdout_years <= 0:
        raise ValueError("Use a train fraction between 0 and 1 and a positive holdout.")
    scaling_window = int(scaling_window_years * 252)
    if scaling_window < 2:
        raise ValueError("The scaling window must contain at least two trading days.")

    market_data = pd.read_csv(feature_path, parse_dates=["date"])
    if expected_dates is None and exchange_calendar is not None:
        validate_dates(market_data)
        validate_symbols(market_data, tradable_symbols, ["SPY"])
        dates = market_data.loc[market_data["symbol"].isin(tradable_symbols), "date"]
        expected_dates = exchange_sessions(dates.min(), dates.max() + pd.Timedelta(days=1),
                                           exchange_calendar)
    requested = list(dict.fromkeys([*asset_features, *spy_features, *market_features, "return_1d"]))
    warmups = {column: FEATURE_WARMUP_PERIODS.get(column, 0) for column in requested}
    if feature_warmup_periods is not None:
        if not isinstance(feature_warmup_periods, dict):
            raise ValueError("feature_warmup_periods must map feature names to session counts.")
        if set(feature_warmup_periods) - set(requested):
            raise ValueError("feature_warmup_periods contains unrequested feature names.")
        warmups.update(feature_warmup_periods)
    validate_feature_data(
        market_data, tradable_symbols, asset_features, spy_features, market_features,
        warmup_periods=warmups, expected_dates=expected_dates,
        feature_bounds={column: FEATURE_BOUNDS[column] for column in requested if column in FEATURE_BOUNDS},
    )
    asset_panel = build_feature_panel(market_data, tradable_symbols, asset_features)
    spy_panel = build_feature_panel(market_data, ["SPY"], spy_features)
    market_panel = (
        market_data.loc[market_data["symbol"] == "SPY"]
        .set_index("date")[market_features].sort_index()
    )
    returns_panel = (
        market_data.loc[market_data["symbol"].isin(tradable_symbols)]
        .pivot(index="date", columns="symbol", values="return_1d")
        .reindex(columns=tradable_symbols).sort_index()
    )
    calendar = (
        returns_panel.index.intersection(asset_panel.index)
        .intersection(spy_panel.index).intersection(market_panel.index).sort_values()
    )
    asset_panel = asset_panel.reindex(calendar)
    spy_panel = spy_panel.reindex(calendar)
    market_panel = market_panel.reindex(calendar)
    complete = (
        asset_panel.notna().all(axis=1)
        & spy_panel.notna().all(axis=1)
        & market_panel.notna().all(axis=1)
        & returns_panel.reindex(calendar).notna().all(axis=1)
    )
    if not complete.any():
        raise ValueError("No dates have all requested features.")
    calendar = calendar[calendar >= calendar[complete][0]]
    asset_panel = asset_panel.reindex(calendar)
    spy_panel = spy_panel.reindex(calendar)
    market_panel = market_panel.reindex(calendar)

    holdout_cutoff = calendar.max() - pd.DateOffset(years=final_holdout_years)
    development_calendar = calendar[calendar <= holdout_cutoff]
    train_end = int(len(development_calendar) * development_train_fraction)
    if train_end < 1:
        raise ValueError("Not enough history before the final holdout.")
    train_cutoff = development_calendar[train_end - 1]

    asset_panel = scale_rolling(asset_panel, scaling_window, observation_clip)
    spy_panel = scale_rolling(spy_panel, scaling_window, observation_clip)
    market_panel = scale_rolling(market_panel, scaling_window, observation_clip)
    complete = (
        asset_panel.notna().all(axis=1)
        & spy_panel.notna().all(axis=1)
        & market_panel.notna().all(axis=1)
    )
    if complete.any():
        first_complete = np.flatnonzero(complete.to_numpy())[0]
        if not complete.iloc[first_complete:].all():
            invalid_dates = calendar[first_complete:][~complete.iloc[first_complete:].to_numpy()]
            raise ValueError(f"Undefined scaled features interrupt daily history: {invalid_dates[:3].tolist()}.")
    calendar = calendar[complete]
    dataset = {
        "dates": calendar.to_numpy(),
        "asset_states": asset_panel.reindex(calendar).to_numpy(dtype=np.float32),
        "spy_states": spy_panel.reindex(calendar).to_numpy(dtype=np.float32),
        "market_states": market_panel.reindex(calendar).to_numpy(dtype=np.float32),
        "asset_returns": returns_panel.reindex(calendar).to_numpy(dtype=np.float32),
    }
    if not all(np.isfinite(values).all() for key, values in dataset.items() if key != "dates"):
        raise ValueError("Prepared features or asset returns contain nonfinite values.")
    masks = {
        "train": np.asarray(calendar <= train_cutoff),
        "development validation": np.asarray((calendar > train_cutoff) & (calendar <= holdout_cutoff)),
        "locked final holdout": np.asarray(calendar > holdout_cutoff),
    }
    if any(mask.sum() < 2 for mask in masks.values()):
        raise ValueError("Each split needs at least two dates after feature scaling.")
    split_summary = pd.DataFrame([
        {"split": name, "first_date": calendar[mask][0],
         "last_date": calendar[mask][-1], "dates": int(mask.sum())}
        for name, mask in masks.items()
    ])
    return (
        slice_dataset(dataset, masks["train"]),
        slice_dataset(dataset, masks["development validation"]),
        split_summary,
    )
