"""
These are helper functions used to create features the agent is trained on.
These expect data provided by yahoo finance.



"""

import numpy as np
import pandas as pd
import yfinance as yf

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

def calculate_cross_sectional_rank(data, feature):
    """Percentile rank among the nine sector ETFs on each date."""
    return (
        data[feature]
        .where(data["tradable"])
        .groupby(data["date"])
        .rank(pct=True)
    )


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

def sort_and_dropna(market_data):
    market_data = (
    market_data.loc[
        market_data["close"].notna(),
        [
            "date",
            "symbol",
            "name",
            "sector",
            "tradable",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dividends",
            "stock_splits",
        ],
    ]
    .sort_values(["date", "symbol"])
    .reset_index(drop=True)
    )
    return market_data


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
):
    """Prepare the same chronological arrays and splits as notebook 03.

    Any prepared feature CSV with the same columns can be used, including a
    separately generated bootstrap path. The final holdout is never evaluated
    by the batch runner.
    """
    from .environment import build_feature_panel, scale_rolling, slice_dataset

    if not 0 < development_train_fraction < 1 or final_holdout_years <= 0:
        raise ValueError("Use a train fraction between 0 and 1 and a positive holdout.")
    scaling_window = int(scaling_window_years * 252)
    if scaling_window < 2:
        raise ValueError("The scaling window must contain at least two trading days.")

    market_data = pd.read_csv(feature_path, parse_dates=["date"])
    if market_data.duplicated(["date", "symbol"]).any():
        raise ValueError("Feature data contains duplicate date/symbol rows.")
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
        returns_panel.dropna().index.intersection(asset_panel.index)
        .intersection(spy_panel.index).intersection(market_panel.index).sort_values()
    )
    asset_panel = asset_panel.reindex(calendar)
    spy_panel = spy_panel.reindex(calendar)
    market_panel = market_panel.reindex(calendar)
    complete = (
        asset_panel.notna().all(axis=1)
        & spy_panel.notna().all(axis=1)
        & market_panel.notna().all(axis=1)
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
