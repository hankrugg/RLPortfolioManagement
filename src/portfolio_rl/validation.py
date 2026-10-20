"""Contracts for normalized market data and unscaled feature data.

The two public validators coordinate the smaller checks below. Validators will
raise ValueError with affected columns, symbols, and sample dates on invalid
input, and return None on success. They must not modify, sort, fill, clip, or
drop data.
"""

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np
import pandas as pd


# Leading trading sessions required by the feature definitions in data.py.
FEATURE_WARMUP_PERIODS = {
    "return_1d": 1, "return_5d": 5, "return_21d": 21, "return_63d": 63,
    "volatility_21d": 21, "rolling_drawdown_63d": 62,
    "cross_sectional_rank_return_21d": 21, "market_relative_return_21d": 21,
    "rolling_market_beta_63d": 63, "sector_breadth_above_sma_63d": 62,
    "sector_return_dispersion_21d": 21, "sector_average_correlation_63d": 62,
    "vix_level": 0, "vix_return_5d": 5,
}
FEATURE_BOUNDS = {
    "volatility_21d": (0, None), "rolling_drawdown_63d": (-1, 0),
    "cross_sectional_rank_return_21d": (0, 1),
    "sector_breadth_above_sma_63d": (0, 1),
    "sector_return_dispersion_21d": (0, None),
    "sector_average_correlation_63d": (-1, 1), "vix_level": (0, None),
}


def _names(values, label, *, allow_empty=True):
    if (isinstance(values, str) or not isinstance(values, Sequence)
            or (not allow_empty and not values)
            or any(not isinstance(value, str) or not value.strip() for value in values)):
        raise ValueError(f"{label} must contain nonempty string names.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must contain unique names.")


def _calendar(data, tradable_symbols, expected_dates):
    dates = (pd.DatetimeIndex(data.loc[data["symbol"].isin(tradable_symbols), "date"])
             .unique().sort_values() if expected_dates is None else expected_dates)
    validate_expected_dates(dates, timezone=data["date"].dt.tz)
    return dates


def _examples(rows):
    return rows[["date", "symbol"]].head(3).to_dict("records")


def validate_expected_dates(expected_dates, *, timezone=None):
    """Require nonempty, unique, chronological daily labels with the given timezone."""
    if (not isinstance(expected_dates, pd.DatetimeIndex) or expected_dates.empty
            or expected_dates.hasnans or not expected_dates.is_unique
            or not expected_dates.is_monotonic_increasing
            or not expected_dates.equals(expected_dates.normalize())):
        raise ValueError("expected_dates must be nonempty, unique, chronological daily dates.")
    if expected_dates.tz != timezone:
        raise ValueError("expected_dates and data must use the same timezone.")


def validate_market_data(
    market_data: pd.DataFrame,
    tradable_symbols: Sequence[str],
    *,
    context_symbols: Sequence[str] = ("SPY", "^VIX"),
    price_columns: Sequence[str] = ("close",),
    expected_dates: pd.DatetimeIndex | None = None,
) -> None:
    """Validate normalized market rows before returns or features are calculated.

    Args:
        market_data: Long-form rows with date, symbol, tradable, and requested
            price columns. Dates must already be parsed and each symbol's rows
            must be chronological. Sort separately without dropping observations.
        tradable_symbols: Nonempty, unique symbols used to construct the portfolio
            and cross-sectional sector features.
        context_symbols: Unique, disjoint symbols required for market context.
        price_columns: Price fields needed by downstream calculations. Require
            numeric, finite, strictly positive values on required trading dates.
        expected_dates: Optional independent trading-session calendar for the
            requested period. Without it, compare coverage against the union of
            observed tradable dates; this cannot detect a session missing from
            every asset. Extra context dates outside that calendar are allowed.

    Check schema, dates, duplicates, symbol availability, tradable flags, coverage,
    and prices, in that order. Optional metadata and unused fields do not need to
    be complete. Do not reject VIX solely for zero volume or SPY for missing sector.

    Raises:
        ValueError: Invalid requirements or market data, including an empty table.
    """
    _names(price_columns, "price_columns", allow_empty=False)
    validate_required_columns(market_data, ["date", "symbol", "tradable", *price_columns])
    validate_dates(market_data, require_sorted=True)
    validate_unique_observations(market_data)
    validate_symbols(market_data, tradable_symbols, context_symbols)
    validate_symbol_roles(market_data, tradable_symbols, context_symbols)
    validate_calendar_coverage(market_data, tradable_symbols, context_symbols,
                               expected_dates=expected_dates)
    calendar = _calendar(market_data, tradable_symbols, expected_dates)
    required = market_data.loc[market_data["date"].isin(calendar)]
    symbols = [*tradable_symbols, *context_symbols]
    validate_numeric_columns(required, price_columns, symbols=symbols)
    validate_finite_values(required, price_columns, symbols=symbols)
    validate_positive_prices(required, price_columns, symbols=symbols)


def validate_feature_data(
    feature_data: pd.DataFrame,
    tradable_symbols: Sequence[str],
    asset_features: Sequence[str],
    spy_features: Sequence[str],
    market_features: Sequence[str],
    *,
    warmup_periods: Mapping[str, int],
    expected_dates: pd.DatetimeIndex | None = None,
    feature_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> None:
    """Validate a constructed feature table before alignment, scaling, or splits.

    Args:
        feature_data: Long-form rows with parsed dates, symbols, return_1d, and
            configured features. Input row order need not be chronological.
        tradable_symbols: Nonempty, unique portfolio symbols requiring asset
            features and daily simple returns.
        asset_features: Feature columns required on every tradable symbol.
        spy_features: Feature columns required on SPY rows.
        market_features: Market-context columns read from SPY rows. These need
            not be present as valid values on every other symbol's rows.
        warmup_periods: Explicit leading-session allowance for every required
            feature and return_1d; zero means no missing values are allowed.
            Counts start at the first expected session, not the first valid
            value. Allowances must follow feature definitions, not observed gaps.
        expected_dates: Independent trading sessions, or None to use the union
            of observed tradable dates with the same limitation as market-data
            validation. Require SPY coverage; VIX need not remain as separate rows
            once its context features have been mapped onto SPY.
        feature_bounds: Optional inclusive bounds for named unscaled features;
            None means no additional feature-specific range checks. Each bound
            can be None for an unbounded side. Do not bound beta arbitrarily.

    Check structure and coverage before numeric values. Reject infinity before
    clipping and reject missing values beyond each declared warm-up. Require a
    nonempty common usable period. Validate returns separately from features;
    large valid positive returns must remain allowed. Do not recalculate features
    or validate the eventual standardized arrays and chronological splits here.

    Raises:
        ValueError: Invalid requirements or feature data, including an empty table.
    """
    for label, columns in (("asset_features", asset_features), ("spy_features", spy_features),
                           ("market_features", market_features)):
        _names(columns, label)
    groups = [(tradable_symbols, list(dict.fromkeys([*asset_features, "return_1d"]))),
              (["SPY"], list(dict.fromkeys([*spy_features, *market_features])))]
    columns = list(dict.fromkeys(column for _, fields in groups for column in fields))
    validate_required_columns(feature_data, ["date", "symbol", *columns])
    validate_dates(feature_data)
    validate_unique_observations(feature_data)
    validate_symbols(feature_data, tradable_symbols, ["SPY"])
    validate_calendar_coverage(feature_data, tradable_symbols, ["SPY"],
                               expected_dates=expected_dates)
    if not isinstance(warmup_periods, Mapping):
        raise ValueError("warmup_periods must map feature names to session counts.")
    missing = set(columns) - warmup_periods.keys()
    if missing:
        raise ValueError(f"Missing warm-up allowances for {sorted(missing)}.")
    if feature_bounds is not None and not isinstance(feature_bounds, Mapping):
        raise ValueError("feature_bounds must map feature names to bounds.")
    bounds = {} if feature_bounds is None else feature_bounds
    if set(bounds) - set(columns):
        raise ValueError(f"Bounds supplied for unrequested features: {sorted(set(bounds) - set(columns))}.")
    calendar = _calendar(feature_data, tradable_symbols, expected_dates)
    required = feature_data.loc[feature_data["date"].isin(calendar)]
    for symbols, fields in groups:
        validate_numeric_columns(required, fields, symbols=symbols)
        validate_finite_values(required, fields, symbols=symbols, allow_missing=True)
        validate_feature_warmup(required, {column: warmup_periods[column] for column in fields},
                                symbols=symbols, expected_dates=calendar)
        validate_feature_bounds(required, {column: bounds[column] for column in fields if column in bounds},
                                symbols=symbols)
    validate_simple_returns(required, symbols=tradable_symbols)


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """Reject empty tables, duplicate column names, or absent required columns.

    Allow extra columns. Report missing or duplicate names before other checks
    attempt to access the affected fields.
    """
    _names(required_columns, "required_columns")
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("No dates: data must be a nonempty DataFrame.")
    duplicates = data.columns[data.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate column names: {duplicates}.")
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}.")


def validate_dates(data: pd.DataFrame, *, require_sorted: bool = False) -> None:
    """Require parsed, nonmissing daily dates with one consistent time convention.

    Reject intraday timestamps and incompatible timezone representations rather
    than silently normalizing them. When require_sorted is True, require each
    symbol's dates to be chronological; global symbol/date row order is irrelevant.
    This check does not establish whether dates are exchange trading sessions.
    """
    validate_required_columns(data, ["date", "symbol"])
    dates = data["date"]
    if not pd.api.types.is_datetime64_any_dtype(dates.dtype):
        raise ValueError("date must contain parsed datetime values with one timezone convention.")
    invalid = dates.isna() | dates.ne(dates.dt.normalize())
    if invalid.any():
        raise ValueError(f"Missing or intraday dates: {_examples(data.loc[invalid])}.")
    if require_sorted:
        for symbol, rows in data.groupby("symbol", sort=False):
            if not rows["date"].is_monotonic_increasing:
                raise ValueError(f"Dates for {symbol} must be chronological.")


def validate_unique_observations(data: pd.DataFrame) -> None:
    """Reject repeated date/symbol pairs, even when their other values agree.

    Include sample duplicated pairs in the error. Never choose or merge rows.
    """
    validate_required_columns(data, ["date", "symbol"])
    duplicates = data.duplicated(["date", "symbol"], keep=False)
    if duplicates.any():
        raise ValueError(f"Data contains duplicate date/symbol rows: {_examples(data.loc[duplicates])}.")


def validate_symbols(
    data: pd.DataFrame,
    tradable_symbols: Sequence[str],
    context_symbols: Sequence[str],
) -> None:
    """Validate requested symbol lists and require each requested symbol to exist.

    Require nonempty string identifiers, unique lists, a nonempty tradable list,
    and disjoint tradable/context sets. Reject missing or blank identifiers in
    data. Extra valid symbols may remain but do not expand the configured universe.
    Presence alone does not establish complete date coverage.
    """
    _names(tradable_symbols, "tradable_symbols", allow_empty=False)
    _names(context_symbols, "context_symbols")
    overlap = set(tradable_symbols) & set(context_symbols)
    if overlap:
        raise ValueError(f"Tradable and context symbols overlap: {sorted(overlap)}.")
    validate_required_columns(data, ["symbol"])
    invalid = ~data["symbol"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    if invalid.any():
        raise ValueError("symbol contains missing, blank, or nonstring identifiers.")
    missing = set([*tradable_symbols, *context_symbols]) - set(data["symbol"])
    if missing:
        raise ValueError(f"Missing required symbols: {sorted(missing)}.")


def validate_symbol_roles(
    data: pd.DataFrame,
    tradable_symbols: Sequence[str],
    context_symbols: Sequence[str],
) -> None:
    """Require boolean tradable flags consistent with the configured universe.

    Require True for configured assets and False for context and extra symbols;
    otherwise cross-sectional feature calculations could use the wrong universe.
    Reject missing flags and strings such as 'False', which are truthy in Python.
    Apply this to market data; prepared feature CSVs need not retain the flag.
    """
    validate_required_columns(data, ["date", "symbol", "tradable"])
    flags = data["tradable"]
    invalid = ~flags.map(lambda value: isinstance(value, (bool, np.bool_)))
    if invalid.any():
        raise ValueError(f"tradable must contain booleans: {_examples(data.loc[invalid])}.")
    invalid = flags.ne(data["symbol"].isin(tradable_symbols))
    if invalid.any():
        raise ValueError(f"tradable flags disagree with the configured universe: {_examples(data.loc[invalid])}.")


def validate_calendar_coverage(
    data: pd.DataFrame,
    tradable_symbols: Sequence[str],
    context_symbols: Sequence[str],
    *,
    expected_dates: pd.DatetimeIndex | None = None,
) -> None:
    """Require one row per required symbol on every expected trading session.

    Validate an explicit calendar as nonempty, unique, chronological daily dates.
    If omitted, use the union of observed tradable dates, never their intersection.
    Reject absent required rows and tradable dates outside an explicit calendar;
    allow context-only dates outside it. Weekdays alone are not an exchange
    calendar. This checks row presence; missing values need separate validation.
    """
    validate_dates(data)
    validate_unique_observations(data)
    validate_symbols(data, tradable_symbols, context_symbols)
    calendar = _calendar(data, tradable_symbols, expected_dates)
    for symbol in [*tradable_symbols, *context_symbols]:
        observed = pd.DatetimeIndex(data.loc[data["symbol"].eq(symbol), "date"])
        missing = calendar.difference(observed)
        if len(missing):
            raise ValueError(f"Missing trading dates for {symbol}: {missing[:3].tolist()}.")
        extra = observed.difference(calendar)
        if symbol in tradable_symbols and len(extra):
            raise ValueError(f"Unexpected trading dates for {symbol}: {extra[:3].tolist()}.")


def validate_numeric_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    symbols: Sequence[str],
) -> None:
    """Require numeric, nonboolean values in the selected symbols and columns.

    Reject numeric strings rather than coercing them. Permit missing values at
    this stage so the separate missingness check can apply warm-up allowances.
    Ignore values on rows outside the requested symbol group.
    """
    validate_required_columns(data, ["date", "symbol", *columns])
    rows = data.loc[data["symbol"].isin(symbols)]
    for column in columns:
        values = rows[column]
        valid = values.map(lambda value: isinstance(value, Real)
                           and not isinstance(value, (bool, np.bool_))) | values.isna()
        if not valid.all():
            raise ValueError(f"{column} must contain numeric, nonboolean values: {_examples(rows.loc[~valid])}.")


def validate_finite_values(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    symbols: Sequence[str],
    allow_missing: bool = False,
) -> None:
    """Reject positive/negative infinity in selected numeric fields before clipping.

    Also reject NaN unless allow_missing is True. Feature checks can allow NaN
    here and validate its position separately; price checks should not. Report
    affected columns and sample date/symbol pairs. Scope rows to the applicable
    trading calendar before calling when extra context dates are permitted.
    """
    validate_numeric_columns(data, columns, symbols=symbols)
    rows = data.loc[data["symbol"].isin(symbols)]
    for column in columns:
        values = rows[column].to_numpy(dtype=float, na_value=np.nan)
        invalid = np.isinf(values) if allow_missing else ~np.isfinite(values)
        if invalid.any():
            raise ValueError(f"{column} contains nonfinite values: {_examples(rows.loc[invalid])}.")


def validate_positive_prices(
    data: pd.DataFrame,
    price_columns: Sequence[str],
    *,
    symbols: Sequence[str],
) -> None:
    """Reject zero or negative required prices after numeric/finite checks.

    Positive closes are needed for log returns. Do not apply this rule to returns,
    volumes, dividends, or split indicators, which can legitimately be zero.
    """
    rows = data.loc[data["symbol"].isin(symbols)]
    for column in price_columns:
        invalid = rows[column].le(0).fillna(False)
        if invalid.any():
            raise ValueError(f"{column} must be strictly positive: {_examples(rows.loc[invalid])}.")


def validate_feature_warmup(
    data: pd.DataFrame,
    warmup_periods: Mapping[str, int],
    *,
    symbols: Sequence[str],
    expected_dates: pd.DatetimeIndex,
) -> None:
    """Allow missing features only within explicitly declared leading sessions.

    Mapping keys select the columns checked for this symbol group. Require
    nonnegative integer allowances, excluding booleans. For allowance N, NaN may
    occur only in the first N expected sessions; valid early observations are
    allowed. Reject any later missing value, including trailing gaps, rather than
    dropping it. Reject allowances leaving no jointly usable observations.
    Row coverage must already be validated so absent rows cannot hide gaps.
    """
    if not isinstance(warmup_periods, Mapping):
        raise ValueError("warmup_periods must map feature names to session counts.")
    calendar = _calendar(data, symbols, expected_dates)
    rows = data.loc[data["symbol"].isin(symbols) & data["date"].isin(calendar)]
    for column, allowance in warmup_periods.items():
        if isinstance(allowance, (bool, np.bool_)) or not isinstance(allowance, Integral) or allowance < 0:
            raise ValueError(f"Warm-up for {column} must be a nonnegative integer.")
        if allowance >= len(calendar):
            raise ValueError(f"No dates remain after the warm-up for {column}.")
        invalid = rows[column].isna() & rows["date"].ge(calendar[allowance])
        if invalid.any():
            raise ValueError(f"Missing {column} beyond its {allowance}-session warm-up: {_examples(rows.loc[invalid])}.")


def validate_simple_returns(
    data: pd.DataFrame,
    *,
    symbols: Sequence[str],
    return_column: str = "return_1d",
) -> None:
    """Reject nonmissing simple asset returns at or below -1, without an upper cap.

    The current pipeline assumes strictly positive adjusted prices, so a total
    loss is outside its supported input contract. NaN placement and finiteness
    are checked separately. This bound applies to simple asset returns, not log
    returns, market-relative features, or standardized observations.
    """
    rows = data.loc[data["symbol"].isin(symbols)]
    invalid = rows[return_column].le(-1).fillna(False)
    if invalid.any():
        raise ValueError(f"{return_column} must exceed -1: {_examples(rows.loc[invalid])}.")


def validate_feature_bounds(
    data: pd.DataFrame,
    bounds: Mapping[str, tuple[float | None, float | None]],
    *,
    symbols: Sequence[str],
    tolerance: float = 1e-10,
) -> None:
    """Check nonmissing unscaled feature values against inclusive supplied bounds.

    For example, breadth lies in [0, 1] and correlation in [-1, 1]. Allow a finite
    nonnegative numerical tolerance; validate ordered finite endpoints when
    supplied. Ignore NaN here because warm-up handles missingness. Scope each
    invocation to features applicable to its symbol group. Never clip violations
    or impose generic outlier limits on mathematically unbounded features.
    """
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative.")
    if not isinstance(bounds, Mapping):
        raise ValueError("bounds must map feature names to (lower, upper) pairs.")
    rows = data.loc[data["symbol"].isin(symbols)]
    for column, limits in bounds.items():
        if not isinstance(limits, (tuple, list)) or len(limits) != 2:
            raise ValueError(f"Bounds for {column} must be a (lower, upper) pair.")
        lower, upper = limits
        if any(value is not None and (isinstance(value, (bool, np.bool_))
               or not isinstance(value, Real) or not np.isfinite(value)) for value in limits):
            raise ValueError(f"Bounds for {column} must be finite numbers or None.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Bounds for {column} must be ordered lower <= upper.")
        values = rows[column]
        invalid = pd.Series(False, index=rows.index)
        if lower is not None:
            invalid |= values.lt(lower - tolerance).fillna(False)
        if upper is not None:
            invalid |= values.gt(upper + tolerance).fillna(False)
        if invalid.any():
            raise ValueError(f"{column} is outside bounds {limits}: {_examples(rows.loc[invalid])}.")
