"""Historical exchange sessions and explicit selection of daily market rows."""

import logging

import pandas as pd

from .validation import validate_dates, validate_expected_dates, validate_unique_observations


def exchange_sessions(start_date, end_date, calendar_name="XNYS"):
    """Return daily session labels in [start_date, end_date), including early closes.

    XNYS includes regular holidays and historical exceptional closures. Bounds
    are calendar dates without timezones; Yahoo's exclusive end convention is
    preserved. No observed prices are used to infer whether an exchange opened.
    """
    import exchange_calendars

    if not isinstance(calendar_name, str) or not calendar_name.strip():
        raise ValueError("calendar_name must name a supported exchange, such as XNYS.")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if (pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None
            or start != start.normalize() or end != end.normalize() or start >= end):
        raise ValueError("Calendar requires daily start_date < end_date without timezones.")
    try:
        calendar = exchange_calendars.get_calendar(
            calendar_name, start=start, end=end - pd.Timedelta(days=1))
    except (ValueError, exchange_calendars.errors.InvalidCalendarName,
            exchange_calendars.errors.NoSessionsError) as error:
        raise ValueError(f"Cannot build exchange calendar {calendar_name!r}: {error}") from error
    sessions = calendar.sessions
    validate_expected_dates(sessions)
    return sessions


def select_exchange_sessions(market_data, sessions):
    """Select calendar sessions without dropping missing observations within them.

    Yahoo's wide response can contain closed-market dates and empty ETF rows.
    Exclude rows by the independently supplied calendar, never by price validity.
    Report the exclusion count through logging; retain every required-session row
    so subsequent validation can reject missing prices. Do not modify input data.
    """
    validate_dates(market_data)
    validate_unique_observations(market_data)
    validate_expected_dates(sessions, timezone=market_data["date"].dt.tz)
    selected = market_data["date"].isin(sessions)
    if not selected.all():
        excluded = market_data.loc[~selected, "date"]
        logging.getLogger(__name__).info(
            "Excluded %d rows on %d non-session dates (examples: %s).",
            len(excluded), excluded.nunique(), excluded.unique()[:3].tolist())
    return market_data.loc[selected].copy()
