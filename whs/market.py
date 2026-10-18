"""CME session clock. Everything is computed in Chicago time so clock changes cannot break it."""
from __future__ import annotations

import pandas as pd

from .config import EXCHANGE_TZ, SESSION_CLOSE_HOUR_CT, SESSION_OPEN_HOUR_CT


def to_ct(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(EXCHANGE_TZ)


def is_open(ts) -> bool:
    """True if CME FX/metals trade at this moment (ignores exchange holidays).

    Week: Sun 17:00 CT -> Fri 16:00 CT. Daily halt 16:00-17:00 CT Mon-Thu.
    """
    ct = to_ct(ts)
    wd, h = ct.dayofweek, ct.hour  # Mon=0 .. Sun=6
    if wd == 5:                     # Saturday
        return False
    if wd == 6:                     # Sunday: only from the evening open
        return h >= SESSION_OPEN_HOUR_CT
    if wd == 4:                     # Friday: until the 16:00 close
        return h < SESSION_CLOSE_HOUR_CT
    return not (SESSION_CLOSE_HOUR_CT <= h < SESSION_OPEN_HOUR_CT)


def expected_hours(start, end) -> list[pd.Timestamp]:
    """Hourly bar-open times (UTC) in [start, end) when the market is open."""
    rng = pd.date_range(pd.Timestamp(start).tz_convert("UTC").ceil("h"),
                        pd.Timestamp(end).tz_convert("UTC"), freq="h", inclusive="left")
    return [t for t in rng if is_open(t)]


def last_complete_hour(now) -> pd.Timestamp | None:
    """Open time (UTC) of the most recent hourly bar that has fully closed while the market was open."""
    t = pd.Timestamp(now).tz_convert("UTC").floor("h") - pd.Timedelta(hours=1)
    for _ in range(24 * 4):
        if is_open(t):
            return t
        t -= pd.Timedelta(hours=1)
    return None
