"""Offline tests for the robot (no network). Run: python -m pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

from whs import bars, market
from whs.config import FUTURES, SPOT

CT = "America/Chicago"


def ct(s):
    return pd.Timestamp(s, tz=CT).tz_convert("UTC")


# ---------------- session clock ----------------
@pytest.mark.parametrize("when,open_", [
    ("2026-10-18 16:59", False),  # Sunday before the open
    ("2026-10-18 17:00", True),   # Sunday open
    ("2026-10-19 15:59", True),   # Monday before the halt
    ("2026-10-19 16:30", False),  # daily halt
    ("2026-10-23 15:00", True),   # Friday afternoon
    ("2026-10-23 16:00", False),  # Friday close
    ("2026-10-24 12:00", False),  # Saturday
    ("2026-11-01 17:00", True),   # Sunday of the US clock change
])
def test_is_open(when, open_):
    assert market.is_open(ct(when)) is open_


def test_last_complete_hour_skips_weekend():
    # Saturday noon: the last complete bar is Friday 15:00 CT
    assert market.last_complete_hour(ct("2026-10-24 12:00")) == ct("2026-10-23 15:00")


# ---------------- bars ----------------
def session_hours(start_ct, n):
    """n open-market hourly bar times from start (CT), skipping halts/weekends."""
    out, t = [], ct(start_ct)
    while len(out) < n:
        if market.is_open(t):
            out.append(t)
        t += pd.Timedelta(hours=1)
    return out


def frame(times, seed=1):
    rng = np.random.default_rng(seed)
    c = 1.10 + np.cumsum(rng.normal(0, 0.0005, len(times)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.0003, "Low": np.minimum(o, c) - 0.0003,
                         "Close": c, "Volume": rng.integers(100, 5000, len(times))},
                        index=pd.DatetimeIndex(times).tz_convert("America/New_York"))


def test_normalise_drops_forming_bar_and_offgrid():
    times = session_hours("2026-10-19 09:00", 5)
    raw = frame(times)
    raw.loc[raw.index[-1] + pd.Timedelta(minutes=30)] = raw.iloc[-1]   # an off-grid row
    now = times[-1] + pd.Timedelta(minutes=20)                            # last bar still forming
    n = bars.normalise(raw.sort_index(), now=now)
    assert list(n["time"]) == times[:-1]


def test_merge_appends_and_revises():
    times = session_hours("2026-10-19 09:00", 6)
    a = bars.normalise(frame(times[:4]))
    b = bars.normalise(frame(times[2:], seed=2))
    m, added, revised = bars.merge(a, b)
    assert len(m) == 6 and added == 2 and revised == 2
    assert m["close"].iloc[2] == b["close"].iloc[0]          # newer copy wins


def test_build_4h_blocks_like_tradingview():
    times = session_hours("2026-10-18 17:00", 23 * 2)       # two full sessions
    h1 = bars.normalise(frame(times))
    h4 = bars.build_4h(h1)
    hours = list(h4["time"].dt.tz_convert(CT).dt.hour)
    assert hours == [17, 21, 1, 5, 9, 13] * 2
    first = h1.iloc[:4]
    assert h4["open"].iloc[0] == first["open"].iloc[0] and h4["close"].iloc[0] == first["close"].iloc[-1]
    assert h4["high"].iloc[0] == first["high"].max() and h4["volume"].iloc[0] == first["volume"].sum()
    # the 13:00 block holds the last 3 hours before the 16:00 close
    last_block = h1[h1["time"].dt.tz_convert(CT).dt.hour.isin([13, 14, 15])].iloc[:3]
    assert h4["volume"].iloc[5] == last_block["volume"].sum()


def test_build_4h_drops_unfinished_block():
    times = session_hours("2026-10-18 17:00", 6)             # 17..22 CT: block 21 has only 2 hours
    h4 = bars.build_4h(bars.normalise(frame(times)), now=times[-1] + pd.Timedelta(hours=1))
    assert list(h4["time"].dt.tz_convert(CT).dt.hour) == [17]


def test_write_read_roundtrip(tmp_path):
    h1 = bars.normalise(frame(session_hours("2026-10-19 09:00", 5)))
    p = tmp_path / "x.csv"
    bars.write(h1, p)
    back = bars.read(p)
    assert (back["time"] == h1["time"]).all() and np.allclose(back["close"], h1["close"])


# ---------------- whole job with a fake source ----------------
def test_run_end_to_end(tmp_path, monkeypatch):
    from jobs import download

    monkeypatch.setattr(download, "DATA", tmp_path / "data")
    monkeypatch.setattr(download, "STATE", tmp_path / "data" / "state.json")
    now = ct("2026-10-21 03:10")                                   # Wed 11:10 UTC+3, market open
    times = [t for t in session_hours("2026-10-19 09:00", 60) if t + pd.Timedelta(hours=1) <= now]
    sent = []

    def fetch_ok(ticker, period="5d"):
        if ticker == SPOT["XAUUSD"]["yahoo"]:
            return pd.DataFrame(), "no data returned"               # optional symbol unavailable
        return frame(times), None

    s = download.run(now=now, fetch=fetch_ok, send=sent.append)
    assert s["problems"] == [], s["problems"]
    assert s["symbols"]["XAUUSD"]["status"] == "unavailable"
    assert any("Data healthy" in m for m in sent)                  # daily OK line after 08:00 local
    assert (tmp_path / "data" / "bars" / "4h" / "6E.csv").exists()

    # one feed goes stale three hours later -> exactly one alert, not repeated
    later = now + pd.Timedelta(hours=3)

    def fetch_stale(ticker, period="5d"):
        if ticker == FUTURES["6J"]["yahoo"]:
            return pd.DataFrame(), "no data returned"
        upto = [t for t in session_hours("2026-10-19 09:00", 80) if t + pd.Timedelta(hours=1) <= later]
        return frame(upto), None

    sent.clear()
    s2 = download.run(now=later, fetch=fetch_stale, send=sent.append)
    keys = {p["key"] for p in s2["problems"]}
    assert "6J:stale" in keys and "6J:source" in keys
    assert len(sent) == 1 and "6J" in sent[0]
    sent.clear()
    download.run(now=later + pd.Timedelta(minutes=5), fetch=fetch_stale, send=sent.append)
    assert sent == []                                              # no repeat for a known problem
    state = json.loads((tmp_path / "data" / "state.json").read_text())
    assert "6J:stale" in state["problems"]
