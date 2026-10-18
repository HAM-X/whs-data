"""WHS Core v1 — Python twin of the Pine Core (must match to 1e-9 on the same bars).

Contract (Stack v2, approved 18 Oct 2026):
  TR             = max(H, C[1]) − min(L, C[1])            (first bar: H − L)
  ATR(14)        = Wilder RMA of TR, i.e. Pine ta.atr(14)
  relative spread= TR ÷ ATR(14)[1]                        (previous bar's ATR: a bar never measures itself)
  close position = (C − min(L, C[1])) ÷ TR                (0 = low, 1 = high)
  effort         = V ÷ median(V of the previous 20 bars with the same Chicago clock slot);
                   fewer than 10 such bars → V ÷ median(V of the previous 20 bars)
  roll window    = 8 sessions before the front month stops (CME FX: 2 business days before the 3rd
                   Wednesday of Mar/Jun/Sep/Dec); gold: from the 20th of Jan/Mar/May/Jul/Sep/Nov to month end.
                   Inside it, effort volume = front + next contract volume.
  Sessions and slots are keyed to Chicago time (America/Chicago), never local time.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

CT = "America/Chicago"


def true_range(high, low, close) -> np.ndarray:
    h, l, c = (np.asarray(x, float) for x in (high, low, close))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.where(np.isnan(pc), h - l, np.maximum(h, pc) - np.minimum(l, pc))
    return tr


def rma(x, length: int) -> np.ndarray:
    """Pine ta.rma: seeded with the SMA of the first `length` values, then (prev*(n−1)+x)/n."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < length:
        return out
    out[length - 1] = x[:length].mean()
    a = 1.0 / length
    for i in range(length, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def atr(high, low, close, length: int = 14) -> np.ndarray:
    return rma(true_range(high, low, close), length)


def rel_spread(high, low, close, length: int = 14) -> np.ndarray:
    tr = true_range(high, low, close)
    a = atr(high, low, close, length)
    prev = np.r_[np.nan, a[:-1]]
    return tr / prev


def close_pos(high, low, close) -> np.ndarray:
    h, l, c = (np.asarray(x, float) for x in (high, low, close))
    pc = np.r_[np.nan, c[:-1]]
    lo = np.where(np.isnan(pc), l, np.minimum(l, pc))
    tr = true_range(h, l, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tr > 0, (c - lo) / tr, 0.5)


def slot_key(times_utc: pd.Series, tf_minutes: int) -> np.ndarray:
    """Chicago clock slot of each bar: (CT hour*60 + minute) // timeframe minutes."""
    ct = pd.to_datetime(times_utc, utc=True).dt.tz_convert(CT)
    return ((ct.dt.hour * 60 + ct.dt.minute) // tf_minutes).to_numpy()


def effort(volume, times_utc: pd.Series, tf_minutes: int = 60, n: int = 20, min_samples: int = 10) -> np.ndarray:
    """Same-slot median over the previous n bars in that slot (current bar excluded).

    Mirrors the Pine ring buffer exactly: only valid volumes are stored; with fewer than
    `min_samples` stored, fall back to the median of the previous 20 bars (Pine ta.median(v, 20)[1],
    which needs 20 bars)."""
    v = np.asarray(volume, float)
    keys = slot_key(times_utc, tf_minutes)
    out = np.full(len(v), np.nan)
    hist: dict[int, list[float]] = {}
    for i, (k, x) in enumerate(zip(keys, v)):
        h = hist.setdefault(int(k), [])
        if len(h) >= min_samples:
            med = float(np.median(h))
        else:
            prev = v[i - 20:i] if i >= 20 else np.array([])
            med = float(np.median(prev)) if len(prev) == 20 and not np.isnan(prev).any() else np.nan
        if not np.isnan(x) and not np.isnan(med) and med > 0:
            out[i] = x / med
        if not np.isnan(x):
            h.append(x)
            if len(h) > n:
                h.pop(0)
    return out


def _third_wednesday(y: int, m: int) -> dt.date:
    d = dt.date(y, m, 15)
    return d + dt.timedelta(days=(2 - d.weekday()) % 7)


def _bdays_back(d: dt.date, k: int) -> dt.date:
    while k:
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:
            k -= 1
    return d


def in_roll_window(root: str, day: dt.date) -> bool:
    """`day` is the CME session date (Chicago)."""
    if root == "GC":
        return day.month in (1, 3, 5, 7, 9, 11) and day.day >= 20
    if root in ("6E", "6B", "6A", "6N", "6J", "6C", "6S"):
        for y in (day.year, day.year + 1):
            for m in (3, 6, 9, 12):
                last = _bdays_back(_third_wednesday(y, m), 2)
                if last >= day:
                    return _bdays_back(last, 8) <= day <= last
    return False


def band(e: float) -> str:
    if e is None or np.isnan(e):
        return "n/a"
    return "≫" if e >= 1.5 else ">" if e >= 1.1 else "≪" if e <= 0.67 else "<" if e <= 0.9 else "≈"
