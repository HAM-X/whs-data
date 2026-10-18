"""Data health checks after each download. Problems are reported once, and again when fixed."""
from __future__ import annotations

import pandas as pd

from .config import EXCHANGE_TZ, FUTURES, LOCAL_TZ, VOLUME_CHECKED
from .market import expected_hours, is_open, last_complete_hour

STALE_AFTER_H = 3        # Yahoo can lag; only flag when the newest bar is 3+ hours behind
GAP_WINDOW_H = 24        # look for missing hours in the last day
FROZEN_BARS = 4          # this many identical flat bars in a row = frozen feed
ZERO_VOL_BARS = 3        # this many zero-volume bars in a row in active hours = broken volume


def check_symbol(key: str, df: pd.DataFrame, now, fetch_error: str | None, kind: str) -> list[dict]:
    now = pd.Timestamp(now).tz_convert("UTC")
    probs: list[dict] = []
    market_open = is_open(now) or is_open(now - pd.Timedelta(hours=1))
    if fetch_error and market_open:
        probs.append(dict(key=f"{key}:source", sym=key, text=f"{key}: download failed ({fetch_error})"))
    if df.empty:
        probs.append(dict(key=f"{key}:empty", sym=key, text=f"{key}: no stored bars yet"))
        return probs
    last = df["time"].iloc[-1]
    target = last_complete_hour(now)
    if market_open and target is not None:
        lag_h = (target - last).total_seconds() / 3600
        if lag_h >= STALE_AFTER_H:
            probs.append(dict(key=f"{key}:stale", sym=key, text=f"{key}: newest bar is {int(lag_h)}h behind",
                              lag=lag_h))
    # missing hours inside the last day, up to the newest stored bar
    if kind == "future":
        start = last - pd.Timedelta(hours=GAP_WINDOW_H)
        have = set(df.loc[df["time"] > start, "time"])
        missing = [t for t in expected_hours(start + pd.Timedelta(hours=1), last) if t not in have]
        if missing:
            probs.append(dict(key=f"{key}:gaps", sym=key, missing=[t.isoformat() for t in missing],
                              text=f"{key}: {len(missing)} missing hour(s) in the last day"))
        tail = df.tail(FROZEN_BARS)
        if len(tail) == FROZEN_BARS and (tail["high"] == tail["low"]).all() and tail["close"].nunique() == 1:
            probs.append(dict(key=f"{key}:frozen", sym=key, text=f"{key}: price frozen for {FROZEN_BARS} bars"))
        if key in VOLUME_CHECKED:
            t = df.tail(ZERO_VOL_BARS)
            active = t["time"].dt.tz_convert(EXCHANGE_TZ).dt.hour.between(2, 15).all()
            if len(t) == ZERO_VOL_BARS and active and (t["volume"].fillna(0) == 0).all():
                probs.append(dict(key=f"{key}:novol", sym=key, text=f"{key}: zero volume for {ZERO_VOL_BARS} active bars"))
    return probs


def collapse_marketwide(probs: list[dict]) -> list[dict]:
    """If most futures are stale or show the same gap at once, it is a holiday or a source outage: say it once."""
    fut = set(FUTURES)
    for kind, label in (("stale", "behind"), ("gaps", "missing hours")):
        hit = {p["sym"] for p in probs if p["key"].endswith(":" + kind) and p["sym"] in fut}
        if len(hit) >= 6:
            probs = [p for p in probs if not (p["key"].endswith(":" + kind) and p["sym"] in fut)]
            probs.append(dict(key=f"ALL:{kind}", sym="ALL",
                              text=f"{len(hit)} of {len(fut)} futures {label} at once — exchange holiday or Yahoo outage"))
    return probs


def alert_text(new: list[dict], resolved: list[str]) -> str | None:
    if not new and not resolved:
        return None
    lines = []
    if new:
        lines.append("⚠️ Data problem")
        lines += [f"• {p['text']}" for p in new]
    if resolved:
        lines.append("✅ Fixed: " + ", ".join(sorted(resolved)))
    return "\n".join(lines)


def local_str(ts) -> str:
    return pd.Timestamp(ts).tz_convert(LOCAL_TZ).strftime("%a %d %b %H:%M")
