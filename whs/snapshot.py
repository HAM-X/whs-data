"""Hourly market snapshot that Pipdon BOT answers from (data/snapshot.json).

Everything is computed from the robot's stored Yahoo bars, in Chicago exchange time, using the
WHS Core definitions (true range with the previous close, ATR(14), effort = volume / median volume
of the same Chicago hour over the last 20 sessions). TradingView remains the price of record.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from .config import EXCHANGE_TZ, FUTURES, LOCAL_TZ, SESSION_OPEN_HOUR_CT, SPOT

# futures -> MT5 spot symbol and direction (CME FX futures are all quoted in USD per foreign unit)
MT5 = {
    "6E": ("EURUSD", "same"), "6B": ("GBPUSD", "same"), "6A": ("AUDUSD", "same"), "6N": ("NZDUSD", "same"),
    "6J": ("USDJPY", "inverted"), "6C": ("USDCAD", "inverted"), "6S": ("USDCHF", "inverted"),
    "GC": ("XAUUSD", "same"),
}
CCY = {"6E": "EUR", "6B": "GBP", "6A": "AUD", "6N": "NZD", "6J": "JPY", "6C": "CAD", "6S": "CHF"}
PIP = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "NZDUSD": 0.0001,
       "USDJPY": 0.01, "USDCAD": 0.0001, "USDCHF": 0.0001, "XAUUSD": 0.01}
# CT hours [start, end) of the three trading sessions inside one CME day
SESSIONS = {"Asia": (17, 2), "London": (2, 7), "New York": (7, 16)}


def r(x, nd=6):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x), nd)


def session_day(times_ct: pd.Series) -> pd.Series:
    """CME trading day label: a bar from 17:00 CT belongs to the next calendar day's session."""
    return (times_ct - pd.Timedelta(hours=SESSION_OPEN_HOUR_CT)).dt.floor("D") + pd.Timedelta(days=1)


def true_range(d: pd.DataFrame) -> pd.Series:
    pc = d["close"].shift()
    return np.maximum(d["high"], pc.fillna(d["high"])) - np.minimum(d["low"], pc.fillna(d["low"]))


def effort(d: pd.DataFrame, n_sessions: int = 20, min_samples: int = 10) -> pd.Series:
    """V / median(V of the same Chicago hour over the previous n sessions); 20-bar median fallback."""
    ct = d["time"].dt.tz_convert(EXCHANGE_TZ)
    hour = ct.dt.hour
    v = d["volume"]
    out = pd.Series(np.nan, index=d.index)
    for h, idx in d.groupby(hour).groups.items():
        vh = v.loc[idx]
        med = vh.shift(1).rolling(n_sessions, min_periods=min_samples).median()
        out.loc[idx] = vh / med
    fallback = v / v.shift(1).rolling(20, min_periods=10).median()
    out = out.where(out.notna(), fallback)
    return out.where(v.notna())


def band(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return "≫" if x >= 1.5 else ">" if x >= 1.1 else "≪" if x <= 0.67 else "<" if x <= 0.9 else "≈"


def third_wednesday(y: int, m: int) -> dt.date:
    d = dt.date(y, m, 15)
    return d + dt.timedelta(days=(2 - d.weekday()) % 7)


def business_days_back(d: dt.date, n: int) -> dt.date:
    while n:
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def roll_info(key: str, today: dt.date) -> dict | None:
    """Roll window (approximate, ignores holidays). CME FX: last trade 2 business days before the
    3rd Wednesday of Mar/Jun/Sep/Dec; handling starts 8 sessions earlier. Gold rolls before first
    notice (end of Jan/Mar/May/Jul/Sep/Nov): flagged from the 20th of those months."""
    if key == "GC":
        if today.month in (1, 3, 5, 7, 9, 11) and today.day >= 20:
            return {"in_roll": True, "note": "gold roll window (approx.): Yahoo and TradingView may sit on different months"}
        return {"in_roll": False}
    if key == "DX":
        return None
    for y in (today.year, today.year + 1):
        for m in (3, 6, 9, 12):
            last = business_days_back(third_wednesday(y, m), 2)
            if last >= today:
                start = business_days_back(last, 8)
                return {"in_roll": start <= today <= last, "last_trade": last.isoformat(),
                        "roll_from": start.isoformat()}
    return None


def instrument(key: str, h1: pd.DataFrame, h4: pd.DataFrame | None, spot: pd.DataFrame | None, now) -> dict:
    if h1.empty:
        return {"key": key, "available": False}
    d = h1.copy().reset_index(drop=True)
    tick = FUTURES[key]["tick"]
    ct = d["time"].dt.tz_convert(EXCHANGE_TZ)
    d["day"] = session_day(ct)
    d["tr"] = true_range(d)
    d["atr"] = d["tr"].rolling(14).mean()
    d["eff"] = effort(d)
    d["rs"] = d["tr"] / d["atr"].shift()
    lo_ = np.minimum(d["low"], d["close"].shift().fillna(d["low"]))
    d["cpos"] = ((d["close"] - lo_) / d["tr"].replace(0, np.nan)).clip(0, 1)

    last = d.iloc[-1]
    days = d.groupby("day").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                close=("close", "last"), n=("time", "size"))
    today = days.iloc[-1]
    prev = days.iloc[-2] if len(days) > 1 else None
    done = days.iloc[:-1].tail(20)
    adr = float((done["high"] - done["low"]).mean()) if len(done) else None
    rng = float(today["high"] - today["low"])

    # week = from Sunday 17:00 CT
    week_start = (ct.iloc[-1] - pd.Timedelta(hours=SESSION_OPEN_HOUR_CT))
    week_start = (week_start - pd.Timedelta(days=(week_start.dayofweek + 1) % 7)).floor("D") + pd.Timedelta(hours=SESSION_OPEN_HOUR_CT)
    wk = d[ct >= week_start]
    pw = d[(ct < week_start) & (ct >= week_start - pd.Timedelta(days=7))]

    sess = {}
    td = d[d["day"] == d["day"].iloc[-1]]
    tct = td["time"].dt.tz_convert(EXCHANGE_TZ).dt.hour
    for name, (a, b) in SESSIONS.items():
        mask = ((tct >= a) | (tct < b)) if a > b else ((tct >= a) & (tct < b))
        s = td[mask]
        if len(s):
            sess[name] = {"high": r(s["high"].max()), "low": r(s["low"].min()), "bars": int(len(s))}

    def bar(row):
        return {"time": row["time"].isoformat(), "local": row["time"].tz_convert(LOCAL_TZ).strftime("%a %H:%M"),
                "open": r(row["open"]), "high": r(row["high"]), "low": r(row["low"]), "close": r(row["close"]),
                "volume": None if pd.isna(row["volume"]) else int(row["volume"]),
                "effort": r(row["eff"], 2), "effort_band": band(row["eff"]),
                "spread_atr": r(row["rs"], 2), "close_pos": r(row["cpos"], 2)}

    atr4 = None
    if h4 is not None and len(h4) > 15:
        atr4 = float(true_range(h4).rolling(14).mean().iloc[-1])

    out = {
        "key": key, "name": FUTURES[key]["name"], "available": True, "tick": tick,
        "context_only": bool(FUTURES[key].get("context")),
        "last": bar(last), "price": r(last["close"]),
        "day": {"open": r(today["open"]), "high": r(today["high"]), "low": r(today["low"]),
                "change": r(last["close"] - prev["close"]) if prev is not None else None,
                "change_pct": r((last["close"] / prev["close"] - 1) * 100, 3) if prev is not None else None,
                "change_ticks": r((last["close"] - prev["close"]) / tick, 0) if prev is not None else None,
                "range": r(rng), "pos_in_range": r((last["close"] - today["low"]) / rng, 2) if rng else None},
        "prev_day": None if prev is None else {"high": r(prev["high"]), "low": r(prev["low"]), "close": r(prev["close"])},
        "week": {"high": r(wk["high"].max()), "low": r(wk["low"].min())} if len(wk) else None,
        "prev_week": {"high": r(pw["high"].max()), "low": r(pw["low"].min())} if len(pw) else None,
        "sessions": sess,
        "atr_1h": r(last["atr"]), "atr_1h_ticks": r(last["atr"] / tick, 1),
        "atr_4h": r(atr4), "atr_4h_ticks": r(atr4 / tick, 1) if atr4 else None,
        "adr20": r(adr), "adr20_ticks": r(adr / tick, 1) if adr else None,
        "adr_used_pct": r(rng / adr * 100, 0) if adr else None,
        "recent": [bar(x) for _, x in d.tail(6).iloc[::-1].iterrows()],
        "roll": roll_info(key, pd.Timestamp(now).tz_convert(EXCHANGE_TZ).date()),
    }
    if key in MT5 and spot is not None and len(spot):
        sym, direction = MT5[key]
        out["mt5"] = {"symbol": sym, "direction": direction, "spot": r(spot["close"].iloc[-1]),
                      "spot_time": spot["time"].iloc[-1].isoformat(), "pip": PIP[sym]}
    elif key in MT5:
        out["mt5"] = {"symbol": MT5[key][0], "direction": MT5[key][1], "spot": None, "pip": PIP[MT5[key][0]]}
    return out


def strength(h1: dict[str, pd.DataFrame]) -> dict:
    """Currency strength (Stack v2): r_i = % change vs USD (USD = 0); strength_i = mean over the
    other 7 of (r_i − r_j). Windows: last 1h bar, current CME day, current week."""
    res = {}
    for label, pick in (("1h", "bar"), ("1d", "day"), ("1w", "week")):
        rets = {"USD": 0.0}
        for k, ccy in CCY.items():
            d = h1.get(k)
            if d is None or len(d) < 3:
                continue
            ct = d["time"].dt.tz_convert(EXCHANGE_TZ)
            if pick == "bar":
                base = d["close"].iloc[-2]
            elif pick == "day":
                day = session_day(ct)
                prev = d[day < day.iloc[-1]]
                base = prev["close"].iloc[-1] if len(prev) else d["open"].iloc[0]
            else:
                ws = (ct.iloc[-1] - pd.Timedelta(hours=SESSION_OPEN_HOUR_CT))
                ws = (ws - pd.Timedelta(days=(ws.dayofweek + 1) % 7)).floor("D") + pd.Timedelta(hours=SESSION_OPEN_HOUR_CT)
                prev = d[ct < ws]
                base = prev["close"].iloc[-1] if len(prev) else d["open"].iloc[0]
            rets[ccy] = (d["close"].iloc[-1] / base - 1) * 100
        ccys = list(rets)
        score = {c: float(np.mean([rets[c] - rets[o] for o in ccys if o != c])) for c in ccys}
        ranked = sorted(score, key=score.get, reverse=True)
        res[label] = [{"ccy": c, "strength_pct": r(score[c], 3), "vs_usd_pct": r(rets[c], 3)} for c in ranked]
    return res


def build(h1: dict, h4: dict, spot: dict, health_summary: dict, now) -> dict:
    now = pd.Timestamp(now).tz_convert("UTC")
    inst = {k: instrument(k, h1.get(k, pd.DataFrame()), h4.get(k), spot.get(MT5.get(k, ("", ""))[0]), now)
            for k in FUTURES}
    return {
        "generated_utc": now.isoformat(),
        "generated_local": now.tz_convert(LOCAL_TZ).strftime("%a %d %b %H:%M"),
        "source": "Yahoo Finance hourly bars (robot copy). TradingView is the price of record.",
        "instruments": inst,
        "strength": strength({k: v for k, v in h1.items() if k in CCY}),
        "health": {"problems": [p["text"] for p in health_summary.get("problems", [])],
                   "symbols": {k: {"status": v["status"], "last_local": v["last_local"]}
                               for k, v in health_summary.get("symbols", {}).items()}},
        "spot": {k: {"price": r(v["close"].iloc[-1]), "time": v["time"].iloc[-1].isoformat()}
                 for k, v in spot.items() if len(v)},
    }
