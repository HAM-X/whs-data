"""Bar storage: normalise, merge new bars into stored files, build 4h bars like TradingView."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import EXCHANGE_TZ, SESSION_OPEN_HOUR_CT

COLUMNS = ["time", "time_ct", "open", "high", "low", "close", "volume"]


def normalise(df: pd.DataFrame, now=None, step=pd.Timedelta(hours=1)) -> pd.DataFrame:
    """Raw OHLCV with a tz-aware index (any tz) -> our columns, UTC, complete bars only."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    d = df.rename(columns=str.lower)
    idx = pd.DatetimeIndex(d.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = pd.DataFrame({
        "time": idx,
        "open": d["open"].to_numpy(dtype=float),
        "high": d["high"].to_numpy(dtype=float),
        "low": d["low"].to_numpy(dtype=float),
        "close": d["close"].to_numpy(dtype=float),
        "volume": d["volume"].fillna(0).to_numpy(dtype=float) if "volume" in d else 0.0,
    })
    out = out.dropna(subset=["open", "high", "low", "close"])
    grid = "h" if step >= pd.Timedelta(hours=1) else f"{int(step.total_seconds() // 60)}min"
    out = out[out["time"] == out["time"].dt.floor(grid)]         # the bar grid of this timeframe (hourly, 15 min, 5 min)
    if now is not None:                                            # drop the bar still forming
        out = out[out["time"] + step <= pd.Timestamp(now).tz_convert("UTC")]
    out["time_ct"] = out["time"].dt.tz_convert(EXCHANGE_TZ)
    return out[COLUMNS].sort_values("time").reset_index(drop=True)


def clean_future(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo quirks on CME futures, found in the step 1.5 cross-check against TradingView:
    - stray bars stamped inside the 16:00-17:00 CT halt (CME is closed) -> dropped
    - the 17:00 CT session-open bar arrives with volume 0 although it traded -> volume marked missing (NaN)
    """
    from .market import is_open
    if df.empty:
        return df
    d = df[df["time"].map(is_open)].copy()
    ct_hour = d["time"].dt.tz_convert(EXCHANGE_TZ).dt.hour
    d.loc[(ct_hour == SESSION_OPEN_HOUR_CT) & (d["volume"] == 0), "volume"] = float("nan")
    return d.reset_index(drop=True)


def read(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    d = pd.read_csv(path)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d["time_ct"] = d["time"].dt.tz_convert(EXCHANGE_TZ)
    return d[COLUMNS]


def write(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["time_ct"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(EXCHANGE_TZ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.to_csv(path, index=False, float_format="%.10g")


def merge(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Append new bars; a bar present in both is replaced by the new copy (Yahoo revises recent bars).

    Returns (merged, bars_added, bars_revised)."""
    if old.empty:
        return new.reset_index(drop=True), len(new), 0
    if new.empty:
        return old, 0, 0
    o = old.set_index("time")
    n = new.set_index("time")
    common = o.index.intersection(n.index)
    cols = ["open", "high", "low", "close", "volume"]
    a, b = o.loc[common, cols].round(10).fillna(-1), n.loc[common, cols].round(10).fillna(-1)
    revised = int((a != b).any(axis=1).sum())
    added = len(n.index.difference(o.index))
    m = pd.concat([o[~o.index.isin(n.index)], n]).sort_index().reset_index()
    return m[COLUMNS], added, revised


def build_4h(h1: pd.DataFrame, now=None) -> pd.DataFrame:
    """4h bars in 4-hour blocks from the 17:00 CT session open (17, 21, 01, 05, 09, 13 CT).

    Matches TradingView's CME 4h bars exactly (checked on 4,327 bars of 6E). The 13:00 block
    ends at the 16:00 close, so it holds 3 hourly bars. Incomplete blocks are left out."""
    if h1.empty:
        return pd.DataFrame(columns=COLUMNS)
    ct = h1["time"].dt.tz_convert(EXCHANGE_TZ)
    shifted = ct - pd.Timedelta(hours=SESSION_OPEN_HOUR_CT)
    session_day = shifted.dt.floor("D")
    offset_h = (shifted - session_day).dt.total_seconds() // 3600
    block_ct = session_day + pd.Timedelta(hours=SESSION_OPEN_HOUR_CT) + pd.to_timedelta((offset_h // 4) * 4, unit="h")
    d = h1.assign(block=block_ct.dt.tz_convert("UTC"))
    g = d.groupby("block", sort=True).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                          close=("close", "last"), volume=("volume", "sum"),
                                          first=("time", "min"), last=("time", "max"))
    g.index.name = "time"
    g = g.reset_index()
    if len(g) and g["first"].iloc[0] > g["time"].iloc[0]:   # history starts mid-block: that block is partial
        g = g.iloc[1:].reset_index(drop=True)
    block_len = pd.Series(4, index=g.index)
    block_len[g["time"].dt.tz_convert(EXCHANGE_TZ).dt.hour == 13] = 3
    g["end"] = g["time"] + pd.to_timedelta(block_len, unit="h")
    if now is not None:
        g = g[g["end"] <= pd.Timestamp(now).tz_convert("UTC")]
    else:  # without a clock, drop only a trailing block that is visibly unfinished
        if len(g) and g["last"].iloc[-1] + pd.Timedelta(hours=1) < g["end"].iloc[-1]:
            g = g.iloc[:-1]
    # 4h files are rebuilt from the 1h file on every run, so a late hourly bar fixes its block next run
    g["time_ct"] = g["time"].dt.tz_convert(EXCHANGE_TZ)
    return g[COLUMNS].reset_index(drop=True)
