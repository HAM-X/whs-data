"""Step 1.5: compare the robot's bars with TradingView exports, bar by bar.

The TradingView files are private (whs-desk), so this runs on Claude's workspace, not on GitHub.
    python -m jobs.crosscheck --tv /path/to/whs-desk/reference/tradingview [--json out.json]

Gate: closes within 1 tick on >= 99% of shared bars, and volume moving in step (rank correlation
>= 0.9, same hour-of-day profile). Days where the two feeds sit on different contracts
(roll timing) are listed separately so they don't hide a real feed problem.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whs import bars
from whs.config import EXCHANGE_TZ, FUTURES

ROOT = Path(__file__).resolve().parents[1]


def load_tv(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d.rename(columns={"Volume": "volume"})[["time", "open", "high", "low", "close", "volume"]]


def compare(key: str, tf: str, robot: pd.DataFrame, tv: pd.DataFrame) -> dict:
    """Prices compared in whole ticks (Yahoo floats carry rounding noise). A bar more than 20 ticks
    off is a roll-timing bar (the feeds sit on different contract months) and is reported apart."""
    tick = FUTURES[key]["tick"]
    m = robot.merge(tv, on="time", suffixes=("_r", "_t"))
    if m.empty:
        return dict(symbol=key, tf=tf, shared=0)
    d = ((m["close_r"] - m["close_t"]) / tick).round().abs()
    roll = d > 20
    c, dc = m[~roll], d[~roll]
    ct_hour = c["time"].dt.tz_convert(EXCHANGE_TZ).dt.hour
    v = c.dropna(subset=["volume_r"])
    v = v[v["volume_t"] > 0]
    ratio = v["volume_r"] / v["volume_t"]
    hour_v = v["time"].dt.tz_convert(EXCHANGE_TZ).dt.hour
    prof_r, prof_t = v.groupby(hour_v)["volume_r"].median(), v.groupby(hour_v)["volume_t"].median()
    return dict(
        symbol=key, tf=tf, shared=len(m),
        first=m["time"].min().isoformat(), last=m["time"].max().isoformat(),
        roll_bars=int(roll.sum()),
        roll_days=int(m.loc[roll, "time"].dt.tz_convert(EXCHANGE_TZ).dt.date.nunique()),
        close_exact=round(float((dc == 0).mean()), 4),
        close_within_1tick=round(float((dc <= 1).mean()), 4),
        close_within_1tick_excl_22ct=round(float((dc[ct_hour != 22] <= 1).mean()), 4),
        close_within_2ticks=round(float((dc <= 2).mean()), 4),
        volume_bars=len(v),
        volume_rank_corr=round(float(v["volume_r"].corr(v["volume_t"], method="spearman")), 3),
        volume_within_20pct=round(float(((ratio > 0.8) & (ratio < 1.25)).mean()), 3),
        volume_ratio_median=round(float(ratio.median()), 3),
        hour_profile_corr=round(float(np.corrcoef(prof_r, prof_t)[0, 1]), 3) if len(prof_r) > 2 else None,
    )


def main(tv_dir: Path, out_json: Path | None):
    rows = []
    for key in FUTURES:
        if key == "DX":       # TradingView side is the Capital.com DXY CFD, a different instrument
            continue
        for tf in ("1h", "4h"):
            rp, tp = ROOT / "data" / "bars" / tf / f"{key}.csv", tv_dir / f"{key}_{tf}.csv"
            if not rp.exists() or not tp.exists():
                rows.append(dict(symbol=key, tf=tf, shared=0, note="file missing"))
                continue
            rows.append(compare(key, tf, bars.read(rp)[["time", "open", "high", "low", "close", "volume"]], load_tv(tp)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    gate = df[df.tf == "1h"]
    ok_price = (gate["close_within_1tick"] >= 0.99).all()
    ok_vol = (gate["volume_rank_corr"] >= 0.9).all() and (gate["hour_profile_corr"] >= 0.9).all()
    print(f"\nGATE price (1h, excl. roll bars, >=99% of closes within 1 tick): {'PASS' if ok_price else 'FAIL'}")
    for _, r in gate[gate["close_within_1tick"] < 0.99].iterrows():
        print(f"   {r.symbol}: {r.close_within_1tick:.2%} within 1 tick ({r.close_within_2ticks:.2%} within 2)")
    print(f"GATE volume (1h, rank corr >=0.9 and hour profile >=0.9): {'PASS' if ok_vol else 'FAIL'}")
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tv", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    main(a.tv, a.json)
