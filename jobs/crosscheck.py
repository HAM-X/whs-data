"""Step 1.5: compare the robot's bars with TradingView exports, bar by bar.

The TradingView files are private (whs-desk), so this runs on Claude's workspace, not on GitHub.
    python -m jobs.crosscheck --tv /path/to/whs-desk/reference/tradingview [--json out.json]

Gate: closes within 1 tick on >= 99% of shared bars, and volume moving in step (correlation of
log volume >= 0.8, same hour-of-day profile). Days where the two feeds sit on different contracts
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
    tick = FUTURES[key]["tick"]
    m = robot.merge(tv, on="time", suffixes=("_r", "_t"))
    if m.empty:
        return dict(symbol=key, tf=tf, shared=0)
    m["dticks"] = (m["close_r"] - m["close_t"]).abs() / tick
    day = m["time"].dt.tz_convert(EXCHANGE_TZ).dt.floor("D")
    day_med = m.groupby(day)["dticks"].transform("median")
    roll = day_med > 2                                   # whole day offset = different contract month
    clean = m[~roll]
    lv_r, lv_t = np.log1p(m["volume_r"]), np.log1p(m["volume_t"])
    hour = m["time"].dt.tz_convert(EXCHANGE_TZ).dt.hour
    prof_r = m.groupby(hour)["volume_r"].median()
    prof_t = m.groupby(hour)["volume_t"].median()
    return dict(
        symbol=key, tf=tf, shared=len(m),
        first=m["time"].min().isoformat(), last=m["time"].max().isoformat(),
        close_within_1tick=round(float((m["dticks"] <= 1.0001).mean()), 4),
        close_within_1tick_excl_roll=round(float((clean["dticks"] <= 1.0001).mean()), 4) if len(clean) else None,
        median_diff_ticks=round(float(m["dticks"].median()), 2),
        roll_mismatch_days=int(day[roll].nunique()),
        roll_mismatch_bars=int(roll.sum()),
        volume_corr_log=round(float(np.corrcoef(lv_r, lv_t)[0, 1]), 3) if len(m) > 2 else None,
        volume_ratio_median=round(float((m["volume_r"] / m["volume_t"].replace(0, np.nan)).median()), 3),
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
    ok_price = (gate["close_within_1tick_excl_roll"] >= 0.99).all()
    ok_vol = (gate["volume_corr_log"] >= 0.8).all() and (gate["hour_profile_corr"] >= 0.9).all()
    print(f"\nGATE price (1h, excl. roll days, >=99% within 1 tick): {'PASS' if ok_price else 'FAIL'}")
    print(f"GATE volume (1h, log corr >=0.8 and hour profile >=0.9): {'PASS' if ok_vol else 'FAIL'}")
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tv", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    main(a.tv, a.json)
