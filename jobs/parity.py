"""R0 parity test: Pine (TradingView export with WHS R0 in export mode) vs the Python Core.

    python -m jobs.parity path/to/export.csv [--tf 60]

Pass = every Core number identical to 1e-9 (relative) on every bar where both are defined.
Effort is recomputed from Pine's own "Core effort volume" column, so roll-week bars (front + next
contract volume, which only TradingView can see) are compared too.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from whs import core

TOL = 1e-9


def col(d: pd.DataFrame, name: str) -> pd.Series:
    for c in d.columns:
        if c.strip().lower() == name.lower():
            return d[c].astype(float)
    raise KeyError(f"column '{name}' not in export (is WHS R0 on the chart with Export mode on?)")


def compare(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = ~np.isnan(a) & ~np.isnan(b)
    rel = np.abs(a[both] - b[both]) / np.maximum(1e-12, np.maximum(np.abs(a[both]), np.abs(b[both])))
    bad = int((rel > TOL).sum())
    only = int((np.isnan(a) != np.isnan(b)).sum())
    return dict(value=name, compared=int(both.sum()), mismatches=bad, worst=float(rel.max()) if len(rel) else 0.0,
                na_disagree=only)


def main(path: str, tf: int, root: str | None = None) -> int:
    d = pd.read_csv(path)
    t = pd.to_datetime(d["time"], utc=True)
    h, l, c = col(d, "high"), col(d, "low"), col(d, "close")
    veff = col(d, "Core effort volume")
    py = {
        "Core TR": core.true_range(h, l, c),
        "Core ATR14": core.atr(h, l, c),
        "Core rel spread": core.rel_spread(h, l, c),
        "Core close pos": core.close_pos(h, l, c),
        "Effort": core.effort(veff, t, tf),
    }
    # TradingView runs the script on history older than the export, so ATR (Wilder memory) and the
    # same-slot buffers start from state the file doesn't contain. Skip a warm-up long enough for both
    # to be fully determined by exported bars: (13/14)^500 < 1e-16, and every slot refilled 20+ times.
    warm = max(500, 30 * int(round(1440 / tf)))
    rows = [compare(col(d, k).iloc[warm:], v[warm:], k) for k, v in py.items()]
    roll_bars = int(col(d, "Core in roll").sum())
    root = root or (re.search(r"(6[EBANJCS]|GC)\d?!?|_(6[EBANJCS]|GC)1_", Path(path).name) or [None, None])[0]
    if root:
        root = re.sub(r"[^A-Z0-9]", "", root)[:2]
        ct = t.dt.tz_convert(core.CT)
        sess = (ct + pd.Timedelta(hours=7)).dt.date
        exp = sess.map(lambda x: core.in_roll_window(root, x)).astype(float).to_numpy()
        got = col(d, "Core in roll").to_numpy()
        rows.append(dict(value=f"Roll window ({root})", compared=len(d), mismatches=int((exp != got).sum()),
                         worst=float(np.abs(exp - got).max()), na_disagree=0))
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    print(f"\nbars {len(d)} (first {warm} = warm-up, not compared) · roll-window bars {roll_bars} · first {t.iloc[0]} · last {t.iloc[-1]}")
    ok = (rep["mismatches"] == 0).all() and (rep["na_disagree"] == 0).all()
    print("R0 PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--tf", type=int, default=60, help="timeframe in minutes (60 for 1h, 240 for 4h)")
    ap.add_argument("--root", help="contract root, e.g. 6E (default: read from the file name)")
    a = ap.parse_args()
    sys.exit(main(a.export, a.tf, a.root))
