"""R0 parity test: Pine (TradingView export with WHS R0 in export mode) vs the Python Core.

    python -m jobs.parity path/to/export.csv [--tf 60]

Pass = every Core number identical to 1e-9 (relative) on every bar where both are defined.
Effort is recomputed from Pine's own "Core effort volume" column, so roll-week bars (front + next
contract volume, which only TradingView can see) are compared too.
"""
from __future__ import annotations

import argparse
import sys

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


def main(path: str, tf: int) -> int:
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
    rows = [compare(col(d, k), v, k) for k, v in py.items()]
    roll_bars = int(col(d, "Core in roll").sum())
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    print(f"\nbars {len(d)} · roll-window bars {roll_bars} · first {t.iloc[0]} · last {t.iloc[-1]}")
    ok = (rep["mismatches"] == 0).all() and (rep["na_disagree"] <= 25).all()
    print("R0 PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--tf", type=int, default=60, help="timeframe in minutes (60 for 1h, 240 for 4h)")
    a = ap.parse_args()
    sys.exit(main(a.export, a.tf))
