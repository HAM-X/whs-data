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

from whs import core, swings

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
    if any(c.strip() == "R1 SBar" for c in d.columns):
        o = col(d, "open")
        py_e = core.effort(veff, t, tf)
        pine_e = col(d, "Effort").to_numpy()
        sb_py = swings.sbars(o, h, l, c)
        tl_py, seq_py = swings.bar_traffic_light(h, l, c, pine_e)          # same effort input as Pine
        exact = lambda name, a, b: dict(value=name, compared=len(a), mismatches=int((np.asarray(a) != np.asarray(b)).sum()),
                                         worst=0.0, na_disagree=0)
        rows.append(exact("R1 SBar", sb_py[warm:], col(d, "R1 SBar").to_numpy()[warm:].astype(int)))
        rows.append(exact("R1 traffic light", tl_py[warm:], col(d, "R1 traffic light").to_numpy()[warm:].astype(int)))
        rows.append(exact("R1 TL sequence", seq_py[warm + 50:], col(d, "R1 TL sequence").to_numpy()[warm + 50:].astype(int)))
        # swings: Pine started on older history, so compare after the first reversal both agree on
        led = swings.ledger(o, h, l, c, pine_e)
        rb_py, rb_pine = led.rbar, col(d, "R1 RBar").to_numpy().astype(int)
        common = [i for i in range(warm, len(d)) if rb_py[i] != 0 and rb_py[i] == rb_pine[i]]
        if common:
            k0 = common[0]
            rows.append(exact("R1 RBar (after sync)", rb_py[k0:], rb_pine[k0:]))
            ext_py = np.full(len(d), np.nan)
            for sw in led.swings:
                ext_py[sw.confirm_i] = sw.extent_pct
            rows.append(compare(col(d, "R1 swing extent").iloc[k0 + 1:], ext_py[k0 + 1:], "R1 swing extent"))
            # swing traffic light needs two same-direction swings of shared history, sequences up to 6 more
            k1 = next((led.swings[j].confirm_i for j, sw in enumerate(led.swings) if sw.confirm_i > k0 and j >= 2
                       and led.swings[j - 2].confirm_i > k0), k0)
            rows.append(exact("R1 swing TL", led.tl[k1:], col(d, "R1 swing TL").to_numpy()[k1:].astype(int)))
            k2 = next((sw.confirm_i for j, sw in enumerate(led.swings) if j >= 8 and led.swings[j - 8].confirm_i > k0), k1)
            rows.append(exact("R1 swing TL sequence", led.seq[k2:], col(d, "R1 swing TL sequence").to_numpy()[k2:].astype(int)))
            rows.append(dict(value="(swings compared)", compared=int((rb_pine[k0:] != 0).sum()), mismatches=0,
                             worst=0.0, na_disagree=0))
        _ = py_e
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    print(f"\nbars {len(d)} (first {warm} = warm-up, not compared) · roll-window bars {roll_bars} · first {t.iloc[0]} · last {t.iloc[-1]}")
    ok = (rep["mismatches"] == 0).all() and (rep["na_disagree"] == 0).all()
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--tf", type=int, default=60, help="timeframe in minutes (60 for 1h, 240 for 4h)")
    ap.add_argument("--root", help="contract root, e.g. 6E (default: read from the file name)")
    a = ap.parse_args()
    sys.exit(main(a.export, a.tf, a.root))
