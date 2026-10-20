"""Stage 7 · Edge Lab v1 — what each Wyckoff entry has actually earned, with honest error bars.

For every automatic range (whs/autorange.chain, the same rule as WHS Range Auto) every entry event the reader finds
(Spring · Shakeout · Test · LPS · SOS bar · BU, and the mirrored UTAD · Test · LPSY · SOW bar · LPSY back-up) is
traded LIVE: entry at the close of the bar on which the event became known, stop = the structure ∓ 0.25 ATR
(SOS/SOW bar: its midpoint), refused if the risk is > 3 ATR or < 0.2 ATR. Each trade is then managed with the Trade
Desk's ADAPTIVE stop (SBar commitments → true-range low; from +1R the one-level structure stop; a climactic bar after
+1R → ½ of it), a 200-bar time cap and 0.05 R costs.

Baseline: for every trade, 3 random entries in the same direction within ±50 bars, stop 1.5 ATR, same management —
an edge must beat what the stop alone would have earned.

Stats per setup: n, win %, Wilson 95 % range of the win rate, average R, expectancy (after costs), "worst case" =
the 5th percentile of the bootstrapped mean, and the baseline's expectancy. PROVEN = n ≥ 30, worst case > 0 and
expectancy above the baseline's — only proven setups may trade at grade A (whs/desk.py EDGE_LAB_PROVEN).
In-sample and the locked hold-out (from 18 Apr 2026) are reported separately.
"""
from __future__ import annotations

import math

import numpy as np

from . import autorange, core, swings

COST_R = 0.05
MAX_BARS = 200
STOP_ATR, MAX_RISK_ATR, MIN_RISK_ATR = 0.25, 3.0, 0.2
SETUP = {  # role → (long name, short name, stop basis)
    "Spring": ("Spring", "UTAD", "low"), "Shakeout": ("Shakeout", "UTAD (shakeout-type)", "low"),
    "Test": ("Test of the spring", "Test of the UTAD", "low"), "Chain": ("LPS", "LPSY", "low"),
    "D_LPS": ("LPS", "LPSY", "low"), "BO": ("SOS bar", "SOW bar", "mid"), "BU": ("BU", "LPSY back-up", "low"),
}


def _prep(o, h, l, c, eff, led):
    c1 = np.r_[c[0], c[:-1]]
    by = {}
    for dd in led.done:
        by.setdefault(dd[5], []).append(dd)
    return dict(h=h, l=l, c=c, eff=eff, atr=core.atr(h, l, c), sb=led.sbar, trlo=np.minimum(l, c1),
                trhi=np.maximum(h, c1), sw=[(dd[5], dd[0], dd[4]) for dd in led.done], by=by, n=len(c))


def adaptive(D, i0, side, e, stop0, max_bars=MAX_BARS):
    """The Trade Desk's Adaptive stop (twin of design/trail_final.py). Returns (R, bars held, best R)."""
    h, l, c, eff, atr, sb, trlo, trhi = (D[k] for k in ("h", "l", "c", "eff", "atr", "sb", "trlo", "trhi"))
    R = abs(e - stop0)
    sup = [p for (cf, dr, p) in D["sw"] if cf <= i0 and dr == -1]
    res = [p for (cf, dr, p) in D["sw"] if cf <= i0 and dr == 1]
    stop, commit, best = stop0, None, 0.0
    end = min(D["n"], i0 + 1 + max_bars)
    for i in range(i0 + 1, end):
        if (side == 1 and l[i] <= stop) or (side == -1 and h[i] >= stop):
            return (stop - e) * side / R, i - i0, best
        best = max(best, ((h[i] if side == 1 else l[i]) - e) * side / R)
        for dd in D["by"].get(i, []):
            (sup if dd[0] == -1 else res).append(dd[4])
        cands = [stop]
        lvA, spA = (res, sup) if side == 1 else (sup, res)
        if lvA:
            lvl = lvA[-1]
            if (c[i] - lvl) * side > 0 and (commit is None or (lvl - commit) * side > 0):
                commit = lvl
                below = [p for p in spA if (p - lvl) * side < 0]
                if best >= 1.0 and below:
                    cands.append(below[-1] - side * 0.1 * atr[i0])
        if np.sign(sb[i]) == side:
            lv = (min(trlo[i], trlo[i - 1]) if abs(sb[i]) == 2 else trlo[i]) if side == 1 else \
                (max(trhi[i], trhi[i - 1]) if abs(sb[i]) == 2 else trhi[i])
            cands.append(lv)
            if best >= 1.0 and not np.isnan(eff[i]) and eff[i] >= 2.5 and trhi[i] - trlo[i] >= 2.0 * atr[i - 1]:
                cands.append((trhi[i] + trlo[i]) / 2)
        ns = max(cands) if side == 1 else min(cands)
        if (ns - stop) * side > 0 and (c[i] - ns) * side > 0:
            stop = ns
    k = end - 1
    return (c[k] - e) * side / R, k - i0, best


def trades(o, h, l, c, eff, v, led=None, ranges_=None):
    """Every entry event of every automatic range, traded live. Returns a list of dicts."""
    o, h, l, c, eff, v = (np.asarray(x, float) for x in (o, h, l, c, eff, v))
    led = led or swings.ledger(o, h, l, c, eff)
    ranges_ = ranges_ if ranges_ is not None else autorange.chain(o, h, l, c, eff, v, led)
    D = _prep(o, h, l, c, eff, led)
    out, seen = [], set()
    for trig, rr in ranges_:
        for ev in rr.events:
            if ev.role not in SETUP or ev.side == 0:
                continue
            key = (ev.role, ev.bar, ev.side)
            if key in seen:
                continue
            seen.add(key)
            side, i0 = ev.side, ev.known
            if i0 >= len(c) - 1 or np.isnan(D["atr"][i0]):
                continue
            name_l, name_s, basis = SETUP[ev.role]
            a = D["atr"][i0]
            struct = (h[ev.bar] + l[ev.bar]) / 2 if basis == "mid" else ev.price
            stop = struct - side * STOP_ATR * a
            e = c[i0]
            risk = (e - stop) * side
            if risk < MIN_RISK_ATR * a or risk > MAX_RISK_ATR * a:
                continue
            r, held, best = adaptive(D, i0, side, e, stop)
            out.append(dict(setup=name_l if side == 1 else name_s, side=side, bar=i0, entry=e, stop=stop,
                            R=r - COST_R, held=held, best=best, range=trig["a1"], state=ev.state))
    return out, D


def baseline(D, trs, k=3, window=50, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    n = D["n"]
    for t in trs:
        for _ in range(k):
            i0 = int(np.clip(t["bar"] + rng.integers(-window, window + 1), 20, n - 2))
            a = D["atr"][i0]
            if np.isnan(a):
                continue
            e = D["c"][i0]
            r, held, best = adaptive(D, i0, t["side"], e, e - t["side"] * 1.5 * a)
            out.append(dict(setup=t["setup"], R=r - COST_R))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (m - w, m + w)


def stats(trs, base=None, boots=2000, seed=11):
    rng = np.random.default_rng(seed)
    out = {}
    for s in sorted({t["setup"] for t in trs}):
        R = np.array([t["R"] for t in trs if t["setup"] == s])
        n = len(R)
        wins = int((R > 0).sum())
        lo, hi = wilson(wins, n)
        worst = float(np.percentile([rng.choice(R, n).mean() for _ in range(boots)], 5)) if n >= 5 else float("nan")
        b = np.array([t["R"] for t in (base or []) if t["setup"] == s])
        bx = float(b.mean()) if len(b) else float("nan")
        exp = float(R.mean())
        out[s] = dict(n=n, win=wins / n, win_lo=lo, win_hi=hi, avg_win=float(R[R > 0].mean()) if wins else 0.0,
                      avg_loss=float(R[R <= 0].mean()) if wins < n else 0.0, exp=exp, worst=worst,
                      big=float((R >= 3).mean()), base=bx,
                      proven=bool(n >= 30 and worst > 0 and (np.isnan(bx) or exp > bx)))
    return out
