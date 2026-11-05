"""WHS Engine R1 in Python — the bar reading, the wave and the parent's commitment. Twin of WHS Engine R1.4 (Pine).

Used by the robot to attach the R1 reading to every trade Hamza opens (the self-scoring journal), and by the replay.

bar     Roman's bar-by-bar template (WTC P1 S7): effort and vs the last bar · demand / supply vs the last bar of the
        same side · spread (TR ÷ ATR[1]) · close in the spread · close to close · intention (2 Up→Yes · 1 Up→No ·
        −1 Down→No · −2 Down→Yes · 0 inside bar).
commit  the commitment line (whs/commit.py): state, SL, SBar steps since the flip.
wave    the commitment's own waves: a wave runs from one RBar to the next (the RBar bar opens it, the next RBar closes it).
        effort = Σ bar effort ("normal bars"), travel = the wave's extreme beyond the RBar close ÷ ATR[1] at the RBar,
        cost = effort per ATR of travel. Compared with the last COMPLETED wave of the same side: dE, dR, cost ×
        (cost × only once the wave is 3+ bars and 0.5+ ATR old — before that it is "young"). Cost is fair mid-wave
        (both parts grow together); effort alone is not (a young wave always has "less effort").
        Replay (5 Nov 2026, TV data, 7,628 1h + 2,159 4h waves, logistic regression): with the wave's travel and duration
        (and the previous wave's) held, effort adds nothing to whether the next same-side wave travels less (1h: effort ×2
        → +1.2 points, z 1.4; in-sample 0.0, hold-out +1.8 · 4h z 0.6) nor to the size of the reaction. A first, coarser
        test (+4.4 points) was regression to the mean — corrected. → like bar effort, a measurement to read, not an edge.
parent  the next timeframe's commitment (1h → 4h), read from its last CLOSED bar (no repainting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import commit, core

LIGHT = {"red": "more effort, less travel", "green": "more effort, more travel", "ease": "less effort, more travel"}


def light(dE, dR) -> str:
    """The R1 traffic-light rule applied to two numbers (bars or waves)."""
    if dE is None or dR is None or not np.isfinite(dE) or not np.isfinite(dR):
        return ""
    return "red" if dE >= 1.1 and dR <= 0.9 else "green" if dE >= 1.1 and dR >= 1.1 else "ease" if dE <= 0.9 and dR >= 1.1 else ""


def cost_band(x) -> str:
    """The wave's cost (effort per ATR of travel) vs the last wave of its side: dearer ≥ 1.25 · cheaper ≤ 0.8."""
    if x is None or not np.isfinite(x):
        return "young"
    return "dearer" if x >= 1.25 else "cheaper" if x <= 0.8 else "even"


def reading(o, h, l, c, v, times: pd.Series, tf_minutes: int) -> dict:
    """Per-bar arrays of the R1 reading (same rules as the Pine script)."""
    o, h, l, c, v = (np.asarray(x, float) for x in (o, h, l, c, v))
    n = len(c)
    eff = core.effort(v, times, tf_minutes)
    tr = core.true_range(h, l, c)
    atr = core.atr(h, l, c)
    atr1 = np.r_[np.nan, atr[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = tr / atr1
    cp = core.close_pos(h, l, c)
    pc = np.r_[np.nan, c[:-1]]
    c2c = np.where(np.isnan(pc), 0, np.sign(c - pc)).astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        dEb = np.r_[np.nan, eff[1:] / eff[:-1]]
        dRb = np.r_[np.nan, rs[1:] / rs[:-1]]
    dEb[~np.isfinite(dEb)] = np.nan
    dRb[~np.isfinite(dRb)] = np.nan
    dSide = np.full(n, np.nan)
    last = {1: np.nan, -1: np.nan}
    for i in range(n):
        s = c2c[i]
        if s != 0 and not np.isnan(eff[i]):
            if last[s] > 0:
                dSide[i] = eff[i] / last[s]
            last[s] = eff[i]
    att = np.zeros(n, int)
    for i in range(1, n):
        up, dn = h[i] > h[i - 1], l[i] < l[i - 1]
        att[i] = 1 if up and not dn else -1 if dn and not up else (1 if h[i] - h[i - 1] >= l[i - 1] - l[i] else -1) if up and dn else 0
    intent = np.where(att == 1, np.where((c2c == 1) & (cp >= 0.5), 2, 1),
                      np.where(att == -1, np.where((c2c == -1) & (cp <= 0.5), -2, -1), 0))
    cm = commit.commitment(o, h, l, c)
    # waves: RBar → next RBar
    wE, wR, wN, wdE, wdR, wC = (np.full(n, np.nan) for _ in range(6))
    lS, lN, lE, lR, ldE, ldR = (np.full(n, np.nan) for _ in range(6))   # the last COMPLETED wave (what the replay tested)
    done = {1: None, -1: None}                     # last completed wave of each side: (E, R)
    last = (np.nan,) * 6
    st, e, ext, c0, a0, k = 0, 0.0, np.nan, np.nan, np.nan, 0
    for i in range(n):
        if cm["rbar"][i] != 0:
            if st != 0 and k:
                R_ = (ext - c0) * st / a0 if a0 > 0 else np.nan
                p = done[st]
                ok = p is not None and p[0] > 0 and np.isfinite(p[1]) and p[1] > 0
                last = (st, k, e, R_, e / p[0] if ok else np.nan, R_ / p[1] if ok and np.isfinite(R_) else np.nan)
                done[st] = (e, R_)
            st, e, ext, c0, a0, k = int(cm["rbar"][i]), 0.0, (h[i] if cm["rbar"][i] > 0 else l[i]), c[i], atr1[i], 0
        lS[i], lN[i], lE[i], lR[i], ldE[i], ldR[i] = last
        if st == 0:
            continue
        k += 1
        e += 0.0 if np.isnan(eff[i]) else eff[i]
        ext = max(ext, h[i]) if st > 0 else min(ext, l[i])
        wE[i], wN[i] = e, k
        wR[i] = (ext - c0) * st / a0 if a0 > 0 else np.nan
        p = done[st]
        if p and p[0] > 0 and np.isfinite(p[1]) and p[1] > 0:
            wdE[i], wdR[i] = e / p[0], wR[i] / p[1]
            if k >= 3 and wR[i] >= 0.5:
                wC[i] = (e / wR[i]) / (p[0] / p[1])
    return dict(effort=eff, rs=rs, cpos=cp, c2c=c2c, dEb=dEb, dRb=dRb, dSide=dSide, att=att, intent=intent, atr=atr,
                state=cm["state"], line=cm["line"], step=cm["step"], rbar=cm["rbar"], sb=cm["sb"],
                wave_effort=wE, wave_travel=wR, wave_bars=wN, wave_dE=wdE, wave_dR=wdR, wave_cost=wC,
                last_side=lS, last_bars=lN, last_effort=lE, last_travel=lR, last_dE=ldE, last_dR=ldR)


def _f(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def last_bar(d: pd.DataFrame, tf_minutes: int = 60, parent: pd.DataFrame | None = None, parent_minutes: int = 240) -> dict | None:
    """The R1 reading of the last bar of `d` (a closed bar), plus the parent timeframe's commitment."""
    if d is None or len(d) < 30:
        return None
    t = pd.to_datetime(d["time"], utc=True).reset_index(drop=True)
    R = reading(*(d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")), t, tf_minutes)
    i = len(d) - 1
    c = float(d["close"].iloc[-1])
    st, ln, a = int(R["state"][i]), R["line"][i], R["atr"][i]
    flips = np.where(R["rbar"][: i + 1] != 0)[0]
    out = {
        "bar": t.iloc[i].isoformat(), "tf": f"{tf_minutes // 60}h" if tf_minutes >= 60 else f"{tf_minutes}m",
        "effort": _f(R["effort"][i]), "band": core.band(R["effort"][i]) if np.isfinite(R["effort"][i]) else "n/a",
        "eff_vs_prev": _f(R["dEb"][i]), "side_vs_prev": _f(R["dSide"][i]), "spread": _f(R["rs"][i]),
        "close_pct": _f(100 * R["cpos"][i], 0), "c2c": int(R["c2c"][i]), "intent": int(R["intent"][i]),
        "sb": int(R["sb"][i]), "rb": int(R["rbar"][i]),
        "state": st, "sl": _f(ln, 6), "step": int(R["step"][i]),
        "since": t.iloc[int(flips[-1])].isoformat() if len(flips) else None,
        "sl_atr": _f(abs(c - ln) / a) if st and np.isfinite(ln) and a > 0 else None,
        "wave": {"bars": int(R["wave_bars"][i]) if np.isfinite(R["wave_bars"][i]) else None,     # the wave in progress, so far
                 "effort": _f(R["wave_effort"][i], 1), "travel": _f(R["wave_travel"][i]),
                 "dE": _f(R["wave_dE"][i]), "dR": _f(R["wave_dR"][i]), "cost": _f(R["wave_cost"][i]),
                 "cost_band": cost_band(R["wave_cost"][i])},
        "last_wave": None if not np.isfinite(R["last_side"][i]) else {                              # the last completed wave
                 "side": int(R["last_side"][i]), "bars": int(R["last_bars"][i]), "effort": _f(R["last_effort"][i], 1),
                 "travel": _f(R["last_travel"][i]), "dE": _f(R["last_dE"][i]), "dR": _f(R["last_dR"][i]),
                 "light": light(R["last_dE"][i], R["last_dR"][i])},
        "parent": None,
    }
    if parent is not None and len(parent) >= 30:
        p = parent.reset_index(drop=True)
        pt = pd.to_datetime(p["time"], utc=True)
        closed = (pt + pd.Timedelta(minutes=parent_minutes)) <= (t.iloc[i] + pd.Timedelta(minutes=tf_minutes))
        p = p[closed.to_numpy()]
        if len(p) >= 30:
            pc = commit.commitment(*(p[k].to_numpy(float) for k in ("open", "high", "low", "close")))
            out["parent"] = {"tf": f"{parent_minutes // 60}h", "state": int(pc["state"][-1]), "sl": _f(pc["line"][-1], 6),
                             "step": int(pc["step"][-1]), "bar": pd.to_datetime(p["time"].iloc[-1], utc=True).isoformat()}
    return out
