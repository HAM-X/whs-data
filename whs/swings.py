"""WHS Engine 2.2 — significant bars, reversal bars, swing ledger, traffic light (Python twin of Pine).

Definitions (Stack v2 Core contract + WTC Part 2):
  SBar (up)      TR ≥ 1.25 × ATR[1], close position ≥ 0.5, close > max(high[1], high[2])   (mirror down)
  Combo SBar     two bars in a row, neither an SBar, each TR ≥ 0.8 × its ATR[1], each closing beyond the
                 previous bar's extreme, together (true range of the pair) ≥ 1.25 × ATR before the pair
  Swing / RBar   an up swing ends at its highest high; the reversal is confirmed by the first bar that
                 closes below the reference low AND has travelled ≥ 0.75 × ATR[1] from the high.
                 Reference low = true-range low of the last up SBar (or combo) in the swing; if the swing
                 has none yet, the low of the extreme bar. That confirming bar is the RBar. The finished
                 swing is drawn back to its extreme (retroactive, flagged). Mirror for down swings.
  Ledger tag     ↑n  extent %  bars  avg effort  R(result vs last same-direction swing)  D/S(effort vs it)
  Traffic light  (Roman, P-S5 JCP and P-S6 SQ) is read SWING by swing, from the ledger deltas against the
                 previous swing in the same direction: dE = avg effort ratio, dR = |extent| ratio.
                 RED    stop     dE ≥ 1.1 and dR ≤ 0.9   effort up, result down (stopping action)
                 GREEN  sync     dE ≥ 1.1 and dR ≥ 1.1   effort and result expand together
                 ease            dE ≤ 0.9 and dR ≥ 1.1   less effort, more result
                    ORANGE present = an ease swing in the OLD direction after a RED (the next attempt still progresses)
                    BLUE   ease    = any other ease swing (continuation with ease)
  TL sequence    RED (direction d) → GREEN (−d) → BLUE (−d) within 6 swings = the reversal to −d is confirmed.
                 ORANGE is optional. A GREEN back in direction d cancels. Red alone is never a reversal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core

SBAR_K, COMBO_EACH, COMBO_SUM, TRAVEL_K = 1.25, 0.8, 1.25, 0.75
TL_NONE, TL_RED, TL_ORANGE, TL_GREEN, TL_BLUE = 0, 1, 2, 3, 4
SEQ_SWINGS = 6
E_UP, E_DN, R_UP, R_DN = 1.1, 0.9, 1.1, 0.9


def ratio_band(x: float) -> str:
    if x is None or np.isnan(x):
        return "·"
    return "≫" if x >= 1.5 else ">" if x >= 1.1 else "≪" if x <= 0.67 else "<" if x <= 0.9 else "≈"


def sbars(o, h, l, c):
    """+1 up SBar, −1 down SBar, +2 / −2 combo SBar (marked on the 2nd bar), 0 none."""
    h, l, c = (np.asarray(x, float) for x in (h, l, c))
    tr = core.true_range(h, l, c)
    atr = core.atr(h, l, c)
    atr1 = np.r_[np.nan, atr[:-1]]
    cp = core.close_pos(h, l, c)
    n = len(c)
    out = np.zeros(n, int)
    for i in range(2, n):
        if np.isnan(atr1[i]):
            continue
        if tr[i] >= SBAR_K * atr1[i]:
            if cp[i] >= 0.5 and c[i] > max(h[i - 1], h[i - 2]):
                out[i] = 1
                continue
            if cp[i] <= 0.5 and c[i] < min(l[i - 1], l[i - 2]):
                out[i] = -1
                continue
    for i in range(3, n):
        if out[i] != 0 or out[i - 1] != 0 or np.isnan(atr1[i - 1]):
            continue
        pair_tr = max(h[i], h[i - 1], c[i - 2]) - min(l[i], l[i - 1], c[i - 2])
        each = tr[i] >= COMBO_EACH * atr1[i] and tr[i - 1] >= COMBO_EACH * atr1[i - 1]
        if each and pair_tr >= COMBO_SUM * atr1[i - 1]:
            if c[i - 1] > h[i - 2] and c[i] > h[i - 1]:
                out[i] = 2
            elif c[i - 1] < l[i - 2] and c[i] < l[i - 1]:
                out[i] = -2
    return out


BAR_WINDOW = 10


def bar_traffic_light(h, l, c, eff):
    """Per-bar traffic light, Roman's deltas applied bar by bar.

    Each bar (direction = close vs previous close) is compared with the previous bar in the SAME
    direction: dE = effort ratio, dR = result ratio (result = TR ÷ ATR[1]).
      RED    stop     dE ≥ 1.1 and dR ≤ 0.9
      GREEN  sync     dE ≥ 1.1 and dR ≥ 1.1
      ease            dE ≤ 0.9 and dR ≥ 1.1 → ORANGE in the old direction while a RED is live, else BLUE
    Sequence RED (d) → GREEN (−d) within 10 bars → BLUE (−d) within 10 bars = ±(−d) on that BLUE bar.
    Returns (tl, seq) arrays."""
    h, l, c, e = (np.asarray(x, float) for x in (h, l, c, eff))
    rs = core.true_range(h, l, c) / np.r_[np.nan, core.atr(h, l, c)[:-1]]
    n = len(c)
    tl, seq = np.zeros(n, int), np.zeros(n, int)
    prev = {1: None, -1: None}
    red_dir, red_i, green_i, stage = 0, -10**9, -10**9, 0
    for i in range(1, n):
        if stage == 1 and i - red_i > BAR_WINDOW:
            stage, red_dir = 0, 0
        if stage == 2 and i - green_i > BAR_WINDOW:
            stage, red_dir = 0, 0
        if np.isnan(e[i]) or np.isnan(rs[i]) or c[i] == c[i - 1]:
            continue
        d = 1 if c[i] > c[i - 1] else -1
        p = prev[d]
        prev[d] = (e[i], rs[i])
        if p is None or p[0] <= 0 or p[1] <= 0:
            continue
        dE, dR = e[i] / p[0], rs[i] / p[1]
        if dE >= E_UP and dR <= R_DN:
            tl[i] = TL_RED
            red_dir, red_i, stage = d, i, 1
        elif dE >= E_UP and dR >= R_UP:
            tl[i] = TL_GREEN
            if stage == 1 and d == -red_dir:
                stage, green_i = 2, i
            elif stage >= 1 and d == red_dir:
                stage, red_dir = 0, 0
        elif dE <= E_DN and dR >= R_UP:
            if stage == 1 and d == red_dir:
                tl[i] = TL_ORANGE
            else:
                tl[i] = TL_BLUE
                if stage == 2 and d == -red_dir:
                    seq[i] = d
                    stage, red_dir = 0, 0
    return tl, seq


@dataclass
class Swing:
    n: int
    direction: int
    start_i: int
    start_p: float
    end_i: int          # extreme bar
    end_p: float
    confirm_i: int      # RBar that confirmed it
    extent_pct: float
    bars: int
    avg_effort: float
    r_vs_prev: float = np.nan
    e_vs_prev: float = np.nan

    tl: int = 0          # TL_NONE / RED / ORANGE / GREEN / BLUE (swing level)
    seq: int = 0         # ±1 when this swing completes RED → GREEN → BLUE

    def tag(self) -> str:
        arrow = "↑" if self.direction == 1 else "↓"
        de = "D" if self.direction == 1 else "S"
        return (f"{arrow}{self.n} {self.extent_pct:+.2f}% {self.bars}b {self.avg_effort:.1f}× "
                f"R{ratio_band(self.r_vs_prev)} {de}{ratio_band(self.e_vs_prev)}")


@dataclass
class SwingResult:
    swings: list = field(default_factory=list)
    rbar: np.ndarray = None       # +1 RBar ending a down swing (bullish), −1 ending an up swing
    sbar: np.ndarray = None
    tl: np.ndarray = None         # swing traffic light, written on the RBar that confirmed the swing
    seq: np.ndarray = None        # ±1 on the RBar of the BLUE swing that completes a sequence


def classify(sw: "Swing", st: dict) -> None:
    """Swing traffic light and the RED → GREEN → BLUE sequence (state kept in `st`)."""
    dE, dR = sw.e_vs_prev, sw.r_vs_prev
    if st["stage"] and sw.n - st["k"] > SEQ_SWINGS:
        st.update(dir=0, stage=0, k=-10**9)
    if np.isnan(dE) or np.isnan(dR):
        sw.tl = TL_NONE
        return
    if dE >= E_UP and dR <= R_DN:
        sw.tl = TL_RED
        st.update(dir=sw.direction, stage=1, k=sw.n)
    elif dE >= E_UP and dR >= R_UP:
        sw.tl = TL_GREEN
        if st["stage"] >= 1 and sw.direction == -st["dir"]:
            st.update(stage=2)
        elif st["stage"] >= 1 and sw.direction == st["dir"]:
            st.update(dir=0, stage=0, k=-10**9)          # sync back in the old direction cancels
    elif dE <= E_DN and dR >= R_UP:
        if st["stage"] == 1 and sw.direction == st["dir"]:
            sw.tl = TL_ORANGE
        else:
            sw.tl = TL_BLUE
            if st["stage"] == 2 and sw.direction == -st["dir"]:
                sw.seq = sw.direction
                st.update(dir=0, stage=0, k=-10**9)
    else:
        sw.tl = TL_NONE


def ledger(o, h, l, c, eff) -> SwingResult:
    h, l, c, e = (np.asarray(x, float) for x in (h, l, c, eff))
    n = len(c)
    sb = sbars(o, h, l, c)
    atr = core.atr(h, l, c)
    atr1 = np.r_[np.nan, atr[:-1]]
    pc = np.r_[np.nan, c[:-1]]
    trlo = np.where(np.isnan(pc), l, np.minimum(l, pc))
    trhi = np.where(np.isnan(pc), h, np.maximum(h, pc))
    rbar = np.zeros(n, int)
    res = SwingResult(rbar=rbar, sbar=sb, tl=np.zeros(n, int), seq=np.zeros(n, int))
    st = {"dir": 0, "stage": 0, "k": -10**9}      # sequence state: red direction, stage (1 red, 2 green), swing no.
    start = next((i for i in range(n) if not np.isnan(atr1[i])), None)
    if start is None or start + 1 >= n:
        return res
    d = 1 if c[start + 1] >= c[start] else -1
    s_i, s_p = start, (l[start] if d == 1 else h[start])
    x_i = start                                   # extreme bar of the current swing
    ref = trlo[start] if d == 1 else trhi[start]  # reversal reference
    has_sb = False
    last = {1: None, -1: None}
    count = 0
    for i in range(start + 1, n):
        # 1) does this bar extend the current swing?
        if (d == 1 and h[i] > h[x_i]) or (d == -1 and l[i] < l[x_i]):
            x_i = i
            if not has_sb:
                ref = trlo[i] if d == 1 else trhi[i]
        # 2) a significant bar in the swing direction moves the reference
        if (d == 1 and sb[i] > 0) or (d == -1 and sb[i] < 0):
            ref = trlo[i] if d == 1 else trhi[i]
            if sb[i] in (2, -2):                    # combo: reference = the pair's true range
                ref = min(trlo[i], trlo[i - 1]) if d == 1 else max(trhi[i], trhi[i - 1])
            has_sb = True
        # 3) reversal bar?
        if i > x_i and not np.isnan(atr1[i]):
            ext = h[x_i] if d == 1 else l[x_i]
            travel = (ext - l[i]) if d == 1 else (h[i] - ext)
            broke = c[i] < ref if d == 1 else c[i] > ref
            if broke and travel >= TRAVEL_K * atr1[i]:
                count += 1
                seg = e[s_i + 1:x_i + 1]
                seg = seg[~np.isnan(seg)]
                sw = Swing(n=count, direction=d, start_i=s_i, start_p=s_p, end_i=x_i, end_p=ext, confirm_i=i,
                           extent_pct=(ext / s_p - 1) * 100, bars=x_i - s_i,
                           avg_effort=float(seg.mean()) if len(seg) else np.nan)
                prev = last[d]
                if prev is not None:
                    sw.r_vs_prev = abs(sw.extent_pct) / abs(prev.extent_pct) if prev.extent_pct else np.nan
                    sw.e_vs_prev = sw.avg_effort / prev.avg_effort if prev.avg_effort else np.nan
                last[d] = sw
                classify(sw, st)
                res.tl[i], res.seq[i] = sw.tl, sw.seq
                res.swings.append(sw)
                rbar[i] = -d
                # new swing in the opposite direction starts at the old extreme
                d = -d
                s_i, s_p = x_i, ext
                seg_i = range(x_i + 1, i + 1)
                x_i = (min(seg_i, key=lambda k: l[k]) if d == -1 else max(seg_i, key=lambda k: h[k]))
                has_sb = False
                ref = trlo[x_i] if d == 1 else trhi[x_i]
                for k in seg_i:                      # SBars already printed in the new direction
                    if (d == 1 and sb[k] > 0) or (d == -1 and sb[k] < 0):
                        ref = trlo[k] if d == 1 else trhi[k]
                        if sb[k] in (2, -2):
                            ref = min(trlo[k], trlo[k - 1]) if d == 1 else max(trhi[k], trhi[k - 1])
                        has_sb = True
    return res
