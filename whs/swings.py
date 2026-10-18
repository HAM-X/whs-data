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
  Traffic light  per bar, with effort e, spread rs = TR/ATR[1], close position cp, direction by close vs C[1]:
                 strong close = up & cp ≥ 0.67, or down & cp ≤ 0.33
                 RED    stop    e ≥ 1.5 and (rs ≤ 0.9 or not a strong close)      effort without result
                 GREEN  sync    e ≥ 1.1 and rs ≥ 1.1 and strong close             effort and result together
                 ORANGE present e ≥ 1.1, neither red nor green                     effort with some result
                 BLUE   ease    e ≤ 0.9 and rs ≥ 0.9 and strong close             less effort, more result
  TL sequence    RED, then a GREEN within 10 bars, then a BLUE in the same direction within 10 bars of it
                 (no opposite GREEN in between) = a swing is allowed to start (Roman's JCP / SQ drills).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core

SBAR_K, COMBO_EACH, COMBO_SUM, TRAVEL_K = 1.25, 0.8, 1.25, 0.75
TL_NONE, TL_RED, TL_ORANGE, TL_GREEN, TL_BLUE = 0, 1, 2, 3, 4
SEQ_WINDOW = 10


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


def traffic_light(h, l, c, eff):
    h, l, c, e = (np.asarray(x, float) for x in (h, l, c, eff))
    tr = core.true_range(h, l, c)
    atr = core.atr(h, l, c)
    rs = tr / np.r_[np.nan, atr[:-1]]
    cp = core.close_pos(h, l, c)
    pc = np.r_[np.nan, c[:-1]]
    tl = np.zeros(len(c), int)
    tdir = np.zeros(len(c), int)
    for i in range(len(c)):
        if np.isnan(e[i]) or np.isnan(rs[i]) or np.isnan(pc[i]):
            continue
        d = 1 if c[i] > pc[i] else -1 if c[i] < pc[i] else 0
        strong = (d == 1 and cp[i] >= 0.67) or (d == -1 and cp[i] <= 0.33)
        tdir[i] = d
        if e[i] >= 1.5 and (rs[i] <= 0.9 or not strong):
            tl[i] = TL_RED
        elif e[i] >= 1.1 and rs[i] >= 1.1 and strong:
            tl[i] = TL_GREEN
        elif e[i] >= 1.1:
            tl[i] = TL_ORANGE
        elif e[i] <= 0.9 and rs[i] >= 0.9 and strong:
            tl[i] = TL_BLUE
    return tl, tdir


def tl_sequences(tl, tdir):
    """+1 / −1 on the BLUE bar that completes RED → GREEN → BLUE in that direction."""
    n = len(tl)
    out = np.zeros(n, int)
    red_i = -10**9
    g_i, g_dir = -10**9, 0
    for i in range(n):
        if tl[i] == TL_RED:
            red_i = i
            g_i, g_dir = -10**9, 0
        elif tl[i] == TL_GREEN and tdir[i] != 0:
            if i - red_i <= SEQ_WINDOW and (g_dir == 0 or g_dir == tdir[i]):
                if g_dir == 0:
                    g_i = i
                g_dir = tdir[i]
            elif g_dir != 0 and tdir[i] != g_dir:      # opposite green breaks the sequence
                g_i, g_dir, red_i = -10**9, 0, -10**9
        elif tl[i] == TL_BLUE and g_dir != 0 and tdir[i] == g_dir and i - g_i <= SEQ_WINDOW:
            out[i] = g_dir
            red_i, g_i, g_dir = -10**9, -10**9, 0
    return out


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
    res = SwingResult(rbar=rbar, sbar=sb)
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
