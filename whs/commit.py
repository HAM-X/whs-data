"""The commitment line — Roman's SBar rules and nothing else (Hamza 5 Nov 2026). Twin of WHS Engine R1.2.

up:   every up SBar (or combo) steps the line to its true-range low (combo: the pair's; a huge bar with
      TR ≥ 2.5 ATR[1]: half of the bar). The line never steps down.
RBar: the first close below the line = commitment to the other side → down: the line goes to the up swing's
      high until the first down SBar moves it. Mirror for down.
"""
from __future__ import annotations

import numpy as np

from . import core, swings


def commitment(o, h, l, c, sb=None) -> dict:
    """Arrays per bar: state (+1 / −1 / 0), line, rbar (±1 on the flip bar), moved (the line stepped), half
    (the step used half a huge bar), step (SBars since the last flip)."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    n = len(c)
    sb = swings.sbars(o, h, l, c) if sb is None else sb
    tr = core.true_range(h, l, c)
    atr = core.atr(h, l, c)
    prev = np.r_[np.nan, c[:-1]]
    trlo = np.where(np.isnan(prev), l, np.minimum(l, prev))
    trhi = np.where(np.isnan(prev), h, np.maximum(h, prev))
    state, line, rbar, moved, half, step = (np.zeros(n, int), np.full(n, np.nan), np.zeros(n, int), np.zeros(n, bool),
                                            np.zeros(n, bool), np.zeros(n, int))
    st, ln, ext, k = 0, np.nan, np.nan, 0
    for i in range(n):
        if st != 0 and ((st == 1 and c[i] < ln) or (st == -1 and c[i] > ln)):
            rbar[i] = -st
            st, ln, k = -st, ext, 0
            ext = h[i] if st == 1 else l[i]
        if st != 0:
            ext = max(ext, h[i]) if st == 1 else min(ext, l[i])
        s = int(np.sign(sb[i]))
        if s != 0 and (st == 0 or s == st) and i >= 1:
            lv = (min(trlo[i], trlo[i - 1]) if abs(sb[i]) == 2 else trlo[i]) if s == 1 else \
                 (max(trhi[i], trhi[i - 1]) if abs(sb[i]) == 2 else trhi[i])
            if not np.isnan(atr[i - 1]) and tr[i] >= 2.5 * atr[i - 1]:
                lv, half[i] = (h[i] + l[i]) / 2, True
            nw = lv if st == 0 else (max(ln, lv) if s == 1 else min(ln, lv))
            if st == 0:
                st, ext = s, (h[i] if s == 1 else l[i])
            moved[i] = not (nw == ln)
            ln, k = nw, k + 1
        state[i], line[i], step[i] = st, ln, k
    return dict(state=state, line=line, rbar=rbar, moved=moved, half=half, step=step, sb=sb)
