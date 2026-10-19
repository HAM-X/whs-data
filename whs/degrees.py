"""Swing degrees (checklist 2.2b): minor swings (the ledger) roll up into MAJOR swings.

A major swing is a series of minor swings. It has three states:
  +1 / −1  trending: it lasts while closes hold the PROTECTED level = the end of the latest finished
           minor counter-swing (at first: the major's start). A close beyond it ends the major at its
           extreme and starts the opposite major from there.
  0        stopped: a CLIMAX (heavy effort, wide bar, closing well off its extreme, at a new major extreme)
           followed by an automatic reaction (a close ≥ AR_ATR × ATR away within AR_N bars) ends the major
           at the climax extreme. Price then works sideways between the climax extreme and the reaction
           extreme — the box a range starts from. A new major starts when price closes out of that box:
           beyond the reaction side by BREAK_H × height (from the box's opposite extreme), or beyond the
           climax side by LEAVE_H × height (the old direction resumes from the box's far extreme).
           Lower lows that stay inside that tolerance are tests of the climax, not a new leg.
           A climax AGAINST the major (a selling climax while the major is up, ≥ COUNTER_ATR below its high)
           means the decline was a major leg of its own: the up major ends at its high, the down major runs
           to the climax, and price goes sideways.

Live walk: minor swings are used only as known on their confirming RBar (led.done); nothing looks ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core

CLX_E = 2.5        # climax: effort ≥ 2.5 ×
CLX_TR = 2.0       # ... TR ≥ 2 × ATR[1]
CLX_OFF = 0.4      # ... closing ≥ 40 % of its range off the extreme
CLX_NEAR = 0.25    # ... with its extreme within 0.25 ATR of the major's extreme (or beyond it)
AR_ATR, AR_N = 1.5, 12   # automatic reaction: a close ≥ 1.5 ATR off the climax extreme within 12 bars
BREAK_H = 0.15     # stopped → new major: close beyond the reaction side by 0.15 × box height
LEAVE_H = 0.5      # ...or beyond the climax side by 0.5 × box height
MIN_BOX = 12       # no breakout of the box within 12 bars of the climax: the reaction side just widens it
AR_END = 1.0       # the reaction ends at its first pullback of 1 ATR (close)
COUNTER_ATR = 3.0  # a climax against the major stops it only after a counter-move of ≥ 3 ATR
COUNTER_MINORS = 2 # ...that already holds ≥ 2 finished minor swings (a structure, not a dip)


@dataclass
class Major:
    direction: int          # +1 / −1, or 0 for a stopped (sideways) stretch
    start_i: int
    start_p: float
    end_i: int
    end_p: float
    confirm_i: int          # bar on which it was known to have ended
    end_kind: str           # "flip" (protected level broke) · "climax" · "break" / "leave" (box exits) · "live"
    minors: int = 0         # finished minor swings inside it
    lo: float = np.nan      # box for a stopped stretch
    hi: float = np.nan
    climax_i: int = -1

    @property
    def extent_pct(self) -> float:
        return (self.end_p / self.start_p - 1.0) * 100.0 if self.direction else 0.0


@dataclass
class MajorResult:
    majors: list = field(default_factory=list)      # finished stretches, oldest first, plus the live one last
    state: np.ndarray = None                        # +1 / −1 / 0 per bar, as known on that bar
    protected: np.ndarray = None                    # protected level per bar (nan when stopped)
    events: list = field(default_factory=list)      # (bar, text) — for alerts


def _is_climax(i, D, h, l, c, eff, atr, x_p):
    if i < 1 or np.isnan(eff[i]) or np.isnan(atr[i - 1]) or atr[i - 1] <= 0:
        return False
    rng = h[i] - l[i]
    tr = max(h[i], c[i - 1]) - min(l[i], c[i - 1])
    if eff[i] < CLX_E or tr < CLX_TR * atr[i - 1] or rng <= 0:
        return False
    if D == -1:
        return (c[i] - l[i]) >= CLX_OFF * rng and l[i] <= x_p + CLX_NEAR * atr[i - 1]
    return (h[i] - c[i]) >= CLX_OFF * rng and h[i] >= x_p - CLX_NEAR * atr[i - 1]


def majors(o, h, l, c, eff, led) -> MajorResult:
    o, h, l, c, eff = (np.asarray(x, float) for x in (o, h, l, c, eff))
    n = len(c)
    atr = core.atr(h, l, c)
    by = {}
    for d in led.done:
        by.setdefault(d[5], []).append(d)
    res = MajorResult(state=np.zeros(n, int), protected=np.full(n, np.nan))
    if not by:
        return res
    first = min(by)
    d0 = by[first][-1]
    D = -d0[0]                                # the major runs in the direction of the minor swing now in progress
    s_i, s_p = d0[3], d0[4]
    x_i = s_i
    P = s_p
    minors = 0
    cx, cd = -1, 0                            # pending climax bar and the direction of the leg it stops
    y_i = s_i
    box = None                                # stopped: dict(lo, hi, cx, from_dir, s_i, low_i, high_i)

    def ext(k, d):
        return h[k] if d == 1 else l[k]

    def start_major(d, i0, p0, i_now):
        nonlocal D, s_i, s_p, x_i, y_i, P, minors, cx, box
        D, s_i, s_p, P, minors, cx, box = d, i0, p0, p0, 0, -1, None
        seg = range(i0, i_now + 1)
        x_i = max(seg, key=lambda k: h[k]) if d == 1 else min(seg, key=lambda k: l[k])
        y_i = x_i

    for i in range(first + 1, n):
        if D != 0:
            if (D == 1 and h[i] > h[x_i]) or (D == -1 and l[i] < l[x_i]):
                x_i = i
            for sw in by.get(i, []):
                if sw[0] == -D:
                    P = sw[4]
                    minors += 1
                elif sw[0] == D:
                    minors += 1
            # the deepest counter-move since the major's extreme
            if x_i == i or y_i < x_i:
                y_i = i
            elif (D == 1 and l[i] < l[y_i]) or (D == -1 and h[i] > h[y_i]):
                y_i = i
            # a climax stops a leg: with the major at its extreme, or against it at the end of a deep counter-move
            if _is_climax(i, D, h, l, c, eff, atr, ext(x_i, D)):
                cx, cd = i, D
            elif (_is_climax(i, -D, h, l, c, eff, atr, ext(y_i, -D))
                  and abs(ext(x_i, D) - ext(y_i, -D)) >= COUNTER_ATR * atr[i - 1]
                  and sum(1 for sw in led.done if x_i < sw[3] and sw[5] <= i) >= COUNTER_MINORS):
                cx, cd = i, -D
            if cx >= 0:
                ce = ext(cx, cd)
                beyond = (cd == -1 and l[i] < ce - CLX_NEAR * atr[cx - 1]) or (cd == 1 and h[i] > ce + CLX_NEAR * atr[cx - 1])
                if beyond or i - cx > AR_N:
                    cx = -1                      # a fresh extreme beyond the climax, or no reaction in time
                elif (cd == -1 and c[i] >= ce + AR_ATR * atr[cx - 1]) or (cd == 1 and c[i] <= ce - AR_ATR * atr[cx - 1]):
                    if cd == D:                  # the major itself is stopped at its extreme
                        e_i = x_i
                        res.majors.append(Major(D, s_i, s_p, e_i, ext(e_i, D), i, "climax", minors, climax_i=cx))
                    else:                        # the counter-move was a major leg of its own, stopped by the climax
                        e_i = y_i
                        res.majors.append(Major(D, s_i, s_p, x_i, ext(x_i, D), i, "flip", minors))
                        res.majors.append(Major(cd, x_i, ext(x_i, D), e_i, ext(e_i, cd), i, "climax", 0, climax_i=cx))
                    res.events.append((i, f"major {'down' if cd == -1 else 'up'} swing stopped by a climax — sideways"))
                    if cd == -1:
                        lo, hi = l[e_i], max(h[k] for k in range(cx, i + 1))
                    else:
                        lo, hi = min(l[k] for k in range(cx, i + 1)), h[e_i]
                    box = dict(lo=lo, hi=hi, cx=cx, from_dir=cd, s_i=e_i, ar_open=True)
                    D, cx = 0, -1
                    res.state[i] = 0
                    continue
            if (D == 1 and c[i] < P) or (D == -1 and c[i] > P):
                e_p = ext(x_i, D)
                res.majors.append(Major(D, s_i, s_p, x_i, e_p, i, "flip", minors))
                res.events.append((i, f"major {'up' if D == 1 else 'down'} swing ended — new major {'down' if D == 1 else 'up'}"))
                start_major(-D, x_i, e_p, i)
            res.state[i] = D
            res.protected[i] = P
        else:
            b = box
            if i - b["cx"] <= MIN_BOX:       # too early for a breakout: the reaction side just widens the box
                b["ar_open"] = True
            if b["ar_open"]:                 # the automatic reaction extends until its first 1-ATR pullback
                if b["from_dir"] == -1:
                    b["hi"] = max(b["hi"], h[i])
                    b["ar_open"] = c[i] > b["hi"] - AR_END * atr[i - 1]
                else:
                    b["lo"] = min(b["lo"], l[i])
                    b["ar_open"] = c[i] < b["lo"] + AR_END * atr[i - 1]
            Hh = b["hi"] - b["lo"]
            fd = b["from_dir"]
            if fd == -1:
                react_break = c[i] > b["hi"] + BREAK_H * Hh
                leave = c[i] < b["lo"] - LEAVE_H * Hh
            else:
                react_break = c[i] < b["lo"] - BREAK_H * Hh
                leave = c[i] > b["hi"] + LEAVE_H * Hh
            if react_break or leave:
                seg = range(b["s_i"], i + 1)
                lo_i = min(seg, key=lambda k: l[k])
                hi_i = max(seg, key=lambda k: h[k])
                kind = "break" if react_break else "leave"
                nd = -fd if react_break else fd
                st_i = lo_i if nd == 1 else hi_i
                inside = range(b["s_i"], i)
                res.majors.append(Major(0, b["s_i"], ext(b["s_i"], fd), i, c[i], i, kind,
                                        lo=min(l[k] for k in inside), hi=max(h[k] for k in inside), climax_i=b["cx"]))
                res.events.append((i, f"price left the sideways box — new major {'up' if nd == 1 else 'down'} swing"))
                start_major(nd, st_i, ext(st_i, -nd), i)
                res.state[i] = D
                res.protected[i] = P
                continue
            res.state[i] = 0
    # the one in progress
    if D != 0:
        res.majors.append(Major(D, s_i, s_p, x_i, ext(x_i, D), -1, "live", minors))
    else:
        b = box
        seg = range(b["s_i"], n)
        res.majors.append(Major(0, b["s_i"], ext(b["s_i"], b["from_dir"]), n - 1, c[-1], -1, "live",
                                lo=min(l[k] for k in seg), hi=max(h[k] for k in seg), climax_i=b["cx"]))
    for m in res.majors:
        if m.direction:
            m.minors = sum(1 for sw in led.swings if sw.start_i >= m.start_i and sw.end_i <= m.end_i)
    return res
