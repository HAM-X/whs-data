"""WHS Engine 2.3 — range workbench, two-sided (Python twin of the WHS Range Pine indicator).

A trading range is a hypothesis until phase C/D proves it (Roman: "the same top first labelled distribution can
resolve as re-accumulation"). So every range is read from BOTH sides at once, and named only by what it does:

  phase A   by the anchor. SC-type (two clicks: the climax low, then the reaction high) or BC-type (climax high,
            then reaction low). The AR may extend; > 2 × the first height = no range. The ST of the climax
            (a swing back into the climax side's zone) ends phase A — in a re-accumulation that ST is at the TOP.
  sides     U (+1) reads the bottom edge as support (springs, LPS) and the top as its breakout (SOS);
            D (−1) reads the top edge (UT/UTAD, LPSY) and the bottom as its breakout (SOW). D is U mirrored (−price).
  phase B   on each edge: a break that comes back within 6 bars before B matures is a phase-B event — bottom:
            ST / mSOW (ST as SOW: effort ≥ 1.5 or a wide down bar) · top: UA (SC-type) / UT (BC-type) / mSOS.
            Swings back into an edge's zone are STs (bottom always; top only in a BC-type range).
  gate      phase C needs a mature B: as long as A, price having visited both halves of the range since B began.
  phase C   per side, testing the extremes of A AND B: beyond them and back within 3 bars = Spring #3 / #2 /
            Shakeout (UTAD on top); beyond the edge but inside the B extremes = LPS (LPSY). Higher lows after
            it = Test / LPS chain. A test that closes beyond its own extreme fails.
  phase D   per side: a close beyond the resistance cluster (all extremes of A+B up to the last pullback) = SOS
            (SOW), confirmed by a back-up holding the edge (BU; LPSY for SOW) or by running 0.5 H clear; a close back
            inside the edge − 0.2 H first fails it (back to the phase it came from). Later pullbacks = LPS (LPSY).
            No phase C seen yet → the pullback extreme the breakout started from becomes the LPS (LPSY) of phase C.
  winner    the first side whose SOS/SOW is confirmed decides the outcome; the other side's open tests fail.
  phase E   price ≥ 0.5 H beyond the edge after a confirmed breakout; the read ends with the swing that left.
  name      outcome up: Re-accumulation if the higher timeframe was trending up when the range began, else
            Accumulation; outcome down: Redistribution if it was trending down, else Distribution.
            Unresolved: both names, each with its evidence so far.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core, swings

ST_ZONE, ST_EFFORT, BREAK_TOL = 0.35, 0.8, 0.15
BACK_BARS, SPRING3_ATR, SPRING3_EFF = 3, 1.0, 1.1
LPS_ZONE = 0.5
HOLD_TOL, LEAVE_H = 0.2, 0.5
AR_MAX = 2.0
CH_FLAT = 0.25        # a channel that moves less than 0.25 H across the range so far is drawn flat
B_MIN = 1.0
SOW_EFF, BACK_BARS_B = 1.5, 6
ST_NEAR = 0.15        # a phase-B swing is named ST only when it gets back within 0.15 H of the lows tested so far

# shown names per side and role
NAMES = {
    1: {"ST": "ST", "B_light": "ST", "B_heavy": "mSOW", "Spring": "Spring", "Shakeout": "Shakeout", "LPS_C": "LPS",
        "Test": "Test", "Chain": "LPS", "BO": "SOS", "BU": "BU", "D_LPS": "LPS", "Fail": "Spring"},
    -1: {"ST": "ST", "B_light": "UA", "B_heavy": "mSOS", "Spring": "UTAD", "Shakeout": "UTAD", "LPS_C": "LPSY",
         "Test": "Test", "Chain": "LPSY", "BO": "SOW", "BU": "LPSY", "D_LPS": "LPSY", "Fail": "UTAD"},
}
# roles drawn on the support side of their own frame (below bars for side +1, above for side −1)
EDGE_ROLES = {"ST", "B_light", "B_heavy", "Spring", "Shakeout", "LPS_C", "Test", "Chain", "BU", "D_LPS", "Fail"}


@dataclass
class Event:
    name: str
    bar: int            # where it is drawn (the extreme)
    price: float        # real price
    state: str          # candidate | confirmed | failed
    known: int          # the bar on which the engine could know it
    side: int = 0       # +1 bullish side (bottom tests, SOS) · −1 bearish side (top tests, SOW) · 0 phase A
    role: str = ""
    effort: float = np.nan
    note: str = ""


@dataclass
class RangeRead:
    kind: str                       # "SC" (stopping a decline) | "BC" (stopping an advance)
    a1: int
    a2: int
    support: float                  # bottom edge (real price)
    resistance: float               # top edge
    events: list = field(default_factory=list)
    phases: list = field(default_factory=list)   # (letter, start bar, end bar)
    profile: tuple = None
    end: int = 0
    live: np.ndarray = None         # phase known on each bar: 0 none · 1 A … 5 E
    outcome: int = 0                # +1 resolved up · −1 down · 0 open
    htf: int = 0                    # higher-timeframe trend when the range began: +1 up · −1 down · 0 none
    name: str = ""                  # Accumulation | Re-accumulation | Distribution | Redistribution | "" (open)
    bull_name: str = ""
    bear_name: str = ""
    bull: list = field(default_factory=list)      # evidence for the bullish outcome
    bear: list = field(default_factory=list)
    channel: tuple = None           # (slope per bar, base bar, lower-line price at base, height); slope 0 = flat


def _effort_at(eff, k, upto=None):
    """Effort at an event: the larger of its bar and the next one (only bars known by `upto`)."""
    top = len(eff) - 1 if upto is None else upto
    vals = [eff[j] for j in (k, k + 1) if 0 <= j <= top and not np.isnan(eff[j])]
    return max(vals) if vals else np.nan


def htf_trend(h, l, n: int = 5):
    """Structure trend per bar from confirmed pivots (n bars each side): +1 when the last two pivot highs AND lows
    rise, −1 when both fall, else 0. A pivot is known n bars after it (no look-ahead)."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    m = len(h)
    out = np.zeros(m, int)
    ph, pl = [], []
    for i in range(2 * n, m):
        k = i - n
        win_h, win_l = h[k - n:i + 1], l[k - n:i + 1]
        if h[k] == win_h.max() and (h[k - n:k] < h[k]).all():
            ph.append(h[k])
        if l[k] == win_l.min() and (l[k - n:k] > l[k]).all():
            pl.append(l[k])
        if len(ph) >= 2 and len(pl) >= 2:
            up = ph[-1] > ph[-2] and pl[-1] > pl[-2]
            dn = ph[-1] < ph[-2] and pl[-1] < pl[-2]
            out[i] = 1 if up else -1 if dn else 0
    return out


class _Side:
    """One side of the range in its own frame (f = +1 real prices, f = −1 negated): its support edge is tested,
    its resistance edge is where its breakout happens."""

    def __init__(self, f, h, l, c, sb, eff, atr, bottom, top, kind, b_start, a1, seed_low, seed_eff, events,
                 label_zone):
        self.f = f
        self.h, self.l, self.c = (h, l, c) if f == 1 else (-l, -h, -c)
        self.sb = sb * f
        self.eff, self.atr = eff, atr
        self.S = bottom if f == 1 else -top
        self.R = top if f == 1 else -bottom
        self.H = self.R - self.S
        self.k, self.base, self.sloped = 0.0, 0, False   # channel: S(j) = S + k·(j − base)
        self.kind = kind
        self.b_start, self.a1 = b_start, a1
        self.ev = events
        self.label_zone = label_zone
        self.st_low = seed_low
        self.edge_eff = seed_eff
        self.low_b = np.nan
        self.pen = None
        self.pen_gate = False
        self.last_brk = None
        self.labelled = set()
        self.C = None                   # the phase-C event
        self.lps_h = None               # hindsight LPS chosen at the breakout
        self.bo = None                  # breakout event (SOS / SOW)
        self.pre = "B"
        self.d_low = np.nan
        self.cl_hi, self.cl_upto = np.nan, None
        self.last_up_end = b_start
        self.phase = "B"
        self.trans = []                 # (letter, bar) transitions after B began
        self.done = False
        self.end = None

    def add(self, role, bar, price_m, state, known, note="", suffix=""):
        nm = NAMES[self.f][role]
        if role == "B_light" and self.f == -1 and self.kind == "BC":
            nm = "UT"
        e = Event(nm + suffix, bar, price_m * self.f, state, known, self.f, role, np.nan, note)
        self.ev.append(e)
        return e

    def set_channel(self, k, base, s0, height):
        self.k, self.base, self.S, self.H, self.sloped = k, base, s0, height, True
        self.R = s0 + height

    def recent(self, bar, p):
        """A test of this edge already named within the last 12 bars that went as far or further."""
        return any(e.side == self.f and e.role in ("ST", "B_light", "B_heavy") and abs(bar - e.bar) <= 2 * BACK_BARS_B
                   and p >= e.price * self.f for e in self.ev)

    def lv(self, j):
        s = self.S + self.k * (j - self.base)
        return s, s + self.H

    def go(self, letter, bar):
        self.phase = letter
        self.trans.append((letter, bar))

    def step(self, i, sw, gate):
        """sw = the swing finished on this bar, in this frame: (dir, start, start_p, end, end_p) or None."""
        h, l, c, H = self.h, self.l, self.c, self.H
        S, R = self.lv(i)
        flr = lambda s_: s_ if (self.sloped or np.isnan(self.low_b)) else min(s_, self.low_b)
        floor = flr(S)
        if sw is not None and sw[0] == 1 and self.phase == "E":
            self.end, self.done = sw[3], True
            return
        if sw is not None and sw[0] == 1:
            self.last_up_end = sw[3]
        if sw is not None and sw[0] == -1 and sw[3] > self.b_start:
            kk, p = sw[3], sw[4]
            e_k = _effort_at(self.eff, kk, i)
            Sk, Rk = self.lv(kk)
            if self.phase == "B" and gate and self.C is None and kk not in self.labelled and flr(Sk) <= p < Sk:
                self.C = self.add("LPS_C", kk, p, "candidate", i, "phase C: a test beyond the edge that held inside the B extremes")
                self.labelled.add(kk)
                self.go("C", kk)
            elif self.phase == "B" and self.C is None and kk not in self.labelled and Sk - BREAK_TOL * H <= p <= Sk + ST_ZONE * H:
                rel = p - (Sk - self.S) if self.sloped else p                       # measured along the channel
                retest = np.isnan(self.st_low) or rel <= self.st_low + ST_NEAR * H  # it goes back to the lows
                if self.label_zone and retest and not self.recent(kk, p):
                    less = not np.isnan(e_k) and not np.isnan(self.edge_eff) and e_k <= ST_EFFORT * self.edge_eff
                    self.add("ST", kk, p, "confirmed" if less else "candidate", i,
                             "on less effort than the climax" if less else "on more effort than the climax")
                rel = p - (Sk - self.S) if self.sloped else p
                self.st_low = rel if np.isnan(self.st_low) else min(self.st_low, rel)
            elif self.phase == "C" and self.C.state != "failed" and p > self.C.price * self.f and kk > self.C.bar \
                    and (np.isnan(self.d_low) or p > self.d_low):
                lpsish = self.C.role == "LPS_C"
                self.add("Chain" if lpsish else "Test", kk, p, "confirmed", i,
                         "a higher low after the phase-C low" if lpsish else "holds a higher low than the spring")
                self.C.state = "confirmed"
                self.d_low = p
            elif self.phase == "D" and self.bo is not None and kk > self.bo.bar and p >= Rk - HOLD_TOL * H \
                    and (np.isnan(self.d_low) or p > self.d_low):
                first = self.bo.state == "candidate"
                self.add("BU" if first else "D_LPS", kk, p, "confirmed", i,
                         "back-up held the edge: the breakout is confirmed" if first else "a higher low in phase D")
                self.bo.state = "confirmed"
                self.d_low = p
            if self.phase in ("B", "C"):              # the A+B extremes up to this pullback = the resistance cluster
                j0 = self.a1 if self.cl_upto is None else self.cl_upto + 1
                if kk >= j0:
                    m = h[j0:kk + 1].max()
                    self.cl_hi = m if np.isnan(self.cl_hi) else max(self.cl_hi, m)
                    self.cl_upto = kk
        # breaks of the support edge
        if self.phase == "B" and self.C is None and self.pen is None and l[i] < (floor if gate else S):
            self.pen, self.pen_gate = i, gate
        if self.pen is not None and self.C is None:
            pen = self.pen
            low_k = pen + int(np.argmin(l[pen:i + 1]))
            e_k = _effort_at(self.eff, low_k, i)
            loud = not np.isnan(e_k) and e_k >= SOW_EFF
            heavy = loud or bool((self.sb[pen:i + 1] < 0).any())
            back = c[i] >= (floor if self.pen_gate else S)
            if back and not self.pen_gate and self.recent(low_k, l[low_k]):
                self.low_b = l[low_k] if np.isnan(self.low_b) else min(self.low_b, l[low_k])
                self.pen = None                        # the same test seen again: no new label
            elif back and not self.pen_gate:
                self.last_brk = (low_k, l[low_k])
                self.add("B_heavy" if heavy else "B_light", low_k, l[low_k], "confirmed", i,
                         "phase B: a break that came back on effort — not the phase-C test" if heavy else
                         "phase B: a test beyond the edge, back inside")
                self.labelled.add(low_k)
                self.low_b = l[low_k] if np.isnan(self.low_b) else min(self.low_b, l[low_k])
                self.pen = None
            elif back:
                shallow = floor - l[low_k] <= SPRING3_ATR * self.atr[low_k] and (np.isnan(e_k) or e_k <= SPRING3_EFF)
                if loud:
                    self.C = self.add("Shakeout", low_k, l[low_k], "candidate", i,
                                      "heavy effort beyond the A+B extremes, back inside — needs a test that holds")
                else:
                    self.C = self.add("Spring", low_k, l[low_k], "confirmed" if shallow else "candidate", i,
                                      "#3 shallow, ordinary effort" if shallow else "#2 needs a test that holds",
                                      " #3" if shallow else " #2")
                self.labelled.add(low_k)
                self.go("C", low_k)
                self.pen = None
            elif i - pen >= (BACK_BARS if self.pen_gate else BACK_BARS_B):
                if self.pen_gate:
                    self.add("Fail", low_k, l[low_k], "failed", i, "no return inside")
                self.pen = None                        # the other side's breakout takes it from here
        if self.C is not None and self.C.state != "failed" and c[i] < self.C.price * self.f:
            self.C.state = "failed"
            self.C.note = "closed beyond its own extreme"
        # breakout through the far edge
        cluster = R if (self.sloped or np.isnan(self.cl_hi)) else max(R, self.cl_hi)
        if self.bo is None and c[i] > cluster and self.phase in ("B", "C"):
            self.trans_mark = len(self.trans)
            self.pre = self.phase
            if self.phase == "B" and self.C is None and not np.isnan(self.st_low):
                j0 = max(self.last_up_end, self.b_start) + 1
                if j0 <= i:
                    kk = j0 + int(np.argmin(l[j0:i + 1]))
                    Sk, _ = self.lv(kk)
                    off = (Sk - self.S) if self.sloped else 0.0
                    fl = (self.st_low if np.isnan(self.low_b) or self.sloped else min(self.st_low, self.low_b)) + off
                    if fl < l[kk] <= Sk + LPS_ZONE * H:
                        old = next((e for e in self.ev if e.side == self.f and e.bar == kk and e.role == "ST"), None)
                        if old is not None:                 # the ST there was the last test: it is the LPS
                            self.ev.remove(old)
                        self.lps_h = self.add("LPS_C", kk, l[kk], "confirmed", i, "the higher low the breakout rose from")
                        self.C = self.lps_h
                        self.go("C", kk)
            if self.C is not None and self.C.state == "candidate":
                self.C.state = "confirmed"
            self.bo = self.add("BO", i, c[i], "candidate", i, "close beyond the A+B extremes")
            self.d_low = np.nan
            self.go("D", i)
        elif self.bo is not None and self.bo.state == "candidate":
            if c[i] < R - HOLD_TOL * H:
                if self.pre == "B" and (self.lps_h is None or self.lps_h is self.C):
                    # a breakout from phase B that failed was phase-B action (the edge logic names it): it goes
                    self.ev[:] = [e for e in self.ev if e is not self.bo and e is not self.lps_h]
                    if self.lps_h is not None:
                        self.C = None
                        self.lps_h = None
                    self.trans = self.trans[:self.trans_mark]
                    self.phase = "B"
                else:
                    self.bo.state = "failed"
                    self.bo.note = "closed back inside before a back-up held"
                    self.go(self.pre, i)
                self.bo = None
            elif h[i] >= R + LEAVE_H * H:
                self.bo.state = "confirmed"
        if self.phase == "D" and self.bo is not None and self.bo.state == "confirmed" and h[i] >= R + LEAVE_H * H:
            self.go("E", i)

    def lose(self, cut):
        """The other side won (its phase C began at bar `cut`). In hindsight this side's tests were phase-B action:
        its phase-C test is renamed to its phase-B name (a failed UTAD reads as a UA, a failed spring as an ST),
        its follow-up tests and breakout go, and anything after `cut` belonged to the winner's move."""
        keep = []
        for e in self.ev:
            if e.side != self.f:
                keep.append(e)
                continue
            if e.bar > cut or e.role in ("Test", "Chain", "BU", "D_LPS", "BO") or e is self.lps_h:
                continue
            if e is self.C or e.role in ("Spring", "Shakeout", "LPS_C", "Fail"):
                e.name = "UT" if (self.f == -1 and self.kind == "BC") else NAMES[self.f]["B_light"]
                e.role, e.state = "B_light", "confirmed"
                e.note = "a phase-C test that failed when the range resolved the other way — phase-B action"
            keep.append(e)
        self.ev[:] = keep


def read(o, h, l, c, eff, vol, a1: int, a2: int, htf: int = 0, profile_rows: int = 24) -> RangeRead:
    """Bar by bar, using only what was known on each bar (swings as finished on their reversal bar)."""
    o, h, l, c, eff, vol = (np.asarray(x, float) for x in (o, h, l, c, eff, vol))
    n = len(c)
    led = swings.ledger(o, h, l, c, eff)
    sb = swings.sbars(o, h, l, c)
    atr = core.atr(h, l, c)
    b2 = a2                                             # the second click; the walk starts after it
    kind = "SC" if l[a1] <= l[a2] and h[a2] >= h[a1] else "BC"
    K = 1 if kind == "SC" else -1                       # the frame in which the climax is the low
    hK, lK, cK = (h, l, c) if K == 1 else (-l, -h, -c)
    S = lK[a1]
    a2 = a1 + int(np.argmax(hK[a1:b2 + 1]))
    R = hK[a2]
    H = R - S
    H0 = H
    ev = []

    def add_a(name, bar, pK, state, known, note=""):
        e = Event(name, bar, pK * K, state, known, 0, name, np.nan, note)
        ev.append(e)
        return e

    sc = add_a("SC" if K == 1 else "BC", a1, S, "candidate", a1)
    ar = add_a("AR", a2, R, "confirmed", a2)
    by_bar = {ci: (dd, si, sp, ei, ep) for (dd, si, sp, ei, ep, ci) in led.done}
    live = np.zeros(n, int)
    live[b2] = 1
    b_start = None
    sides = {}
    winner = 0
    end = n - 1
    visited = [False, False]                            # since B began: lower half, upper half
    channel = None
    for i in range(b2 + 1, n):
        if b_start is None:
            # ---- phase A (in the climax frame)
            if hK[i] > R:
                if hK[i] - S > AR_MAX * H0:
                    ar.state, ar.note = "failed", "the reaction kept going: no range, the trend turned"
                    sc.state = "failed"
                    live[i] = 1
                    end = i
                    break
                R, a2, H = hK[i], i, hK[i] - S
                ar.bar, ar.price, ar.note = i, R * K, "extended past the second click"
            if cK[i] < S - BREAK_TOL * H:
                sc.state, sc.note = "failed", "closed back beyond the climax: the trend resumed"
                live[i] = 1
                end = i
                break
            live[i] = 1
            sw = by_bar.get(i)
            if sw is not None and sw[0] * K == -1 and sw[1] >= a2 and sw[3] > a2:
                p = sw[4] * K
                if S - BREAK_TOL * H <= p <= S + ST_ZONE * H:
                    e_k, e_c = _effort_at(eff, sw[3], i), _effort_at(eff, a1, i)
                    less = not np.isnan(e_k) and not np.isnan(e_c) and e_k <= ST_EFFORT * e_c
                    st = add_a("ST", sw[3], p, "confirmed" if less else "candidate", i,
                               "on less effort than the climax" if less else "on more effort than the climax")
                    st.side, st.role = K, "ST"
                    if less:
                        sc.state = "confirmed"
                    b_start = sw[3]
                    bottom, top = (S, R) if K == 1 else (-R, -S)
                    seed = {K: (p, e_c), -K: (-R, _effort_at(eff, a2, i))}
                    for f in (1, -1):
                        sides[f] = _Side(f, h, l, c, sb, eff, atr, bottom, top, kind, b_start, a1,
                                         seed[f][0], seed[f][1], ev, label_zone=(f == 1 or kind == "BC"))
                        sides[f].labelled.add(sw[3])
                    live[i] = 2
            continue
        # ---- phase B onward: both sides
        bottom, top = sides[1].S, sides[1].R
        mid = (bottom + top) / 2
        visited[0] = visited[0] or l[i] <= mid
        visited[1] = visited[1] or h[i] >= mid
        gate = visited[0] and visited[1] and (i - b_start) >= B_MIN * (b_start - a1)
        if gate and channel is None:
            channel = _fit_channel(h, l, a1, a2 if K == -1 else a1, b_start, i, sides[1].H)
            if channel[0] != 0.0:
                slope, bA, pA, hc = channel
                sides[1].set_channel(slope, bA, pA, hc)
                sides[-1].set_channel(-slope, bA, -(pA + hc), hc)
        sw = by_bar.get(i)
        for f in (1, -1):
            sd = sides[f]
            if sd.done or (winner and f != winner):
                continue
            swf = None if sw is None else (sw[0] * f, sw[1], sw[2] * f, sw[3], sw[4] * f)
            sd.step(i, swf, gate)
            if not winner and sd.bo is not None and sd.bo.state == "confirmed":
                winner = f
                cut = next((bar for letter, bar in sd.trans if letter == "C"), sd.bo.bar)
                sides[-f].lose(cut)
        live[i] = "ABCDE".index(_merged_phase(sides, winner)) + 1
        if winner and sides[winner].done:
            end = sides[winner].end
            break
    # phases
    phases = [["A", a1, b_start if b_start is not None else end]]
    if b_start is not None:
        phases.append(["B", b_start, None])
        if winner:
            src = sides[winner]
        else:
            src = max(sides.values(), key=lambda s: ("BCDE".index(s.phase), s.trans[-1][1] if s.trans else -1))
        for letter, bar in src.trans:
            if letter == phases[-1][0]:
                continue
            phases[-1][2] = bar
            phases.append([letter, bar, None])
    phases[-1][2] = end
    for e in ev:
        e.effort = _effort_at(eff, e.bar)
    # profile: climax → the bar before the winning breakout (or the end of the read)
    bo_bar = sides[winner].bo.bar if winner else None
    p_end = max(a1, (bo_bar if bo_bar is not None else end + 1) - 1)
    lo_p, hi_p = l[a1:p_end + 1].min(), h[a1:p_end + 1].max()
    edges = np.linspace(lo_p, hi_p, profile_rows + 1)
    prof = np.zeros(profile_rows)
    for j in range(a1, p_end + 1):
        if np.isnan(vol[j]) or h[j] <= l[j]:
            continue
        ov = np.clip(np.minimum(edges[1:], h[j]) - np.maximum(edges[:-1], l[j]), 0, None)
        prof += vol[j] * ov / (h[j] - l[j])
    bottom, top = (S, R) if K == 1 else (-R, -S)
    if channel is not None and channel[0] != 0.0:
        rr_channel = channel
    else:
        rr_channel = (0.0, a1, bottom, top - bottom)
    rr = RangeRead(kind, a1, a2, bottom, top, ev, [tuple(p) for p in phases], (edges, prof, int(np.argmax(prof))),
                   end, live, winner, htf)
    rr.channel = rr_channel
    rr.bull_name = "Re-accumulation" if htf == 1 else "Accumulation"
    rr.bear_name = "Redistribution" if htf == -1 else "Distribution"
    rr.name = rr.bull_name if winner == 1 else rr.bear_name if winner == -1 else ""
    rr.bull, rr.bear = _evidence(ev, htf, sides)
    return rr


def _fit_channel(h, l, a1, a_low_bar, b_start, i, H):
    """Sloping range (Roman: most real ranges slope): the lower line through the phase-A low and the lowest low of
    phase B, the upper line parallel through the highest high of A+B. Flat when it moves < CH_FLAT·H across the
    range so far."""
    pA = l[a_low_bar]
    seg = l[b_start:i + 1]
    bB = b_start + int(np.argmin(seg))
    if bB <= a_low_bar:
        return (0.0, a1, pA, H)
    slope = (l[bB] - pA) / (bB - a_low_bar)
    if abs(slope * (i - a1)) < CH_FLAT * H:
        return (0.0, a1, pA, H)
    js = np.arange(a1, i + 1)
    lower = pA + slope * (js - a_low_bar)
    if (l[a1:i + 1] < lower - 1e-12).any():           # the line must hold the lows it was drawn through
        return (0.0, a1, pA, H)
    hc = float((h[a1:i + 1] - lower).max())
    return (slope, a_low_bar, pA, hc)


def _merged_phase(sides, winner):
    if winner:
        return sides[winner].phase
    return max((s.phase for s in sides.values()), key=lambda p: "BCDE".index(p))


def _evidence(ev, htf, sides):
    """Plain evidence for each outcome, from what the range has shown so far."""
    out = {1: [], -1: []}
    if htf:
        out[htf].append("higher timeframe trending " + ("up" if htf == 1 else "down"))
    for f in (1, -1):
        mine = [e for e in ev if e.side == f]
        word = "lows" if f == 1 else "highs"
        if any(e.role == "ST" and e.state == "confirmed" for e in mine):
            out[f].append(f"tests of the {word} on less effort")
        if any(e.role in ("Spring", "Shakeout", "LPS_C") and e.state == "confirmed" for e in mine):
            out[f].append("phase C " + ("spring / LPS" if f == 1 else "UTAD / LPSY") + " held")
        tests = [e.price * f for e in mine if e.role in ("ST", "B_light", "B_heavy", "LPS_C", "Chain", "Test",
                                                          "Spring", "Shakeout")]
        if len(tests) >= 2 and tests[-1] > tests[0]:
            out[f].append("higher lows" if f == 1 else "lower highs")
        if any(e.role == "B_heavy" and e.side == -f for e in ev):
            out[f].append("UA as a minor SOS in B (demand)" if f == 1 else "ST as SOW in B (supply)")
        if sides and f in sides and sides[f].bo is not None:
            out[f].append(("SOS" if f == 1 else "SOW") + (" confirmed" if sides[f].bo.state == "confirmed" else " (candidate)"))
    return out[1], out[-1]
