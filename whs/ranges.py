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
AR_RET_END = 0.618    # auto: a pullback retracing ≥ 61.8 % of the AR so far ends the AR (the next event has begun)…
AR_RET_ATR = 3.0      # …once the AR is ≥ 3 ATR
SO_ATR, SO_SPEED, SO_EFF = 5.0, 0.5, 2.0   # shakeout after a BC: ≥ 5 ATR, ≥ 0.5 ATR per bar, climactic effort ≥ 2×
CH_FLAT = 0.25        # a channel that moves less than 0.25 H across the range so far is drawn flat
CH_MAX = 1.0          # ... and one that moves more than 1 H is a trend, not a range: flat too
B_MIN = 1.0
SOW_EFF, BACK_BARS_B = 1.5, 6
LEFT_SWINGS = 2       # a range is LEFT only after 2 pullbacks hold outside it; until then it may still be going on
REENTER = 0.25        # a close 0.25 H back inside after a confirmed breakout = the breakout failed, the range goes on
ST_NEAR = 0.15        # a phase-B swing is named ST only when it gets back within 0.15 H of the lows tested so far

# shown names per side and role
NAMES = {
    1: {"ST": "ST", "B_light": "ST", "B_heavy": "mSOW", "Spring": "Spring", "Shakeout": "Shakeout", "LPS_C": "LPS",
        "Test": "Test", "Chain": "LPS", "BO": "SOS", "BU": "BU", "D_LPS": "LPS", "Fail": "Spring"},
    -1: {"ST": "ST", "B_light": "UA", "B_heavy": "mSOS", "Spring": "UTAD", "Shakeout": "UTAD", "LPS_C": "LPSY",
         "Test": "Test", "Chain": "LPSY", "BO": "SOW", "BU": "LPSY", "D_LPS": "LPSY", "Fail": "UTAD"},
}
ALERT_ROLES = {"B_heavy", "Spring", "Shakeout", "LPS_C", "Test", "BO", "BU"}
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
    hidden: bool = False   # set aside while the other side's breakout stands (restored if it fails)


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
    left: bool = False              # price left the range and stayed out (2 pullbacks held outside)
    box_end: int = 0                # last bar of the range box (where it left, or now while it is still alive)
    alerts: list = field(default_factory=list)    # (bar, text) — what the indicator alerts on, in order


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
        self.left = False
        self.away = False
        self.refit = False
        self.out_ok = 0
        self.e_start = None
        self.reentered = None
        self.undo = []
        self.alerts = None              # shared alert list (set by read)

    def add(self, role, bar, price_m, state, known, note="", suffix=""):
        nm = NAMES[self.f][role]
        if role == "B_light" and self.f == -1 and self.kind == "BC":
            nm = "UT"
        e = Event(nm + suffix, bar, price_m * self.f, state, known, self.f, role, np.nan, note)
        self.ev.append(e)
        if self.alerts is not None and role in ALERT_ROLES and state != "failed":
            self.alerts.append((known, e.name + ("?" if state == "candidate" else "")))
        return e

    def set_channel(self, k, base, s0, height):
        self.k, self.base, self.S, self.H, self.sloped = k, base, s0, height, True
        self.R = s0 + height

    def recent(self, bar, p, window=2 * BACK_BARS_B):
        """A test of this edge already named within the last `window` bars that went as far or further."""
        return any(e.side == self.f and not e.hidden and e.role in ("ST", "B_light", "B_heavy") and abs(bar - e.bar) <= window
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
        # after a confirmed breakout: did price come back into the range? then the range is still going on
        if self.phase in ("D", "E") and self.bo is not None and self.bo.state == "confirmed" and c[i] < R - REENTER * H:
            self.reentered = i
            return
        # in E: the range is left once LEFT_SWINGS pullbacks have held outside it
        if self.phase == "E" and sw is not None and sw[0] == -1 and sw[3] > self.e_start:
            Sk, Rk = self.lv(sw[3])
            if sw[4] >= Rk - HOLD_TOL * H:
                self.out_ok += 1
                if self.out_ok >= LEFT_SWINGS:
                    self.left, self.done, self.end = True, True, i
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
                    and (np.isnan(self.d_low) or p > self.d_low) and p <= self.C.price * self.f + LPS_ZONE * H \
                    and not any(e.side == self.f and not e.hidden and e.role in ("Test", "Chain")
                                and kk - e.bar <= 2 * BACK_BARS_B for e in self.ev):
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
        if self.away and c[i] >= S:
            self.away = False
        if self.phase == "B" and self.C is None and self.pen is None and not self.away and l[i] < (floor if gate else S):
            self.pen, self.pen_gate = i, gate
        if self.pen is not None and self.C is None:
            pen = self.pen
            low_k = pen + int(np.argmin(l[pen:i + 1]))
            e_k = _effort_at(self.eff, low_k, i)
            loud = not np.isnan(e_k) and e_k >= SOW_EFF
            heavy = loud or bool((self.sb[pen:i + 1] < 0).any())
            back = c[i] >= (floor if self.pen_gate else S)
            further = not any(e.side == self.f and not e.hidden and e.role in ("B_light", "B_heavy")
                              and l[low_k] >= e.price * self.f for e in self.ev)   # a new extreme of this edge?
            if back and not self.pen_gate and (self.recent(low_k, l[low_k], 4 * BACK_BARS_B) or (not heavy and not further)):
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
                self.away = True                       # no new test until price is back inside
        if self.C is not None and self.C.state != "failed" and c[i] < self.C.price * self.f:
            # the phase-C test failed: in hindsight it was spring-type action in phase B (Roman: a minor SOW / SOS);
            # its extreme becomes a new B extreme, and phase C must come again later
            e = self.C
            e.name, e.role, e.state = NAMES[self.f]["B_heavy"], "B_heavy", "confirmed"
            e.note = "a phase-C test that failed (closed beyond its extreme) — phase-B action, the range goes on"
            p_ = e.price * self.f
            self.low_b = p_ if np.isnan(self.low_b) else min(self.low_b, p_)
            k_ = max((n_ for n_, t in enumerate(self.trans) if t[0] == "C"), default=None)
            if k_ is not None and self.phase in ("B", "C"):
                self.trans = self.trans[:k_]
                self.phase = "B"
            for x in self.ev:                          # its tests go with it
                if x.side == self.f and not x.hidden and x.role in ("Test", "Chain") and x.bar > e.bar:
                    x.hidden = True
            self.C = None
            self.d_low = np.nan
            self.refit = True
            if self.alerts is not None:
                self.alerts.append((i, f"{NAMES[self.f]['Spring']} failed — read as {e.name} in phase B, the range goes on"))
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
                    self.bo.hidden = True               # the range goes on; the poke is part of it
                    self.trans = self.trans[:self.trans_mark]   # and the phases read as if it never was
                    self.phase = self.pre
                self.bo = None
            elif h[i] >= R + LEAVE_H * H:
                self.bo.state = "confirmed"
                self.trans_d = len(self.trans)
        if self.phase == "D" and self.bo is not None and self.bo.state == "confirmed" and h[i] >= R + LEAVE_H * H:
            self.e_start = i
            self.go("E", i)

    def lose(self, cut):
        """The other side's breakout is confirmed (its phase C began at bar `cut`). While it stands, this side's tests
        read as phase-B action: its phase-C test takes its phase-B name (a failed UTAD reads as a UA, a failed spring
        as an ST), its follow-ups and breakout are set aside, and anything after `cut` belonged to the winner's move.
        All of it is undone by restore() if that breakout fails."""
        self.undo = []
        for e in self.ev:
            if e.side != self.f or e.hidden:
                continue
            if e.bar > cut or e.role in ("Test", "Chain", "BU", "D_LPS", "BO") or e is self.lps_h:
                self.undo.append(("hide", e))
                e.hidden = True
            elif e is self.C or e.role in ("Spring", "Shakeout", "LPS_C", "Fail"):
                self.undo.append(("ren", e, e.name, e.role, e.state, e.note))
                e.name = "UT" if (self.f == -1 and self.kind == "BC") else NAMES[self.f]["B_light"]
                e.role, e.state = "B_light", "confirmed"
                e.note = "a phase-C test that failed when the range resolved the other way — phase-B action"

    def restore(self):
        for u in reversed(self.undo):
            if u[0] == "hide":
                u[1].hidden = False
            else:
                e = u[1]
                e.name, e.role, e.state, e.note = u[2], u[3], u[4], u[5]
        self.undo = []

    def unwind(self, i):
        """This side's confirmed breakout failed at bar i: the breakout and its back-ups go, the phases roll back to
        before it, and the excursion joins the range's extremes. Returns the excursion's extreme (bar, frame price)."""
        bo = self.bo
        j = bo.bar + int(np.argmax(self.h[bo.bar:i + 1]))
        for e in self.ev:
            if e.side == self.f and not e.hidden and (e is bo or (e.role in ("BU", "D_LPS") and e.bar >= bo.bar)):
                e.hidden = True
        if self.lps_h is not None and self.lps_h.known >= bo.known:
            self.lps_h.hidden = True
            if self.C is self.lps_h:
                self.C = None
            self.lps_h = None
        self.trans = self.trans[:self.trans_mark]
        self.phase = self.trans[-1][0] if self.trans else "B"
        if self.phase in ("D", "E"):
            self.phase = "B"
        self.bo = None
        self.d_low = np.nan
        self.out_ok, self.e_start, self.reentered, self.done = 0, None, None, False
        m = self.h[bo.bar:i + 1].max()
        self.cl_hi = m if np.isnan(self.cl_hi) else max(self.cl_hi, m)
        self.cl_upto = i
        self.labelled.add(j)
        return j, self.h[j]


def read(o, h, l, c, eff, vol, a1: int, a2: int, htf: int = 0, profile_rows: int = 24, auto: bool = False,
         kind: str | None = None) -> RangeRead:
    """Bar by bar, using only what was known on each bar (swings as finished on their reversal bar).
    auto (WHS Range Auto): the AR is the composite reaction from the climax — it extends through pullbacks that hold
    (higher lows after an SC); once a pullback breaks that structure the AR is final, and a close beyond it by the
    break tolerance before any ST means the reaction kept going: no range."""
    o, h, l, c, eff, vol = (np.asarray(x, float) for x in (o, h, l, c, eff, vol))
    n = len(c)
    led = swings.ledger(o, h, l, c, eff)
    sb = swings.sbars(o, h, l, c)
    atr = core.atr(h, l, c)
    b2 = a2                                             # the second click; the walk starts after it
    if kind is None:
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

    def _shakeout(k1, k2, size):
        """The reaction from the climax bar k1 to its extreme k2 is a SHAKEOUT: big (≥ SO_ATR × ATR), fast
        (≥ SO_SPEED ATR per bar on average) and on climactic effort somewhere in it (≥ SO_EFF)."""
        a = atr[k1] if not np.isnan(atr[k1]) else np.nan
        if np.isnan(a) or a <= 0 or k2 <= k1:
            return False
        e = eff[k1:k2 + 1]
        e = e[~np.isnan(e)]
        return size >= SO_ATR * a and size / (k2 - k1) >= SO_SPEED * a and len(e) > 0 and e.max() >= SO_EFF
    ar_open, ar_floor = True, S                         # auto: the AR's composite is still building / its last hold

    ar_hi = S                                           # the AR's extreme as known so far (start-up pass)

    def ar_track(sw, top=None):
        nonlocal ar_open, ar_floor, ar_hi
        if ar_open and sw[0] * K == 1 and sw[1] >= a1:
            ar_hi = max(ar_hi, sw[4] * K)
        if ar_open and sw[0] * K == -1 and sw[1] >= a1:
            hi = ar_hi if top is None else top
            deep = hi - S >= AR_RET_ATR * atr[a1] and hi - sw[4] * K >= AR_RET_END * (hi - S)
            if sw[4] * K < ar_floor or deep:
                ar_open = False
            else:
                ar_floor = sw[4] * K

    if auto:                                            # the list as the indicator holds it on the start bar
        snap = [lst for ci, lst in getattr(led, "live", []) if ci <= b2]
        for sw in (snap[-1] if snap else []):
            ar_track(sw)
    live = np.zeros(n, int)
    live[b2] = 1
    b_start = None
    sides = {}
    winner = 0
    end = n - 1
    visited = [False, False]                            # since B began: lower half, upper half
    channel = None
    alerts = []
    last_phase = "B"
    bull_name = "Re-accumulation" if htf == 1 else "Accumulation"
    bear_name = "Redistribution" if htf == -1 else "Distribution"
    for i in range(b2 + 1, n):
        if b_start is None:
            # ---- phase A (in the climax frame)
            if auto and not ar_open and cK[i] > R + BREAK_TOL * H:
                ar.state, ar.note = "failed", "the reaction kept going: no range, the trend turned"
                sc.state = "failed"
                live[i] = 1
                end = i
                break
            if hK[i] > R and (ar_open or not auto):
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
            if auto and sw is not None:
                ar_track(sw, R)
            if sw is not None and sw[0] * K == -1 and sw[1] >= a2 and sw[3] > a2:
                p = sw[4] * K
                if S - BREAK_TOL * H <= p <= S + ST_ZONE * H:
                    e_k, e_c = _effort_at(eff, sw[3], i), _effort_at(eff, a1, i)
                    less = not np.isnan(e_k) and not np.isnan(e_c) and e_k <= ST_EFFORT * e_c
                    st = add_a("ST", sw[3], p, "confirmed" if less else "candidate", i,
                               "on less effort than the climax" if less else "on more effort than the climax")
                    if auto and K == -1 and _shakeout(a1, a2, H):
                        # BC → SHAKEOUT → AR (WTC S2 p15 / S3 p14, Roman's alternative labelling; AAPL May 2010):
                        # the fast, climactic drop off the BC is a shakeout, the rally that buys it back is the AR
                        ar.name, ar.note = "Shakeout", "a fast, climactic drop off the BC, bought back"
                        st.name, st.note = "AR", "the rally that bought the shakeout back"
                    st.side, st.role = K, "ST"
                    if less:
                        sc.state = "confirmed"
                    b_start = sw[3]
                    bottom, top = (S, R) if K == 1 else (-R, -S)
                    bottom_flat, top_flat, H_flat = bottom, top, top - bottom
                    seed = {K: (p, e_c), -K: (-R, _effort_at(eff, a2, i))}
                    for f in (1, -1):
                        sides[f] = _Side(f, h, l, c, sb, eff, atr, bottom, top, kind, b_start, a1,
                                         seed[f][0], seed[f][1], ev, label_zone=(f == 1 or kind == "BC"))
                        sides[f].labelled.add(sw[3])
                        sides[f].alerts = alerts
                    alerts.append((i, "phase B"))
                    live[i] = 2
            continue
        # ---- phase B onward: both sides
        bottom, top = sides[1].lv(i)
        mid = (bottom + top) / 2
        visited[0] = visited[0] or l[i] <= mid
        visited[1] = visited[1] or h[i] >= mid
        gate = visited[0] and visited[1] and (i - b_start) >= B_MIN * (b_start - a1)
        busy = any(sd.bo is not None for sd in sides.values())
        if gate and not busy and (channel is None or sides[1].refit or sides[-1].refit):
            # fitted in the climax frame: an SC-type range hangs on its lows, a BC-type range on its highs
            ck = _fit_channel(hK, lK, a1, a1, b_start, i, H_flat)
            sides[1].refit = sides[-1].refit = False
            if K == 1:
                channel = ck
            else:                                       # back to real prices: lower line = −(upper line in frame K)
                kk_, bA_, pA_, hc_ = ck
                channel = (-kk_, bA_, -(pA_ + hc_), hc_) if kk_ != 0.0 else (0.0, a1, bottom_flat, H_flat)
            slope, bA, pA, hc = channel
            if slope != 0.0:
                sides[1].set_channel(slope, bA, pA, hc)
                sides[-1].set_channel(-slope, bA, -(pA + hc), hc)
            else:
                sides[1].set_channel(0.0, a1, bottom_flat, H_flat)
                sides[-1].set_channel(0.0, a1, -top_flat, H_flat)
                sides[1].sloped = sides[-1].sloped = False
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
                alerts.append((i, ("SOS" if f == 1 else "SOW") + " confirmed — " +
                               ("bullish: " + bull_name if f == 1 else "bearish: " + bear_name)))
        if winner and sides[winner].reentered is not None:
            # the breakout failed: price came back in — the range is still going on
            W, O = sides[winner], sides[-winner]
            was = NAMES[winner]["BO"]
            j, pw = W.unwind(i)
            O.restore()
            po = -pw                                    # the excursion's extreme in the other side's frame
            if gate and (O.C is None or O.C.state == "failed"):
                O.C = O.add("Spring", j, po, "candidate", i, "the breakout failed: price came back into the range")
                O.labelled.add(j)
                O.go("C", j)
                now = O.C.name
            else:
                ex = O.add("B_light", j, po, "confirmed", i, "the breakout failed: phase-B action, the range goes on")
                O.labelled.add(j)
                O.low_b = po if np.isnan(O.low_b) else min(O.low_b, po)
                now = ex.name
            alerts.append((i, f"{was} failed — price is back in the range, it is still going on ({was} → {now})"))
            winner = 0
            W.refit = True
        cur = _merged_phase(sides, winner)
        live[i] = "ABCDE".index(cur) + 1
        if cur != last_phase:
            if cur in ("B", "C"):
                alerts.append((i, f"phase {cur}"))
            last_phase = cur
        if winner and sides[winner].left:
            end = sides[winner].end
            alerts.append((i, "left the range — " + (bull_name if winner == 1 else bear_name) + " confirmed"))
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
    ev[:] = [e for e in ev if not e.hidden]
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
    rr.left = bool(winner and sides[winner].left)
    rr.box_end = sides[winner].e_start if rr.left else end
    rr.alerts = alerts
    rr.bull_name = "Re-accumulation" if htf == 1 else "Accumulation"
    rr.bear_name = "Redistribution" if htf == -1 else "Distribution"
    rr.name = rr.bull_name if winner == 1 else rr.bear_name if winner == -1 else ""
    rr.bull, rr.bear = _evidence(ev, htf, sides)
    return rr


def _fit_channel(h, l, a1, a_low_bar, b_start, i, H):
    """Sloping range (Roman: most real ranges slope; lower line through the lows of A and B, upper line parallel
    through the AR / UA highs). The B point is the lowest low of the LATER half of B, so the two points are far
    apart; the highs must lean the same way (else it is a triangle: flat); the tilt across the range so far must be
    between CH_FLAT·H (below: flat) and CH_MAX·H (above: not a range, flat); the line must hold every low."""
    pA = l[a_low_bar]
    mid = b_start + (i - b_start) // 2
    if i - mid < 2:
        return (0.0, a1, pA, H)
    bB = mid + int(np.argmin(l[mid:i + 1]))
    if bB - a_low_bar < max(4, (i - a1) // 3):
        return (0.0, a1, pA, H)
    slope = (l[bB] - pA) / (bB - a_low_bar)
    tilt = abs(slope * (i - a1))
    if tilt < CH_FLAT * H or tilt > CH_MAX * H:
        return (0.0, a1, pA, H)
    first_hi = h[a1:mid + 1].max()
    later_hi = h[mid:i + 1].max()
    if (later_hi - first_hi) * slope < 0 and abs(later_hi - first_hi) > CH_FLAT * H:
        return (0.0, a1, pA, H)                        # highs lean the other way: a triangle, keep it flat
    js = np.arange(a1, i + 1)
    lower = pA + slope * (js - a_low_bar)
    if (l[a1:i + 1] < lower - 0.1 * H).any():         # the line must hold the lows it was drawn through
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
