"""WHS Engine 2.3 — range workbench (Python twin of the Pine range section).

The trader anchors a range with two clicks: the bar that stopped the trend (SC for accumulation, BC for
distribution) and the extreme of the automatic reaction (AR). From there the engine reads events, their states
and the phases. Distribution is accumulation mirrored (prices negated), so one reader serves both.

Levels   accumulation: support = SC low, creek = AR high       distribution: resistance = BC high, ice = AR low
Events   (accumulation names; distribution in brackets)
  SC [BC]     the anchor. candidate → confirmed by an ST on clearly less effort (≤ ST_EFFORT × SC effort)
  AR          the second anchor (confirmed by the trader)
  ST          a down swing after AR ending in the lower ST_ZONE of the range (not below support − BREAK_TOL·H).
              Less effort than SC → confirmed (and the SC with it); more effort → candidate, a later ST may confirm
  ST / mSOW   [ST / mSOS] in an immature phase B a trade below support that returns inside within 6 bars is an ST
              in B — or, on effort ≥ 1.5 or with a down SBar, an "ST as SOW" (minor SOW). It stays phase B and
              lowers the floor phase C must test (the lows of A AND B). No return → breakdown.
  Gate        phase C needs a mature B: B as long as A, and a rally into the upper half of the range since B began
  Spring [UTAD]  after the gate, price trades below the A+B lows and closes back above them within BACK_BARS bars.
              Effort ≥ 1.5 at the low = Shakeout, candidate until a test holds a higher low.
              #3 = shallow (≤ SPRING3_ATR) on ordinary effort (≤ SPRING3_EFF): confirmed at once.
              #2 = deeper or heavier: candidate until a test holds a higher low, then confirmed.
              failed = a close below the spring low (or no return inside → breakdown).
  UA          [UT] in phase B, price trades above the creek and closes back below within BACK_BARS bars
  LPS [LPSY]  no spring: when the SOS prints, the pullback low it rose from (lowest low since the last up swing's
              top) if that is a higher low than the ST in the lower half of the range — known only in hindsight
  SOS [SOW]   a close beyond the creek; confirmed when price holds beyond it (no close back below
              creek − HOLD_TOL·H) for HOLD_BARS bars, failed if it closes back below that line first
  BU          a pullback swing between the SOS and leaving the range that held the creek (when one exists)
  UA needs price to fall back into the range (≥ UA_DROP·H below the creek) before any SOS; else it was the SOS start
Phases   A  SC → the first ST · B  building cause → C  spring / test / LPS → D  SOS → E  price leaves the range
         (≥ LEAVE_H × H beyond the creek). Without a spring or LPS, B runs straight into D.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core, swings

ST_ZONE, ST_EFFORT, BREAK_TOL = 0.35, 0.8, 0.15
BACK_BARS, SPRING3_ATR, SPRING3_EFF = 3, 1.0, 1.1
LPS_ZONE = 0.5
HOLD_TOL, HOLD_BARS, LEAVE_H = 0.2, 6, 0.5
UA_DROP = 0.25
AR_MAX = 2.0          # phase A: an AR that runs past 2 × the first range height means no range formed
B_MIN, B_ROUND = 1.0, 0.5   # phase C needs a mature phase B: as long as phase A, with a rally into the upper half
SOW_EFF, BACK_BARS_B = 1.5, 6   # a phase-B break on effort ≥ 1.5 or a down SBar = ST as SOW; it gets 6 bars to return

NAMES_ACC = {"SC": "SC", "AR": "AR", "ST": "ST", "Spring": "Spring", "UA": "UA", "LPS": "LPS", "SOS": "SOS",
             "mSOW": "mSOW", "Shakeout": "Shakeout",
             "BU": "BU", "Test": "Test"}
NAMES_DIST = {"SC": "BC", "AR": "AR", "ST": "ST", "Spring": "UTAD", "UA": "UT", "LPS": "LPSY", "SOS": "SOW",
              "mSOW": "mSOS", "Shakeout": "UTAD",
              "BU": "BU", "Test": "Test"}


@dataclass
class Event:
    name: str
    bar: int            # where it is drawn (the extreme)
    price: float
    state: str          # candidate | confirmed | failed
    known: int          # the bar on which the engine could know it
    effort: float = np.nan
    note: str = ""


@dataclass
class RangeRead:
    kind: str                       # "accumulation" | "distribution"
    a1: int
    a2: int
    support: float                  # SC low / AR low (ice) for distribution
    resistance: float               # AR high (creek) / BC high
    events: list = field(default_factory=list)
    phases: list = field(default_factory=list)   # (letter, start bar, end bar)
    profile: tuple = None           # (row edges, volume per row, poc row)
    end: int = 0                    # last bar of the range read
    live: np.ndarray = None         # phase as known on each bar: 0 none · 1 A … 5 E (what Pine exports)


def _effort_at(eff, k, upto=None):
    """Effort at an event: the larger of its bar and the next one (only bars known by `upto`)."""
    top = len(eff) - 1 if upto is None else upto
    vals = [eff[j] for j in (k, k + 1) if 0 <= j <= top and not np.isnan(eff[j])]
    return max(vals) if vals else np.nan


def read(o, h, l, c, eff, vol, a1: int, a2: int, kind: str | None = None, profile_rows: int = 24) -> RangeRead:
    """Bar by bar, using only what was known on each bar (swings as finished on their reversal bar) —
    the same walk the Pine range section makes."""
    o, h, l, c, eff, vol = (np.asarray(x, float) for x in (o, h, l, c, eff, vol))
    led = swings.ledger(o, h, l, c, eff)                # swings on the real prices; mirrored below
    sb_real = swings.sbars(o, h, l, c)
    if kind is None:                                   # SC low under the AR high → accumulation
        kind = "accumulation" if l[a1] <= l[a2] and h[a2] >= h[a1] else "distribution"
    acc = kind == "accumulation"
    k_ = 1 if acc else -1
    if not acc:                                        # mirror: distribution read as accumulation of −price
        o, h, l, c = -o, -l, -h, -c
    names = NAMES_ACC if acc else NAMES_DIST
    sbm = sb_real * k_                                  # SBars in the mirrored frame (negative = down)
    n = len(c)
    atr = core.atr(h, l, c)
    S = l[a1]
    b2 = a2                                             # the second click; the walk starts after it
    a2 = a1 + int(np.argmax(h[a1:a2 + 1]))             # AR = highest high up to the second click
    R = h[a2]
    H = R - S
    ev = []

    def add(key, bar, price, state, known, note=""):
        e = Event(names[key], bar, price, state, known, _effort_at(eff, bar), note)
        ev.append(e)
        return e

    sc = add("SC", a1, S, "candidate", a1)
    ar = add("AR", a2, R, "confirmed", a2)
    phases = [["A", a1, None]]
    phase = "A"
    st_low = np.nan
    low_b = np.nan                                      # lowest low of phase-B breaks below support
    b_start = None
    rallied = False                                     # phase B has rallied into the upper half
    pen_gate = False
    labelled = set()                                    # bars already named by a break (no double ST)
    spring = lps = sos = None
    sos_hold = 0
    bu = None
    by_bar = {}                                         # reversal bar → the swing finished on it (mirrored)
    for (dd, si, sp, ei, ep, ci) in led.done:
        by_bar[ci] = (dd * k_, si, sp * k_, ei, ep * k_)
    end = n - 1
    pen = None                                          # first bar of a penetration below support
    up = None                                           # first bar above the creek (phase B)
    ua_wait = None                                      # UA high waiting for price to fall back into the range
    H0 = H
    a2_click = a2
    last_up_end = a2
    live = np.zeros(n, int)
    live[b2] = 1
    for i in range(b2 + 1, n):
        live[i] = "ABCDE".index(phase) + 1
        # phase A: the reaction may still be running (AR extends) — or the trend simply resumes (no range)
        if phase == "A":
            if h[i] > R:
                if h[i] - S > AR_MAX * H0:
                    ar.state, ar.note = "failed", "the reaction kept going: no range, the trend turned"
                    sc.state = "failed"
                    end = i
                    live[i] = "ABCDE".index(phase) + 1
                    break
                R, a2, H = h[i], i, h[i] - S
                ar.bar, ar.price, ar.effort, ar.note = i, R, _effort_at(eff, i), "extended past the second click"
            if c[i] < S - BREAK_TOL * H:
                sc.state, sc.note = "failed", "closed back beyond the SC: the trend resumed"
                end = i
                live[i] = "ABCDE".index(phase) + 1
                break
        # the swing finished on this bar (if any)
        sw = by_bar.get(i)
        if sw is not None and sw[0] == 1 and phase == "E":
            end = sw[3]                                 # the swing that left the range is over: the read ends
            live[i] = "ABCDE".index(phase) + 1
            break
        if sw is not None and sw[0] == 1:
            last_up_end = sw[3]
        if sw is not None and sw[0] == -1 and sw[1] >= a2_click and sw[3] > a2:
            kk, p = sw[3], sw[4]
            e_k = _effort_at(eff, kk, i)
            if (phase in ("A", "B") and spring is None and lps is None and kk not in labelled
                    and S - BREAK_TOL * H <= p <= S + ST_ZONE * H):
                e_sc = _effort_at(eff, a1, i)
                less = not np.isnan(e_k) and not np.isnan(e_sc) and e_k <= ST_EFFORT * e_sc
                add("ST", kk, p, "confirmed" if less else "candidate", i,
                    "on less effort than the SC" if less else "on more effort than the SC")
                if less and sc.state == "candidate":
                    sc.state = "confirmed"
                if phase == "A":
                    phases[-1][2] = kk
                    phases.append(["B", kk, None])
                    phase = "B"
                    b_start = kk
                st_low = p if np.isnan(st_low) else min(st_low, p)
            elif spring is not None and spring.state == "candidate" and p > spring.price and kk > spring.bar:
                add("Test", kk, p, "confirmed", i, "holds a higher low than the spring")
                spring.state = "confirmed"
            elif phase == "D" and sos is not None and kk > sos.bar and p >= R - HOLD_TOL * H:
                bu = (kk, p, i)
        # phase B maturity: phase C may start only after B has lasted as long as A and rallied into the upper half
        if phase == "B" and b_start is not None and h[i] >= S + B_ROUND * H and i > b_start:
            rallied = True
        gate = phase == "B" and b_start is not None and rallied and (i - b_start) >= B_MIN * (b_start - a1)
        floor = S if np.isnan(low_b) else min(S, low_b)    # Phase C tests the lows of A AND B
        # a trade below support: in an immature B it is an ST (or ST as SOW); after the gate it is the spring
        if phase == "B" and spring is None and pen is None and l[i] < (floor if gate else S):
            pen, pen_gate = i, gate
        if pen is not None and spring is None:
            low_k = pen + int(np.argmin(l[pen:i + 1]))
            e_k = _effort_at(eff, low_k, i)
            loud = not np.isnan(e_k) and e_k >= SOW_EFF                 # above-average effort at the low
            heavy = loud or bool((sbm[pen:i + 1] < 0).any())             # ...or a wide down bar (SOW character)
            back = c[i] >= (floor if pen_gate else S)
            if back and not pen_gate:
                key = "mSOW" if heavy else "ST"
                add(key, low_k, l[low_k], "confirmed", i,
                    "phase B: ST as SOW — supply still present, not the spring" if heavy else
                    "phase B: ST below the SC low, back inside")
                labelled.add(low_k)
                low_b = l[low_k] if np.isnan(low_b) else min(low_b, l[low_k])
                pen = None
            elif back:
                depth = floor - l[low_k]
                shallow = depth <= SPRING3_ATR * atr[low_k] and (np.isnan(e_k) or e_k <= SPRING3_EFF)
                if loud:
                    spring = add("Shakeout", low_k, l[low_k], "candidate", i,
                                 "heavy effort below the A+B lows, back inside — needs a test that holds a higher low")
                else:
                    spring = add("Spring", low_k, l[low_k], "confirmed" if shallow else "candidate", i,
                                 "#3 shallow, ordinary effort" if shallow else "#2 needs a test that holds a higher low")
                    spring.name += " #3" if shallow else " #2"
                labelled.add(low_k)
                phases[-1][2] = low_k
                phases.append(["C", low_k, None])
                phase = "C"
                pen = None
            elif i - pen >= (BACK_BARS if pen_gate else BACK_BARS_B):
                add("Spring" if pen_gate else "mSOW", low_k, l[low_k], "failed", i, "no return inside: breakdown")
                end = i
                live[i] = "ABCDE".index(phase) + 1
                break
        if spring is not None and spring.state != "failed" and c[i] < spring.price:
            spring.state = "failed"
            spring.note = "closed below the spring low"
        # UA: a phase-B trade above the creek that closes back inside, then falls back into the range
        if phase == "B" and sos is None:
            if h[i] > R and up is None and ua_wait is None:
                up = i
            if up is not None:
                if c[i] > R:
                    up = None                           # closed above: the SOS rule takes it
                elif c[i] < R:
                    ua_wait = up + int(np.argmax(h[up:i + 1]))
                    up = None
            if ua_wait is not None and l[i] <= R - UA_DROP * H:
                add("UA", ua_wait, h[ua_wait], "confirmed", i, "trade beyond the range edge fell back inside")
                ua_wait = None
        # SOS: a close beyond the creek
        if sos is None and c[i] > R and phase in ("B", "C"):
            ua_wait = None
            up = None
            # LPS (no spring): the pullback low this SOS rose from — a higher low than the ST in the lower half
            if phase == "B" and spring is None and not np.isnan(st_low):
                j0 = max(last_up_end, a2) + 1
                kk = j0 + int(np.argmin(l[j0:i + 1]))
                fl = st_low if np.isnan(low_b) else min(st_low, low_b)
                if fl < l[kk] <= S + LPS_ZONE * H:
                    lps = add("LPS", kk, l[kk], "confirmed", i, "higher low than the ST; the SOS rose from it")
                    phases[-1][2] = kk
                    phases.append(["C", kk, None])
            sos = add("SOS", i, c[i], "candidate", i, "close beyond the range edge")
            sos_hold = 0
            bu = None
            phases[-1][2] = i
            phases.append(["D", i, None])
            phase = "D"
        elif sos is not None and sos.state == "candidate":
            if c[i] < R - HOLD_TOL * H:
                sos.state = "failed"
                sos.note = "closed back inside the range"
                phases[-1][2] = i
                phases.append(["B", i, None])
                phase = "B"
                sos = None
            else:
                sos_hold += 1
                if sos_hold >= HOLD_BARS:
                    sos.state = "confirmed"
        if phase == "D" and sos is not None and sos.state == "confirmed" and h[i] >= R + LEAVE_H * H:
            if bu is not None:
                add("BU", bu[0], bu[1], "confirmed", bu[2], "back-up held the range edge")
            phases[-1][2] = i
            phases.append(["E", i, None])
            phase = "E"
        live[i] = "ABCDE".index(phase) + 1
    phases[-1][2] = end
    # range volume profile: the cause built inside the range, SC → the bar before the SOS (or the end of the read)
    # raw volume (contracts traded at each price), spread evenly across each bar's high–low
    p_end = next((e.bar for e in ev if e.name == names["SOS"] and e.state != "failed"), end + 1) - 1
    lo_p, hi_p = l[a1:p_end + 1].min(), h[a1:p_end + 1].max()
    edges = np.linspace(lo_p, hi_p, profile_rows + 1)
    prof = np.zeros(profile_rows)
    for j in range(a1, p_end + 1):
        if np.isnan(vol[j]) or h[j] <= l[j]:
            continue
        ov = np.clip(np.minimum(edges[1:], h[j]) - np.maximum(edges[:-1], l[j]), 0, None)
        prof += vol[j] * ov / (h[j] - l[j])
    if not acc:                                        # mirror back
        S, R = -S, -R
        for e in ev:
            e.price = -e.price
        edges = -edges[::-1]
        prof = prof[::-1]
    for e in ev:                                       # shown effort: full information
        e.effort = _effort_at(eff, e.bar)
    return RangeRead(kind, a1, a2, min(S, R), max(S, R), ev, [tuple(p) for p in phases],
                     (edges, prof, int(np.argmax(prof))), end, live)
