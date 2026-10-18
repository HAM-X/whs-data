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
  Spring [UTAD]  price trades below support and closes back inside within BACK_BARS bars.
              #3 = shallow (≤ SPRING3_ATR) on ordinary effort (≤ SPRING3_EFF): confirmed at once.
              #2 = deeper or heavier: candidate until a test holds a higher low, then confirmed.
              failed = a close below the spring low (or no return inside → breakdown).
  UA          [UT] in phase B, price trades above the creek and closes back below within BACK_BARS bars
  LPS [LPSY]  no spring yet: a down swing in phase B that holds a higher low than the ST in the lower half.
              candidate → confirmed when an SOS follows without breaking it; failed if a close below it
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

NAMES_ACC = {"SC": "SC", "AR": "AR", "ST": "ST", "Spring": "Spring", "UA": "UA", "LPS": "LPS", "SOS": "SOS",
             "BU": "BU", "Test": "Test"}
NAMES_DIST = {"SC": "BC", "AR": "AR", "ST": "ST", "Spring": "UTAD", "UA": "UT", "LPS": "LPSY", "SOS": "SOW",
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


def _effort_at(eff, k):
    vals = [eff[j] for j in (k, k + 1) if 0 <= j < len(eff) and not np.isnan(eff[j])]
    return max(vals) if vals else np.nan


def read(o, h, l, c, eff, vol, a1: int, a2: int, kind: str | None = None, profile_rows: int = 24) -> RangeRead:
    o, h, l, c, eff, vol = (np.asarray(x, float) for x in (o, h, l, c, eff, vol))
    if kind is None:                                   # SC low under the AR high → accumulation
        kind = "accumulation" if l[a1] <= l[a2] and h[a2] >= h[a1] else "distribution"
    acc = kind == "accumulation"
    if not acc:                                        # mirror: distribution read as accumulation of −price
        o, h, l, c = -o, -l, -h, -c
    names = NAMES_ACC if acc else NAMES_DIST
    n = len(c)
    atr = core.atr(h, l, c)
    S = l[a1]
    a2 = a1 + int(np.argmax(h[a1:a2 + 1]))             # AR = highest high up to the second click
    R = h[a2]
    H = R - S
    led = swings.ledger(o, h, l, c, eff)
    ev = []

    def add(key, bar, price, state, known, note=""):
        e = Event(names[key], bar, price, state, known, _effort_at(eff, bar), note)
        ev.append(e)
        return e

    sc = add("SC", a1, S, "candidate", a1)
    ar = add("AR", a2, R, "confirmed", a2)
    phases = [["A", a1, None]]
    phase = "A"
    st_eff, st_low = np.nan, np.nan
    spring = lps = sos = None
    sos_hold = 0
    done_swings = sorted((s for s in led.swings if s.start_i >= a2), key=lambda s: s.confirm_i)
    sw_k = 0
    out_i = n - 1
    i = a2 + 1
    pen = None                                          # first bar of a penetration below support
    up = None                                           # first bar above the creek (phase B)
    ua_wait = None                                      # UA high waiting for price to fall back into the range
    H0 = H
    while i < n:
        # phase A: the reaction may still be running (AR extends) — or the trend simply resumes (no range)
        if phase == "A":
            if h[i] > R:
                if h[i] - S > AR_MAX * H0:
                    ar.state, ar.note = "failed", "the reaction kept going: no range, the trend turned"
                    sc.state = "failed"
                    out_i = i
                    break
                R, a2, H = h[i], i, h[i] - S
                ar.bar, ar.price, ar.effort, ar.note = i, R, _effort_at(eff, i), "extended past the second click"
            if c[i] < S - BREAK_TOL * H:
                sc.state, sc.note = "failed", "closed back beyond the SC: the trend resumed"
                out_i = i
                break
        # swings that became known on this bar
        while sw_k < len(done_swings) and done_swings[sw_k].confirm_i <= i:
            s = done_swings[sw_k]
            sw_k += 1
            if s.direction != -1 or s.end_i <= a2:
                continue
            k, p = s.end_i, s.end_p
            e_k = _effort_at(eff, k)
            if phase in ("A", "B") and spring is None and S - BREAK_TOL * H <= p <= S + ST_ZONE * H and lps is None:
                less = not np.isnan(e_k) and not np.isnan(sc.effort) and e_k <= ST_EFFORT * sc.effort
                st = add("ST", k, p, "confirmed" if less else "candidate", i,
                         "on less effort than the SC" if less else "on more effort than the SC")
                if less and sc.state == "candidate":
                    sc.state = "confirmed"
                if phase == "A":
                    phases[-1][2] = k
                    phases.append(["B", k, None])
                    phase = "B"
                st_eff = e_k if np.isnan(st_eff) else min(st_eff, e_k)
                st_low = p if np.isnan(st_low) else min(st_low, p)
            elif phase == "B" and spring is None and not np.isnan(st_low) and st_low < p <= S + LPS_ZONE * H and k > a2:
                if lps is not None and lps.state == "candidate":
                    lps.state = "failed" if p < lps.price else lps.state
                lps = add("LPS", k, p, "candidate", i, "higher low than the ST, before any spring")
            elif spring is not None and spring.state == "candidate" and p > spring.price and k > spring.bar:
                add("Test", k, p, "confirmed", i, "holds a higher low than the spring")
                spring.state = "confirmed"
        # spring: a trade below support after the ST
        if phase == "B" and spring is None and pen is None and l[i] < S:
            pen = i
        if pen is not None and spring is None:
            low_k = pen + int(np.argmin(l[pen:i + 1]))
            if c[i] >= S:
                depth = S - l[low_k]
                e_k = _effort_at(eff, low_k)
                shallow = depth <= SPRING3_ATR * atr[low_k] and (np.isnan(e_k) or e_k <= SPRING3_EFF)
                spring = add("Spring", low_k, l[low_k], "confirmed" if shallow else "candidate", i,
                             "#3 shallow, ordinary effort" if shallow else "#2 needs a test that holds a higher low")
                spring.name += " #3" if shallow else " #2"
                phases[-1][2] = low_k
                phases.append(["C", low_k, None])
                phase = "C"
                pen = None
            elif i - pen >= BACK_BARS:
                add("Spring", low_k, l[low_k], "failed", i, "no return inside: breakdown")
                out_i = i
                break
        if spring is not None and spring.state != "failed" and c[i] < spring.price:
            spring.state = "failed"
            spring.note = "closed below the spring low"
        if lps is not None and lps.state == "candidate" and c[i] < lps.price:
            lps.state = "failed"
        # UA: a phase-B trade above the creek that closes back inside, then falls back into the range
        if phase == "B" and sos is None:
            if h[i] > R and up is None and ua_wait is None:
                up = i
            if up is not None:
                if c[i] > R:
                    up = None                                  # closed above: the SOS rule takes it
                elif c[i] < R:
                    ua_wait = up + int(np.argmax(h[up:i + 1]))
                    up = None
            if ua_wait is not None and l[i] <= R - UA_DROP * H:
                add("UA", ua_wait, h[ua_wait], "confirmed", i, "trade beyond the range edge fell back inside")
                ua_wait = None
        # SOS: a close beyond the creek
        if sos is None and c[i] > R and phase in ("B", "C"):
            ua_wait = None
            sos = add("SOS", i, c[i], "candidate", i, "close beyond the range edge")
            sos_hold = 0
            if lps is not None and lps.state == "candidate":
                lps.state = "confirmed"
                phases[-1][2] = lps.bar
                phases.append(["C", lps.bar, None])
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
        if phase == "D" and h[i] >= R + LEAVE_H * H and sos is not None and sos.state == "confirmed":
            phases[-1][2] = i
            phases.append(["E", i, None])
            phase = "E"
            out_i = i
            # BU: a pullback swing between the SOS and leaving the range that held the creek
            bu = next((s for s in done_swings if s.direction == -1 and sos.bar < s.end_i <= i
                       and s.end_p >= R - HOLD_TOL * H), None)
            if bu is not None:
                add("BU", bu.end_i, bu.end_p, "confirmed", bu.confirm_i, "back-up held the range edge")
            break
        i += 1
    end = out_i if phase == "E" else min(i, n - 1)
    # phase E runs to the end of the swing that left the range (or the last bar)
    leave = next((s for s in led.swings if s.start_i <= end <= s.end_i and s.direction == 1), None)
    phases[-1][2] = (leave.end_i if leave is not None else n - 1) if phase == "E" else end
    # range volume profile: the cause built inside the range, SC → the SOS (or the end of the read)
    # raw volume (contracts traded at each price), spread evenly across each bar's high–low
    p_end = next((e.bar for e in ev if e.name == names["SOS"] and e.state != "failed"), end) - 1
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
    rr = RangeRead(kind, a1, a2, min(S, R), max(S, R), ev, [tuple(p) for p in phases],
                   (edges, prof, int(np.argmax(prof))), end)
    return rr
