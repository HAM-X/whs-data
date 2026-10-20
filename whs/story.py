"""Wyckoff Story (checklist 2.4): Roman's four steps, filled in from a range read.

Each step names EFFORT and RESULT with the measured numbers, then the bias line:
    #1 A   the stopping action: climax effort and where it closed, the automatic reaction, the secondary test
    #2 B   cause building: effort on reactions (supply) vs rallies (demand) and how it changes, tests of the edges
    #3 C   the test: spring / shakeout / UTAD / LPS(Y), its effort and whether it held
    #4 D/E the trend out: SOS / SOW effort and result, back-up / last point of support, whether price left
    BIAS   name · evidence k of n · fails below / above <price>

Built only from what the range read already knows (the read is a live walk), so the story at bar t never
uses bars after t. Swings come from the ledger as finished on their reversal bars.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core

UP_WORDS = ("rising", "flat", "falling")


@dataclass
class Story:
    steps: list = field(default_factory=list)     # (number, phase letters, text)
    bias: str = ""
    bias_dir: int = 0                              # +1 bullish · −1 bearish · 0 open
    fails: float = np.nan                          # the price that invalidates the bias
    evidence: tuple = (0, 0)                       # (for, against)


def _trend(vals):
    vals = vals[-4:]
    if len(vals) < 2:
        return ""
    r = vals[-1] / vals[0] if vals[0] else np.nan
    return UP_WORDS[0] if r >= 1.15 else UP_WORDS[2] if r <= 0.87 else UP_WORDS[1]


def _seq(vals, k=4):
    return " → ".join(f"{v:.1f}×" for v in vals[-k:])


def _ev(rr, *roles, side=None):
    return [e for e in rr.events if not e.hidden and e.role in roles and (side is None or e.side == side)]


def _phase_span(rr, letter):
    for p in rr.phases:
        if p[0] == letter:
            return p[1], (p[2] if len(p) > 2 else rr.end)
    return None


def _avg(eff, a, b):
    v = [x for x in eff[a:b + 1] if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


def build(o, h, l, c, eff, rr, led) -> Story:
    h, l, c, eff = (np.asarray(x, float) for x in (h, l, c, eff))
    st = Story()
    sc = rr.kind == "SC"
    lowword, highword = ("low", "high") if sc else ("high", "low")
    # ---- #1 phase A ----
    clx = next((e for e in rr.events if e.role in ("SC", "BC")), None)
    ar = next((e for e in rr.events if e.role == "AR"), None)
    parts = []
    if clx is not None:
        k = clx.bar
        rng = h[k] - l[k]
        off = ((c[k] - l[k]) if sc else (h[k] - c[k])) / rng * 100 if rng > 0 else np.nan
        parts.append(f"{clx.name} on {clx.effort:.1f}×, closed {off:.0f}% off its {lowword}")
    if clx is not None and ar is not None:
        ext = (ar.price / clx.price - 1) * 100
        parts.append(f"{ar.name} {ext:+.2f}% in {ar.bar - clx.bar} bars on {_avg(eff, clx.bar + 1, ar.bar):.1f}×")
    sts = _ev(rr, "ST", side=1 if sc else -1)
    if sts:
        s0 = sts[0]
        cmp_ = "" if clx is None or np.isnan(clx.effort) else (" (less than the climax)" if s0.effort < clx.effort else " (MORE than the climax)")
        parts.append(f"{s0.name} on {s0.effort:.1f}×{cmp_}")
    if parts:
        st.steps.append((1, "A", " · ".join(parts)))
    # ---- #2 phase B ----
    b = _phase_span(rr, "B")
    if b is not None:
        b0, b1 = b
        known = [d for d in led.done if d[5] <= rr.end]
        inside = [d for d in known if d[1] >= b0 and d[3] <= b1]
        dn = [_avg(eff, d[1] + 1, d[3]) for d in inside if d[0] == -1]
        up = [_avg(eff, d[1] + 1, d[3]) for d in inside if d[0] == 1]
        dn = [x for x in dn if not np.isnan(x)]
        up = [x for x in up if not np.isnan(x)]
        parts = []
        # Roman: rising effort on reactions = supply growing (distribution signature); flat / falling = absorbed
        sv = {"rising": "supply growing", "flat": "supply absorbed", "falling": "supply drying up"}
        dv = {"rising": "demand growing", "flat": "demand steady", "falling": "demand fading"}
        if dn:
            parts.append(f"reactions {_seq(dn)}" + (f" ({sv[_trend(dn)]})" if _trend(dn) else ""))
        if up:
            parts.append(f"rallies {_seq(up)}" + (f" ({dv[_trend(up)]})" if _trend(up) else ""))
        for e in _ev(rr, "B_heavy", "B_light"):
            if b0 <= e.bar <= b1:
                short = "a failed C test" if e.role == "B_heavy" and "failed" in e.note else "heavy" if e.role == "B_heavy" else "back inside"
                parts.append(f"{e.name} on {e.effort:.1f}× ({short})")
        if parts:
            st.steps.append((2, "B", " · ".join(parts[:5])))
    # ---- #3 phase C ----
    cs = _ev(rr, "Spring", "Shakeout", "LPS_C", "Test")
    if cs:
        parts = []
        for e in cs[:2]:
            held = "held" if e.state == "confirmed" else "failed" if e.state == "failed" else "to be confirmed"
            parts.append(f"{e.name} on {e.effort:.1f}×, {held}")
        st.steps.append((3, "C", " · ".join(parts)))
    # ---- #4 phase D / E ----
    bo = _ev(rr, "BO")
    if bo:
        e = bo[-1]
        base = cs[-1].price if cs else (rr.support if e.side == 1 else rr.resistance)
        ext = (e.price / base - 1) * 100
        sync = " in sync" if e.effort >= 1.1 else " on low effort"
        parts = [f"{e.name} on {e.effort:.1f}×, {ext:+.2f}% from the {'LPS' if e.side == 1 else 'LPSY'}{sync}"]
        for x in _ev(rr, "BU", "D_LPS"):
            parts.append(f"{x.name} on {x.effort:.1f}×" + (" held" if x.state == "confirmed" else ""))
        if rr.left:
            parts.append("price left the range")
        st.steps.append((4, "D" if not rr.left else "D · E", " · ".join(parts[:3])))
    # ---- bias ----
    nb, nr = len(rr.bull), len(rr.bear)
    if rr.outcome:
        f = rr.outcome
        st.bias_dir = f
        lows = [e.price for e in rr.events if not e.hidden and e.side == f and e.role in ("ST", "Spring", "Shakeout", "LPS_C", "Test", "BU", "D_LPS", "Chain")]
        if f == 1:
            st.fails = max([p for p in lows if p < c[rr.end]] or [rr.support])
        else:
            st.fails = min([p for p in lows if p > c[rr.end]] or [rr.resistance])
        n_for, n_ag = (nb, nr) if f == 1 else (nr, nb)
        st.evidence = (n_for, n_ag)
        st.bias = f"{rr.name.upper()} · evidence {n_for} of {n_for + n_ag} · fails {'below' if f == 1 else 'above'} {st.fails:.5g}"
    else:
        st.evidence = (nb, nr)
        st.bias = f"OPEN · {rr.bull_name} {nb} vs {rr.bear_name} {nr} · watching {rr.resistance:.5g} / {rr.support:.5g}"
    return st
