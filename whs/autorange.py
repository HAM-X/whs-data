"""Automatic range anchors (WHS Range Auto) — from the course (WTC Part 1 S2; Forte, Anatomy of a Trading Range):

  "CHoCH happens twice per range. In Phase A (trending → non-trending: the stop of the old trend) … the largest
   reaction within the uptrend [mirror: the largest RALLY within the downtrend], increase of spreads, increase of
   volume, break of the trendline." The AR after the SC is that largest rally; "the high of the AR helps define
   the TR top". "Phase A can end without dramatic spread/volume, but dramatic selling is better" → the climax is
   evidence, the CHoCH is the condition. "A climax is only a candidate until a swing reversal confirms it."

  Rallies are measured as COMPOSITE swings: a rally keeps going through minor pullbacks that hold above its start
  and retrace < 50 % of it (the AR of Jan 2025 on 6E 4h is three minor up swings). Live walk on the minor swings as
  known on their reversal bars:
    trend leg       from the most extreme point of the last LOOKBACK swings (highest high for a decline) to now;
                    it must already contain ≥ MIN_COUNTER composite rallies, and fall ≥ LEG_K × the AR
    stopping action the AR starts at the leg's lowest low (SC) — climactic if effort ≥ CLX_EFF near it
    AR = CHoCH      as soon as the composite rally from that low is bigger than every earlier rally of the leg
                    (× CHOCH_K): ① = the SC, ② = the rally's high so far (the AR may still extend in Phase A)
"""
from __future__ import annotations

import numpy as np

LOOKBACK = 50
MIN_COUNTER = 1
CHOCH_K = 1.0
LEG_K = 1.0
LEG_K0 = 2.0
CLX_EFF = 2.0
OUTSIDE = 0.25


def live_swings(led):
    cur = []
    for d in led.done:
        dr, s_i, s_p, e_i, e_p, conf = d
        cur = [x for x in cur if x[1] < s_i]
        if cur and cur[-1][3] > s_i:
            x = cur[-1]
            cur[-1] = (x[0], x[1], x[2], s_i, s_p, x[5])
        cur.append(d)
        yield conf, list(cur)


def composites(sw, dr):
    """Composite swings in direction dr (+1 rallies) by structure: a rally keeps going while every pullback makes a
    HIGHER LOW (holds above the rally's last low); it ends when a pullback breaks that low. Returns the finished ones
    and the live one: (start bar, start price, end bar, end price)."""
    out = []
    cur, floor = None, None
    for x in sw:
        if x[0] == dr:
            if cur is None:
                cur, floor = [x[1], x[2], x[3], x[4]], x[2]
            elif (x[4] - cur[3]) * dr > 0:
                cur[2], cur[3] = x[3], x[4]
        elif cur is not None:
            if (x[4] - floor) * dr < 0:
                out.append(tuple(cur))
                cur = None
            else:
                floor = x[4]
    return out, cur


def detect(h, l, eff, led, need_climax=False, choch_k=CHOCH_K, min_counter=MIN_COUNTER, leg_k=LEG_K, accept=None,
           spans=None):
    """CHoCH triggers, live. accept(trigger) → True | False | "retry" decides whether a trigger becomes the range in
    force (chain). spans() → [(start, end, low, high)] of the ranges shown so far: swings that start inside a range's
    box (AR, ST, tests) are range action, not counter-moves of the trend, so they are left out of the comparison."""
    out, seen = [], {}
    for conf, sw in live_swings(led):
        for dr in (1, -1):
            win = sw[-LOOKBACK:]
            done, cur = composites(win, dr)
            if cur is None:
                continue
            a1, p1, a2, p2 = cur
            if (a1, dr) in seen and seen[(a1, dr)] != "retry":
                continue
            # trend leg: from the most extreme opposite point before the AR's start
            before = [x for x in win if x[3] <= a1]
            if not before:
                continue
            opp = [(x[1], x[2]) for x in before if x[0] == -dr] + [(x[3], x[4]) for x in before if x[0] == dr]
            leg_i, leg_p = (max if dr == 1 else min)(opp, key=lambda q: q[1] * 1.0)
            ext_ok = all((x[4] - p1) * dr >= -1e-12 for x in before if x[3] >= leg_i and x[0] == -dr)
            if not ext_ok:
                continue
            sp = spans() if spans else []
            rallies = [r for r in done if r[0] >= leg_i and r[2] <= a1
                       and not any(b0 <= r[0] <= b1 and lo <= r[1] <= hi for b0, b1, lo, hi in sp)]
            size = (p2 - p1) * dr
            if len(rallies) < min_counter:
                # a straight leg (no counter-move of its own outside a range): the reaction is trivially the largest —
                # take it only when the leg is a real trend, at least LEG_K0 × the reaction
                if (leg_p - p1) * dr < LEG_K0 * size:
                    continue
                biggest = 0.0
            else:
                biggest = max((r[3] - r[1]) * dr for r in rallies)
            if size <= choch_k * biggest or (leg_p - p1) * dr < leg_k * size:
                continue
            ce = np.nanmax(eff[max(0, a1 - 2):a1 + 3])
            climax = bool(not np.isnan(ce) and ce >= CLX_EFF)
            if need_climax and not climax:
                continue
            r = dict(dir=dr, a1=a1, a2=a2, conf=conf, kind="SC" if dr == 1 else "BC", climax=climax,
                     ratio=size / biggest if biggest else float("inf"), rallies=len(rallies), effort=ce, leg=leg_i)
            ok = True if accept is None else accept(r)
            seen[(a1, dr)] = ok
            if ok is True:
                out.append(r)
    return out


def chain(o, h, l, c, eff, v, led, htf_at=lambda i: 0, **kw):
    """The ranges the indicator shows, newest last: every CHoCH trigger starts a range unless its climax sits inside
    the range in force (± OUTSIDE × H) — then it is that range's own action (an ST, a test, a UTAD, a spring).
    Each range is read by the range reader (whs/ranges.py), which extends the AR in phase A and drops a range whose
    reaction kept going (the trend turned). Returns [(trigger, RangeRead)]."""
    from . import ranges
    out = []

    def accept(r):
        k = r["conf"] + 1
        if out:
            q, _ = out[-1]
            try:
                now = ranges.read(o[:k], h[:k], l[:k], c[:k], eff[:k], v[:k], q["a1"], q["a2"], htf=q["htf"], auto=True)
                failed = any(e.role in ("SC", "BC") and e.state == "failed" for e in now.events)
                if not now.left and not failed:
                    lo, hi = min(now.support, now.resistance), max(now.support, now.resistance)
                    H = hi - lo
                    p1 = l[r["a1"]] if r["dir"] == 1 else h[r["a1"]]
                    if lo - OUTSIDE * H <= p1 <= hi + OUTSIDE * H:
                        # a new extreme beyond ① whose reaction also goes beyond ② engulfs the range: the old climax
                        # was only preliminary support / supply (PS / PSY) — the new one is the climax
                        p2 = h[r["a2"]] if r["dir"] == 1 else l[r["a2"]]
                        under = (p1 < lo) if r["dir"] == 1 else (p1 > hi)
                        over = (p2 > hi) if r["dir"] == 1 else (p2 < lo)
                        if not under:
                            return False
                        if not over:
                            return "retry"
                        out.pop()
            except Exception:
                pass
        r["htf"] = htf_at(r["a1"])
        try:
            rr = ranges.read(o, h, l, c, eff, v, r["a1"], r["a2"], htf=r["htf"], auto=True)
        except Exception:
            return False
        out.append((r, rr))
        return True

    def spans():
        # a range that died in phase A (the reaction kept going / the climax gave way) was no range: its swings
        # stay counter-moves of the trend
        return [(q["a1"], max(q["a1"] + 1, rr.box_end or rr.end), min(rr.support, rr.resistance),
                 max(rr.support, rr.resistance)) for q, rr in out if any(p[0] != "A" for p in rr.phases)]

    detect(h, l, eff, led, accept=accept, spans=spans, **kw)
    return out
