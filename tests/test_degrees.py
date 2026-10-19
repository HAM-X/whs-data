"""Swing degrees (2.2b): minor swings roll up into major swings; a climax stops a major (sideways box)."""
from pathlib import Path

import numpy as np
import pandas as pd

from whs import core, degrees, swings

FIX = Path(__file__).parent / "fixtures" / "6E_4h_2024Q4_2025Q1.csv"


def _load():
    d = pd.read_csv(FIX)
    t = pd.to_datetime(d.time, utc=True)
    eff = core.effort(d.volume, t, 240)
    return d, eff


def _run(d, eff, n=None):
    n = n or len(d)
    o, h, l, c = (d[k].to_numpy(float)[:n] for k in ("open", "high", "low", "close"))
    led = swings.ledger(o, h, l, c, eff[:n])
    return led, degrees.majors(o, h, l, c, eff[:n], led)


def test_majors_alternate_and_chain():
    d, eff = _load()
    led, mr = _run(d, eff)
    ms = mr.majors
    assert len(ms) >= 4
    trend = [m for m in ms if m.direction]
    assert len(led.swings) / len(trend) >= 2.0           # a major is a series of minor swings
    for a, b in zip(ms, ms[1:]):
        if a.direction and b.direction:
            assert b.direction == -a.direction and b.start_i == a.end_i   # a flip starts where the last major ended
        if a.direction and b.direction == 0:
            assert a.end_kind == "climax" and b.start_i == a.end_i        # a box starts at the climax extreme


def test_majors_never_look_ahead():
    d, eff = _load()
    _, full = _run(d, eff)
    for cut in (200, 300, 400):
        _, part = _run(d, eff, cut)
        done_full = [(m.direction, m.start_i, m.end_i, m.end_kind) for m in full.majors if 0 <= m.confirm_i < cut]
        done_part = [(m.direction, m.start_i, m.end_i, m.end_kind) for m in part.majors if m.end_kind != "live"]
        assert done_part == done_full


def test_a_climax_stops_the_major_and_price_goes_sideways():
    # a steady decline, a selling climax closing well off its low, then a reaction and sideways chop
    steps = ([-0.0020] * 6 + [0.0025] * 3) * 11            # a stair-step decline: minor swings down and up
    n0 = len(steps)
    c = list(1.20 + np.cumsum(steps))
    o = [c[0]] + c[:-1]
    h = [max(a, b) + 0.0006 for a, b in zip(o, c)]
    l = [min(a, b) - 0.0006 for a, b in zip(o, c)]
    v = [1000.0] * n0
    last = c[-1]
    o.append(last); l.append(last - 0.0120); h.append(last + 0.0003); c.append(last - 0.0040); v.append(9000.0)
    base = c[-1]
    for k in range(60):
        x = base + 0.0030 + 0.0012 * np.sin(k / 3)
        o.append(c[-1]); c.append(x); h.append(max(o[-1], x) + 0.0005); l.append(min(o[-1], x) - 0.0005); v.append(1000.0)
    o, h, l, c, v = (np.array(a, float) for a in (o, h, l, c, v))
    eff = v / np.median(v)
    led = swings.ledger(o, h, l, c, eff)
    mr = degrees.majors(o, h, l, c, eff, led)
    kinds = [(m.direction, m.end_kind) for m in mr.majors]
    assert (-1, "climax") in kinds
    stop = next(m for m in mr.majors if m.end_kind == "climax")
    assert stop.end_i == n0                                   # the major ends at the climax low
    assert mr.majors[-1].direction == 0                       # and price is still sideways
