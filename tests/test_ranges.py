"""Engine 2.3 tests: range workbench (events, states, phases, profile)."""
from pathlib import Path

import numpy as np
import pandas as pd

from whs import core, ranges

FIX = Path(__file__).parent / "fixtures" / "6E_4h_2024Q4_2025Q1.csv"


def _load():
    d = pd.read_csv(FIX)
    t = pd.to_datetime(d.time, utc=True)
    lab = t.dt.tz_convert("Africa/Nairobi").dt.strftime("%Y-%m-%d %H:%M").tolist()
    eff = core.effort(d.volume, t, 240)
    return d, lab, eff


def _read(d, eff, a1, a2, sign=1):
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    if sign == -1:
        o, h, l, c = -o, -l, -h, -c
    return ranges.read(o, h, l, c, eff, d.volume.to_numpy(float), a1, a2)


def test_eurusd_q1_2025_accumulation():
    # SC 13 Jan, AR 27 Jan, ST 3 Feb (tariff gap) / 10 Feb, LPS 28 Feb, SOS 4 Mar (German fiscal news), E after
    d, lab, eff = _load()
    a1 = lab.index("2025-01-13 10:00")
    a2 = lab.index("2025-01-27 14:00")
    r = _read(d, eff, a1, a2)
    assert r.kind == "SC" and r.outcome == 1 and r.name == "Accumulation"
    names = {e.name: e for e in r.events}
    assert names["SOS"].state == "confirmed" and names["LPS"].state == "confirmed"
    assert [p[0] for p in r.phases] == ["A", "B", "C", "D", "E"]
    assert "SOS confirmed" in r.bull and not any("SOW" in x for x in r.bear)
    assert lab[names["SOS"].bar].startswith("2025-03-0")
    edges, prof, poc = r.profile
    assert prof.sum() > 0 and edges[0] <= r.support and edges[-1] >= r.support


def test_distribution_is_accumulation_mirrored():
    # the same bars upside down: a BC-type range that resolves down, with the same phase-C/D story on the top side
    d, lab, eff = _load()
    a1, a2 = lab.index("2025-01-13 10:00"), lab.index("2025-01-27 14:00")
    acc = _read(d, eff, a1, a2)
    dist = _read(d, eff, a1, a2, sign=-1)
    assert dist.kind == "BC" and dist.outcome == -1 and dist.name == "Distribution"
    key = lambda r, side: [(e.role, e.bar, e.state) for e in r.events if e.side == side and e.role not in ("ST", "B_light")]
    assert key(acc, 1) == key(dist, -1)
    assert [p[:2] for p in acc.phases] == [p[:2] for p in dist.phases]
    assert np.isclose(dist.support, -acc.resistance) and np.isclose(dist.resistance, -acc.support)


def test_names_follow_the_higher_timeframe():
    d, lab, eff = _load()
    a1, a2 = lab.index("2025-01-13 10:00"), lab.index("2025-01-27 14:00")
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    up = ranges.read(o, h, l, c, eff, d.volume.to_numpy(float), a1, a2, htf=1)
    assert up.name == "Re-accumulation" and up.bear_name == "Distribution"
    dn = ranges.read(-o, -l, -h, -c, eff, d.volume.to_numpy(float), a1, a2, htf=-1)
    assert dn.name == "Redistribution" and dn.bull_name == "Accumulation"


def test_no_range_when_the_reaction_keeps_going():
    # a low, a bounce chosen as the "AR", then the rally just keeps going: the range never formed
    c = np.r_[np.linspace(110, 100, 40), np.linspace(100, 104, 10), np.linspace(104, 125, 60)]
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c) + 0.2, np.minimum(o, c) - 0.2
    a1 = int(np.argmin(l[:45]))
    r = ranges.read(o, h, l, c, np.ones(len(c)), np.ones(len(c)), a1, 49)
    ar = next(e for e in r.events if e.name == "AR")
    assert ar.state == "failed" and [p[0] for p in r.phases] == ["A"] and r.outcome == 0


def _early_break_range():
    seg=lambda a,b,n: list(np.linspace(a,b,n+1))[1:]
    c=[120.0]+seg(120,100,30)
    c+= [99.2]                         # SC close (bar a1)
    a1=len(c)-1
    c+=seg(99.2,105.5,8); a2=len(c)-1  # AR
    c+=seg(105.5,99.6,6)               # ST
    c+=seg(99.6,103.5,5)               # rally that turns the ST swing
    c+=seg(103.5,98.6,4)               # drop into the break
    brk=len(c); c+=[97.4]              # heavy break bar
    c+=seg(97.4,99.0,2)                # back inside
    c+=seg(99.0,104.5,8)               # rally into the upper half
    c+=seg(104.5,100.5,6); c+=seg(100.5,104.0,6); c+=seg(104.0,98.2,7)
    spr=len(c); c+=[97.2]              # dip under the A+B lows (96.9?) -> see lows below
    c+=seg(97.2,99.5,2); c+=seg(99.5,106.8,10); c+=seg(106.8,110,8); c+=seg(110,108,4)
    c=np.array(c); o=np.r_[c[0],c[:-1]]
    h=np.maximum(o,c)+0.35; l=np.minimum(o,c)-0.35
    l[a1]=98.0; h[a1]=100.6
    l[brk]=96.8; l[spr]=96.55; h[spr]=max(h[spr],97.6)
    eff=np.ones(len(c)); eff[a1]=2.4; eff[a1+1]=1.5; eff[brk]=2.6; eff[spr]=0.8; eff[spr+1]=0.9
    return o,h,l,c,eff,a1,a2,brk,spr


def test_early_phase_b_break_is_st_as_sow_not_the_spring():
    # Roman (HD 1990): a break below support EARLY in phase B on rising effort is "ST as SOW" — still phase B.
    # Phase C comes only after a mature B (as long as A, with a rally into the upper half) and tests the A+B lows.
    o, h, l, c, eff, a1, a2, brk, spr = _early_break_range()
    r = ranges.read(o, h, l, c, eff, np.ones(len(c)), a1, a2)
    ev = {e.bar: e for e in r.events}
    assert ev[brk].name == "mSOW" and ev[brk].state == "confirmed"
    assert ev[spr].name.startswith("Spring")
    starts = {p[0]: p[1] for p in r.phases}
    assert starts["B"] < brk < starts["C"] == spr
    assert [p[0] for p in r.phases] == ["A", "B", "C", "D", "E"]
