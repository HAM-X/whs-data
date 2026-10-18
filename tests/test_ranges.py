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
    assert r.kind == "accumulation"
    names = {e.name: e for e in r.events}
    assert names["SOS"].state == "confirmed" and names["LPS"].state == "confirmed"
    assert [e.name for e in r.events] == ["SC", "AR", "ST", "ST", "LPS", "SOS"]
    assert [p[0] for p in r.phases] == ["A", "B", "C", "D", "E"]
    assert lab[names["SOS"].bar].startswith("2025-03-0")
    edges, prof, poc = r.profile
    assert prof.sum() > 0 and edges[0] <= r.support and edges[-1] >= r.support


def test_distribution_is_accumulation_mirrored():
    d, lab, eff = _load()
    a1, a2 = lab.index("2025-01-13 10:00"), lab.index("2025-01-27 14:00")
    acc = _read(d, eff, a1, a2)
    dist = _read(d, eff, a1, a2, sign=-1)
    assert dist.kind == "distribution"
    m = dict(zip(ranges.NAMES_ACC.values(), ranges.NAMES_DIST.values()))
    assert [(m[e.name.split(" ")[0]], e.bar, e.state) for e in acc.events] == \
           [(e.name.split(" ")[0], e.bar, e.state) for e in dist.events]
    assert [p[:2] for p in acc.phases] == [p[:2] for p in dist.phases]
    assert np.isclose(dist.support, -acc.resistance) and np.isclose(dist.resistance, -acc.support)


def test_no_range_when_the_reaction_keeps_going():
    # a low, a bounce chosen as the "AR", then the rally just keeps going: the range never formed
    c = np.r_[np.linspace(110, 100, 40), np.linspace(100, 104, 10), np.linspace(104, 125, 60)]
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c) + 0.2, np.minimum(o, c) - 0.2
    a1 = int(np.argmin(l[:45]))
    r = ranges.read(o, h, l, c, np.ones(len(c)), np.ones(len(c)), a1, 49)
    ar = next(e for e in r.events if e.name == "AR")
    assert ar.state == "failed" and r.phases == [("A", a1, r.phases[0][2])]
