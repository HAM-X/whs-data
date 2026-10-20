"""Auto ranges: the change of character (WTC P1 S2) — the AR must be bigger than every earlier reaction of the trend."""
from pathlib import Path

import numpy as np
import pandas as pd

from whs import autorange, core, swings

FIX = Path(__file__).parent / "fixtures" / "6E_4h_2024Q4_2025Q1.csv"


def _chain():
    d = pd.read_csv(FIX)
    t = pd.to_datetime(d.time, utc=True)
    lab = t.dt.tz_convert("Africa/Nairobi").dt.strftime("%Y-%m-%d %H:%M").tolist()
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    eff = core.effort(d.volume, t, 240)
    led = swings.ledger(o, h, l, c, eff)
    return lab, h, l, autorange.chain(o, h, l, c, eff, v, led)


def test_q1_2025_sc_and_ar_found_by_choch():
    lab, h, l, ch = _chain()
    got = [(lab[r["a1"]], lab[[e for e in rr.events if e.name == "AR"][0].bar]) for r, rr in ch]
    assert ("2025-01-13 10:00", "2025-01-27 14:00") in got        # the Q1 2025 accumulation, as Hamza marked it


def test_composite_rally_holds_through_higher_lows():
    # up 1→3, pullback to 2 (higher low), up to 5 → one rally 1→5; then a pullback to 0.5 breaks it
    sw = [(1, 0, 1.0, 1, 3.0, 2), (-1, 1, 3.0, 2, 2.0, 3), (1, 2, 2.0, 3, 5.0, 4), (-1, 3, 5.0, 4, 0.5, 5)]
    done, cur = autorange.composites(sw, 1)
    assert done == [(0, 1.0, 3, 5.0)] and cur is None


def test_ar_must_beat_every_earlier_rally():
    lab, h, l, ch = _chain()
    for r, rr in ch:
        assert r["ratio"] > autorange.CHOCH_K


def test_deep_pullback_ends_a_big_move():
    # down 10 → 4 (6 ATR with atr=1), then a rally back to 8.5 (75 %) ends the reaction even though it holds a lower high
    import numpy as np
    sw = [(-1, 0, 10.0, 1, 4.0, 2), (1, 1, 4.0, 2, 8.5, 3), (-1, 2, 8.5, 3, 3.0, 4)]
    done, cur = autorange.composites(sw, -1, np.ones(10))
    assert done == [(0, 10.0, 1, 4.0)] and cur == [2, 8.5, 3, 3.0]


def test_small_first_leg_is_not_ended_by_a_deep_pullback():
    import numpy as np
    sw = [(-1, 0, 10.0, 1, 9.0, 2), (1, 1, 9.0, 2, 9.8, 3), (-1, 2, 9.8, 3, 5.0, 4)]
    done, cur = autorange.composites(sw, -1, np.ones(10))     # 1 ATR first leg: the 80 % pullback does not end it
    assert done == [] and cur == [0, 10.0, 3, 5.0]
