import numpy as np
import pandas as pd

from whs import commit, r1


def _bars(c, vol=None, start="2026-10-01 00:00"):
    c = np.asarray(c, float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.05
    l = np.minimum(o, c) - 0.05
    t = pd.date_range(start, periods=len(c), freq="h", tz="UTC")
    v = np.full(len(c), 1000.0) if vol is None else np.asarray(vol, float)
    return pd.DataFrame(dict(time=t, open=o, high=h, low=l, close=c, volume=v))


def test_light_rule():
    assert r1.light(1.4, 0.7) == "red" and r1.light(1.2, 1.3) == "green" and r1.light(0.8, 1.2) == "ease"
    assert r1.light(1.0, 1.0) == "" and r1.light(np.nan, 1) == ""


def test_intention_and_close_to_close():
    d = _bars([10, 10.2, 10.1, 10.4, 10.3, 10.1] * 8)
    R = r1.reading(d.open, d.high, d.low, d.close, d.volume, d.time, 60)
    i = 3                                    # higher high, higher low, closes up at its high → Up → Yes
    assert R["att"][i] == 1 and R["c2c"][i] == 1 and R["intent"][i] == 2
    i = 5                                    # lower low, lower high, closes down → Down → Yes
    assert R["att"][i] == -1 and R["intent"][i] == -2


def test_commitment_matches_commit_py():
    rng = np.random.default_rng(3)
    d = _bars(10 + np.cumsum(rng.normal(0, 0.2, 400)))
    R = r1.reading(d.open, d.high, d.low, d.close, d.volume, d.time, 60)
    cm = commit.commitment(d.open, d.high, d.low, d.close)
    assert (R["state"] == cm["state"]).all() and np.allclose(R["line"], cm["line"], equal_nan=True)


def test_wave_counts_effort_and_travel_from_the_rbar():
    rng = np.random.default_rng(8)
    d = _bars(10 + np.cumsum(rng.normal(0, 0.25, 600)), vol=rng.uniform(500, 1500, 600))
    R = r1.reading(d.open, d.high, d.low, d.close, d.volume, d.time, 60)
    fl = np.where(R["rbar"] != 0)[0]
    assert len(fl) >= 4
    a, b = fl[1], fl[2]
    s = R["rbar"][a]
    assert R["wave_bars"][b - 1] == b - a
    assert np.isclose(R["wave_effort"][b - 1], np.nansum(R["effort"][a:b]))
    ext = d.high[a:b].max() if s > 0 else d.low[a:b].min()
    atr1 = R["atr"][a - 1]
    assert np.isclose(R["wave_travel"][b - 1], (ext - d.close[a]) * s / atr1)
    # dE / dR: each wave vs the previous wave of the SAME side (waves alternate → two waves back)
    E = [np.nansum(R["effort"][x:y]) for x, y in zip(fl[:-1], fl[1:])]
    T = [R["wave_travel"][y - 1] for y in fl[1:]]
    checked = 0
    for j in range(2, len(E)):
        y = fl[j + 1] - 1
        assert R["rbar"][fl[j]] == R["rbar"][fl[j - 2]]
        if E[j - 2] > 0 and T[j - 2] > 0:
            assert np.isclose(R["wave_dE"][y], E[j] / E[j - 2]) and np.isclose(R["wave_dR"][y], T[j] / T[j - 2])
            checked += 1
    assert checked >= 2


def test_last_bar_reads_the_parent_from_closed_bars_only():
    rng = np.random.default_rng(5)
    d = _bars(10 + np.cumsum(rng.normal(0, 0.2, 800)))
    p = d.set_index("time").resample("4h", origin="epoch").agg(dict(open="first", high="max", low="min", close="last", volume="sum")).dropna().reset_index()
    out = r1.last_bar(d, 60, p, 240)
    assert out["tf"] == "1h" and out["parent"]["tf"] == "4h"
    last_close = pd.Timestamp(out["bar"]) + pd.Timedelta(hours=1)
    assert pd.Timestamp(out["parent"]["bar"]) + pd.Timedelta(hours=4) <= last_close
    assert out["state"] in (1, -1) and out["sl"] is not None and out["wave"]["bars"] >= 1
