"""WHS Core v1 contract tests (Python twin of the Pine Core)."""
import datetime as dt

import numpy as np
import pandas as pd

from whs import core


def test_true_range_uses_previous_close():
    h, l, c = [10, 12, 11], [9, 11, 8], [9.5, 11.5, 10]
    assert list(core.true_range(h, l, c)) == [1, 12 - 9.5, 11.5 - 8]   # gap up counts, gap down counts


def test_rma_matches_pine_seed():
    x = np.arange(1, 21, dtype=float)
    r = core.rma(x, 14)
    assert np.isnan(r[12]) and r[13] == x[:14].mean()
    assert np.isclose(r[14], (r[13] * 13 + x[14]) / 14)


def test_rel_spread_uses_previous_atr():
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 1, 60))
    h, l = c + 1, c - 1
    rs = core.rel_spread(h, l, c)
    a = core.atr(h, l, c)
    assert np.isclose(rs[30], core.true_range(h, l, c)[30] / a[29])


def test_close_pos_bounds():
    cp = core.close_pos([10, 12, 12], [9, 11, 10], [9.5, 12, 10])
    assert cp[1] == 1.0 and cp[2] == 0.0


def test_effort_same_slot_median_excludes_current_bar():
    t = pd.Series(pd.date_range("2026-01-05 15:00", periods=24 * 30, freq="h", tz="UTC"))
    v = np.where(t.dt.hour == 14, 1000.0, 100.0)          # one busy slot
    e = core.effort(v, t, 60)
    hour = t.dt.hour.to_numpy()
    assert np.allclose(e[24 * 25:][hour[24 * 25:] == 14], 1.0)   # busy hour vs its own history = 1
    assert np.allclose(e[24 * 25:][hour[24 * 25:] == 3], 1.0)


def test_effort_4h_slots_are_unique():
    t = pd.Series(pd.to_datetime(["2026-03-02 17:00", "2026-03-02 21:00", "2026-03-03 01:00",
                                  "2026-03-03 05:00", "2026-03-03 09:00", "2026-03-03 13:00"]).tz_localize("America/Chicago"))
    assert sorted(core.slot_key(t.dt.tz_convert("UTC"), 240)) == [0, 1, 2, 3, 4, 5]


def test_roll_windows():
    # 6E Dec 2026: 3rd Wednesday 16 Dec → last trade Mon 14 Dec → roll handling from Wed 2 Dec
    assert not core.in_roll_window("6E", dt.date(2026, 12, 1))
    assert core.in_roll_window("6E", dt.date(2026, 12, 2))
    assert core.in_roll_window("6E", dt.date(2026, 12, 14))
    assert not core.in_roll_window("6E", dt.date(2026, 12, 15))
    assert core.in_roll_window("GC", dt.date(2026, 11, 20)) and not core.in_roll_window("GC", dt.date(2026, 12, 20))


def test_band_edges():
    assert [core.band(x) for x in (1.5, 1.1, 1.0, 0.9, 0.67)] == ["≫", ">", "≈", "<", "≪"]
