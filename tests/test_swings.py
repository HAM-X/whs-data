"""Engine 2.2 tests: significant bars, traffic light sequence, swing ledger."""
import numpy as np

from whs import swings


def bars_from_closes(closes, spread=0.4):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + spread / 2
    l = np.minimum(o, c) - spread / 2
    return o, h, l, c


def test_sbar_needs_size_close_half_and_breakout():
    c = [100] * 20 + [100.2, 99.9, 100.1, 103.0]          # last bar: big, closes high, beyond 2 prior highs
    o, h, l, cc = bars_from_closes(c)
    sb = swings.sbars(o, h, l, cc)
    assert sb[-1] == 1 and (sb[:-1] == 0).all()


def test_tl_sequence_red_green_blue():
    tl = np.array([0, 1, 0, 2, 3, 0, 4, 0])
    td = np.array([0, -1, 0, 1, 1, 0, 1, 0])
    seq = swings.tl_sequences(tl, td)
    assert seq[6] == 1 and seq.sum() == 1


def test_tl_red_alone_is_not_a_reversal():
    tl = np.array([0, 1, 0, 4, 0, 4])
    td = np.array([0, -1, 0, 1, 0, 1])
    assert (swings.tl_sequences(tl, td) == 0).all()


def test_tl_opposite_green_breaks_sequence():
    tl = np.array([1, 3, 3, 4])
    td = np.array([-1, 1, -1, 1])
    assert (swings.tl_sequences(tl, td) == 0).all()


def test_ledger_zigzag_alternates_and_confirms_late():
    leg = list(np.linspace(100, 106, 12)) + list(np.linspace(106, 101, 12))[1:] + list(np.linspace(101, 107, 12))[1:]
    c = [100.0] * 20 + leg
    o, h, l, cc = bars_from_closes(c)
    eff = np.ones(len(c))
    r = swings.ledger(o, h, l, cc, eff)
    dirs = [s.direction for s in r.swings]
    assert all(a != b for a, b in zip(dirs, dirs[1:]))          # directions alternate
    for s in r.swings:
        assert s.confirm_i > s.end_i                            # confirmed after the extreme (retroactive)
        assert r.rbar[s.confirm_i] == -s.direction
    up = [s for s in r.swings if s.direction == 1]
    assert up and abs(up[0].end_p - h[31]) < 1e-9               # first up swing tops at the 106 high


def test_tag_format():
    s = swings.Swing(n=9, direction=1, start_i=0, start_p=1.0, end_i=13, end_p=1.0111, confirm_i=15,
                     extent_pct=1.11, bars=13, avg_effort=1.4, r_vs_prev=1.2, e_vs_prev=1.2)
    assert s.tag() == "↑9 +1.11% 13b 1.4× R> D>"
