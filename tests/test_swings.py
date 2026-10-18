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


def mk(n, d, dE, dR):
    s = swings.Swing(n=n, direction=d, start_i=0, start_p=1, end_i=1, end_p=1, confirm_i=2, extent_pct=1, bars=1,
                     avg_effort=1, r_vs_prev=dR, e_vs_prev=dE)
    return s


def run(seq):
    st = {"dir": 0, "stage": 0, "k": -10**9}
    out = []
    for s in seq:
        swings.classify(s, st)
        out.append((s.tl, s.seq))
    return out


def test_tl_top_sequence_red_orange_green_blue():
    # rally stops (RED up) → lower-high attempt on less effort (ORANGE up) → decline expands (GREEN down) → eases (BLUE down)
    out = run([mk(1, 1, 1.6, 0.6), mk(2, -1, 1.0, 1.0), mk(3, 1, 0.8, 1.2), mk(4, -1, 1.5, 1.6),
               mk(5, 1, 1.0, 1.0), mk(6, -1, 0.7, 1.3)])
    assert [o[0] for o in out] == [swings.TL_RED, 0, swings.TL_ORANGE, swings.TL_GREEN, 0, swings.TL_BLUE]
    assert out[-1][1] == -1 and sum(o[1] for o in out) == -1


def test_tl_red_alone_is_not_a_reversal():
    out = run([mk(1, 1, 1.6, 0.6), mk(2, -1, 0.7, 1.3), mk(3, 1, 1.0, 1.0)])
    assert all(o[1] == 0 for o in out)


def test_tl_green_back_in_old_direction_cancels():
    out = run([mk(1, 1, 1.6, 0.6), mk(2, -1, 1.5, 1.5), mk(3, 1, 1.5, 1.5), mk(4, -1, 0.7, 1.3)])
    assert all(o[1] == 0 for o in out)


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


def test_bar_traffic_light_sequence():
    # flat history, then: up bar stopping (more effort, less result) → down bar sync → down bar ease
    c = [100.0, 100.5, 100.0, 100.5, 100.0, 100.5, 100.0] * 3
    h = [x + 0.5 for x in c]; l = [x - 0.5 for x in c]
    e = [1.0] * len(c)
    c += [100.5, 99.5, 98.1]; h += [100.6, 100.5, 99.5]; l += [100.2, 99.2, 98.0]
    e += [2.0, 2.2, 0.8]
    tl, seq = swings.bar_traffic_light(h, l, c, e)
    assert list(tl[-3:]) == [swings.TL_RED, swings.TL_GREEN, swings.TL_BLUE]
    assert seq[-1] == -1
