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


def _walk(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.standard_t(3, n) * 0.1)
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 0.06, n))
    return o, np.maximum(o, c) + w, np.minimum(o, c) - w, c


def test_v2_ledger_invariants_on_a_random_walk():
    o, h, l, c = _walk()
    r = swings.ledger(o, h, l, c, np.ones(len(c)))
    S = r.swings
    assert len(S) > 50
    for a, b in zip(S, S[1:]):
        assert a.direction != b.direction                       # directions alternate
        assert a.end_i == b.start_i and a.end_p == b.start_p    # each swing starts where the last one ended
    for s in S:
        assert s.confirm_i > s.start_i       # (a shifted turn can move a swing's end past its live RBar)
        seg_h, seg_l = h[s.start_i + 1:s.end_i + 1], l[s.start_i + 1:s.end_i + 1]   # the turning bar itself excluded
        if s.direction == 1:
            assert s.end_p == seg_h.max()                       # ends at its highest high
            assert l[s.start_i + 1:s.end_i].min(initial=np.inf) >= s.start_p   # never trades below its start
        else:
            assert s.end_p == seg_l.min()
            assert h[s.start_i + 1:s.end_i].max(initial=-np.inf) <= s.start_p


def test_v1_rules_remain_available():
    o, h, l, c = _walk()
    v1 = swings.ledger(o, h, l, c, np.ones(len(c)), rules=swings.RULES_V1).swings
    v2 = swings.ledger(o, h, l, c, np.ones(len(c))).swings
    assert v1 and v2 and [(s.start_i, s.end_i) for s in v1] != [(s.start_i, s.end_i) for s in v2]


def test_climactic_bar_keeps_the_reference():
    # quiet drift down, one climactic up bar (≈13 × ATR), a small follow-through SBar, a dip that closes below
    # the follow-through low but far above the climactic low, then a new high.
    c = list(100 - np.arange(30) * 0.03)
    o = [c[0]] + c[:-1]
    h = [max(a, b) + 0.08 for a, b in zip(o, c)]
    l = [min(a, b) - 0.08 for a, b in zip(o, c)]
    base = c[-1]
    k = len(c)
    rows = [(base, base + 1.7, base - 0.1, base + 1.6),         # climactic bar
            (base + 1.6, base + 2.1, base + 1.5, base + 2.05),  # follow-through SBar (much smaller)
            (base + 2.05, base + 2.07, base + 1.35, base + 1.4),  # dip closes below the follow-through low
            (base + 1.4, base + 1.5, base + 1.25, base + 1.3),
            (base + 1.3, base + 1.65, base + 1.28, base + 1.6),
            (base + 1.6, base + 2.0, base + 1.55, base + 1.95),
            (base + 1.95, base + 2.5, base + 1.9, base + 2.45),
            (base + 2.45, base + 3.0, base + 2.4, base + 2.95),
            (base + 2.95, base + 3.0, base + 2.2, base + 2.25),   # turn down
            (base + 2.25, base + 2.3, base + 1.6, base + 1.65),
            (base + 1.65, base + 1.7, base + 1.0, base + 1.05)]
    for r in rows:
        o.append(r[0]); h.append(r[1]); l.append(r[2]); c.append(r[3])
    o, h, l, c = (np.array(x) for x in (o, h, l, c))
    sb = swings.sbars(o, h, l, c)
    assert sb[k] == 1 and sb[k + 1] == 1
    ups = lambda rules: [s for s in swings.ledger(o, h, l, c, np.ones(len(c)), rules=rules).swings
                         if s.direction == 1 and s.start_i == k]
    v1, v2 = ups(swings.RULES_V1), ups(None)
    assert v1 and v1[0].end_i == k + 1                          # v1: the dip ended the swing at the follow-through
    assert v2 and v2[0].end_i == k + 7                          # v2: one swing from the climactic low to the top
