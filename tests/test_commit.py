import numpy as np

from whs import commit


def test_commitment_steps_up_and_flips_on_the_close_below():
    # a rally of wide up bars (SBars) then a close below the last SBar's true-range low
    c = np.array([10, 10.1, 10, 10.1, 10, 10.1, 10, 10.1, 10, 10.1, 10, 10.1, 10, 10.1, 10, 10.1, 11, 12, 13, 12.9, 11.5], float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.05
    l = np.minimum(o, c) - 0.05
    r = commit.commitment(o, h, l, c)
    up = np.where(r["sb"] > 0)[0]
    assert len(up) >= 2 and r["state"][up[-1]] == 1
    assert r["line"][up[-1]] >= r["line"][up[0]]                  # never steps down
    assert r["rbar"][-1] == -1 and r["state"][-1] == -1           # the close below the line = RBar
    assert r["half"][16]                                          # a huge bar: the line goes to half of it
    assert r["line"][-1] == (h[-1] + l[-1]) / 2 <= max(h[up[0]:])  # flipped to the swing high, then the huge down SBar moved it
