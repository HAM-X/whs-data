import numpy as np
import pandas as pd

from whs import calls, cards


def _bars(n=300, drift=0.0004):
    t = pd.date_range("2026-10-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(3)
    c = 1.10 + np.cumsum(rng.normal(drift, 0.0008, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(dict(time=t, open=o, high=np.maximum(o, c) + 0.0004, low=np.minimum(o, c) - 0.0004, close=c,
                             volume=rng.integers(500, 1500, n).astype(float)))


def test_call_logged_and_scored():
    d = _bars()
    i = 150
    dec = {"6E 1h": dict(action="BUY ZONE", side=1, stop=float(d.low[i] - 0.001), target=1.2, price=float(d.close[i]),
                         bar=d.time[i].isoformat(), atr=0.0015)}
    allc = []
    made = calls.new_calls([("6E 1h", "WAIT", "BUY ZONE")], dec, allc)
    assert len(made) == 1 and allc[0]["stop"] < dec["6E 1h"]["stop"]          # buffer 0.25 ATR below the structure
    assert calls.new_calls([("6E 1h", "WAIT", "BUY ZONE")], dec, allc) == []    # same bar → no duplicate
    calls.score(allc, {("6E", "1h"): d})
    c = allc[0]
    assert c["status"] in ("open", "closed") and c["held"] > 0
    txt = calls.report(allc, d.time.iloc[-1])
    assert "Robot self-grade" in txt


def test_watch_zone():
    dec = {"action": "WAIT", "levels": [1.10, 1.12]}
    assert cards.watch(1.1003, 0.002, dec)["edge"] == "support"
    assert cards.watch(1.0950, 0.002, dec)["beyond"] is True
    assert cards.watch(1.1198, 0.002, dec)["edge"] == "resistance"
    assert cards.watch(1.1100, 0.002, dec) is None
    assert cards.watch(1.1003, 0.002, dict(dec, action="TREND ↑")) is None
