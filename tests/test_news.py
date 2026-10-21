import numpy as np
import pandas as pd

from whs import news

RAW = [
    {"title": "CPI m/m", "country": "USD", "date": "2026-10-14T08:30:00-04:00", "impact": "High", "forecast": "0.3%", "previous": "0.4%"},
    {"title": "German ZEW", "country": "EUR", "date": "2026-10-14T05:00:00-04:00", "impact": "Medium", "forecast": "", "previous": "10"},
    {"title": "RBA Gov Speaks", "country": "AUD", "date": "2026-10-15T02:00:00-04:00", "impact": "High", "forecast": "", "previous": ""},
    {"title": "Bank Holiday", "country": "CNY", "date": "2026-10-15T00:00:00-04:00", "impact": "Holiday"},
]


def test_normalise_merge_and_blackout():
    ev = news.normalise(RAW)
    assert [e["ccy"] for e in ev] == ["USD", "EUR", "AUD"]           # CNY dropped
    assert ev[0]["t"] == "2026-10-14T12:30:00+00:00"
    cal = news.merge([], ev, "2026-10-14T00:00Z")
    cal = news.merge(cal, [dict(ev[0], actual="0.5%")], "2026-10-14T13:00Z")
    assert len(cal) == 3 and next(e for e in cal if e["ccy"] == "USD")["actual"] == "0.5%"
    at = pd.Timestamp("2026-10-14T12:00Z")
    assert news.blackout(cal, "6E", at) and news.blackout(cal, "GC", at)    # USD → every chart
    assert not news.blackout(cal, "6E", "2026-10-14T14:00Z")               # 90 min after → free
    assert news.blackout(cal, "6A", "2026-10-15T06:30Z") and not news.blackout(cal, "6J", "2026-10-15T06:30Z")
    assert not news.blackout(cal, "6E", "2026-10-14T09:00Z")               # Medium never blocks
    old = news.merge(cal, [], "2026-11-20T00:00Z")
    assert old == []                                                       # kept 21 days only


def test_pre_alert_once():
    cal = news.normalise(RAW)
    lines, ids = news.pre_alerts(cal, "2026-10-14T11:30Z", [])
    assert len(lines) == 1 and "CPI" in lines[0] and "every chart" in lines[0]
    again, ids2 = news.pre_alerts(cal, "2026-10-14T12:00Z", ids)
    assert again == [] and ids2 == ids


def _bars(moves, vol):
    t = pd.date_range("2026-10-13T00:00Z", periods=len(moves), freq="h")
    c = 1.10 + np.cumsum(moves)
    o = np.r_[1.10, c[:-1]]
    return pd.DataFrame(dict(time=t, open=o, high=np.maximum(o, c) + 0.0002, low=np.minimum(o, c) - 0.0002, close=c, volume=vol))


def test_reaction_verdicts():
    e = {"id": "x", "t": "2026-10-14T12:30:00+00:00", "ccy": "EUR", "title": "CPI", "impact": "High"}
    n = 60
    base = np.full(n, 0.0001)
    b = 36                                                  # 2026-10-14 12:00 bar
    up = base.copy(); up[b] = 0.0040; up[b + 1:b + 4] = 0.0001
    r = news.reaction(e, _bars(up, np.full(n, 1000.0)))
    assert r["verdict"] == "followed through" and r["move"] > 0.75
    fade = base.copy(); fade[b] = 0.0040; fade[b + 1] = -0.0045
    assert news.reaction(e, _bars(fade, np.full(n, 1000.0)))["verdict"].startswith("faded")
    vol = np.full(n, 1000.0); vol[b] = 50000.0
    flat = base.copy(); flat[b] = 0.0
    assert news.reaction(e, _bars(flat, vol))["verdict"].startswith("absorbed")
