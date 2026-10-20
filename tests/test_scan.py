"""Stage 3 scan: automatic ranges from climax-stopped boxes, and alerts only for what is new."""
from pathlib import Path

import pandas as pd

from whs import scan

FIX = Path(__file__).parent / "fixtures" / "6E_4h_2024Q4_2025Q1.csv"


def _bars(n=None):
    d = pd.read_csv(FIX)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d.iloc[:n].reset_index(drop=True) if n else d


def test_read_one_gives_majors_and_live_ranges():
    r = scan.read_one(_bars(), "4h", scan.daily(_bars()))
    assert r["major"]["state"] in (-1, 0, 1) and r["bars"] == len(_bars())
    for g in r["ranges"]:
        assert g["support"] < g["resistance"] and g["story"] and g["bias"]


def test_news_is_quiet_first_then_only_new():
    d = _bars()
    a = {"6E 4h": scan.read_one(d.iloc[:300].reset_index(drop=True), "4h", scan.daily(d.iloc[:300]))}
    lines, marks = scan.news(a, {})
    assert lines == [] and marks["6E 4h"]["last"] == a["6E 4h"]["last"]
    again, _ = scan.news(a, marks)
    assert again == []                                   # nothing new → nothing sent
    b = {"6E 4h": scan.read_one(d, "4h", scan.daily(d))}
    later, marks2 = scan.news(b, marks)
    assert all(l.split(" · ")[0].endswith("6E 4h") for l in later)
    assert scan.news(b, marks2)[0] == []


def test_morning_brief_ranks_charts_and_lists_events():
    from whs import brief
    d = _bars()
    sc = {"6E 4h": scan.read_one(d, "4h", scan.daily(d))}
    snap = {"generated_local": "Tue 08:12", "health": {}, "instruments": {"DX": {"day": {"change_pct": 0.1}}},
            "strength": {"1d": [{"ccy": "EUR"}, {"ccy": "USD"}]}}
    txt = brief.build(snap, sc, now=d["time"].iloc[-1] + pd.Timedelta(hours=1))
    assert txt.startswith("☀️ Morning brief") and "EUR > USD" in txt and ("Top 3" in txt)
