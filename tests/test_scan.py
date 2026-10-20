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


def test_trade_desk_stop_translate_and_grade():
    from whs import desk
    # stop: structure − 0.5 ATR(1h) − 1 tick, rounded down (Roman ex. 1: BU low 1.0896 → 1.0889 with ATR1h 0.0013)
    assert abs(desk._round_away(1.0896 - (0.5 * 0.0013 + 0.00005), 0.00005, 1) - 1.0889) < 1e-9
    # translate: 6E same direction, 6J inverted (Roman ex. A / B)
    assert abs(desk.to_mt5("6E", 1.0889, 1.0909, 1.0874) - 1.08541) < 1e-4
    assert abs(desk.to_mt5("6J", 0.00669, 0.00671, 149.60) - 150.047) < 0.01
    assert desk.lots("6E", 1.0909, 1.0889, 1.0909, 1.0874, 500) == 2.5
    assert desk.grade_of(9, "BU") == "B"          # A stays capped at B until Edge Lab proves the setup
    assert desk.grade_of(5, "BU") == "C" and desk.grade_of(3, "BU") == "F"


def test_panel_heat_and_persistence():
    from whs import panel
    rho = {"6E": {"6B": 0.78, "6C": 0.55}, "6B": {"6E": 0.78, "6C": 0.5}, "6C": {"6E": 0.55, "6B": 0.5}}
    assert abs(panel.effective_heat({"6E": 0.5, "6B": 0.5, "6C": 0.5}, rho) - 1.29) < 0.01
    st = {w: [{"ccy": c} for c in order] for w, order in
          (("1h", "JPY CHF EUR GBP USD AUD NZD CAD".split()), ("1d", "JPY EUR CHF GBP USD AUD CAD NZD".split()),
           ("1w", "CHF JPY EUR USD GBP AUD NZD CAD".split()))}
    p = panel.persistence(st)
    assert p["JPY"].startswith("strong") and p["CAD"].startswith("weak")


def test_edge_lab_proven_setups_may_grade_a():
    from whs import desk
    assert desk.grade_of(9, "Spring") == "A" and desk.grade_of(9, "UTAD") == "A" and desk.grade_of(9, "Shakeout") == "A"
    assert desk.grade_of(9, "SOS bar") == "A" and desk.grade_of(9, "SOW bar") == "A"
    assert desk.grade_of(9, "Test of the spring") == "B" and desk.grade_of(9, "LPS") == "B"
