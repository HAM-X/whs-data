import numpy as np

from whs import campaign as cp


def ev(n, r, s, p, k, ef=1.0, st="confirmed"):
    return dict(n=n, r=r, s=s, p=p, k=k, ef=ef, st=st)


BASE = [ev("BC", "BC", 0, 4157, "2026-10-01"), ev("AR", "AR", 0, 3955, "2026-10-03")]


def rng(extra, kind="BC"):
    return dict(S=3955.0, R=4157.0, kind=kind, phase="C", events=BASE + extra)


def test_unbacked_utad_is_provisional_and_the_sow_makes_it_count():
    r1 = cp.read(rng([ev("UTAD", "Spring", -1, 4166, "2026-10-22", ef=0.7)]), 4100)
    it = next(i for i in r1["items"] if i["key"] == "c_provisional")
    assert it["w"] == 0 and "provisional" in it["text"]
    r2 = cp.read(rng([ev("UTAD", "Spring", -1, 4166, "2026-10-22", ef=0.7), ev("SOW", "BO", -1, 3950, "2026-10-24", ef=1.4)]), 3990)
    assert r2["side"] == -1 and r2["label"] == "DISTRIBUTION" and r2["score"] <= -5
    assert "UTAD high" in r2["invalid"] and "4166" in r2["invalid"]
    assert "LPSY" in r2["watch"] and "sow_heavy" in {i["key"] for i in r2["items"]}


def test_effort_grades_the_secondary_test_and_heavy_pushes():
    r = cp.read(rng([ev("ST", "ST", 1, 3960, "2026-10-05", ef=0.6), ev("mSOW", "B_heavy", 1, 3958, "2026-10-08", ef=1.5)], kind="SC"), 4000)
    t = {i["key"]: i for i in r["items"]}
    assert t["st_low"]["side"] == 1 and "drying up" in t["st_low"]["text"]
    assert t["bheavy_bottom"]["side"] == -1 and t["bheavy_bottom"]["replay"] == "✓"


def test_balanced_read_names_both_edges():
    r = cp.read(dict(S=1.0, R=1.1, kind=None, phase="B", events=[]), 1.05)
    assert r["side"] == 0 and "1.1" in r["confirm"] and r["chance_up"] == 0.5


def test_swing_evidence_follows_the_nine_tests():
    sw = [dict(d=-1, sp=1.10, ep=1.00, eff=1.4), dict(d=1, sp=1.00, ep=1.08, eff=1.3),
          dict(d=-1, sp=1.08, ep=1.03, eff=0.7), dict(d=1, sp=1.03, ep=1.09, eff=1.2)]
    items = {i["key"]: i for i in cp.evidence(dict(events=[]), sw)}
    assert items["supports"]["side"] == 1 and items["tops"]["side"] == 1 and items["contract_dn"]["side"] == 1
    assert items["activity"]["side"] == 1


def test_pnf_count_and_objectives():
    close = np.array([10, 9, 8, 9, 10, 9, 8, 9, 10, 9, 8, 9, 10, 11, 12], float)
    pc = cp.pnf_count(close, 1.0)
    assert pc["cols"] >= 6 and 8 <= pc["line"] <= 11
    rd = dict(side=1, inv_p=7.5, S=8.0, R=10.0)
    o = cp.objectives(rd, close, 2.0)                          # box = 1.0
    assert o["cons"] == 7.5 + o["cols"] and o["aggr"] > o["cons"]


def test_touch_grades():
    assert cp.touch_grade(100, 101, 99.6, 100.8, 0.7, 2.0, -1)[1] == 1            # no supply at the support
    assert "absorption" in cp.touch_grade(100, 101, 97, 100.5, 1.6, 2.0, -1)[0]
    assert cp.touch_grade(100, 100.2, 97, 97.3, 1.4, 2.0, -1)[1] == -1            # supply in control
    assert cp.touch_grade(100, 100.5, 99.5, 100, 1.0, 2.0, -1)[1] == 0


def test_quiet_alerts_announce_a_read_change_only_when_it_holds(tmp_path, monkeypatch):
    import pandas as pd
    from jobs import download
    from whs import cards
    monkeypatch.setattr(download, "DATA", tmp_path)
    monkeypatch.setattr(cards, "LAST_RECS", {}, raising=False)
    monkeypatch.setattr(cards, "LAST_ATRS", {}, raising=False)
    def dec(label, side, price):
        st = dict(id="r1", label=label, side=side, grade="lean" if side else "none", S=3955.0, R=4157.0, chance=0.5, inv=None, phase="B")
        return {"_camp": {"GC": {"4h": st}}, "_story": {"GC": "📖 <b>GC · the story</b>\nline"}, "GC 1h": {"price": price}}
    state, now = {}, pd.Timestamp("2026-10-30T10:00:00Z")
    assert download.hierarchy_alerts(now, dec("BALANCED", 0, 4050), {}, state) == []          # first run records
    assert download.hierarchy_alerts(now, dec("leaning DISTRIBUTION", -1, 4050), {}, state) == []   # pending
    m = download.hierarchy_alerts(now, dec("leaning DISTRIBUTION", -1, 4050), {}, state)
    assert len(m) == 1 and "the read changed" in m[0] and "leaning DISTRIBUTION" in m[0]
    m = download.hierarchy_alerts(now, dec("leaning DISTRIBUTION", -1, 3940), {}, state)
    assert len(m) == 1 and "confirmation" in m[0] and "below 3955" in m[0]
    assert download.hierarchy_alerts(now, dec("leaning DISTRIBUTION", -1, 3930), {}, state) == []   # once only
    import json
    log = json.loads((tmp_path / "htr_log.json").read_text())
    assert any(c["kind"] == "campaign" and c["side"] == -1 and c["wick"] for c in log)
