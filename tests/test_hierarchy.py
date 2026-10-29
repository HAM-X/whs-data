import pandas as pd

from whs import hierarchy as hz


def ev(n, r, p, b, st="confirmed", s=0):
    return dict(n=n, r=r, p=p, b=b, k=b, st=st, s=s)


P4 = dict(id="2026-10-01T00:00:00+00:00", support=3955.0, resistance=4157.0, phase="B", outcome=0, left=False,
          ev2=[ev("BC", "BC", 4157.0, "2026-10-01T00:00:00+00:00"), ev("AR", "AR", 3955.0, "2026-10-03T00:00:00+00:00"),
               ev("ST", "ST", 4150.0, "2026-10-08T00:00:00+00:00")])
C1 = dict(id="2026-10-20T00:00:00+00:00", support=4110.0, resistance=4166.0, phase="D", outcome=-1, left=False,
          ev2=[ev("BC", "BC", 4160.0, "2026-10-20T00:00:00+00:00"), ev("AR", "AR", 4110.0, "2026-10-20T06:00:00+00:00"),
               ev("UTAD #3", "Spring", 4166.0, "2026-10-22T00:00:00+00:00", s=-1), ev("SOW", "BO", 4108.0, "2026-10-23T00:00:00+00:00", s=-1)])


def recs():
    return {("GC", "4h"): {"ranges": [P4]}, ("GC", "1h"): {"ranges": [C1]}}


def test_child_at_the_parent_top_reads_as_minor_distribution():
    out = hz.describe("GC", recs(), {("GC", "1h"): "SUPPLY in control"}, 4115.0)
    t = out["text"]
    assert "at its top (UT / UTAD level)" in t and "own phase C: UTAD #3 4166" in t and "resolved DOWN" in t
    assert "own phase C (UTAD #3 4166) is done" in t and "supply is present" in t and "resolved DOWN, back into the 4h range" in t and "4h middle 4056" in t and "support 3955" in t
    assert "35% (chance 40%, n 40)" in t                                # the measured odds are always printed
    ch = out["frames"][0]["children"][0]
    assert ch["zone"] == 1 and ch["back_in"] and ch["outcome"] == -1


def test_child_must_form_after_the_parent_ar_and_sit_at_an_edge():
    early = dict(C1, id="2026-10-02T00:00:00+00:00")
    middle = dict(C1, id="2026-10-20T00:00:00+00:00", support=4040.0, resistance=4070.0)
    assert hz.children(P4, {"ranges": [early, middle]}) == []


def test_edges_confirmed_and_touch_episodes():
    assert hz.edges_confirmed(P4)
    assert not hz.edges_confirmed(dict(P4, ev2=P4["ev2"][1:]))      # no climax confirmed
    assert not hz.edges_confirmed(dict(P4, phase="E"))
    t = pd.date_range("2026-10-29 10:00", periods=6, freq="15min", tz="UTC")
    b = pd.DataFrame(dict(time=t, high=[4100, 4156.5, 4158, 4100, 4060, 4157.0], low=[4090, 4140, 4140, 4050, 4040, 4120],
                          close=[4095, 4150, 4150, 4060, 4050, 4150]))
    armed = {}
    ts = hz.touches("GC", {("GC", "4h"): {"ranges": [P4]}}, b, t[0] - pd.Timedelta(minutes=15), {("GC", "4h"): 20.0}, armed)
    assert [x["t"][11:16] for x in ts] == ["10:15", "11:15"]       # one per episode, re-armed after the pullback
    msg = hz.touches_text(ts)
    assert msg.count("4h") >= 1 and "B setups at best" in msg and "4h edges: first touches reached the middle first 40% (chance 33%, n 38)" in msg


def test_corrections_need_two_runs():
    f_old = {"a": dict(key="6E", tf="1h", rid="r1", fam="Spring", name="Spring #3", price=1.1135, bar="2026-10-20T10:00:00+00:00", side=1)}
    renamed = {"b": dict(key="6E", tf="1h", rid="r1", fam="B_heavy", name="mSOW", price=1.1135, bar="2026-10-20T10:00:00+00:00", side=1)}
    new, corr, pend = hz.diff_facts(f_old, renamed, {}, {"r1"})
    assert corr == [] and "a" in pend                                # first sighting: pending only
    new, corr, pend = hz.diff_facts(f_old, renamed, pend, {"r1"})
    assert len(corr) == 1 and corr[0]["now"] == "mSOW"
    assert "was phase B" in hz.correction_text(corr[0])
    back = dict(f_old)                                               # the label came back: no correction
    new, corr, pend = hz.diff_facts(f_old, back, {"a": f_old["a"]}, {"r1"})
    assert corr == [] and pend == {}


def test_facts_skip_lower_than_15m_and_new_fact_text():
    r = {("6E", "5m"): {"ranges": [C1]}, ("6E", "1h"): {"ranges": [C1]}}
    f = hz.facts(r)
    assert all(v["tf"] == "1h" for v in f.values()) and len(f) == 4
    s = hz.fact_text(next(v for v in f.values() if v["fam"] == "Spring"))
    assert "provisional" in s and "stand 49%" in s


def test_forward_claims_are_scored_against_chance():
    c = hz.log_claim("child", "GC", "4h", -1, 4115.0, 4056.0, 4166.0, "2026-10-29T10:00:00+00:00")
    assert c["chance"] == round(51 / (51 + 59), 3)
    assert hz.log_claim("child", "GC", "4h", -1, 4115.0, 4130.0, 4166.0, "2026-10-29T10:00:00+00:00") is None
    t = pd.date_range("2026-10-29 10:15", periods=4, freq="15min", tz="UTC")
    b = pd.DataFrame(dict(time=t, high=[4120, 4118, 4090, 4070], low=[4100, 4080, 4050, 4040], close=[4110, 4085, 4060, 4045]))
    log = hz.score_claims([c, dict(c, inv=4105.0, t="2026-10-29T10:00:00+00:00")], {"GC": b})
    assert [x["status"] for x in log] == ["hit", "miss"]
    assert "reached the range middle first 50% of 2" in hz.claims_report(log)


def test_sow_backs_the_phase_c_test():
    f = hz.facts({("GC", "1h"): {"ranges": [C1]}})
    sow = next(v for v in f.values() if v["fam"] == "BO")
    s = hz.fact_text(sow, f)
    assert "it backs the UTAD #3 4166" in s and "such labels stand 77% (was 49%)" in s
