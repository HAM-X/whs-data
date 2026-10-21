import numpy as np

from whs import cards, decide, vsa


def _rec(phase, events, outcome=0, state=0):
    return {"major": {"state": state, "protected": 1.1}, "ranges": [{"id": "2026-10-14T15:00:00+00:00", "name": "Accumulation",
            "phase": phase, "outcome": outcome, "support": 1.10, "resistance": 1.12, "events": events}]}


def test_decisions():
    assert decide.chart(_rec("B", []))["action"] == "WAIT"
    ev = [["t", "SC", "confirmed", 1.10], ["t", "Spring", "confirmed", 1.098]]
    d = decide.chart(_rec("C", ev, 1))
    assert d["action"] == "BUY ZONE" and "1.098" in d["invalid"]
    d = decide.chart(_rec("C", [["t", "UTAD", "confirmed", 1.123]], -1))
    assert d["action"] == "SELL ZONE"
    assert decide.chart(_rec("D", [], 1))["action"] == "BUY PULLBACKS"
    assert decide.chart({"major": {"state": -1, "protected": 2.0}, "ranges": []})["action"] == "TREND ↓"
    b = [{"ccy": "EUR", "title": "CPI", "t": "2026-10-21T09:00:00+00:00"}]
    assert decide.chart(_rec("C", ev, 1), b)["action"] == "NO NEW ENTRIES"
    for r in (_rec("A", []), _rec("C", ev, 1), {"major": {"state": 1, "protected": 1.0}, "ranges": []}):
        assert decide.chart(r)["short"]


def test_alerts_only_on_actionable_changes():
    dec = {"6E 4h": {"action": "BUY ZONE"}, "6B 4h": {"action": "TREND ↑"}, "6A 4h": {"action": "WAIT"}, "_today": "x"}
    seen = {"6E 4h": "WAIT", "6B 4h": "WAIT", "6A 4h": "SELL ZONE"}
    assert cards.changes(dec, seen) == [("6E 4h", "WAIT", "BUY ZONE"), ("6A 4h", "SELL ZONE", "WAIT")]
    assert cards.changes(dec, {}) == []


def test_vsa_symbols_and_bar():
    assert [vsa.sym(x) for x in (2.0, 1.2, 1.0, 0.8, 0.5)] == [">>", ">", "=", "<", "<<"]
    n = 40
    c = 1.0 + np.cumsum(np.full(n, 0.001))
    o = c - 0.0008
    h, l = c + 0.0001, o - 0.0001
    eff = np.full(n, 1.0)
    eff[-1] = 2.0
    b = vsa.bar(o, h, l, c, eff, n - 1)
    assert b["dir"] == 1 and b["v"] == ">>" and b["code"] == "D>"        # more effort, normal spread, closes high
    eff[-1] = 0.5
    o2, c2 = o.copy(), c.copy()
    c2[-1] = c2[-2] + 0.00005                                             # tiny up bar on low effort
    assert vsa.bar(o2, np.maximum(h, c2), l, c2, eff, n - 1)["code"] == "D<"
