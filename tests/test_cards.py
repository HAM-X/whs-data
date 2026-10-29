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


def test_dead_setup_is_never_an_entry():
    """Audit 29 Oct: 6J 15m BUY PULLBACKS was alerted while price sat below its own invalidation."""
    r = dict(_rec("D", [], 1), close=1.115)                     # SOS done, but price is back below R 1.12
    d = decide.chart(r)
    assert d["action"] == "WAIT" and d["failed"] == "BUY PULLBACKS" and "below 1.12" in d["do"]
    assert "side" not in d and "stop" not in d
    assert decide.chart(dict(_rec("D", [], 1), close=1.125))["action"] == "BUY PULLBACKS"
    ev = [["t", "SC", "confirmed", 1.10], ["t", "Spring", "confirmed", 1.098]]
    assert decide.chart(dict(_rec("C", ev, 1), close=1.097))["action"] == "WAIT"     # closed under the spring low
    assert decide.chart(dict(_rec("C", [["t", "UTAD", "confirmed", 1.123]], -1), close=1.124))["action"] == "WAIT"
    assert decide.chart(dict(_rec("D", [], -1), close=1.09))["action"] == "SELL RALLIES"


def test_alerts_only_for_fresh_phase_c_setups():
    dec = {"6E 4h": {"action": "BUY ZONE", "setup": True}, "6N 4h": {"action": "BUY ZONE", "setup": False},
           "6B 4h": {"action": "TREND ↑"}, "6C 4h": {"action": "BUY PULLBACKS"},
           "6A 4h": {"action": "WAIT", "failed": "SELL ZONE"}, "6J 4h": {"action": "WAIT"}, "_today": "x"}
    seen = {"6E 4h": "WAIT", "6N 4h": "WAIT", "6B 4h": "WAIT", "6C 4h": "WAIT", "6A 4h": "SELL ZONE", "6J 4h": "SELL RALLIES"}
    assert cards.changes(dec, seen) == [("6E 4h", "WAIT", "BUY ZONE"), ("6A 4h", "SELL ZONE", "failed")]
    assert cards.changes(dec, {}) == []


def test_entry_window_closes_after_two_bars():
    ev = [["2026-10-14T15:00:00+00:00", "SC", "confirmed", 1.10], ["2026-10-20T10:00:00+00:00", "Spring", "confirmed", 1.098]]
    r = dict(_rec("C", ev, 1), close=1.105, last="2026-10-20T12:00:00+00:00")
    d = decide.chart(r, tf="1h")
    assert d["action"] == "BUY ZONE" and d["setup"]
    d = decide.chart(dict(r, last="2026-10-20T14:00:00+00:00"), tf="1h")
    assert d["action"] == "WAIT" and d["used"] == "BUY ZONE" and "4 bars ago" in d["do"]
    assert decide.chart(dict(r, last="2026-10-20T14:00:00+00:00"), tf="4h")["setup"]      # 0.5 bar on 4h
    t = [["2026-10-14T15:00:00+00:00", "SC", "confirmed", 1.10], ["2026-10-20T10:00:00+00:00", "Test", "confirmed", 1.101]]
    assert not decide.chart(dict(_rec("C", t, 1), close=1.105, last="2026-10-20T11:00:00+00:00"), tf="1h")["setup"]


def test_board_flow_replaces_vote():
    assert cards.direction({"6E 1h": {"flow": 1}}, "6E")[0] == 1
    assert cards.direction({"6E 1h": {"flow": 0}, "6E 4h": {"action": "TREND ↑", "major": 1}}, "6E") == (0, "◆ NONE no tested read")


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


def test_ltf_alerts_only_for_entries_at_an_htf_edge_once():
    nest = {"side": 1, "tf": "4h", "line": "x"}
    dec = {"6E 15m": {"action": "BUY ZONE", "strong": True, "nest": nest}, "6B 15m": {"action": "BUY ZONE", "strong": False, "nest": nest},
           "6A 5m": {"action": "WAIT", "strong": False, "nest": nest}}
    seen = {"6E 15m": "WAIT", "6B 15m": "WAIT", "6A 5m": "BUY ZONE"}
    assert cards.changes(dec, seen) == [("6E 15m", "WAIT", "BUY ZONE at the 4h edge")]
    assert cards.changes(dec, {"6E 15m": "BUY ZONE ✅"}) == []           # already alerted


def test_nested_edges():
    d4 = {"action": "WAIT", "levels": [1.10, 1.12]}
    assert cards.nested(1.1005, d4, None)["side"] == 1
    assert cards.nested(1.0960, d4, None)["side"] == 1                 # spring territory below support
    assert cards.nested(1.1195, d4, None)["side"] == -1
    assert cards.nested(1.1100, d4, None)["side"] == 0
    assert cards.nested(1.2000, d4, None)["side"] == 0


def test_spring_number_and_phase_d_still_make_the_entry():
    """Replay 29 Oct: 'Spring #3' (and phase D on the same bar) never became a BUY ZONE — 5 of 127 events did."""
    ev = [["2026-10-14T15:00:00+00:00", "SC", "confirmed", 1.10], ["2026-10-20T10:00:00+00:00", "Spring #3", "confirmed", 1.098],
          ["2026-10-20T10:00:00+00:00", "ST", "confirmed", 1.099]]
    r = dict(_rec("D", ev, 0), close=1.104, last="2026-10-20T11:00:00+00:00")
    d = decide.chart(r, tf="1h")
    assert d["action"] == "BUY ZONE" and d["setup"] and d["stop"] == 1.098 and d["side"] == 1
    assert decide.chart(dict(r, close=1.097), tf="1h")["action"] != "BUY ZONE"          # already below the spring
    assert decide.chart(dict(r, last="2026-10-20T15:00:00+00:00"), tf="1h")["action"] != "BUY ZONE"   # window closed
    u = [["2026-10-14T15:00:00+00:00", "BC", "confirmed", 1.13], ["2026-10-20T10:00:00+00:00", "UTAD #3", "confirmed", 1.124]]
    d = decide.chart(dict(_rec("D", u, 0), close=1.12, last="2026-10-20T10:00:00+00:00"), tf="1h")
    assert d["action"] == "SELL ZONE" and d["setup"] and d["side"] == -1


def test_candidate_spring_is_the_poe1_moment_and_failed_is_not():
    ev = [["2026-10-14T15:00:00+00:00", "SC", "confirmed", 1.10], ["2026-10-20T10:00:00+00:00", "Spring #3", "candidate", 1.098]]
    r = dict(_rec("C", ev, 0), close=1.103, last="2026-10-20T10:00:00+00:00")
    assert decide.chart(r, tf="1h")["action"] == "BUY ZONE"
    ev[1][2] = "failed"
    assert decide.chart(dict(r, ranges=[dict(r["ranges"][0], events=ev)]), tf="1h")["action"] != "BUY ZONE"
