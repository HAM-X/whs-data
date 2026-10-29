import pandas as pd

from jobs import audit


def test_audit_scores_claims_against_later_bars():
    t = pd.date_range("2026-10-01", periods=40, freq="h", tz="UTC")
    c = [1.0 + 0.001 * i for i in range(40)]                       # a steady rise
    b = pd.DataFrame(dict(time=t, open=c, high=[x + 0.0005 for x in c], low=[x - 0.0005 for x in c], close=c, volume=1))
    snaps = [{"6E 1h": dict(bar=t[5].isoformat(), atr=0.001, action="TREND ↑", verdict="SUPPLY in control", flow=1)},
             {"6E 1h": dict(bar=t[6].isoformat(), atr=0.001, action="TREND ↓", verdict="BALANCED", flow=0)},
             {"_at": "x"}]
    sc = audit.score(audit.claims(snaps), {("6E", "1h"): b})
    assert sc["res"]["decision"] == [1, 1, 0]          # TREND ↑ right, TREND ↓ wrong
    assert sc["res"]["verdict"] == [0, 1, 0]
    assert sc["res"]["flow"] == [1, 0, 0]
    assert "50% right" in audit.text(sc, 7, 3)
