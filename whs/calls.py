"""Edge Lab 7.4 · the robot grades itself, live.

Every time a chart's decision becomes an entry (BUY ZONE · SELL ZONE · BUY PULLBACKS · SELL RALLIES on 4h / 1h, or
a 15m / 5m entry AT a 4h / 1h edge) the robot logs a CALL: entry = that bar's close, stop = the decision's
invalidation ∓ 0.25 ATR (as in the Edge Lab), managed with the same Adaptive stop, 200-bar cap, 0.05 R costs.
Each hour every open call is re-scored on that timeframe's bars. data/calls.json keeps them all.

Report (Saturday morning + bot /calls): live calls by kind vs what the backtest said they earn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import core, edgelab, swings

ENTRY = {"BUY ZONE", "SELL ZONE", "BUY PULLBACKS", "SELL RALLIES"}
BACKTEST = {"ZONE": ("phase-C reversal (Spring / Shakeout / UTAD)", 1.65), "PULLBACK": ("SOS / SOW then LPS / BU", 0.42)}
STOP_ATR, MIN_RISK_ATR, MAX_RISK_ATR = 0.25, 0.2, 3.0


def kind(action: str) -> str:
    return "ZONE" if action.endswith("ZONE") else "PULLBACK"


def new_calls(changes: list, dec: dict, calls: list) -> list[dict]:
    """changes = cards.changes() output. Returns the calls opened now (also appended to `calls`)."""
    out = []
    have = {(c["key"], c["tf"], c["bar"]) for c in calls}
    for k, old, new in changes:
        d = dec.get(k) or {}
        act = d.get("action")
        if act not in ENTRY or d.get("stop") is None or d.get("price") is None:
            continue
        key, tf = k.split()
        side, e, atr = d["side"], d["price"], d.get("atr") or 0
        stop = d["stop"] - side * STOP_ATR * atr
        risk = (e - stop) * side
        if not atr or risk < MIN_RISK_ATR * atr or risk > MAX_RISK_ATR * atr or (key, tf, d.get("bar")) in have:
            continue
        c = dict(id=f"{k} {d.get('bar')}", key=key, tf=tf, action=act, kind=kind(act), side=side, bar=d.get("bar"),
                 entry=float(e), stop=float(stop), target=d.get("target"), status="open", r=0.0, best=0.0, held=0)
        calls.append(c)
        out.append(c)
    return out


def score(calls: list, data: dict) -> list[dict]:
    """Re-score every open call. data = {(key, tf): DataFrame of bars}. Returns the calls that closed now."""
    closed = []
    by = {}
    for c in calls:
        if c["status"] == "open":
            by.setdefault((c["key"], c["tf"]), []).append(c)
    for (key, tf), cs in by.items():
        d = data.get((key, tf))
        if d is None or len(d) < 50:
            continue
        t = pd.to_datetime(d["time"], utc=True)
        o, h, l, c_, v = (d[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
        mins = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}[tf]
        eff = core.effort(v, t, mins)
        led = swings.ledger(o, h, l, c_, eff)
        D = edgelab._prep(o, h, l, c_, eff, led)
        for c in cs:
            i0 = int(np.searchsorted(t.values, pd.Timestamp(c["bar"]).to_datetime64()))
            if i0 >= len(t) or t.iloc[i0] != pd.Timestamp(c["bar"]):
                if i0 < 1:
                    c.update(status="void")
                continue
            r, held, best = edgelab.adaptive(D, i0, c["side"], c["entry"], c["stop"])
            c.update(r=round(float(r) - edgelab.COST_R, 2), best=round(float(best), 2), held=int(held))
            if held >= edgelab.MAX_BARS or i0 + held < len(t) - 1:          # stopped out, or the time cap
                c.update(status="closed", closed=t.iloc[min(i0 + held, len(t) - 1)].isoformat())
                closed.append(c)
    return closed


def report(calls: list, now, days: int = 7) -> str:
    now = pd.Timestamp(now)
    done = [c for c in calls if c["status"] == "closed"]
    week = [c for c in done if pd.Timestamp(c.get("closed", c["bar"])) >= now - pd.Timedelta(days=days)]
    opn = [c for c in calls if c["status"] == "open"]
    out = ["<b>Robot self-grade</b>  — its own calls, scored live with the Adaptive stop"]

    def line(cs, label):
        if not cs:
            return f"{label}: none yet"
        R = np.array([c["r"] for c in cs])
        return f"{label}: {len(cs)} calls · win {int(round(100 * (R > 0).mean()))}% · avg {R.mean():+.2f}R · total {R.sum():+.1f}R"

    out.append(line(week, f"Last {days} days"))
    out.append(line(done, "All time"))
    for kd, (name, bt) in BACKTEST.items():
        cs = [c for c in done if c["kind"] == kd]
        if cs:
            out.append(f"• {name}: live {np.mean([c['r'] for c in cs]):+.2f}R on {len(cs)} vs backtest {bt:+.2f}R")
    for tfs, label in ((("4h", "1h"), "4h / 1h"), (("15m", "5m"), "15m / 5m at an edge")):
        cs = [c for c in done if c["tf"] in tfs]
        if cs:
            out.append(f"• {label}: {len(cs)} calls, avg {np.mean([c['r'] for c in cs]):+.2f}R")
    if opn:
        out.append("<b>Open</b>  " + " · ".join(f"{c['key']} {c['tf']} {c['action']} {c['r']:+.1f}R" for c in opn[-6:]))
    last = sorted(done, key=lambda c: c.get("closed", ""))[-6:]
    if last:
        out.append("<b>Last closed</b>")
        out += [f"• {c['key']} {c['tf']} {c['action']} {pd.Timestamp(c['bar']).tz_convert('Africa/Nairobi').strftime('%d %b %H:%M')}"
                f" → {c['r']:+.2f}R (best {c['best']:+.1f}R)" for c in reversed(last)]
    out.append("<i>Live numbers need ~30 calls per kind before they mean much — the backtest stays the guide until then.</i>")
    return "\n".join(out)
