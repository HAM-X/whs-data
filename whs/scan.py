"""Stage 3 · the robot reads the market by itself (no clicks).

For each future and timeframe:
  1. effort, the swing ledger (minor swings) and major swings (whs/degrees.py)
  2. AUTOMATIC RANGES by the change of character (whs/autorange.py, the same rule as WHS Range Auto): after a
     climax the automatic reaction is bigger than every earlier reaction of the trend in its direction —
       ① = the climax, ② = the AR (extends while its pullbacks hold)
     each read live with the two-sided range reader (whs/ranges.py) and its Wyckoff Story (whs/story.py)
  3. what changed since the last run becomes Telegram alerts (major swing events, range events, phases,
     breakouts confirmed / back in the range / left the range)

Everything is a live walk: the read at the last bar uses only bars up to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import autorange, core, degrees, desk, ranges, story, swings, vsa
from .config import EXCHANGE_TZ, FUTURES

TF_MIN = {"1h": 60, "4h": 240, "15m": 15, "5m": 5, "30m": 30, "1d": 1440, "1w": 10080}
RECENT = {"1h": 360, "4h": 300, "15m": 400, "5m": 500, "30m": 400, "1d": 250, "1w": 150}       # look for ranges whose climax is within this many bars of now
REACT_N = degrees.AR_N + degrees.MIN_BOX


def daily(d: pd.DataFrame) -> pd.DataFrame:
    ct = d["time"].dt.tz_convert(EXCHANGE_TZ) + pd.Timedelta(hours=7)   # a bar from 17:00 CT belongs to the next day
    return d.assign(day=ct.dt.date).groupby("day").agg(time=("time", "last"), high=("high", "max"), low=("low", "min"))


def htf_at(htf_bars: pd.DataFrame, t_utc: pd.Timestamp) -> int:
    """Higher-timeframe structure trend known at time t (the last COMPLETED higher bar before it)."""
    if htf_bars is None or len(htf_bars) < 15:
        return 0
    tr = ranges.htf_trend(htf_bars["high"], htf_bars["low"])
    ht = pd.to_datetime(htf_bars["time"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    k = np.searchsorted(ht, pd.Timestamp(t_utc).as_unit("ns").value, side="left") - 2
    return int(tr[max(k, 0)])


def auto_anchors(h, l, mr: degrees.MajorResult, n: int, recent: int) -> list[tuple[int, int, str, bool]]:
    """(① climax extreme bar, ② reaction extreme bar, kind, box still live) for each recent climax-stopped box."""
    out = []
    for j, m in enumerate(mr.majors):
        if m.direction != 0 or m.start_i < n - recent or j == 0:
            continue
        sc = mr.majors[j - 1].direction == -1          # the box after a down leg starts at a selling climax
        a1 = m.start_i
        hi_end = min(n - 1, a1 + REACT_N)
        if hi_end <= a1:
            continue
        seg = np.arange(a1 + 1, hi_end + 1)
        a2 = int(seg[np.argmax(h[seg])]) if sc else int(seg[np.argmin(l[seg])])
        out.append((a1, a2, "SC" if sc else "BC", m.end_kind == "live"))
    return out


def read_one(d: pd.DataFrame, tf: str, htf_bars: pd.DataFrame | None, ctx: dict | None = None) -> dict:
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    t = d["time"]
    eff = core.effort(v, t, TF_MIN[tf])
    n = len(c)
    led = swings.ledger(o, h, l, c, eff)
    mr = degrees.majors(o, h, l, c, eff, led)
    live = mr.majors[-1] if mr.majors else None
    out = {"bars": n, "last": t.iloc[-1].isoformat(), "close": float(c[-1]),
           "major": None, "major_events": [], "ranges": [], "plans": []}
    if live is not None:
        out["major"] = {"state": int(live.direction), "since": t.iloc[live.start_i].isoformat(),
                        "protected": None if np.isnan(mr.protected[-1]) else float(mr.protected[-1]),
                        "box": None if live.direction else [float(live.lo), float(live.hi)]}
    out["major_events"] = [(t.iloc[b].isoformat(), txt) for b, txt in mr.events]
    try:
        chain = autorange.chain(o, h, l, c, eff, v, led, htf_at=lambda i: htf_at(htf_bars, t.iloc[i]))
    except Exception as e:          # one odd chart must never stop the scan
        print(f"[auto range error] {tf}: {type(e).__name__}: {e}")
        chain = []
    for trig, rr in chain[-3:]:
        a1 = trig["a1"]
        a2 = next(e.bar for e in rr.events if e.name == "AR")
        if a1 < n - RECENT[tf]:
            continue
        ph = rr.phases[-1][0] if rr.phases else "A"
        failed = any(e.role in ("SC", "BC") and e.state == "failed" for e in rr.events)
        if not rr.events or rr.left or failed:
            continue                 # already over · the reaction kept going / the climax gave way (no range)
        st = story.build(o, h, l, c, eff, rr, led)
        if ctx:
            try:
                out["plans"] += desk.plans(ctx["key"], tf, h, l, c, rr, st, ctx["atr1h"], ctx["atr4h"], ctx["tick"],
                                           htf_at(htf_bars, t.iloc[-1]), ctx.get("strength"), late=ctx.get("late", False))
            except Exception as e:
                print(f"[desk error] {ctx['key']} {tf}: {type(e).__name__}: {e}")
        out["ranges"].append({
            "id": t.iloc[a1].isoformat(), "kind": rr.kind, "a2": t.iloc[a2].isoformat(),
            "phase": ph, "outcome": rr.outcome, "left": bool(rr.left),
            "name": rr.name or f"{rr.bull_name} / {rr.bear_name}",
            "support": float(min(rr.support, rr.resistance)), "resistance": float(max(rr.support, rr.resistance)),
            "events": [(t.iloc[e.known].isoformat(), e.name, e.state, float(e.price)) for e in rr.events if not e.hidden],
            # full event detail for the range hierarchy (whs/hierarchy.py): extreme bar, known bar, role, side
            "ev2": [dict(b=t.iloc[e.bar].isoformat(), k=t.iloc[e.known].isoformat(), n=e.name, r=e.role, s=int(e.side),
                         st=e.state, p=float(e.price), ef=None if np.isnan(e.effort) else round(float(e.effort), 2))
                    for e in rr.events if not e.hidden],
            "alerts": [(t.iloc[b].isoformat(), txt) for b, txt in rr.alerts],
            "story": [f"#{k} {p}: {txt}" for k, p, txt in st.steps], "bias": st.bias,
        })
    # the swings and the effort/result verdict the campaign read uses (same as the replay that measured it)
    try:
        vs = vsa.read(o, h, l, c, eff, led, n_swings=4, n_bars=3)
        sws = [s for s in led.swings if s.confirm_i is not None and s.confirm_i <= n - 1][-6:]
        out["sw"] = [dict(d=int(s.direction), sp=float(s.start_p), ep=float(s.end_p), bars=int(s.bars), eff=round(float(s.avg_effort), 3),
                          code=x["code"]) for s, x in zip(sws[-len(vs["swings"]):] if vs["swings"] else [], vs["swings"])]
        out["verdict"] = vs["verdict"]
    except Exception as e:
        print(f"[scan swings error] {tf}: {type(e).__name__}: {e}")
        out["sw"], out["verdict"] = [], ""
    return out


def _atr_now(d):
    if d is None or len(d) < 20:
        return None
    return float(core.atr(d["high"].to_numpy(float), d["low"].to_numpy(float), d["close"].to_numpy(float))[-1])


def scan(h1: dict[str, pd.DataFrame], h4: dict[str, pd.DataFrame], strength: list | None = None,
         late: dict | None = None) -> dict:
    """strength: snapshot['strength']['1d'] rows (ccy, vs_usd_pct) — for the Trade Desk grade."""
    res = {}
    vs = {x["ccy"]: x.get("vs_usd_pct") for x in (strength or [])}
    for key in FUTURES:
        if FUTURES[key].get("context"):
            continue
        for tf, src in (("1h", h1), ("4h", h4)):
            d = src.get(key)
            if d is None or len(d) < 200:
                continue
            d = d.reset_index(drop=True)
            htf_bars = h4.get(key) if tf == "1h" else daily(d)
            a1, a4 = _atr_now(h1.get(key)), _atr_now(h4.get(key))
            ccy = desk.CCY.get(key)
            srank = vs.get(ccy) if ccy else (-vs["USD"] if key == "GC" and vs.get("USD") is not None else None)
            ctx = dict(key=key, atr1h=a1, atr4h=a4, tick=FUTURES[key]["tick"], strength=srank,
                       late=bool((late or {}).get(key))) if a1 and a4 else None
            try:
                res[f"{key} {tf}"] = read_one(d, tf, htf_bars, ctx)
            except Exception as e:
                print(f"[scan error] {key} {tf}: {type(e).__name__}: {e}")
    return res


ICON = {"confirmed": "●", "still going on": "↺", "left the range": "⚑"}


def _fmt_time(iso: str) -> str:
    return pd.Timestamp(iso).tz_convert("Africa/Nairobi").strftime("%a %d %b %H:%M")


def news(scan_now: dict, last_seen: dict) -> tuple[list[str], dict]:
    """Alert lines for everything that happened after the last scan (per symbol/timeframe), and the new marks.
    last_seen[key] = {"last": time of the last bar scanned, "ranges": ids of the ranges already announced}."""
    lines, marks = [], {}
    for key, r in scan_now.items():
        seen = last_seen.get(key)
        marks[key] = {"last": r["last"], "ranges": [g["id"] for g in r["ranges"]],
                      "plans": sorted(set(seen.get("plans", []) if seen else []) | {p["id"] for p in r.get("plans", [])})[-50:]}
        if not seen:                # first scan of this chart: remember, do not flood
            continue
        t0, known = seen["last"], set(seen.get("ranges", []))
        for when, txt in r["major_events"]:
            if when > t0:
                lines.append(f"{key} · {txt}")
        for g in r["ranges"]:
            rid = _fmt_time(g["id"])[4:]
            if g["id"] not in known:
                lines.append(f"📐 {key} · new range from the {rid} climax ({g['kind']}) · {g['support']:.5g} – {g['resistance']:.5g}")
            for when, txt in g["alerts"]:
                if when > t0:
                    icon = next((v for k, v in ICON.items() if k in txt), "📐")
                    lines.append(f"{icon} {key} · range from {rid} · {txt}")
    return lines, marks


def fresh_plans(scan_now: dict, last_seen: dict) -> list[dict]:
    """Trade plans not announced before (grade F is never sent). Silent on a chart's first scan."""
    out = []
    for key, r in scan_now.items():
        seen = last_seen.get(key)
        if not seen:
            continue
        done = set(seen.get("plans", []))
        out += [p for p in r.get("plans", []) if p["id"] not in done and p["grade"] != "F"]
    return out


def plan_line(p: dict) -> str:
    """One line for a plan that is not a trade (stop too wide or R/R too low)."""
    why = (f"stop too wide ({abs(p['entry'] - p['stop']):.5g} > 1.5 × ATR 4h) — wait for an {'LPS' if p['side'] == 1 else 'LPSY'}"
           if p["refused"] else f"R/R {p['rr']} below {p['min_rr']} — no trade")
    return f"⛔ {p['key']} {p['tf']} · {p['setup']} · {why}"

