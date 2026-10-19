"""Stage 3 · the robot reads the market by itself (no clicks).

For each future and timeframe:
  1. effort, the swing ledger (minor swings) and major swings (whs/degrees.py)
  2. every climax-stopped sideways box of the major degree becomes an AUTOMATIC RANGE:
       ① = the climax extreme (the box's start), ② = the reaction extreme (the top of the automatic rally after a
       selling climax, the bottom of the reaction after a buying climax)
     and is read live with the two-sided range reader (whs/ranges.py) and its Wyckoff Story (whs/story.py)
  3. what changed since the last run becomes Telegram alerts (major swing events, range events, phases,
     breakouts confirmed / back in the range / left the range)

Everything is a live walk: the read at the last bar uses only bars up to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import core, degrees, ranges, story, swings
from .config import EXCHANGE_TZ, FUTURES

TF_MIN = {"1h": 60, "4h": 240}
RECENT = {"1h": 360, "4h": 300}       # look for ranges whose climax is within this many bars of now
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


def read_one(d: pd.DataFrame, tf: str, htf_bars: pd.DataFrame | None) -> dict:
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    t = d["time"]
    eff = core.effort(v, t, TF_MIN[tf])
    n = len(c)
    led = swings.ledger(o, h, l, c, eff)
    mr = degrees.majors(o, h, l, c, eff, led)
    live = mr.majors[-1] if mr.majors else None
    out = {"bars": n, "last": t.iloc[-1].isoformat(), "close": float(c[-1]),
           "major": None, "major_events": [], "ranges": []}
    if live is not None:
        out["major"] = {"state": int(live.direction), "since": t.iloc[live.start_i].isoformat(),
                        "protected": None if np.isnan(mr.protected[-1]) else float(mr.protected[-1]),
                        "box": None if live.direction else [float(live.lo), float(live.hi)]}
    out["major_events"] = [(t.iloc[b].isoformat(), txt) for b, txt in mr.events]
    for a1, a2, kind, box_live in auto_anchors(h, l, mr, n, RECENT[tf]):
        try:
            rr = ranges.read(o, h, l, c, eff, v, a1, a2, htf=htf_at(htf_bars, t.iloc[a1]))
        except Exception as e:      # one odd range must never stop the scan
            print(f"[range error] {tf} {t.iloc[a1]}: {type(e).__name__}: {e}")
            continue
        ph = rr.phases[-1][0] if rr.phases else "A"
        if not rr.events or rr.left or (not box_live and ph == "A"):
            continue                 # no range · already over · a box price left before it built any cause
        st = story.build(o, h, l, c, eff, rr, led)
        out["ranges"].append({
            "id": t.iloc[a1].isoformat(), "kind": rr.kind, "a2": t.iloc[a2].isoformat(),
            "phase": ph, "outcome": rr.outcome, "left": bool(rr.left),
            "name": rr.name or f"{rr.bull_name} / {rr.bear_name}",
            "support": float(min(rr.support, rr.resistance)), "resistance": float(max(rr.support, rr.resistance)),
            "events": [(t.iloc[e.known].isoformat(), e.name, e.state, float(e.price)) for e in rr.events if not e.hidden],
            "alerts": [(t.iloc[b].isoformat(), txt) for b, txt in rr.alerts],
            "story": [f"#{k} {p}: {txt}" for k, p, txt in st.steps], "bias": st.bias,
        })
    return out


def scan(h1: dict[str, pd.DataFrame], h4: dict[str, pd.DataFrame]) -> dict:
    res = {}
    for key in FUTURES:
        if FUTURES[key].get("context"):
            continue
        for tf, src in (("1h", h1), ("4h", h4)):
            d = src.get(key)
            if d is None or len(d) < 200:
                continue
            d = d.reset_index(drop=True)
            htf_bars = h4.get(key) if tf == "1h" else daily(d)
            try:
                res[f"{key} {tf}"] = read_one(d, tf, htf_bars)
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
        marks[key] = {"last": r["last"], "ranges": [g["id"] for g in r["ranges"]]}
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
