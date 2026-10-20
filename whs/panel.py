"""Stage 5 · Market Panel v2 (robot side) — screens the market, never labels a chart. Output: data/panel.json.

  1 strength     the snapshot's currency strength on 1h / 1d / 1w; PERSISTENT = same sign and top/bottom 3
                 on all three windows (a one-window move is a spike)
  2 sweet spot   when DX confirms a new swing high (minor ledger, 4h), each currency's low now vs its low at
                 DX's previous swing high: a HIGHER low = accumulated while the dollar pushed (bullish for it).
                 Mirror on a DX swing low: a LOWER high = distributed (bearish for it).
  5 ADR used     today's range ÷ 20-day ADR; ≥ 75 % → breakout entries are LATE (tests / LPS unaffected)
  6 themes       60-day correlations of daily returns, recomputed each run; blocs at ρ ≥ 0.7.
                 Feeds the Guard: positions signed by USD direction (long any FX future or gold = short USD),
                 effective heat = √(Σᵢ Σⱼ wᵢ wⱼ ρᵢⱼ)
  Top 3          strength persistence + structure state (scan) + sweet spot − lateness
(3 OI quadrants and 4 COT are added once their data sources are wired.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import core, swings
from .config import EXCHANGE_TZ
from .snapshot import CCY

KEYS = ["6E", "6B", "6A", "6N", "6J", "6C", "6S", "GC"]
LATE = 75.0
BLOC = 0.7


def daily_closes(h1: dict) -> pd.DataFrame:
    cols = {}
    for k in KEYS:
        d = h1.get(k)
        if d is None or len(d) < 50:
            continue
        ct = d["time"].dt.tz_convert(EXCHANGE_TZ) + pd.Timedelta(hours=7)
        cols[k] = d.assign(day=ct.dt.date).groupby("day")["close"].last()
    return pd.DataFrame(cols).dropna(how="all")


def themes(h1: dict, days: int = 60) -> dict:
    px = daily_closes(h1)
    ret = px.pct_change().iloc[-days:]
    rho = ret.corr(min_periods=int(days * 0.6)).round(2)
    keys = list(rho.columns)
    blocs, used = [], set()
    for k in keys:
        if k in used:
            continue
        grp = [k] + [j for j in keys if j != k and j not in used and rho.loc[k, j] >= BLOC]
        if len(grp) > 1:
            blocs.append(grp)
            used |= set(grp)
    return {"days": days, "rho": {k: {j: (None if pd.isna(rho.loc[k, j]) else float(rho.loc[k, j])) for j in keys} for k in keys},
            "blocs": blocs}


def effective_heat(positions: dict, rho: dict) -> float:
    """positions: key → signed risk % (+ = long the future = short USD). √(Σ Σ wᵢ wⱼ ρᵢⱼ)."""
    ks = list(positions)
    s = 0.0
    for i in ks:
        for j in ks:
            r = 1.0 if i == j else (rho.get(i, {}).get(j) or 0.0)
            s += positions[i] * positions[j] * r
    return float(np.sqrt(max(s, 0.0)))


def persistence(strength: dict) -> dict:
    out = {}
    for c in [x["ccy"] for x in strength.get("1d", [])]:
        ranks = []
        for w in ("1h", "1d", "1w"):
            lst = [x["ccy"] for x in strength.get(w, [])]
            ranks.append(lst.index(c) if c in lst else None)
        if None in ranks:
            continue
        n = len(strength["1d"])
        if all(r <= 2 for r in ranks):
            out[c] = "strong (all 3 windows)"
        elif all(r >= n - 3 for r in ranks):
            out[c] = "weak (all 3 windows)"
        elif ranks[0] <= 1 and ranks[2] >= n - 3 or ranks[0] >= n - 2 and ranks[2] <= 2:
            out[c] = "spike against the week"
    return out


def sweet_spot(h4: dict) -> dict:
    """On DX's MAJOR swings (4h): when its latest major up swing made a new high above the previous one, each
    currency's low during that push vs its low during the previous push (higher low = accumulated → bullish).
    Mirror for a new DX major low (lower high = distributed → bearish)."""
    from . import degrees
    dx = h4.get("DX")
    if dx is None or len(dx) < 100:
        return {}
    o, h, l, c = (dx[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    one = np.ones(len(c))
    mr = degrees.majors(o, h, l, c, one, swings.ledger(o, h, l, c, one))
    t = dx["time"]
    out = {}
    for d, name in ((1, "dx_high"), (-1, "dx_low")):
        legs = [m for m in mr.majors if m.direction == d]
        if len(legs) < 2:
            continue
        a, b = legs[-2], legs[-1]
        new = (b.end_p > a.end_p) if d == 1 else (b.end_p < a.end_p)
        res = {}
        for k, ccy in CCY.items():
            x = h4.get(k)
            if x is None:
                continue
            win = lambda m: x[(x["time"] >= t.iloc[m.start_i]) & (x["time"] <= t.iloc[m.end_i])]
            if d == 1:
                pa, pb = float(win(a)["low"].min()), float(win(b)["low"].min())
                if np.isfinite(pa) and np.isfinite(pb):
                    res[ccy] = {"dir": 1, "text": "higher low — accumulated while the dollar pushed"} if pb > pa else {"dir": 0, "text": "lower low"}
            else:
                pa, pb = float(win(a)["high"].max()), float(win(b)["high"].max())
                if np.isfinite(pa) and np.isfinite(pb):
                    res[ccy] = {"dir": -1, "text": "lower high — distributed while the dollar fell"} if pb < pa else {"dir": 0, "text": "higher high"}
        out[name] = {"at": t.iloc[b.end_i].isoformat(), "prev": t.iloc[a.end_i].isoformat(), "new_extreme": bool(new),
                     "read": res if new else {}}
    return out


def build(snap: dict, scan: dict, h1: dict, h4: dict) -> dict:
    st = snap.get("strength") or {}
    pers = persistence(st)
    ss = sweet_spot(h4)
    th = themes(h1)
    late = {k: (snap["instruments"][k].get("adr_used_pct") or 0) >= LATE for k in KEYS if k in snap.get("instruments", {})}
    # Top 3 by instrument: structure (best of 1h/4h scan) + strength persistence + sweet spot − lateness
    vs = {x["ccy"]: x.get("vs_usd_pct") for x in st.get("1d", [])}
    rows = []
    for k in KEYS:
        ccy = CCY.get(k)
        pts, why = 0.0, []
        for tf, w in (("4h", 1.5), ("1h", 1.0)):
            r = scan.get(f"{k} {tf}") or {}
            g = (r.get("ranges") or [None])[-1]
            if g:
                p = {"C": 3, "D": 2, "B": 1}.get(g["phase"], 0) * w
                if p:
                    pts += p
                    why.append(f"{tf} range phase {g['phase']}")
        if ccy and ccy in pers:
            pts += 1.5
            why.append(f"{ccy} {pers[ccy]}")
        for side in ("dx_high", "dx_low"):
            rd = ((ss.get(side) or {}).get("read") or {}).get(ccy) if ccy else None
            if rd and rd["dir"]:
                pts += 1.5
                why.append(f"{ccy} {rd['text'].split(' — ')[0]} vs DX ({'bullish' if rd['dir'] == 1 else 'bearish'})")
        if late.get(k):
            pts -= 1
            why.append("ADR ≥ 75 % used — breakouts late")
        rows.append({"key": k, "score": round(pts, 2), "why": why})
    rows.sort(key=lambda x: -x["score"])
    return {"persistence": pers, "sweet_spot": ss, "themes": th, "late": late,
            "top3": [r for r in rows if r["score"] > 0][:3], "ranking": rows}
