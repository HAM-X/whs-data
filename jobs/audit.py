"""Heal fix 8 · the weekly self-audit — the robot grades EVERY claim it published, not only its calls.

Walks the git history of data/decisions.json for the last N days (needs a full clone: the audit workflow checks
out with fetch-depth 0), takes each chart's claim once per closed bar, and scores it against the bars that came
after with the same rule as the 29 Oct audit and the Heal study:
  m = (close[t+H] − close[t]) ÷ ATR × direction · right ≥ +0.2 · wrong ≤ −0.2 · else flat
  H = 6 bars on 4h, 12 on 1h, 16 on 15m, 24 on 5m
Claims: the chart decision (trend / entry), the VSA verdict, the 1h flow (BLUE ease of movement), phase-C setups
(target vs stop) and failed setups. Output: a Telegram text + KV "audit:latest". Usage: python -m jobs.audit [days]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
H = {"4h": 6, "1h": 12, "15m": 16, "5m": 24}
UPS = {"TREND ↑", "BUY ZONE", "BUY PULLBACKS"}
DNS = {"TREND ↓", "SELL ZONE", "SELL RALLIES"}


def history(days: int) -> list[dict]:
    log = subprocess.run(["git", "log", f"--since={days} days ago", "--format=%H", "--", "data/decisions.json"],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
    out = []
    for h in reversed(log):
        r = subprocess.run(["git", "show", f"{h}:data/decisions.json"], cwd=ROOT, capture_output=True, text=True)
        try:
            out.append(json.loads(r.stdout))
        except Exception:
            continue
    return out


def claims(snaps: list[dict]) -> dict:
    """One record per (chart, closed bar): the latest snapshot that showed it."""
    c = {}
    for s in snaps:
        for k, d in s.items():
            if k.startswith("_") or not isinstance(d, dict) or not d.get("bar") or not d.get("atr"):
                continue
            c[(k, d["bar"])] = d
    return c


def score(cl: dict, bars: dict) -> dict:
    res = {n: [0, 0, 0] for n in ("decision", "verdict", "flow")}
    ent = []
    for (k, bar), d in cl.items():
        key, tf = k.split()
        b = bars.get((key, tf))
        if b is None:
            continue
        t = pd.to_datetime(b["time"], utc=True)
        j = np.searchsorted(t.values, pd.Timestamp(bar).to_datetime64())
        if j >= len(t) or t.iloc[j] != pd.Timestamp(bar) or j + H[tf] >= len(t):
            continue
        c = b["close"].to_numpy(float)
        mv = (c[j + H[tf]] - c[j]) / d["atr"]
        v = d.get("verdict") or ""
        for name, dirn in (("decision", 1 if d["action"] in UPS else -1 if d["action"] in DNS else 0),
                           ("verdict", 1 if v.startswith("DEMAND") else -1 if v.startswith("SUPPLY") else 0),
                           ("flow", (d.get("flow") or 0) if tf == "1h" else 0)):
            if dirn:
                m = mv * dirn
                res[name][0 if m >= 0.2 else 1 if m <= -0.2 else 2] += 1
        if d.get("setup") and d.get("stop") is not None and d.get("target") is not None:
            side, stop, tg = d["side"], d["stop"] - d["side"] * 0.25 * d["atr"], d["target"]
            hi, lo = b["high"].to_numpy(float), b["low"].to_numpy(float)
            out = "open"
            for i in range(j + 1, min(len(c), j + 201)):
                if (lo[i] <= stop) if side == 1 else (hi[i] >= stop):
                    out = "stop"
                    break
                if (hi[i] >= tg) if side == 1 else (lo[i] <= tg):
                    out = "target"
                    break
            ent.append((k, d["action"], bar, out))
    return dict(res=res, ent=sorted(set(ent)))


def text(sc: dict, days: int, n_snap: int) -> str:
    o = [f"<b>Weekly self-audit</b> — every claim of the last {days} days, scored on what price did next ({n_snap} runs)"]
    for name, lab in (("decision", "Chart decisions (trend / entry)"), ("verdict", "VSA verdict"), ("flow", "1h flow (ease)")):
        r, w, f = sc["res"][name]
        o.append(f"• {lab}: " + (f"{100 * r / (r + w):.0f}% right ({r} of {r + w}, {f} flat)" if r + w else "no claims scored yet"))
    if sc["ent"]:
        hit = sum(1 for e in sc["ent"] if e[3] == "target")
        stp = sum(1 for e in sc["ent"] if e[3] == "stop")
        o.append(f"• Phase-C setups: {len(sc['ent'])} · target {hit} · stopped {stp} · open {len(sc['ent']) - hit - stp}")
    else:
        o.append("• Phase-C setups: none this week")
    o.append("<i>A coin scores 50 %. Two years of replay: decisions 48–51 %, flow 56 %. Numbers this week are noise "
             "unless they hold for months.</i>")
    return "\n".join(o)


def main(days: int = 7, send=None, kv=None):
    from whs import bars as barsmod
    snaps = history(days)
    cl = claims(snaps)
    bars = {}
    for tf in H:
        for p in (DATA / "bars" / tf).glob("*.csv"):
            bars[(p.stem, tf)] = barsmod.read(p)
    sc = score(cl, bars)
    msg = text(sc, days, len(snaps))
    print(msg)
    if send:
        send(msg)
    if kv:
        kv("audit:latest", dict(at=pd.Timestamp.utcnow().isoformat(), text=msg, **sc))
    return sc


if __name__ == "__main__":
    from whs import cfkv, telegram
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7, send=lambda m: telegram.send(m, html=True), kv=cfkv.put_json)
