"""Stage 3.4 · morning brief v1 (sent once a day at the first robot run from 08:00 UTC+3).

Built only from the robot's own files: data/snapshot.json (prices, levels, strength, health) and
data/scan.json (major swings, automatic ranges, Story). TradingView stays the price of record.

  header   date · data health
  dollar   DX day change · currency strength (1 day) strongest → weakest
  top 3    the charts most worth opening this morning, scored from the Wyckoff read:
           phase C +3 (a test is happening) · D +2 · B +1 · E 0 · new events in the last 12 h +2 ·
           a major-swing event in the last 12 h +1 · 4h counts 1.5× a 1h read
           each with: range name + phase, the Story's bias line, the levels that matter
  overnight  every Wyckoff event of the last 12 h, oldest first
"""
from __future__ import annotations

import pandas as pd

PH_SCORE = {"A": 0.5, "B": 1, "C": 3, "D": 2, "E": 0}
MAJ = {1: "↑ major up", -1: "↓ major down", 0: "↔ sideways"}


def _t(iso):
    return pd.Timestamp(iso)


def _local(iso):
    return _t(iso).tz_convert("Africa/Nairobi").strftime("%a %H:%M")


def score(key: str, r: dict, since: pd.Timestamp) -> tuple[float, list[str]]:
    pts, why = 0.0, []
    tfw = 1.5 if key.endswith("4h") else 1.0
    g = (r.get("ranges") or [None])[-1]
    if g:
        p = PH_SCORE.get(g["phase"], 0)
        if p:
            pts += p
            why.append(f"phase {g['phase']}")
        fresh = [a for a in g["alerts"] if _t(a[0]) >= since]
        if fresh:
            pts += 2
            why.append("new range events")
    if any(_t(w) >= since for w, _ in r.get("major_events", [])):
        pts += 1
        why.append("major swing event")
    return pts * tfw, why


def build(snap: dict, scan: dict, now=None) -> str:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    since = now - pd.Timedelta(hours=12)
    loc = now.tz_convert("Africa/Nairobi")
    out = [f"☀️ Morning brief · {loc.strftime('%a %d %b')} · data {snap.get('generated_local', '?')} UTC+3"]
    bad = [k for k, v in (snap.get("health") or {}).items() if isinstance(v, dict) and v.get("problems")]
    out.append("Data: " + ("⚠️ check " + ", ".join(bad) if bad else "🟢 all feeds fresh"))
    # dollar + strength
    dx = (snap.get("instruments") or {}).get("DX") or {}
    dch = (dx.get("day") or {}).get("change_pct")
    st = (snap.get("strength") or {}).get("1d") or []
    if st:
        out.append("")
        out.append(f"Dollar {'' if dch is None else f'{dch:+.2f}% today'} · strength (1d): " +
                   " > ".join(x["ccy"] for x in st))
    # top 3
    ranked = sorted(((score(k, r, since), k, r) for k, r in scan.items()), key=lambda x: -x[0][0])
    top = [x for x in ranked if x[0][0] > 0][:3]
    out.append("")
    out.append("<b>Top 3 to open</b>" if top else "<b>Top 3</b>: nothing building — trend days, wait for structure.")
    for n, ((pts, why), k, r) in enumerate(top, 1):
        m = r.get("major") or {}
        g = (r.get("ranges") or [None])[-1]
        line = f"{n}. <b>{k}</b> — {MAJ.get(m.get('state'), '—')}"
        if g:
            line += f" · {g['name']} · phase {g['phase']}"
        out.append(line)
        if g:
            out.append(f"   {g['bias']}")
            out.append(f"   range {g['support']:.5g} – {g['resistance']:.5g}" +
                       (f" · major holds {'above' if m.get('state') == 1 else 'below'} {m['protected']:.5g}"
                        if m.get("protected") is not None and m.get("state") else ""))
        out.append(f"   <i>why: {', '.join(why)}</i>")
    # overnight events
    ev = []
    for k, r in scan.items():
        for w, txt in r.get("major_events", []):
            if _t(w) >= since:
                ev.append((w, f"{k} · {txt}"))
        for g in r.get("ranges", []):
            for w, txt in g.get("alerts", []):
                if _t(w) >= since:
                    ev.append((w, f"{k} · {txt}"))
    out.append("")
    if ev:
        out.append("<b>Overnight</b> (last 12 h)")
        for w, txt in sorted(ev)[-12:]:
            out.append(f"• {_local(w)} {txt}")
    else:
        out.append("Overnight: no Wyckoff events.")
    out.append("")
    out.append("<i>/story 6E 4h for any chart's full read · TradingView is the price of record.</i>")
    return "\n".join(out)
