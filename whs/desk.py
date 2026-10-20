"""Stage 4 · Trade Desk (robot side): Grade → Trigger → Stop → Translate, as a plan card.

The robot turns a fresh, confirmed Wyckoff entry event of a live range into a complete plan:
  TRIGGER  the 6 points of entry (mirror for shorts):
             Spring / Shakeout (C)  stop below its low           min R/R 3.0
             Test of the spring (C) stop below the test low      3.0
             LPS (D, after the SOS)  stop below the LPS low       2.5
             SOS bar (D)            stop below the bar midpoint  2.0
             BU (D)                 stop below the BU low        2.5
           only when the event became known on one of the last 2 bars (actionable now)
  STOP     structural − 0.5 × ATR(14, 1h) − 1 tick of spread, rounded away to the tick;
           wider than 1.5 × ATR(14, 4h) → REFUSED (wait for an LPS / LPSY), never tightened
  TARGET   phase C entries: first the far edge, then the far edge + range height (measured move);
           phase D entries: the far edge + range height. R/R to the measured move.
  GATES    stop too wide → refused · R/R below the trigger's minimum → not a trade (both reported as one line)
  GRADE    10 points (Roman's proforma): higher timeframe not against (2) · currency strength agrees (2) ·
           structure: confirmed entry event (3) · story evidence for the side ≥ 75 % (2, ≥ 50 % 1) ·
           R/R ≥ the trigger's minimum (1). 8–10 A · 6–7 B · 4–5 C · 0–3 F (no plan sent).
           Until Edge Lab proves a setup (n ≥ 30), A is capped at B.
  RISK     A 0.5 % · B 0.25 % · C 0.125 % (settled 18 Oct 2026)
  TRANSLATE  MT5 level: same-direction pairs spot × level ÷ future; inverted (6J 6C 6S) spot × future ÷ level.
"""
from __future__ import annotations

import math

import numpy as np

from .snapshot import MT5, PIP, CCY

TRIG = {  # role → (name, min R/R, stop basis)
    "Spring": ("Spring", 3.0, "low"), "Shakeout": ("Shakeout", 3.0, "low"), "Test": ("Test of the spring", 3.0, "low"),
    "Chain": ("LPS", 2.5, "low"), "D_LPS": ("LPS", 2.5, "low"),   # (a phase-C LPS is only known at the SOS → the SOS bar)
    "BO": ("SOS bar", 2.0, "mid"), "BU": ("BU", 2.5, "low"),
}
SHORT_NAME = {"Spring": "UTAD", "Shakeout": "UTAD", "Test of the spring": "Test of the UTAD", "LPS": "LPSY",
              "SOS bar": "SOW bar", "BU": "LPSY (back-up)"}
RISK = {"A": 0.5, "B": 0.25, "C": 0.125, "F": 0.0}
EDGE_LAB_PROVEN: set = set()          # setups proven in Edge Lab (n ≥ 30, worst case > 0) — none yet


def _round_away(p, tick, side):
    """Round a stop away from price: down for a long's stop, up for a short's."""
    q = p / tick
    return (math.floor(q + 1e-9) if side == 1 else math.ceil(q - 1e-9)) * tick


def to_mt5(key, level, fut_now, spot_now):
    if key not in MT5 or spot_now is None or not fut_now:
        return None
    return spot_now * fut_now / level if MT5[key][1] == "inverted" else spot_now * level / fut_now


def lots(key, entry, stop, fut_now, spot_now, risk_usd, xau_contract=100.0):
    if key == "GC":
        return math.floor(risk_usd / (abs(entry - stop) * xau_contract) * 100) / 100
    sym = MT5[key][0]
    e5, s5 = to_mt5(key, entry, fut_now, spot_now), to_mt5(key, stop, fut_now, spot_now)
    if e5 is None:
        return None
    pips = abs(e5 - s5) / PIP[sym]
    pv = (1000 / spot_now if sym == "USDJPY" else 10 / spot_now) if sym.startswith("USD") else 10.0
    return math.floor(risk_usd / (pips * pv) * 100) / 100


def grade_of(points: int, setup: str) -> str:
    g = "A" if points >= 8 else "B" if points >= 6 else "C" if points >= 4 else "F"
    if g == "A" and setup not in EDGE_LAB_PROVEN:
        g = "B"
    return g


def plans(key, tf, h, l, c, rr, st, atr1h, atr4h, tick, htf, strength_rank, n_recent=2, late=False):
    """Plans for the entry events of range read `rr` that became known on the last `n_recent` bars."""
    n = len(c)
    out = []
    lo_edge, hi_edge = min(rr.support, rr.resistance), max(rr.support, rr.resistance)
    H = hi_edge - lo_edge
    for e in rr.events:
        if e.hidden or e.state != "confirmed" or e.role not in TRIG or e.known < n - n_recent or e.side == 0:
            continue
        side = e.side
        if e.role == "BO" and side != rr.outcome:
            continue
        name, min_rr, basis = TRIG[e.role]
        if side == -1:
            name = SHORT_NAME.get(name, name)
        entry = float(c[-1])
        if basis == "mid":
            struct = (h[e.bar] + l[e.bar]) / 2
        else:
            struct = l[e.bar] if side == 1 else h[e.bar]
        stop = _round_away(struct - side * (0.5 * atr1h + tick), tick, side)
        dist = abs(entry - stop)
        refused = dist > 1.5 * atr4h or (entry - stop) * side <= 0
        phase_c = e.role in ("Spring", "Shakeout", "Test")
        t1 = hi_edge if side == 1 else lo_edge
        t2 = hi_edge + H if side == 1 else lo_edge - H
        rr_mm = (t2 - entry) * side / dist if dist > 0 else 0.0
        # grade
        pts, why = 0, []
        if htf != -side:
            pts += 2
            why.append("higher timeframe not against")
        if strength_rank is not None and strength_rank * side > 0:
            pts += 2
            why.append("currency strength agrees")
        pts += 3
        why.append(f"{name} confirmed")
        ev_for, ev_ag = (len(rr.bull), len(rr.bear)) if side == 1 else (len(rr.bear), len(rr.bull))
        share = ev_for / max(1, ev_for + ev_ag)
        if share >= 0.75:
            pts += 2
            why.append(f"story evidence {ev_for} of {ev_for + ev_ag}")
        elif share >= 0.5:
            pts += 1
            why.append(f"story evidence {ev_for} of {ev_for + ev_ag} (mixed)")
        if rr_mm >= min_rr:
            pts += 1
            why.append(f"R/R {rr_mm:.1f} ≥ {min_rr}")
        if late and e.role == "BO":                  # Market Panel: ≥ 75 % of the ADR used → a breakout entry is late
            pts -= 1
            why.append("LATE: ≥ 75 % of the day's range already used (−1)")
        g = grade_of(pts, name)
        out.append(dict(id=f"{key} {tf} {e.role} {e.bar}", key=key, tf=tf, side=side, setup=name, phase="C" if phase_c else "D",
                        entry=entry, stop=stop, struct=float(struct), t1=float(t1) if phase_c else None, t2=float(t2),
                        rr=round(rr_mm, 2), min_rr=min_rr, points=pts, grade=g, why=why, refused=bool(refused),
                        rr_low=bool(rr_mm < min_rr),
                        cap=1.5 * atr4h, range=rr.name or f"{rr.bull_name} / {rr.bear_name}", bias=st.bias))
    return out


def card(p, fut_now, spot_now, balance=100000.0):
    """Telegram plan card (HTML)."""
    d = lambda x: f"{x:.{max(0, -int(math.floor(math.log10(abs(x)))) + 4)}g}" if x else "—"
    sd = "LONG" if p["side"] == 1 else "SHORT"
    capped = " (A until Edge Lab proves it)" if p["points"] >= 8 and p["grade"] == "B" else ""
    head = f"🎯 <b>{p['key']} {p['tf']} · {sd} · {p['setup']} (phase {p['phase']})</b> · grade <b>{p['grade']}</b> {p['points']}/10{capped}"
    if p["refused"]:
        return head + f"\n⛔ Stop too wide ({abs(p['entry'] - p['stop']):.5g} > cap {p['cap']:.5g} = 1.5 × ATR 4h). " \
                      f"Wait for an {'LPS' if p['side'] == 1 else 'LPSY'}; never tighten the stop."
    risk = RISK[p["grade"]]
    lines = [head, f"{p['range']} · {p['bias']}",
             f"Future: entry {p['entry']:.6g} · stop {p['stop']:.6g} (structure {p['struct']:.6g})" +
             (f" · T1 {p['t1']:.6g}" if p["t1"] else "") + f" · T2 {p['t2']:.6g} · R/R {p['rr']}"]
    if p["key"] in MT5 and spot_now:
        sym = MT5[p["key"]][0]
        e5, s5 = to_mt5(p["key"], p["entry"], fut_now, spot_now), to_mt5(p["key"], p["stop"], fut_now, spot_now)
        inv = MT5[p["key"]][1] == "inverted"
        act = ("SELL" if p["side"] == 1 else "BUY") if inv else ("BUY" if p["side"] == 1 else "SELL")
        nd = 3 if sym == "USDJPY" else 2 if sym == "XAUUSD" else 5
        lt = lots(p["key"], p["entry"], p["stop"], fut_now, spot_now, balance * risk / 100) if risk else 0
        lines.append(f"MT5 {sym} {act}: entry {e5:.{nd}f} · stop {s5:.{nd}f} → <b>{lt:.2f} lots</b> per ${balance:,.0f} at {risk}%")
    lines.append(f"<i>why: {'; '.join(p['why'])}</i>")
    lines.append(f"<i>/size {p['key']} {p['entry']:.6g} {p['stop']:.6g} {p['grade']} &lt;balance&gt; · manage: ⅓ at 2R, BE once BU holds, SBar staircase</i>")
    return "\n".join(lines)
