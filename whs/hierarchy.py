"""Ranges within ranges (Hamza 29 Oct 2026) — 1D → 4h → 1h → 15m, facts only, no entry signals.

For every future and timeframe the live range (the newest one whose AR is confirmed, not left) is the PARENT; the
next lower timeframe's ranges that formed inside the parent's TOP or BOTTOM zone after the parent's AR are its
CHILDREN (a minor range at the UT / Spring level). Each child is read with its own events, phase, own phase C, and
the child timeframe's effort/result verdict. The robot then says what the structure points to — e.g. "minor
distribution at the 4h top, supply present → the structure points to a test of the 4h support" — together with
what that has done in the Heal study (live-walk replay), so a description is never dressed up as an edge.

Also here:
  touches(…)      every touch of a confirmed range edge (any of the four timeframes), once per episode — B setups
  facts(…)        the confirmed events of every live range, each with how often such a label stands (replay)
  diff_facts(…)   new facts + CORRECTIONS (an earlier label renamed / failed / removed), a correction only after
                  it held for two runs in a row (replay: 9.7 % of 1h changes flipped back; one-read flickers)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import campaign as cp, core

TF_ORDER = ["1d", "4h", "1h", "15m"]
CHILD = {"1d": "4h", "4h": "1h", "1h": "15m"}
TF_MIN = {"1d": 1440, "4h": 240, "1h": 60, "15m": 15}
TF_NAME = {"1d": "1D", "4h": "4h", "1h": "1h", "15m": "15m"}

# Heal study (29 Oct 2026, replay v2, 1h, 1,945 confirmed facts): share of each kind of label that was never
# contradicted later. Phase-C labels are provisional; after the SOS / SOW that follows them they stand far more often.
STANDS = {"SC": 100, "BC": 100, "PS": 100, "PSY": 100, "ST": 95, "AR": 82, "B_light": 88, "B_heavy": 75, "BO": 69,
          "BU": 64, "D_LPS": 75, "Spring": 49, "Shakeout": 61, "Test": 58, "LPS_C": 45, "Chain": 8}
STANDS_AFTER_D = {"Spring": 77, "Test": 79, "Shakeout": 73, "LPS_C": 56}
PHASE_C = {"Spring", "Shakeout", "Test", "LPS_C"}
# Heal study (29 Oct 2026, replay v2 + empirical control): from a child range at the parent's edge whose effort/result
# points back into the parent, how often price reached the parent's MIDDLE before closing beyond the child range — vs
# the same race started at 60 random bars of the same future and timeframe (keeps the wick/close asymmetry and drift).
ODDS = {("1d", "4h"): dict(n=18, hit=56, chance=39, verdict="promising, too few cases",
                           inside=dict(n=11, hit=82, chance=56, verdict="promising, too few cases")),
        ("4h", "1h"): dict(n=57, hit=32, chance=35, verdict="no edge",
                           inside=dict(n=40, hit=35, chance=40, verdict="no edge")),
        ("1h", "15m"): dict(n=14, hit=36, chance=45, verdict="no edge so far (15m history is 2 months)")}
# First touch of a confirmed edge: reached the range middle before closing beyond the touch bar + 0.25 ATR.
TOUCH_ODDS = {"1d": dict(n=13, hit=38, chance=29), "4h": dict(n=38, hit=40, chance=33), "1h": dict(n=208, hit=30, chance=30)}


def touch_history(tfs) -> str:
    """'1h edges: first touches reached the middle first 30% (chance 30%, n 208)' for the timeframes in a message."""
    out = []
    for tf in dict.fromkeys(tfs):
        o = TOUCH_ODDS.get(tf)
        out.append(f"{TF_NAME[tf]} edges: first touches reached the middle first {o['hit']}% (chance {o['chance']}%, n {o['n']})"
                   if o else f"{TF_NAME[tf]} edges: not measured yet")
    return "; ".join(out) + " — no proven edge"


def _p(x) -> str:
    return f"{x:.5g}" if x is not None else "—"


def _fam(e: dict) -> str:
    return e.get("r") or str(e.get("n", "")).split(" #")[0]


def live_range(rec: dict | None) -> dict | None:
    """The parent range: newest range with a confirmed AR that has not left (scan already drops failed ones)."""
    for g in reversed((rec or {}).get("ranges") or []):
        if g.get("left") or g["resistance"] <= g["support"]:
            continue
        ev = g.get("ev2") or []
        if any(_fam(e) == "AR" and e["st"] == "confirmed" for e in ev):
            return g
    return None


def zone(g: dict, price: float) -> int:
    """+1 top zone (≥ 80 % of the height, up to 30 % above it) · −1 bottom zone · 0 elsewhere."""
    H = g["resistance"] - g["support"]
    pos = (price - g["support"]) / H
    return 1 if 0.8 <= pos <= 1.3 else -1 if -0.3 <= pos <= 0.2 else 0


def confirmed(g: dict) -> list[dict]:
    return [e for e in (g.get("ev2") or []) if e["st"] == "confirmed"]


def event_line(g: dict, n: int = 6) -> str:
    """'SC 1.1142 ✓ · AR 1.1296 ✓ · ST 1.1135 ✓' — the confirmed events in order (the last n)."""
    ev = sorted(confirmed(g), key=lambda e: e["k"])[-n:]
    return " · ".join(f"{e['n']} {_p(e['p'])} ✓" for e in ev) or "no confirmed events yet"


def children(P: dict, crec: dict | None) -> list[dict]:
    """Child ranges (next lower timeframe) sitting in P's top / bottom zone, formed after P's AR became known."""
    ar = [e for e in confirmed(P) if _fam(e) == "AR"]
    if not ar or not crec:
        return []
    t_ar = pd.Timestamp(ar[0]["k"])
    out = []
    for g in crec.get("ranges") or []:
        if pd.Timestamp(g["id"]) < t_ar or g["resistance"] <= g["support"]:
            continue
        z = zone(P, (g["support"] + g["resistance"]) / 2)
        if z:
            out.append(dict(g, zone=z))
    return out


def own_phase_c(g: dict, z: int) -> str | None:
    """The child's own phase-C test at the parent edge: a UTAD / UT at the top, a Spring / Shakeout at the bottom."""
    names = ("UTAD", "UT") if z == 1 else ("Spring", "Shakeout")
    ev = [e for e in confirmed(g) if str(e["n"]).startswith(names)]
    return f"{ev[-1]['n']} {_p(ev[-1]['p'])}" if ev else None


def _overlap(f: dict, g: dict) -> float:
    lo, hi = max(f["S"], g["support"]), min(f["R"], g["resistance"])
    return max(0.0, hi - lo) / max(f["R"] - f["S"], g["resistance"] - g["support"])


def daily_bars(h1: pd.DataFrame) -> pd.DataFrame:
    """1D bars from 1h by CME session (a bar from 17:00 Chicago belongs to the next day); days with < 12 bars dropped."""
    from .config import EXCHANGE_TZ
    t = pd.to_datetime(h1["time"], utc=True)
    day = (t.dt.tz_convert(EXCHANGE_TZ) + pd.Timedelta(hours=7)).dt.date
    g = h1.assign(time=t, day=day).groupby("day")
    d = pd.DataFrame(dict(time=g.time.first(), open=g.open.first(), high=g.high.max(), low=g.low.min(),
                          close=g.close.last(), volume=g.volume.sum(), n=g.time.size()))
    return d[d.n >= 12].drop(columns="n").reset_index(drop=True)


def describe(key: str, recs: dict, verdicts: dict, price: float) -> dict:
    """One hierarchy block per future. recs[(key, tf)] = scan record, verdicts[(key, tf)] = VSA verdict text."""
    lines, frames = [], []
    shown = []
    for tf in TF_ORDER:
        P = live_range(recs.get((key, tf)))
        if not P:
            continue
        same = next((f for f in shown if _overlap(f, P) >= 0.8), None)
        shown.append(dict(tf=tf, S=P["support"], R=P["resistance"]))
        z = zone(P, price)
        pos = (price - P["support"]) / (P["resistance"] - P["support"])
        where = ("price above it" if pos > 1.3 else "price below it" if pos < -0.3 else
                 "price in its TOP zone" if z == 1 else "price in its BOTTOM zone" if z == -1 else "price inside, away from the edges")
        if same:                        # the lower timeframe sees the same box: say so, don't repeat the levels
            lines.append(f"<b>{TF_NAME[tf]}</b> same range as the {TF_NAME[same['tf']]} · phase {P['phase']} here · {event_line(P, 5)}")
        else:
            lines.append(f"<b>{TF_NAME[tf]}</b> range {_p(P['support'])} – {_p(P['resistance'])} · phase {P['phase']} · "
                         f"{event_line(P, 5)} · {where}")
        fr = dict(tf=tf, S=P["support"], R=P["resistance"], phase=P["phase"], zone=z, children=[])
        ctf = CHILD.get(tf)
        for c in children(P, recs.get((key, ctf))) if ctf else []:
            z2 = c["zone"]
            pc = own_phase_c(c, z2)
            out = c.get("outcome", 0)
            res = ("resolved DOWN" if out == -1 else "resolved UP" if out == 1 else f"phase {c['phase']}")
            ver = verdicts.get((key, ctf), "")
            vs = 1 if ver.startswith("DEMAND") else -1 if ver.startswith("SUPPLY") else 0
            back_in = vs == -z2 or out == -z2
            side_word = "top (UT / UTAD level)" if z2 == 1 else "bottom (Spring level)"
            l2 = (f"   ↳ at its {side_word}: <b>{TF_NAME[ctf]}</b> range {_p(c['support'])} – {_p(c['resistance'])} · "
                  f"{event_line(c, 5)}" + (f" · own phase C: {pc}" if pc else "") + f" · {res}")
            lines.append(l2)
            if ver:
                lines.append(f"     {TF_NAME[ctf]} effort/result: {ver.split(' —')[0]}")
            if back_in:
                tgt = P["support"] if z2 == 1 else P["resistance"]
                mid = (P["support"] + P["resistance"]) / 2
                why = []
                if vs == -z2:
                    why.append("effort/result: supply is present (demand weak) at the top" if z2 == 1
                               else "effort/result: demand is present (supply weak) at the bottom")
                if out == -z2:
                    why.append(f"the minor range resolved {'DOWN' if z2 == 1 else 'UP'}, back into the {TF_NAME[tf]} range")
                if pc:
                    why.insert(0, f"its own phase C ({pc}) is done")
                what = "; ".join(why)
                lines.append(f"     reading: {what} — the structure points to the {TF_NAME[tf]} middle {_p(mid)}, "
                             f"then a test of the {TF_NAME[tf]} {'support' if z2 == 1 else 'resistance'} {_p(tgt)}")
                o = ODDS.get((tf, ctf))
                inside = P["support"] <= price <= P["resistance"]
                if o and inside and o.get("inside"):
                    o, how = o["inside"], " with price inside the parent"
                else:
                    how = ""
                lines.append("     history: " + (f"{TF_NAME[tf]}→{TF_NAME[ctf]} cases like this{how} reached the middle first "
                                                  f"{o['hit']}% (chance {o['chance']}%, n {o['n']}) — {o['verdict']}"
                                                  if o else "not measured yet on this pair — the weekly audit is logging it"))
            ext = [e["p"] for e in confirmed(c)] + [c["resistance"] if z2 == 1 else c["support"]]
            fr["children"].append(dict(tf=ctf, S=c["support"], R=c["resistance"], zone=z2, phase=c["phase"],
                                       outcome=out, own_c=pc, back_in=bool(back_in), id=c["id"],
                                       inv=max(ext) if z2 == 1 else min(ext), target=(P["support"] + P["resistance"]) / 2))
        frames.append(fr)
    head = f"<b>{key} · ranges within ranges</b>"
    if not frames:
        return dict(text=head + "\nno live range with a confirmed AR on 1D–15m", frames=[])
    return dict(text="\n".join([head] + lines), frames=frames)


def touches(key: str, recs: dict, bars_15m: pd.DataFrame | None, since, atrs: dict, armed: dict) -> list[dict]:
    """Touches of confirmed range edges (1D, 4h, 1h, 15m) by the 15m bars since `since`. `armed[(key, tf, range
    id, side)]` is False while an episode is running; it re-arms once a close is 30 % of the height back inside."""
    out = []
    if bars_15m is None or not len(bars_15m):
        return out
    tt = pd.to_datetime(bars_15m["time"], utc=True)
    try:                                   # the touch bar, graded by effort and result (whs/campaign.touch_grade)
        eff15 = core.effort(bars_15m["volume"].to_numpy(float), tt, 15)
        atr15 = core.atr(bars_15m["high"].to_numpy(float), bars_15m["low"].to_numpy(float), bars_15m["close"].to_numpy(float))
    except Exception:
        eff15 = atr15 = None
    pos = {ts: i for i, ts in enumerate(tt)}
    b = bars_15m[tt > pd.Timestamp(since)]
    for tf in TF_ORDER:
        P = live_range(recs.get((key, tf)))
        if not P or not edges_confirmed(P):
            continue
        S, R, H = P["support"], P["resistance"], P["resistance"] - P["support"]
        a = atrs.get((key, tf)) or 0.1 * H
        for _, x in b.iterrows():
            for side, hit, lvl in ((1, x["high"] >= R - 0.1 * a, R), (-1, x["low"] <= S + 0.1 * a, S)):
                k = f"{key}|{tf}|{P['id']}|{side}"
                if side == 1 and x["close"] <= R - 0.3 * H:
                    armed[k] = True
                if side == -1 and x["close"] >= S + 0.3 * H:
                    armed[k] = True
                if hit and armed.get(k, True):
                    armed[k] = False
                    i = pos.get(pd.Timestamp(x["time"]).tz_convert("UTC") if pd.Timestamp(x["time"]).tzinfo else pd.Timestamp(x["time"], tz="UTC"))
                    grade, gs = ("undecided (no effort data)", 0)
                    if i is not None and eff15 is not None:
                        grade, gs = cp.touch_grade(float(x["open"]), float(x["high"]), float(x["low"]), float(x["close"]),
                                                   float(eff15[i]), float(atr15[i - 1] if i else atr15[i]), side)
                    inv = float(x["high"] + 0.25 * a) if side == 1 else float(x["low"] - 0.25 * a)
                    out.append(dict(key=key, tf=tf, side=side, level=lvl, t=pd.Timestamp(x["time"]).isoformat(),
                                    beyond=bool(x["high"] > R) if side == 1 else bool(x["low"] < S),
                                    events=event_line(P, 4), phase=P["phase"], id=P["id"],
                                    c=float(x["close"]), inv=inv, target=(S + R) / 2, grade=grade, gsign=gs))
    return out


def edges_confirmed(P: dict) -> bool:
    """A range's edges are confirmed once its climax (SC / BC) AND its AR are confirmed; phase E = it is over."""
    fams = {_fam(e) for e in confirmed(P)}
    return P.get("phase") != "E" and "AR" in fams and bool(fams & {"SC", "BC"})


def touches_text(ts: list[dict]) -> str:
    """One message per future per run: every confirmed edge it touched, the same level on two timeframes once."""
    ts = sorted(ts, key=lambda t: (TF_ORDER.index(t["tf"]), t["t"]))
    groups = []
    for t in ts:
        g = next((g for g in groups if g[0]["side"] == t["side"] and abs(g[0]["level"] - t["level"]) <= 1e-9 + 2e-4 * abs(t["level"])), None)
        (g.append(t) if g else groups.append([t]))
    key = ts[0]["key"]
    out = [f"📏 <b>{key} · price touched a confirmed range edge</b>"]
    for g in groups:
        t = g[0]
        edge = "TOP" if t["side"] == 1 else "BOTTOM"
        when = pd.Timestamp(t["t"]).tz_convert("Africa/Nairobi").strftime("%a %H:%M")
        tfs = " · ".join(dict.fromkeys(TF_NAME[x["tf"]] for x in g))
        out.append(f"• <b>{tfs}</b> range {edge} {_p(t['level'])}{' (and beyond)' if t['beyond'] else ''} · {when} · phase {t['phase']} · {t['events']}")
        if t.get("grade"):
            out.append(f"   the touch bar: <b>{t['grade']}</b>")
    out.append(f"<i>B setups at best. History: {touch_history(t['tf'] for t in ts)}. Graded touches sat at their own "
               f"chance too (replay, 259 touches) — the grade describes the bar, it is not an edge.</i>")
    return "\n".join(out)


def touch_text(t: dict, block: str | None = None) -> str:
    edge = "TOP" if t["side"] == 1 else "BOTTOM"
    when = pd.Timestamp(t["t"]).tz_convert("Africa/Nairobi").strftime("%a %H:%M")
    s = (f"📏 <b>{t['key']} {TF_NAME[t['tf']]} · price touched the range {edge} {_p(t['level'])}</b>"
         f"{' (and went beyond it)' if t['beyond'] else ''} · {when}\n"
         f"phase {t['phase']} · {t['events']}\n"
         f"<i>A B setup at best. History: {touch_history([t['tf']])}.</i>")
    if block:
        s += "\n" + block
    return s


def facts(recs: dict) -> dict:
    """{fact id: dict(chart, tf, fam, name, price, bar)} for every confirmed event of every live range."""
    out = {}
    for (key, tf), rec in recs.items():
        if tf not in TF_MIN:
            continue
        for g in (rec or {}).get("ranges") or []:
            for e in confirmed(g):
                fid = f"{key}|{tf}|{g['id']}|{_fam(e)}|{e['s']}|{e['b'][:13]}"
                out[fid] = dict(key=key, tf=tf, rid=g["id"], fam=_fam(e), name=e["n"], price=e["p"], bar=e["b"], side=e["s"])
    return out


def _same(a: dict, b: dict, tol_bars: int = 6) -> bool:
    if a["fam"] != b["fam"] or a["side"] != b["side"] or a["rid"] != b["rid"]:
        return False
    if a["fam"] in ("AR", "SC", "BC", "PS", "PSY"):
        return True
    dt = abs((pd.Timestamp(a["bar"]) - pd.Timestamp(b["bar"])).total_seconds()) / 60 / TF_MIN[a["tf"]]
    return dt <= tol_bars


def diff_facts(prev: dict, cur: dict, pending: dict, live_ranges: set) -> tuple[list, list, dict]:
    """New facts, confirmed corrections, and the pending corrections to re-check next run.
    A previous fact with no matching current fact (same range, family, side, bar ± 6 bars) while its range is still
    read = a correction candidate; it is announced only if it is still missing on the NEXT run."""
    new = [f for fid, f in cur.items() if not any(_same(f, p) for p in prev.values())]
    corrections, nxt = [], {}
    for fid, f in prev.items():
        if f["rid"] not in live_ranges or any(_same(f, c) for c in cur.values()):
            continue
        if fid in pending:
            rep = [c for c in cur.values() if c["rid"] == f["rid"] and c["side"] == f["side"] and
                   abs((pd.Timestamp(c["bar"]) - pd.Timestamp(f["bar"])).total_seconds()) / 60 / TF_MIN[f["tf"]] <= 6]
            corrections.append(dict(f, now=rep[0]["name"] if rep else None))
        else:
            nxt[fid] = f
    return new, corrections, nxt


ALERT_FAMS = {"SC", "BC", "AR", "Spring", "Shakeout", "Test", "BO"}


def alert_fact(f: dict) -> bool:
    """Which new facts are worth a message: the climax, the AR, phase-C tests and the SOS / SOW; on 15m only the
    Spring / Shakeout / UTAD and the SOS / SOW (the rest goes into /htr)."""
    if f["tf"] == "15m":
        return f["fam"] in ("Spring", "Shakeout", "BO")
    return f["fam"] in ALERT_FAMS


def backs(f: dict, cur: dict | None) -> dict | None:
    """The phase-C test an SOS / SOW confirms: same range, same side, before it."""
    if f["fam"] != "BO" or not cur:
        return None
    ev = [c for c in cur.values() if c["rid"] == f["rid"] and c["side"] == f["side"] and c["fam"] in PHASE_C
          and pd.Timestamp(c["bar"]) < pd.Timestamp(f["bar"])]
    return max(ev, key=lambda c: c["bar"]) if ev else None


def fact_text(f: dict, cur: dict | None = None) -> str:
    when = pd.Timestamp(f["bar"]).tz_convert("Africa/Nairobi").strftime("%a %d %b %H:%M")
    rate = STANDS.get(f["fam"])
    prov = " · provisional: phase-C labels are revised about half the time" if f["fam"] in PHASE_C else ""
    s = (f"🧾 <b>{f['key']} {TF_NAME[f['tf']]} · {f['name']} confirmed</b> at {_p(f['price'])} ({when})"
         + (f" · such labels stand {rate}%" if rate is not None else "") + prov)
    c = backs(f, cur)
    if c:                       # the fact confirming itself: the SOS / SOW that follows a phase-C test
        s += (f"\n✅ it backs the {c['name']} {_p(c['price'])} — after the {f['name'].split(' #')[0]}, "
              f"such labels stand {STANDS_AFTER_D.get(c['fam'], STANDS.get(c['fam']))}% (was {STANDS.get(c['fam'])}%)")
    return s


def correction_text(f: dict) -> str:
    when = pd.Timestamp(f["bar"]).tz_convert("Africa/Nairobi").strftime("%a %d %b %H:%M")
    now = f"now read as <b>{f['now']}</b>" if f.get("now") else "no longer counted as an event"
    extra = " (it was phase B — the phase-C test is still to come or is a newer one)" if f["fam"] in PHASE_C and f.get("now") else ""
    return (f"✏️ <b>Correction · {f['key']} {TF_NAME[f['tf']]}</b> · the {f['name']} of {when} ({_p(f['price'])}) is {now}{extra}. "
            f"<i>Corrections stick 85% of the time (replay).</i>")


def log_claim(kind: str, key: str, tf: str, side: int, price: float, target: float, inv: float, when) -> dict | None:
    """A forward record for the weekly audit: price must reach `target` before a CLOSE beyond `inv`.
    `chance` = the random-walk odds of that race (distance to the invalidation ÷ both distances)."""
    dt, di = (target - price) * side, (price - inv) * side
    if dt <= 0 or di <= 0:
        return None
    return dict(kind=kind, key=key, tf=tf, side=side, t=pd.Timestamp(when).isoformat(), c=price, target=target, inv=inv,
                chance=round(di / (di + dt), 3), status="open")


def score_claims(log: list, bars_15m: dict, cap: int = 960) -> list:
    """Close every open claim whose race finished on the 15m bars (cap ≈ 10 trading days)."""
    for c in log:
        if c["status"] != "open":
            continue
        b = bars_15m.get(c["key"])
        if b is None:
            continue
        t = pd.to_datetime(b["time"], utc=True)
        i = int(np.searchsorted(t.values, pd.Timestamp(c["t"]).to_datetime64(), side="right"))
        hi, lo, cl = b["high"].to_numpy(float), b["low"].to_numpy(float), b["close"].to_numpy(float)
        s = c["side"]
        for j in range(i, min(len(b), i + cap)):
            if (lo[j] <= c["target"]) if s == -1 else (hi[j] >= c["target"]):
                c["status"] = "hit"
                break
            wick = c.get("wick")             # campaign claims: both sides on the wick (R + ½H vs S − ½H)
            if ((hi[j] if wick else cl[j]) > c["inv"]) if s == -1 else ((lo[j] if wick else cl[j]) < c["inv"]):
                c["status"] = "miss"
                break
        else:
            if len(b) - i >= cap:
                c["status"] = "miss"
    return log


def campaign_claim(key: str, st: dict, price: float, when) -> dict | None:
    """The read as a forward claim: the range resolves to its side first (R + ½H up / S − ½H down)."""
    H = st["R"] - st["S"]
    if not st["side"] or H <= 0 or st.get("grade") == "resolved":
        return None
    up, dn = st["R"] + 0.5 * H, st["S"] - 0.5 * H
    c = log_claim("campaign", key, st["tf"], st["side"], price, up if st["side"] > 0 else dn, dn if st["side"] > 0 else up, when)
    if c:
        c.update(wick=True, label=st["label"], grade=st["grade"], phase=st["phase"])
    return c


def claims_report(log: list) -> str:
    out = []
    for kind, lab in (("campaign", "Campaign read (resolves to its side first)"), ("child", "Child range at a parent edge, pointing back in"),
                      ("touch", "First touch of a confirmed edge")):
        done = [c for c in log if c["kind"] == kind and c["status"] != "open"]
        if not done:
            out.append(f"• {lab}: no finished cases yet ({sum(1 for c in log if c['kind'] == kind)} open)")
            continue
        hit = sum(c["status"] == "hit" for c in done)
        ch = 100 * sum(c["chance"] for c in done) / len(done)
        what = "resolved to its side first" if kind == "campaign" else "reached the range middle first"
        out.append(f"• {lab}: {what} {100 * hit / len(done):.0f}% of {len(done)} (random-walk chance {ch:.0f}%)")
    return "\n".join(out)


# ───────────────────────────── the campaign story (Hamza 30 Oct 2026) ─────────────────────────────
# What the course-logic read (whs/campaign.py) has added over the position chance in the replay (Heal study no.3,
# 514 ranges, bootstrap by range): points of "called the side it resolved to" above chance.
TRACK = {"all": (3.3, 514), "A": (3.8, 416), "B": (5.7, 218), "C": (-2.1, 109), "D": (1.8, 207)}
PNF_NOTE = "A yardstick, not a magnet (replay: hit about as often as any target of the same size)"
ROLE = ("campaign", "structure", "test", "test")


def _rng(P: dict) -> dict:
    return dict(S=P["support"], R=P["resistance"], kind=P.get("kind"), phase=P.get("phase"), outcome=P.get("outcome"),
                events=P.get("ev2") or [])


def _major(rec: dict | None) -> int:
    m = (rec or {}).get("major")
    return int(m.get("state") or 0) if isinstance(m, dict) else int(m or 0)


def reads(key: str, recs: dict, price: float, bars: dict | None = None, atrs: dict | None = None) -> list[dict]:
    """The campaign read of every live range of `key`, top-down (1D → 15m), with the minor ranges at its edges."""
    frames = {f["tf"]: f for f in describe(key, recs, {k: (r or {}).get("verdict", "") for k, r in recs.items()}, price)["frames"]}
    out = []
    for tf in TF_ORDER:
        rec = recs.get((key, tf))
        P = live_range(rec)
        if not P:
            continue
        kids = (frames.get(tf) or {}).get("children") or []
        rd = cp.read(_rng(P), price, sw=(rec or {}).get("sw"), verdict=(rec or {}).get("verdict", ""), major=_major(rec), children=kids)
        rd.update(tf=tf, id=P["id"], kids=kids, edges=edges_confirmed(P), events=event_line(P, 5))
        b = (bars or {}).get((key, tf))
        a = (atrs or {}).get((key, tf))
        if b is not None and a:
            t = pd.to_datetime(b["time"], utc=True)
            cl = b["close"].to_numpy(float)[int(np.searchsorted(t.values, pd.Timestamp(P["id"]).to_datetime64())):]
            rd["obj"] = cp.objectives(rd, cl, a)
        out.append(rd)
    return out


def _odds_line(rd: dict) -> str:
    if rd["chance_up"] is None:
        return "▸ <b>Odds</b> — the structure range has resolved; odds come back with the next range"
    up = round(100 * rd["chance_up"])
    s = f"▸ <b>Odds</b> it resolves up first {up}% · down {100 - up}% (from where price sits). "
    if not rd["side"]:
        return s + "The read has no side to call yet"
    tr = TRACK.get(rd["phase"]) or TRACK["all"]
    return s + f"Reads like this beat chance by {tr[0]:+.1f} pts in the replay (phase {rd['phase']}, n {tr[1]} ranges)"


def story_text(key: str, rds: list[dict], price: float | None = None) -> str:
    if not rds:
        return f"📖 <b>{key} · the story</b>\nno live range with a confirmed AR on 1D–15m — no campaign to read"
    lines = [f"📖 <b>{key} · the story</b>"]
    live = [r for r in rds if r["grade"] != "resolved"]
    struct = next((r for r in live if r["tf"] == "4h"), None) or (live[0] if live else rds[0])
    shown = []
    for i, rd in enumerate(rds):
        same = next((x for x in shown if _overlap(dict(S=x["S"], R=x["R"]), dict(support=rd["S"], resistance=rd["R"])) >= 0.8), None)
        role = "campaign" if i == 0 else "structure" if rd is struct else "test"
        if same:
            role = f"(same box as the {TF_NAME[same['tf']]})"
        shown.append(rd)
        where = f" · {cp.where_text(rd, price)}" if price is not None and not same else ""
        lines.append(f"<b>{TF_NAME[rd['tf']]} {role}</b> {_p(rd['S'])} – {_p(rd['R'])} · phase {rd['phase']} · "
                     f"<b>{cp.label_text(rd)}</b>{where}")
        why = cp.items_text(rd, 3)
        if why:
            lines.append(f"   {why}")
    lines.append(f"▸ <b>Confirms</b> {struct['confirm']} ({TF_NAME[struct['tf']]})")
    if struct["side"] and struct["grade"] != "resolved":
        lines.append(f"▸ <b>Invalidates</b> {struct['invalid']}")
    lines.append(f"▸ <b>Watching next</b> {struct['watch']}")
    o = struct.get("obj")
    if o and struct["side"] and struct["grade"] != "resolved":
        edge = struct["S"] if struct["side"] < 0 else struct["R"]
        beyond = (o["cons"] < edge) if struct["side"] < 0 else (o["cons"] > edge)
        lines.append(f"▸ <b>Count</b> {_p(o['cons'])} (from the {'top' if struct['side'] < 0 else 'low'}) – {_p(o['aggr'])} "
                     f"(from the count line), {o['cols']} columns"
                     + ("" if beyond else f" — the cause built so far does not yet reach beyond the {'support' if struct['side'] < 0 else 'resistance'}")
                     + f". {PNF_NOTE}")
    lines.append(_odds_line(struct))
    return "\n".join(lines)


def state_of(rds: list[dict]) -> dict:
    """What the quiet alerts compare run to run: per timeframe the range and its read."""
    return {rd["tf"]: dict(id=rd["id"], label=rd["label"], side=rd["side"], grade=rd["grade"], S=rd["S"], R=rd["R"], chance=rd["chance_up"],
                           inv=rd.get("inv_p"), phase=rd["phase"]) for rd in rds}
