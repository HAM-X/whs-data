"""Robot v2 · every chart's decision, its VSA in notation, its card image, the market board and the message templates.

build() is called every robot run after the scan and the news. It returns
  decisions  {"6E 4h": {action, tone, short, do, trigger, invalid, levels, control, caption, vsa}, …, "_today": text}
  images     {"card:6E:4h": png path, …, "board": png path}
The bot reads data/decisions.json (texts) and Cloudflare KV img:<name> (images) — nothing is computed in the bot.

TEMPLATES (Telegram HTML; one shape everywhere: title line · action · 3–5 labelled lines · next commands)
  chart    <b>6S · 1h</b>  ▸ <b>WAIT</b> / do / Trigger / Invalid or Levels / Control / News / → /vsa 6S 1h
  alert    🔔 <b>6S 1h · WAIT → BUY ZONE</b> + the chart caption (only when the action changes to or from an actionable one)
  vsa      verdict / Swings <pre> / Last bars <pre> / what the symbols mean (one line)
  today    Act on / Watch / News / Data
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from . import core, decide, hierarchy, news, render, scan as scanmod, swings, vsa
from .config import FUTURES

LOCAL = "Africa/Nairobi"
LAST_RECS: dict = {}     # (key, tf) → scan record of the last build (1D, 4h, 1h, 15m) — for the hierarchy alerts
LAST_ATRS: dict = {}
SIGNAL_ALERTS = False   # Hamza 29 Oct: "I don't want your signals, I want the fact as it comes and confirms itself" —
                        # no 🔔 for phase-C setups; they stay on the cards / board and are still logged as calls
ACTIONABLE = {"BUY ZONE", "SELL ZONE"}   # heal: only phase-C entries alert (phase D / WATCH: no edge in the replay)


def flow(led, last: int) -> int:
    """+1 / −1 when the last finished swing is BLUE (ease of movement: more result on less effort than the previous
    swing of the same side — WTC P-S5 traffic light), else 0. The only direction read that beat a coin out of sample."""
    sws = [x for x in led.swings if x.confirm_i is not None and x.confirm_i <= last]
    return int(sws[-1].direction) if sws and sws[-1].tl == swings.TL_BLUE else 0


def act_ok(d: dict) -> bool:
    """An entry worth acting on: a phase-C setup (Spring / Shakeout / UTAD) inside its 2-bar window, not failed (heal fixes 1–3)."""
    return bool(d.get("setup")) and d.get("action") in ("BUY ZONE", "SELL ZONE")


def _loc(iso, fmt="%a %H:%M"):
    return pd.Timestamp(iso).tz_convert(LOCAL).strftime(fmt)


def _p(x):
    return f"{x:.5g}" if x is not None else "—"


def vsa_text(key: str, tf: str, t: pd.Series, vs: dict) -> str:
    sw, bs = [], []
    for s in vs["swings"]:
        sw.append(f"{'▲ rally   ' if s['dir'] == 1 else '▼ reaction'} {_loc(t.iloc[s['end']].isoformat(), '%d %b %H:%M')}"
                  f"  V{s['v']:<2} R{s['r']:<2} → {s['code']:<2} {s['dom']}")
        sw.append(f"   {s['text']}")
    for b in vs["bars"]:
        bs.append(f"{_loc(t.iloc[b['i']].isoformat(), '%d %H:%M')}  {b['expr'].split(' → ')[0]:<22} → {b['code']}")
        bs.append(f"   {b['text']}")
    return "\n".join([f"<b>{key} · {tf} · VSA</b>   {_esc(vs['verdict'])}", "",
                      "<b>Swings</b>  each vs the last swing of the same side", "<pre>" + _esc("\n".join(sw)) + "</pre>",
                      "<b>Last bars</b>", "<pre>" + _esc("\n".join(bs)) + "</pre>",
                      "<i>V volume · R result (spread or travel) · D demand · S supply · " + _esc(">> much more · > more · < less · << much less") + "</i>"])


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def caption(key: str, tf: str, dec: dict, g: dict | None, control: str, news_line: str) -> str:
    lines = [f"<b>{key} · {tf}</b>  ▸ <b>{_esc(dec['action'])}</b>", _esc(dec["do"]), "",
             f"<b>Trigger</b>  {_esc(dec['trigger'])}"]
    if dec.get("invalid") and dec["invalid"] != "—":
        lines.append(f"<b>Invalid</b>  {_esc(dec['invalid'])}")
    elif g:
        lines.append(f"<b>Levels</b>  {_p(g['support'])} – {_p(g['resistance'])}")
    lines.append(f"<b>Control</b>  {_esc(control)}")
    lines.append(f"<b>News</b>  {_esc(news_line)}")
    lines.append(f"<i>/vsa {key} {tf} · /plan {key} long|short entry stop</i>")
    return "\n".join(lines)[:1024]


LTF_TAIL = {"15m": 2000, "5m": 3000}         # bars read on the lower timeframes (≈ 3 weeks of 15m, ≈ 10 days of 5m)
BULL = {"BUY ZONE", "BUY PULLBACKS"}
BEAR = {"SELL ZONE", "SELL RALLIES"}


def nested(price: float, d4: dict | None, d1: dict | None) -> dict:
    """Where price sits in the 4h / 1h ranges: side +1 at or just below a support (spring territory), −1 at or just
    above a resistance, 0 inside. A 15m / 5m setup only matters at those edges."""
    for tf, d in (("4h", d4), ("1h", d1)):
        if not d or not d.get("levels"):
            continue
        S, R = d["levels"]
        H = R - S
        if H <= 0:
            continue
        pos = (price - S) / H
        if -0.5 <= pos <= 0.25:
            where = "below" if pos < 0 else "at"
            return dict(side=1, tf=tf, level=S, line=f"price {where} the {tf} support {_p(S)} ({pos:+.2f} of the {tf} range)")
        if 0.75 <= pos <= 1.5:
            where = "above" if pos > 1 else "at"
            return dict(side=-1, tf=tf, level=R, line=f"price {where} the {tf} resistance {_p(R)} ({pos:+.2f} of the {tf} range)")
        if 0 < pos < 1:
            return dict(side=0, tf=tf, level=None, line=f"price inside the {tf} range {_p(S)}–{_p(R)}, away from its edges")
    acts = " · ".join(f"{tf} {d['action']}" for tf, d in (("4h", d4), ("1h", d1)) if d)
    return dict(side=0, tf=None, level=None, line=f"no 4h / 1h range at price ({acts})")


WATCH_ATR = 0.3          # an edge is "in play" within 0.3 ATR of it (or up to half the range beyond it)


def watch(price: float, atr: float, dec: dict) -> dict | None:
    """Price at the edge of a WAITING range: the place a spring / UTAD — the course's POE 1 — would happen."""
    if dec.get("action") not in ("WAIT", "WATCH") or not dec.get("levels") or not atr or np.isnan(atr):
        return None
    S, R = dec["levels"]
    H = R - S
    if H <= 0:
        return None
    if S - 0.5 * H <= price <= S + WATCH_ATR * atr:
        return dict(edge="support", level=S, side=1, beyond=bool(price < S))
    if R - WATCH_ATR * atr <= price <= R + 0.5 * H:
        return dict(edge="resistance", level=R, side=-1, beyond=bool(price > R))
    return None


def watch_text(k: str, w: dict) -> str:
    key, tf = k.split()
    if w["side"] == 1:
        what = ("below" if w["beyond"] else "at") + f" its support {_p(w['level'])}"
        nxt = "don't buy the edge. Only a spring that CLOSES back above it is a BUY ZONE"
    else:
        what = ("above" if w["beyond"] else "at") + f" its resistance {_p(w['level'])}"
        nxt = "don't sell the edge. Only a UTAD that CLOSES back below it is a SELL ZONE"
    # heal fix 5 · 22-month replay (1h + 4h, 1,590 edge visits): price broke > 1 ATR through the edge 52–63 % of the
    # time and turned back into the range only 26–39 % — an edge is a place to wait, not a reason to expect a turn
    return (f"👀 <b>{k} · price {what}</b>\n{_esc(nxt)}\n"
            f"<i>History: from here price broke through more often than it turned (≈ 6 in 10 vs 3 in 10).</i>\n"
            f"<i>Prepare: /zoom {key} for the 15m · /zoom {key} 5m · /why {key} {tf}</i>")


def _extras(vs: dict, rec: dict) -> dict:
    sw = vs["swings"][-1] if vs["swings"] else None
    b = vs["bars"][-1] if vs["bars"] else None
    return dict(major=(rec.get("major") or {}).get("state"),
                sw=dict(dir=sw["dir"], code=sw["code"], v=sw["v"], r=sw["r"], text=sw["text"], pct=sw["pct"]) if sw else None,
                lb=dict(expr=b["expr"], code=b["code"], text=b["text"]) if b else None)


UPS = {"TREND ↑", "BUY ZONE", "BUY PULLBACKS"}
DNS = {"TREND ↓", "SELL ZONE", "SELL RALLIES"}


def direction(dec: dict, key: str) -> tuple[int, str]:
    """Heal fix 4 · the board no longer adds trend votes into UP / DOWN / MIXED (49.5 % right live, 48–51 % on 22
    months of replay — a coin). It shows the one read that beat the coin out of sample: the course's BLUE swing
    (ease of movement, WTC P-S5 traffic light) on the 1h — the last finished swing travelled further on less effort
    than the one before. Replay: 53 % in-sample, 58 % hold-out over the next 12 h. Weak evidence, said as such."""
    f = (dec.get(f"{key} 1h") or {}).get("flow") or 0
    if f == 1:
        return 1, "▲ EASE 1h · weak 56%"
    if f == -1:
        return -1, "▼ EASE 1h · weak 56%"
    return 0, "◆ NONE no tested read"


SHORT_ACT = {"BUY ZONE": "BUY ZONE", "SELL ZONE": "SELL ZONE", "BUY PULLBACKS": "BUY PB", "SELL RALLIES": "SELL RALLY",
             "WATCH": "WATCH", "WAIT": "WAIT", "TREND ↑": "TREND ↑", "TREND ↓": "TREND ↓", "SIDEWAYS": "SIDEWAYS",
             "NO NEW ENTRIES": "NEWS"}


def do_line(d4: dict, d1: dict, d15: dict, d5: dict) -> str:
    """One instruction per pair. Only a fresh, unbroken phase-C setup is an instruction to act."""
    for tf, d in (("15m", d15), ("5m", d5)):
        if d.get("strong"):
            return f"{tf} {d['action']} at the HTF edge — {d['trigger']}"
    for tf, d in (("4h", d4), ("1h", d1)):
        if act_ok(d):
            return f"{tf} {d['action']} — {d['trigger']} · invalid {d.get('invalid', '—')}"
        if d.get("action") == "NO NEW ENTRIES":
            return f"no new entries — {d['do']}"
    for tf, d in (("4h", d4), ("1h", d1)):
        if d.get("failed") or d.get("used"):
            return f"{tf} {d['do']}"
    for tf, d in (("4h", d4), ("1h", d1)):
        if d.get("action") in ("WAIT", "WATCH") and d.get("levels"):
            S, R = d["levels"]
            return f"wait ({tf}) — act only on a spring that closes back above {_p(S)} or a UTAD back below {_p(R)}"
    for tf, d in (("4h", d4), ("1h", d1)):
        if d.get("action") in BULL | BEAR:
            return f"{tf} phase D ({d['action'].lower()}) — information only, no tested edge"
    a4 = d4.get("action")
    if a4 in ("TREND ↑", "TREND ↓"):
        return f"4h {a4} — nothing at a level; {d4.get('invalid', '')}"
    return d4.get("do", "—")


def hourly_texts(dec: dict, changes: list, up: list, now) -> tuple[str, str]:
    """(caption for the board photo ≤ 1024, detail text ≤ 4000): the hourly update."""
    keys = [k for k in ("6E", "6B", "6A", "6N", "6J", "6C", "6S", "GC") if f"{k} 4h" in dec]
    loc = now.tz_convert(LOCAL)
    act = [(k, d) for k, d in dec.items() if not k.startswith("_") and act_ok(d) and (not k.endswith("m") or d.get("aligned"))]
    edge = [k for k in keys if any((dec.get(f"{k} {tf}") or {}).get("nest", {}).get("side") for tf in ("15m", "5m"))]
    soon = [e for e in up if pd.Timestamp(e["t"]) <= now + pd.Timedelta(hours=2)]
    dl = (dec.get("_dollar") or "").split("\n")[0].replace("<b>", "").replace("</b>", "").replace("Dollar  ▸ ", "")
    cap = [f"<b>Hourly · {loc.strftime('%a %H:%M')}</b>  {_esc(dl)}"]
    cap.append("<b>Phase-C</b>  " + (" · ".join(f"{k} {_esc(d['action'])}" for k, d in act[:4]) if act else "no event now"))
    if edge:
        cap.append("<b>At a 4h/1h edge</b>  " + " · ".join(edge))
    if changes:
        cap.append("<b>Changed</b>  " + " · ".join(f"{k} {_esc(o)}→{_esc(n)}" for k, o, n in changes[:5]))
    cap.append("<b>News ≤2h</b>  " + (" · ".join(f"{e['ccy']} {_esc(e['title'][:24])} {_loc(e['t'], '%H:%M')}" for e in soon[:3])
                                       if soon else "none"))
    body = []
    for k in keys:
        d4, d1, d15, d5 = (dec.get(f"{k} {tf}") or {} for tf in ("4h", "1h", "15m", "5m"))
        _, lab = direction(dec, k)
        rng = f" {_p(d4['levels'][0])}–{_p(d4['levels'][1])}" if d4.get("levels") else ""
        lines = [f"<b>{k}</b>  {lab}  ·  4h {_esc(d4.get('action', '—'))}{_esc(rng)} · 1h {_esc(d1.get('action', '—'))} · "
                 f"15m {_esc(d15.get('action', '—'))} · 5m {_esc(d5.get('action', '—'))}"]
        sws = []
        for tf, d in (("4h", d4), ("1h", d1)):
            sw = d.get("sw")
            if sw:
                sws.append(f"{tf} {'▲' if sw['dir'] == 1 else '▼'} V{sw['v']} R{sw['r']} → {sw['code']} {sw['text']}")
        if sws:
            lines.append("   swing  " + _esc(" · ".join(sws)))
        bs = []
        for tf, d in (("1h", d1), ("5m", d5)):
            lb = d.get("lb")
            if lb:
                bs.append(f"{tf} {lb['expr'].split(' close')[0]} → {lb['code']} {lb['text']}")
        if bs:
            lines.append("   bar  " + _esc(" · ".join(bs)))
        ns = d15.get("nest") or {}
        if ns.get("side"):
            lines.append("   edge  " + _esc(ns["line"]) + (" ◐" if d15.get("aligned") or d5.get("aligned") else ""))
        lines.append("   <b>do</b>  " + _esc(do_line(d4, d1, d15, d5)))
        body.append("\n".join(lines))
    text = "\n\n".join(body) + "\n\n<i>/chart · /zoom · /vsa · /why any pair · /hourly off to stop these</i>"
    return "\n".join(cap)[:1024], text[:4000]


def why_text(key: str, tf: str, rec: dict, dec: dict) -> str:
    g = (rec.get("ranges") or [None])[-1]
    out = [f"<b>{key} · {tf} · the Wyckoff story</b>  ▸ <b>{_esc(dec['action'])}</b>", ""]
    if not g:
        m = rec.get("major") or {}
        out.append(_esc({1: "Major trend up", -1: "Major trend down"}.get(m.get("state"), "Sideways") +
                        (f" since {_loc(m['since'], '%d %b %H:%M')}" if m.get("since") else "") +
                        (f" · holds while price stays {'above' if m.get('state') == 1 else 'below'} {_p(m.get('protected'))}"
                         if m.get("protected") is not None and m.get("state") else "")))
        out.append("No live range: the robot reads a range once a climax stops the trend and the AR confirms it.")
        return "\n".join(out)
    out.append(f"<b>{_esc(g['name'])}</b> · phase {g['phase']} · {_p(g['support'])} – {_p(g['resistance'])}")
    ev = [f"{'✓' if st == 'confirmed' else '·'} {_loc(iso, '%d %b %H:%M')}  {name:<8} {_p(pr)}" for iso, name, st, pr in g["events"]]
    if ev:
        out.append("<pre>" + _esc("\n".join(ev[-10:])) + "</pre>")
    out += [_esc(x) for x in g.get("story") or []]
    out.append(f"<b>Bias</b>  {_esc(g.get('bias', ''))}")
    out.append("<i>✓ confirmed · candidate</i>")
    return "\n".join(out)


def build(now, scan: dict, cal: list, h1: dict, h4: dict, health_ok: bool = True, outdir: str | None = None,
          ltf: dict | None = None, snap: dict | None = None) -> tuple[dict, dict]:
    now = pd.Timestamp(now).tz_convert("UTC")
    outdir = outdir or tempfile.mkdtemp(prefix="whs_cards_")
    dec_all, images, board_rows = {}, {}, []
    recs, verdicts, atrs = {}, {}, {}
    up24 = news.grouped(news.upcoming(cal, now, 24))
    for key in FUTURES:
        if FUTURES[key].get("context") or f"{key} 4h" not in scan:
            continue
        nxt = [e for e in up24 if key in news.keys_for(e["ccy"])]
        news_line = (f"{nxt[0]['ccy']} {nxt[0]['title']} {_loc(nxt[0]['t'])} — no new entries ±60 min" if nxt
                     else "no red-folder release in the next 24 h")
        row = {"key": key}
        for tf, src, mins in (("4h", h4, 240), ("1h", h1, 60)):
            rec = scan.get(f"{key} {tf}") or {}
            d = src.get(key)
            if d is None or len(d) < 60:
                continue
            d = d.reset_index(drop=True)
            t = pd.to_datetime(d["time"], utc=True)
            o, h, l, c, v = (d[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
            eff = core.effort(v, t, mins)
            led = swings.ledger(o, h, l, c, eff)
            vs = vsa.read(o, h, l, c, eff, led, n_swings=4, n_bars=5)
            dec = decide.chart(rec, news.blackout(cal, key, now), _loc, tf)
            g = (rec.get("ranges") or [None])[-1]
            lr = next((s for s in reversed(vs["swings"]) if s["dir"] == 1), None)
            lx = next((s for s in reversed(vs["swings"]) if s["dir"] == -1), None)
            ctl = vs["verdict"].split(" ")[0] + "".join(
                f" · {nm} {s['code']} {s['text']}" for nm, s in (("rally", lr), ("reaction", lx)) if s)
            png = os.path.join(outdir, f"card_{key}_{tf}.png")
            try:
                render.chart_card(key, tf, d, eff, rec, dec, vs, news_line, png)
                images[f"card:{key}:{tf}"] = png
            except Exception as e:
                print(f"[card error] {key} {tf}: {type(e).__name__}: {e}")
            recs[(key, tf)], verdicts[(key, tf)], atrs[(key, tf)] = rec, vs["verdict"], float(core.atr(h, l, c)[-1])
            dec_all[f"{key} {tf}"] = dict(dec, control=ctl, verdict=vs["verdict"], levels=[g["support"], g["resistance"]] if g else None,
                                          flow=flow(led, len(c) - 1),
                                          caption=caption(key, tf, dec, g, ctl, news_line), vsa=vsa_text(key, tf, t, vs),
                                          why=why_text(key, tf, rec, dec), price=float(c[-1]),
                                          bar=t.iloc[-1].isoformat(), atr=float(core.atr(h, l, c)[-1]), **_extras(vs, rec))
            dec_all[f"{key} {tf}"]["watch"] = watch(float(c[-1]), dec_all[f"{key} {tf}"]["atr"], dec_all[f"{key} {tf}"])
            row["d4" if tf == "4h" else "d1"] = dec
            if tf == "4h":
                row["ctrl"] = vs["verdict"]
                row["ctrl_why"] = f"rally {lr['code'] if lr else '?'} · reaction {lx['code'] if lx else '?'}"
        if "d4" in row and "d1" in row:
            board_rows.append(row)
        # ---- 15m / 5m: their own structure, read against the 4h / 1h range edges ----
        for tf in ("15m", "5m"):
            try:
                d = (ltf or {}).get(tf, {}).get(key)
                if d is None or len(d) < 300:
                    continue
                d = d.tail(LTF_TAIL[tf]).reset_index(drop=True)
                rec = scanmod.read_one(d, tf, h1.get(key))
                t = pd.to_datetime(d["time"], utc=True)
                o, h, l, c, v = (d[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
                eff = core.effort(v, t, scanmod.TF_MIN[tf])
                led = swings.ledger(o, h, l, c, eff)
                vs = vsa.read(o, h, l, c, eff, led, n_swings=4, n_bars=5)
                dec = decide.chart(rec, news.blackout(cal, key, now), _loc, tf)
                g = (rec.get("ranges") or [None])[-1]
                ns = nested(float(c[-1]), dec_all.get(f"{key} 4h"), dec_all.get(f"{key} 1h"))
                act = dec["action"]
                aligned = (ns["side"] == 1 and (act in BULL or (act in ("WATCH", "WAIT") and vs["verdict"].startswith("DEMAND")))) or \
                          (ns["side"] == -1 and (act in BEAR or (act in ("WATCH", "WAIT") and vs["verdict"].startswith("SUPPLY"))))
                strong = aligned and act_ok(dec)
                tag = (" ✅ entry at the " + ns["tf"] + " edge" if strong else " ◐ supportive at the " + ns["tf"] + " edge" if aligned else "")
                lr = next((x for x in reversed(vs["swings"]) if x["dir"] == 1), None)
                lx = next((x for x in reversed(vs["swings"]) if x["dir"] == -1), None)
                ctl = vs["verdict"].split(" ")[0] + "".join(f" · {nm} {x['code']} {x['text']}" for nm, x in (("rally", lr), ("reaction", lx)) if x)
                cap = (f"<b>{key} · {tf}</b>  ▸ <b>{_esc(act)}</b>{_esc(tag)}\n<b>HTF</b>  {_esc(ns['line'])}\n" +
                       caption(key, tf, dec, g, ctl, news_line).split("\n", 1)[1])[:1024]
                png = os.path.join(outdir, f"card_{key}_{tf}.png")
                hl = []
                for tf2, dd in (("4h", dec_all.get(f"{key} 4h")), ("1h", dec_all.get(f"{key} 1h"))):
                    for nm, lv in zip(("support", "resistance"), (dd or {}).get("levels") or []):
                        same = next((x for x in hl if abs(x[1] - lv) <= 1e-9 + abs(lv) * 1e-5), None)
                        if same:                     # the same edge on 4h and 1h → one label
                            hl[hl.index(same)] = (same[0].replace(f" {nm}", f" · {tf2} {nm}"), lv)
                        else:
                            hl.append((f"{tf2} {nm}", lv))
                try:
                    render.chart_card(key, tf, d, eff, rec, dict(dec, action=act, htf=ns["line"] + tag), vs, news_line, png, htf_lines=hl)
                    images[f"card:{key}:{tf}"] = png
                except Exception as e:
                    print(f"[card error] {key} {tf}: {type(e).__name__}: {e}")
                if tf in hierarchy.TF_MIN:          # the hierarchy runs 1D → 15m, never lower (Hamza 29 Oct)
                    recs[(key, tf)], verdicts[(key, tf)], atrs[(key, tf)] = rec, vs["verdict"], float(core.atr(h, l, c)[-1])
                dec_all[f"{key} {tf}"] = dict(dec, control=ctl, verdict=vs["verdict"], levels=[g["support"], g["resistance"]] if g else None,
                                              caption=cap, vsa=vsa_text(key, tf, t, vs), why=why_text(key, tf, rec, dec),
                                              nest=ns, aligned=bool(aligned), strong=bool(strong), price=float(c[-1]),
                                              bar=t.iloc[-1].isoformat(), atr=float(core.atr(h, l, c)[-1]), **_extras(vs, rec))
            except Exception as e:
                print(f"[ltf read error] {key} {tf}: {type(e).__name__}: {e}")
        # ---- 1D + the range hierarchy (1D → 4h → 1h → 15m) ----
        try:
            d1 = hierarchy.daily_bars(h1[key]) if h1.get(key) is not None else None
            if d1 is not None and len(d1) >= 120:
                d1 = d1.tail(400).reset_index(drop=True)
                o, h, l, c, v = (d1[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
                eff = core.effort(v, d1["time"], 1440)
                recs[(key, "1d")] = scanmod.read_one(d1, "1d", None)
                verdicts[(key, "1d")] = vsa.read(o, h, l, c, eff, swings.ledger(o, h, l, c, eff))["verdict"]
                atrs[(key, "1d")] = float(core.atr(h, l, c)[-1])
            px = float(h1[key]["close"].iloc[-1])
            hz = hierarchy.describe(key, recs, verdicts, px)
            dec_all.setdefault("_htr", {})[key] = hz["text"]
            dec_all.setdefault("_htr_frames", {})[key] = hz["frames"]
        except Exception as e:
            print(f"[hierarchy error] {key}: {type(e).__name__}: {e}")
    global LAST_RECS, LAST_ATRS
    LAST_RECS, LAST_ATRS = recs, atrs
    png = os.path.join(outdir, "board.png")
    try:
        rows4 = []
        for row in board_rows:
            k = row["key"]
            rows4.append(dict(row, tfs=[(tf, dec_all.get(f"{k} {tf}")) for tf in ("4h", "1h", "15m", "5m")],
                              dirn=direction(dec_all, k)))
        render.board(rows4, now.tz_convert(LOCAL).strftime("%a %d %b %H:%M") + " UTC+3", png)
        images["board"] = png
    except Exception as e:
        print(f"[board error] {type(e).__name__}: {e}")
    dec_all["_today"] = today_text(dec_all, up24, now, health_ok)
    dec_all["_setups"] = setups_text(dec_all, now)
    try:
        dec_all["_dollar"] = dollar_text(scan, h1, h4, snap, cal, now)
    except Exception as e:
        print(f"[dollar error] {type(e).__name__}: {e}")
        dec_all["_dollar"] = "No dollar read this hour."
    dec_all["_at"] = now.isoformat()
    return dec_all, images


def today_text(dec: dict, up24: list, now, health_ok: bool) -> str:
    out = [f"<b>Today · {now.tz_convert(LOCAL).strftime('%a %d %b')}</b>"]
    act = [(k, d) for k, d in dec.items() if not k.startswith("_") and act_ok(d)]
    watch = [(k, d) for k, d in dec.items() if not k.startswith("_") and d["action"] == "WATCH"]
    wait, seenr = [], set()
    for k, d in sorted(((k, d) for k, d in dec.items() if not k.startswith("_") and d["action"] == "WAIT" and d.get("levels")),
                       key=lambda kd: kd[1]["short"].split(" ")[1] if kd[1]["short"].startswith("phase") else "Z", reverse=True):
        rid = (k.split()[0], tuple(round(x, 6) for x in d["levels"]))
        if rid not in seenr:                     # the same range on 4h and 1h → once
            seenr.add(rid)
            wait.append((k, d))
    out.append("")
    if act:
        out.append("<b>Phase-C events</b>  (check on your chart — no edge proven)")
        out += [f"• <b>{k}</b> {_esc(d['action'])} — {_esc(d['short'])}" for k, d in act[:5]]
    else:
        out.append("<b>Phase-C events</b>  none right now.")
    if watch or wait:
        out.append("<b>Watch</b>  (a test of the range edge decides these)")
        out += [f"• {k} {_esc(d['action'])} — {_esc(d['short'])}" for k, d in (watch + wait)[:4]]
    out.append("<b>News</b>  " + (" · ".join(f"{e['ccy']} {_esc(e['title'][:28])} {_loc(e['t'], '%H:%M')}" for e in up24[:4])
                                  if up24 else "no red-folder release in 24 h"))
    out.append("<b>Data</b>  " + ("🟢 fresh" if health_ok else "⚠️ a feed has a problem — trust TradingView"))
    out.append("<i>/chart 6E 4h · /vsa 6E · /plan · /news</i>")
    return "\n".join(out)[:1024]


def setups_text(dec: dict, now) -> str:
    items = [(k, d) for k, d in dec.items() if not k.startswith("_")]
    act = [(k, d) for k, d in items if act_ok(d) and (not k.endswith("m") or d.get("aligned"))]
    edge = [(k, d) for k, d in items if k.endswith("m") and d.get("nest", {}).get("side") and not (act_ok(d) and d.get("aligned"))]
    watch = [(k, d) for k, d in items if d["action"] == "WATCH" and not k.endswith("m")]
    out = [f"<b>Setups · {now.tz_convert(LOCAL).strftime('%a %H:%M')}</b>", ""]
    out.append("<b>Phase-C events</b>  (check on your chart — no edge proven)" if act else "<b>Phase-C events</b>  none right now")
    out += [f"• <b>{k}</b> {_esc(d['action'])} — {_esc(d['short'])}" + (" ✅ at the HTF edge" if d.get("strong") else "") for k, d in act[:6]]
    if edge:
        out.append("<b>At a 4h / 1h edge</b>  (where a 15m / 5m entry comes from)")
        seen = set()
        for k, d in edge:
            key = k.split()[0]
            if (key, d["nest"]["tf"]) in seen:
                continue
            seen.add((key, d["nest"]["tf"]))
            ltfs = " · ".join(f"{tf} {dec[f'{key} {tf}']['action']}{' ◐' if dec[f'{key} {tf}'].get('aligned') else ''}"
                              for tf in ("15m", "5m") if f"{key} {tf}" in dec)
            out.append(f"• {key} — {_esc(d['nest']['line'])} → {_esc(ltfs)}")
    if watch:
        out.append("<b>Watch</b>  (phase C forming)")
        out += [f"• {k} — {_esc(d['short'])}" for k, d in watch[:4]]
    out.append("<i>/zoom 6E for the 15m · /zoom 6E 5m · ◐ = the lower timeframe leans the edge's way</i>")
    return "\n".join(out)


CCY_KEY = {"EUR": "6E", "GBP": "6B", "AUD": "6A", "NZD": "6N", "JPY": "6J", "CAD": "6C", "CHF": "6S"}


def dollar_text(scan: dict, h1: dict, h4: dict, snap: dict | None, cal: list, now) -> str:
    acts, verd = {}, {}
    for tf, src, mins in (("4h", h4, 240), ("1h", h1, 60)):
        d = src.get("DX")
        if d is None or len(d) < 100:
            continue
        d = d.reset_index(drop=True)
        rec = scanmod.read_one(d, tf, h4.get("DX") if tf == "1h" else None)
        acts[tf] = decide.chart(rec, None, _loc, tf)["action"]
        t = pd.to_datetime(d["time"], utc=True)
        o, h, l, c, v = (d[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
        eff = core.effort(v, t, mins)
        verd[tf] = vsa.read(o, h, l, c, eff, swings.ledger(o, h, l, c, eff))["verdict"].split(" ")[0]
    up = {"TREND ↑", "BUY ZONE", "BUY PULLBACKS"}
    dn = {"TREND ↓", "SELL ZONE", "SELL RALLIES"}
    sc = sum(w * (1 if acts.get(tf) in up else -1 if acts.get(tf) in dn else 0) for tf, w in (("4h", 2), ("1h", 1)))
    sc += sum(0.5 * (1 if verd.get(tf) == "DEMAND" else -1 if verd.get(tf) == "SUPPLY" else 0) for tf in ("4h", "1h"))
    state = "USD STRONG" if sc >= 1.5 else "USD WEAK" if sc <= -1.5 else "USD MIXED"
    st = ((snap or {}).get("strength") or {}).get("1d") or []
    order = [x["ccy"] for x in st]
    out = [f"<b>Dollar</b>  ▸ <b>{state}</b>",
           f"DX 4h {_esc(acts.get('4h', '—'))} · 1h {_esc(acts.get('1h', '—'))} · control 4h {verd.get('4h', '—')} · 1h {verd.get('1h', '—')}"]
    if order:
        out.append("<b>Strength (1 day)</b>  " + " &gt; ".join(order))
        fx = [c for c in order if c in CCY_KEY]
        if state == "USD STRONG" and fx:
            out.append(f"<b>Favour</b>  shorts on the weakest: {', '.join(CCY_KEY[c] for c in fx[-2:])} · longs against the dollar need grade A")
        elif state == "USD WEAK" and fx:
            out.append(f"<b>Favour</b>  longs on the strongest: {', '.join(CCY_KEY[c] for c in fx[:2])} · gold leans up with a weak dollar")
        else:
            out.append("<b>Favour</b>  no dollar tailwind — trade each chart on its own structure")
    usd = [e for e in news.grouped(news.upcoming(cal, now, 24)) if e["ccy"] == "USD"]
    out.append("<b>US news</b>  " + (" · ".join(f"{_esc(e['title'][:30])} {_loc(e['t'], '%H:%M')}" for e in usd[:3]) if usd else "none in 24 h"))
    out.append("<i>Strength is context only — Edge Lab found no edge in it on its own.</i>")
    return "\n".join(out)


def changes(dec: dict, seen: dict) -> list[tuple[str, str, str]]:
    """(chart, old action, new action) where the move is to or from an actionable decision.
    15m / 5m charts alert only when the new decision is an entry AT a 4h / 1h edge (aligned)."""
    out = []
    for k, d in dec.items():
        if k.startswith("_"):
            continue
        old = seen.get(k)
        if k.endswith("m"):                      # 15m / 5m: only a NEW entry at a 4h / 1h edge alerts
            cur = d["action"] + " ✅" if d.get("strong") else d["action"]
            if d.get("strong") and old is not None and old != cur:
                out.append((k, old.replace(" ✅", ""), d["action"] + " at the " + d["nest"]["tf"] + " edge"))
            continue
        if old is None or old == d["action"]:
            continue
        if act_ok(d):
            out.append((k, old, d["action"]))
        elif old in ACTIONABLE and (d.get("failed") or d.get("used")):
            out.append((k, old, "failed" if d.get("failed") else "window closed"))
    return out
