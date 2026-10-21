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

from . import core, decide, news, render, swings, vsa
from .config import FUTURES

LOCAL = "Africa/Nairobi"
ACTIONABLE = {"BUY ZONE", "SELL ZONE", "BUY PULLBACKS", "SELL RALLIES", "WATCH"}


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


def build(now, scan: dict, cal: list, h1: dict, h4: dict, health_ok: bool = True, outdir: str | None = None) -> tuple[dict, dict]:
    now = pd.Timestamp(now).tz_convert("UTC")
    outdir = outdir or tempfile.mkdtemp(prefix="whs_cards_")
    dec_all, images, board_rows = {}, {}, []
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
            dec = decide.chart(rec, news.blackout(cal, key, now), _loc)
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
            dec_all[f"{key} {tf}"] = dict(dec, control=ctl, verdict=vs["verdict"], levels=[g["support"], g["resistance"]] if g else None,
                                          caption=caption(key, tf, dec, g, ctl, news_line), vsa=vsa_text(key, tf, t, vs))
            row["d4" if tf == "4h" else "d1"] = dec
            if tf == "4h":
                row["ctrl"] = vs["verdict"]
                row["ctrl_why"] = f"rally {lr['code'] if lr else '?'} · reaction {lx['code'] if lx else '?'}"
        if "d4" in row and "d1" in row:
            board_rows.append(row)
    png = os.path.join(outdir, "board.png")
    try:
        render.board(board_rows, now.tz_convert(LOCAL).strftime("%a %d %b %H:%M") + " UTC+3", png)
        images["board"] = png
    except Exception as e:
        print(f"[board error] {type(e).__name__}: {e}")
    dec_all["_today"] = today_text(dec_all, up24, now, health_ok)
    dec_all["_at"] = now.isoformat()
    return dec_all, images


def today_text(dec: dict, up24: list, now, health_ok: bool) -> str:
    out = [f"<b>Today · {now.tz_convert(LOCAL).strftime('%a %d %b')}</b>"]
    act = [(k, d) for k, d in dec.items() if not k.startswith("_") and d["action"] in ACTIONABLE - {"WATCH"}]
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
        out.append("<b>Act on</b>")
        out += [f"• <b>{k}</b> {_esc(d['action'])} — {_esc(d['short'])}" for k, d in act[:5]]
    else:
        out.append("<b>Act on</b>  nothing — no chart has a confirmed entry.")
    if watch or wait:
        out.append("<b>Watch</b>  (a test of the range edge decides these)")
        out += [f"• {k} {_esc(d['action'])} — {_esc(d['short'])}" for k, d in (watch + wait)[:4]]
    out.append("<b>News</b>  " + (" · ".join(f"{e['ccy']} {_esc(e['title'][:28])} {_loc(e['t'], '%H:%M')}" for e in up24[:4])
                                  if up24 else "no red-folder release in 24 h"))
    out.append("<b>Data</b>  " + ("🟢 fresh" if health_ok else "⚠️ a feed has a problem — trust TradingView"))
    out.append("<i>/chart 6E 4h · /vsa 6E · /plan · /news</i>")
    return "\n".join(out)[:1024]


def changes(dec: dict, seen: dict) -> list[tuple[str, str, str]]:
    """(chart, old action, new action) where the move is to or from an actionable decision."""
    out = []
    for k, d in dec.items():
        if k.startswith("_"):
            continue
        old = seen.get(k)
        if old is not None and old != d["action"] and (d["action"] in ACTIONABLE or old in ACTIONABLE):
            out.append((k, old, d["action"]))
    return out
