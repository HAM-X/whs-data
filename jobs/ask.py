"""Ask the robot anything, fresh: any pair, any timeframe, any analysis — fetched from Yahoo at the moment you ask.

Pipdon BOT sends your words here (robot-now workflow, input ask=...) when its hourly read is too old or missing, or
when you add "now". This job pulls only what the question needs, reads it with the same engine as the hourly robot
and replies in Telegram (≈ 1 min).

  python -m jobs.ask "gc 15m vsa trend"        python -m jobs.ask "/zoom 6e 5m"        python -m jobs.ask "euro daily why"

PAIR       6E 6B 6A 6N 6J 6C 6S GC DX or their names (euro, cable, aussie, kiwi, yen, cad, franc, gold, dollar)
TIMEFRAME  5m 15m 30m 1h 4h 1d (daily) 1w (weekly) · default 4h
ANALYSIS   chart (card) · vsa · effort / result (bar table) · trend · swings · why (story) · zoom (= chart on 15m) ·
           all · several at once ("vsa trend swings") · default: chart
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from whs import bars, cards, core, decide, degrees, news, render, scan as scanmod, swings, telegram, vsa  # noqa: E402
from whs.config import FUTURES  # noqa: E402
from whs.sources import yahoo  # noqa: E402

ALIASES = {"6E": ["6e", "eur", "euro", "eurusd", "fiber"], "6B": ["6b", "gbp", "pound", "cable", "gbpusd", "sterling"],
           "6A": ["6a", "aud", "aussie", "audusd"], "6N": ["6n", "nzd", "kiwi", "nzdusd"], "6J": ["6j", "jpy", "yen", "usdjpy"],
           "6C": ["6c", "cad", "loonie", "usdcad"], "6S": ["6s", "chf", "swissy", "franc", "usdchf"],
           "GC": ["gc", "gold", "xau", "xauusd"], "DX": ["dx", "dxy", "dollar", "usdx", "usd"]}
TFS = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "60m": "1h", "4h": "4h", "1d": "1d", "d": "1d", "daily": "1d",
       "day": "1d", "1w": "1w", "w": "1w", "weekly": "1w", "week": "1w"}
FETCH = {"5m": ("5m", "30d"), "15m": ("15m", "60d"), "30m": ("30m", "60d"), "1h": ("1h", "730d"), "4h": ("1h", "730d"),
         "1d": ("1d", "10y"), "1w": ("1wk", "max")}
STEP = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1), "1w": pd.Timedelta(days=7)}
TAIL = {"5m": 3000, "15m": 2000, "30m": 2000, "1h": 3000, "4h": 1500, "1d": 1500, "1w": 600}
HTF_OF = {"5m": ("1h", "4h"), "15m": ("1h", "4h"), "30m": ("1h", "4h"), "1h": ("4h", "1d"), "4h": ("1d", "1w"), "1d": ("1w", None),
          "1w": (None, None)}
WORDS = {"chart": "chart", "card": "chart", "zoom": "chart", "ltf": "chart", "vsa": "vsa", "effort": "effort", "result": "effort",
         "volume": "effort", "bars": "effort", "trend": "trend", "structure": "trend", "swing": "swings", "swings": "swings",
         "why": "why", "story": "why", "wyckoff": "why", "all": "all", "everything": "all", "full": "all"}


def parse(text: str) -> tuple[str | None, str, list[str]]:
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("/", " ")).split()
    key = next((k for k, names in ALIASES.items() for w in words if w in names), None)
    tf = next((TFS[w] for w in words if w in TFS), None)
    if tf is None:
        tf = "15m" if any(w in ("zoom", "ltf") for w in words) else "4h"
    want = []
    for w in words:
        a = WORDS.get(w)
        if a and a not in want:
            want.append(a)
    if "all" in want:
        want = ["chart", "vsa", "trend", "swings", "effort", "why"]
    return key, tf, want or ["chart"]


def load(key: str, tf: str, now) -> pd.DataFrame:
    iv, per = FETCH[tf]
    raw, err = yahoo.fetch(FUTURES[key]["yahoo"], per, iv)
    if err:
        raise RuntimeError(f"Yahoo {key} {iv}: {err}")
    step = STEP["1h" if tf == "4h" else tf]
    d = bars.normalise(raw, now=now, step=step)
    if tf not in ("1d", "1w") and not FUTURES[key].get("context"):
        d = bars.clean_future(d)
    if tf == "4h":
        d = bars.build_4h(d, now=now)
    return d.tail(TAIL[tf]).reset_index(drop=True)


def read(key: str, tf: str, d: pd.DataFrame, htf: pd.DataFrame | None, cal: list, now) -> dict:
    t = pd.to_datetime(d["time"], utc=True)
    o, h, l, c, v = (d[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    eff = core.effort(v, t, scanmod.TF_MIN[tf])
    led = swings.ledger(o, h, l, c, eff)
    rec = scanmod.read_one(d, tf, htf)
    dec = decide.chart(rec, news.blackout(cal, key, now) if key != "DX" else None, cards._loc, tf)
    vs = vsa.read(o, h, l, c, eff, led, n_swings=6, n_bars=8)
    return dict(d=d, t=t, o=o, h=h, l=l, c=c, eff=eff, led=led, rec=rec, dec=dec, vs=vs)


def effort_text(key, tf, r) -> str:
    if np.isnan(r["eff"][-20:]).all():
        return (f"<b>{key} · {tf} · effort vs result</b>\n{key} has no volume on Yahoo (the ICE dollar index) — "
                "effort can't be read; use the futures' effort and /trend {key} for its structure.")
    rows = []
    for b in r["vs"]["bars"]:
        i = b["i"]
        rows.append(f"{cards._loc(r['t'].iloc[i].isoformat(), '%d %H:%M')}  {'▲' if b['dir'] == 1 else '▼' if b['dir'] == -1 else '•'} "
                    f"V {r['eff'][i]:4.1f}× {b['v']:<2}  R {b['r']:<2} close {b['close']:>3}%  → {b['code']}")
        rows.append(f"   {b['text']}")
    return (f"<b>{key} · {tf} · effort vs result</b>  (last {len(r['vs']['bars'])} bars)\n<pre>" + cards._esc("\n".join(rows)) +
            "</pre>\n<i>V = volume ÷ the same session slot's usual · R = spread vs ATR · close = where in the bar it closed</i>")


def trend_text(key, tf, r) -> str:
    mr = degrees.majors(r["o"], r["h"], r["l"], r["c"], r["eff"], r["led"])
    live = mr.majors[-1] if mr.majors else None
    out = [f"<b>{key} · {tf} · trend</b>  ▸ <b>{cards._esc(r['dec']['action'])}</b>"]
    if live is not None:
        st = {1: "UP", -1: "DOWN", 0: "SIDEWAYS"}[int(live.direction)]
        prot = mr.protected[-1]
        out.append(f"<b>Major</b>  {st} since {cards._loc(r['t'].iloc[live.start_i].isoformat(), '%d %b %H:%M')}" +
                   ("" if np.isnan(prot) else f" · holds while price stays {'above' if live.direction == 1 else 'below'} {prot:.5g}"))
        if not live.direction:
            out.append(f"<b>Box</b>  {live.lo:.5g} – {live.hi:.5g} (a climax stopped the trend)")
    sws = [s for s in r["led"].swings][-6:]
    if sws:
        hi = [s.end_p for s in sws if s.direction == 1]
        lo = [s.end_p for s in sws if s.direction == -1]
        hh = "higher highs" if len(hi) >= 2 and hi[-1] > hi[-2] else "lower highs" if len(hi) >= 2 else ""
        hl = "higher lows" if len(lo) >= 2 and lo[-1] > lo[-2] else "lower lows" if len(lo) >= 2 else ""
        out.append(f"<b>Swings</b>  {', '.join(x for x in (hh, hl) if x) or 'not enough swings'}")
    out.append(f"<b>Control</b>  {cards._esc(r['vs']['verdict'])}")
    ev = [x for x in mr.events][-3:]
    if ev:
        out.append("<b>Recent</b>  " + " · ".join(f"{cards._loc(r['t'].iloc[b].isoformat(), '%d %b %H:%M')} {cards._esc(txt)}" for b, txt in ev))
    return "\n".join(out)


def swings_text(key, tf, r) -> str:
    rows = []
    for s in r["vs"]["swings"]:
        rows.append(f"{'▲' if s['dir'] == 1 else '▼'} {cards._loc(r['t'].iloc[s['end']].isoformat(), '%d %b %H:%M')}  {s['pct']:+6.2f}%"
                    f"  {s['bars']:>3} bars  {s['effort']:.1f}×  V{s['v']:<2} R{s['r']:<2} → {s['code']:<2} {s['dom']}")
        rows.append(f"   {s['text']}")
    return f"<b>{key} · {tf} · swings</b>  {cards._esc(r['vs']['verdict'])}\n<pre>" + cards._esc("\n".join(rows)) + "</pre>"


def answer(text: str, now=None, send_text=telegram.send, send_photo=telegram.send_photo) -> list[str]:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    key, tf, want = parse(text)
    if key is None:
        send_text("Which pair? e.g. “gc 15m vsa trend”.", html=True)
        return []
    cal = []
    try:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "data" / "calendar.json"
        cal = json.loads(p.read_text()) if p.exists() else []
    except Exception:
        pass
    d = load(key, tf, now)
    h_a, h_b = HTF_OF[tf]
    htf_d = {x: load(key, x, now) for x in (h_a, h_b) if x}
    r = read(key, tf, d, htf_d.get(h_a), cal, now)
    htf_dec = {}
    for x, dx in htf_d.items():
        rr = scanmod.read_one(dx, x, None)
        g = (rr.get("ranges") or [None])[-1]
        dd = decide.chart(rr, None, cards._loc, x)
        dd["levels"] = [g["support"], g["resistance"]] if g else None
        htf_dec[x] = dd
    ns = cards.nested(float(r["c"][-1]), htf_dec.get(h_a) if h_a else None, htf_dec.get(h_b) if h_b else None)
    if ns["tf"] is None and (h_a or h_b):
        ns["line"] = ns["line"].replace("4h / 1h", " / ".join(x for x in (h_a, h_b) if x))
    age = f"⚡ live · {cards._loc(r['t'].iloc[-1].isoformat(), '%H:%M')} bar"
    sent = []
    g = (r["rec"].get("ranges") or [None])[-1]
    nxt = [e for e in news.grouped(news.upcoming(cal, now, 24)) if key in news.keys_for(e["ccy"])]
    news_line = (f"{nxt[0]['ccy']} {nxt[0]['title']} {cards._loc(nxt[0]['t'])} — no new entries ±60 min" if nxt
                 else "no red-folder release in the next 24 h")
    lr = next((s for s in reversed(r["vs"]["swings"]) if s["dir"] == 1), None)
    lx = next((s for s in reversed(r["vs"]["swings"]) if s["dir"] == -1), None)
    ctl = r["vs"]["verdict"].split(" ")[0] + "".join(f" · {nm} {s['code']} {s['text']}" for nm, s in (("rally", lr), ("reaction", lx)) if s)
    for a in want:
        if a == "chart":
            hl = [(f"{x} {nm}", lv) for x in (h_a, h_b) if x and htf_dec.get(x, {}).get("levels")
                  for nm, lv in zip(("support", "resistance"), htf_dec[x]["levels"])]
            png = os.path.join(tempfile.mkdtemp(), "ask.png")
            render.chart_card(key, tf, r["d"], r["eff"], r["rec"], dict(r["dec"], htf=ns["line"]), r["vs"], news_line, png,
                              htf_lines=hl)
            cap = (f"{age}\n<b>{key} · {tf}</b>  ▸ <b>{cards._esc(r['dec']['action'])}</b>\n<b>HTF</b>  {cards._esc(ns['line'])}\n" +
                   cards.caption(key, tf, r["dec"], g, ctl, news_line).split("\n", 1)[1])
            send_photo(png, cap[:1024])
            sent.append("chart")
        elif a == "vsa":
            send_text(age + "\n" + cards.vsa_text(key, tf, r["t"], r["vs"]), html=True)
            sent.append("vsa")
        elif a == "effort":
            send_text(age + "\n" + effort_text(key, tf, r), html=True)
            sent.append("effort")
        elif a == "trend":
            send_text(age + "\n" + trend_text(key, tf, r) + f"\n<b>HTF</b>  {cards._esc(ns['line'])}", html=True)
            sent.append("trend")
        elif a == "swings":
            send_text(age + "\n" + swings_text(key, tf, r), html=True)
            sent.append("swings")
        elif a == "why":
            send_text(age + "\n" + cards.why_text(key, tf, r["rec"], r["dec"]), html=True)
            sent.append("why")
    print(f"::notice title=ask::{text} → {key} {tf} {','.join(sent)}")
    return sent


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or os.environ.get("ASK", "")
    try:
        answer(q)
    except Exception as e:
        print(f"::error title=ask::{type(e).__name__}: {e}")
        telegram.send(f"⚠️ Could not read that fresh ({type(e).__name__}). Try /today or the same question again.")
        raise
