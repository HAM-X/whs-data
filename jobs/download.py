"""Hourly job: download bars, store them, build 4h, check health, alert through Pipdon BOT.

Run:  python -m jobs.download            (normal: last 5 days, merged into stored files)
      python -m jobs.download --backfill (first run: Yahoo's full ~2 years of 1h bars)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from whs import bars, brief, cfkv, desk, health, news, panel, scan, snapshot, telegram
from whs.config import FUTURES, LOCAL_TZ, SPOT
from whs.market import is_open
from whs.sources import yahoo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE = DATA / "state.json"


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"problems": {}, "last_ok_day": None}


def run(backfill: bool = False, now=None, fetch=yahoo.fetch_1h, send=telegram.send, news_fetch=None,
        cards_on: bool = False, ltf_fetch=None) -> dict:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    report, problems = {}, []
    universe = [(k, v, "future") for k, v in FUTURES.items()] + [(k, v, "spot") for k, v in SPOT.items()]
    for key, meta, kind in universe:
        path = DATA / "bars" / "1h" / f"{key}.csv"
        old = bars.read(path)
        period = "2y" if (backfill or old.empty) else "5d"
        raw, err = fetch(meta["yahoo"], period=period)
        new = bars.normalise(raw, now=now)
        if kind == "future" and not meta.get("context"):
            new = bars.clean_future(new)
        merged, added, revised = bars.merge(old, new)
        if kind == "future" and not meta.get("context"):
            merged = bars.clean_future(merged)      # also cleans bars stored before this rule existed
        if not merged.empty:
            bars.write(merged, path)
            if kind == "future":
                bars.write(bars.build_4h(merged, now=now), DATA / "bars" / "4h" / f"{key}.csv")
        optional_missing = meta.get("optional") and merged.empty
        health_kind = "spot" if meta.get("context") else kind   # context series: freshness checks only
        probs = [] if optional_missing else health.check_symbol(key, merged, now, err, health_kind)
        problems += probs
        report[key] = dict(
            kind=kind, yahoo=meta["yahoo"], bars=len(merged), added=added, revised=revised,
            first=merged["time"].iloc[0].isoformat() if len(merged) else None,
            last=merged["time"].iloc[-1].isoformat() if len(merged) else None,
            last_local=health.local_str(merged["time"].iloc[-1]) if len(merged) else None,
            error=err, status="unavailable" if optional_missing else ("ok" if not probs else "problem"),
        )
        print(f"{key:7s} {kind:6s} bars={len(merged):6d} +{added:<5d} rev={revised:<3d} "
              f"last={report[key]['last_local']} {'ERR ' + err if err else ''}")

    problems = health.collapse_marketwide(problems)
    state = load_state()
    before = state.get("problems", {})
    now_keys = {p["key"]: p["text"] for p in problems}
    new = [p for p in problems if p["key"] not in before]
    resolved = [k.split(":")[0] if not k.startswith("ALL") else "market-wide" for k in before if k not in now_keys]
    msg = health.alert_text(new, resolved)
    if msg:
        send(msg)

    # one calm line per trading day after 08:00 your time, only when everything is healthy
    local = now.tz_convert(LOCAL_TZ)
    today = local.strftime("%Y-%m-%d")
    if not problems and local.hour >= 8 and state.get("last_ok_day") != today and is_open(now):
        fut_last = max(pd.Timestamp(report[k]["last"]) for k in FUTURES if report[k]["last"])
        n_spot = sum(1 for k in SPOT if report[k]["status"] == "ok")
        send(f"🟢 Data healthy · all {len(FUTURES)} futures fresh (last bar {health.local_str(fut_last)}) · "
             f"{n_spot} spot pairs ok")
        state["last_ok_day"] = today

    state["problems"] = now_keys
    DATA.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))
    summary = dict(run_utc=now.isoformat(), run_local=health.local_str(now), backfill=backfill,
                   problems=problems, symbols=report)
    (DATA / "health.json").write_text(json.dumps(summary, indent=1, default=str))
    write_snapshot(summary, now)
    if ltf_fetch is not None:
        write_ltf_bars(now, ltf_fetch)
    write_news(now, send, news_fetch)
    write_scan(send, now)
    made = write_cards(now, not problems) if cards_on else None
    write_brief(now, send, made)
    return summary


def write_cards(now, health_ok: bool = True, send_photo=telegram.send_photo, kv=cfkv.put_bundle, alerts: bool = True):
    """Robot v2: a decision + VSA + card image per chart, the market board → data/decisions.json + the bot's KV;
    a 🔔 card when a chart's decision moves to or from an actionable one."""
    try:
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty-slice warnings on short intraday windows
        from whs import cards         # needs matplotlib; imported here so a run without it still collects data
        sc = json.loads((DATA / "scan.json").read_text())
        h1 = {k: bars.read(DATA / "bars" / "1h" / f"{k}.csv") for k in FUTURES}
        h4 = {k: bars.read(DATA / "bars" / "4h" / f"{k}.csv") for k in FUTURES}
        ltf = {tf: {k: bars.read(DATA / "bars" / tf / f"{k}.csv") for k in FUTURES} for tf in LTF}
        snap = json.loads((DATA / "snapshot.json").read_text()) if (DATA / "snapshot.json").exists() else {}
        dec, imgs = cards.build(now, sc, load_calendar(), h1, h4, health_ok, ltf=ltf, snap=snap)
        state = load_state()
        seen = state.get("decisions")
        ch = cards.changes(dec, seen) if seen is not None else []
        # Edge Lab 7.4: log the robot's own entries and score every open one (data/calls.json)
        from whs import calls as callsmod
        cpath = DATA / "calls.json"
        try:
            allc = json.loads(cpath.read_text()) if cpath.exists() else []
        except Exception:
            allc = []
        if alerts:
            callsmod.new_calls(callsmod.entries(dec), dec, allc)
        series = {(k, "4h"): h4[k] for k in h4} | {(k, "1h"): h1[k] for k in h1} | \
                 {(k, tf): ltf[tf][k] for tf in ltf for k in ltf[tf]}
        closed = callsmod.score(allc, series)
        dec["_calls"] = callsmod.report(allc, now)
        if alerts:
            cpath.write_text(json.dumps(allc, indent=0, default=str))
        (DATA / "decisions.json").write_text(json.dumps(dec, separators=(",", ":"), default=str))
        print(f"[cards] {len(imgs)} images, {kv(imgs)} uploaded · calls {len(allc)} ({len(closed)} closed now)")
        if alerts and seen is not None and is_open(now):
            for k, old, new in (ch[:4] if cards.SIGNAL_ALERTS else []):
                key, tf = k.split()
                cap = f"🔔 <b>{k} · {old} → {new}</b>\n" + dec[k]["caption"]
                if imgs.get(f"card:{key}:{tf}"):
                    send_photo(imgs[f"card:{key}:{tf}"], cap[:1024])
            # 👀 watch zones: price arrives at the edge of a waiting 4h / 1h range (once per edge)
            was = state.get("watch") or {}
            sent = 0 if "watch" in state else 99           # the first run only records the zones (no burst of alerts)
            for k, d in dec.items():
                w = d.get("watch") if not k.startswith("_") and not k.endswith("m") else None
                tag = f"{w['edge']}@{w['level']:.6g}" if w else None
                if w and was.get(k) != tag and sent < 3 and imgs.get("card:" + k.replace(" ", ":")):
                    send_photo(imgs["card:" + k.replace(" ", ":")], cards.watch_text(k, w)[:1024])
                    sent += 1
            for c in closed:
                txt = (f"📗 <b>Robot call closed · {c['key']} {c['tf']} {c['action']}</b> → {c['r']:+.2f}R "
                       f"(best {c['best']:+.1f}R, {c['held']} bars)")
                try:
                    telegram.send(txt, html=True)
                except TypeError:
                    telegram.send(txt)
        # the HOURLY UPDATE: board (4 timeframes) + overview caption, then one block per pair. Silent, except the first
        # one of the day from 08:00 (it replaces the morning brief). /hourly off in the bot stops it.
        cap, body = cards.hourly_texts(dec, ch, news.grouped(news.upcoming(load_calendar(), now, 24)), pd.Timestamp(now).tz_convert("UTC"))
        dec["_hourly"] = {"caption": cap, "body": body}
        (DATA / "decisions.json").write_text(json.dumps(dec, separators=(",", ":"), default=str))
        if kv is cfkv.put_bundle:
            cfkv.put_json("dec:latest", dec)
        if alerts and is_open(now) and imgs.get("board") and (cfkv.get_text("hourly") or "on") != "off":
            local = pd.Timestamp(now).tz_convert(LOCAL_TZ)
            first = local.hour >= 8 and state.get("last_brief_day") != local.strftime("%Y-%m-%d")
            try:
                send_photo(imgs["board"], cap, silent=not first)
            except TypeError:
                send_photo(imgs["board"], cap)
            try:
                telegram.send(body, html=True, silent=not first)
            except TypeError:
                telegram.send(body)
            if first:
                state["last_brief_day"] = local.strftime("%Y-%m-%d")
        if alerts:
            state["decisions"] = {k: d["action"] if not k.endswith("m") or not d.get("strong") else d["action"] + " ✅"
                                  for k, d in dec.items() if not k.startswith("_")}
            state["watch"] = {k: f"{d['watch']['edge']}@{d['watch']['level']:.6g}" for k, d in dec.items()
                              if not k.startswith("_") and not k.endswith("m") and d.get("watch")}
            local = pd.Timestamp(now).tz_convert(LOCAL_TZ)          # Saturday morning: the weekly self-grade
            wk = local.strftime("%G-%V")
            if local.weekday() == 5 and local.hour >= 8 and state.get("last_grade_week") != wk:
                try:
                    telegram.send(dec["_calls"], html=True)
                except TypeError:
                    telegram.send(dec["_calls"])
                state["last_grade_week"] = wk
            STATE.write_text(json.dumps(state, indent=1))
        return dec, imgs
    except Exception as e:   # cards must never break data collection
        print(f"[cards error] {type(e).__name__}: {e}")
        return None


LTF = {"15m": (pd.Timedelta(minutes=15), 60), "5m": (pd.Timedelta(minutes=5), 25)}   # step, days kept


def write_ltf_bars(now, fetch=yahoo.fetch) -> None:
    """15m and 5m bars of the futures (Yahoo keeps 60 days) → data/bars/15m, data/bars/5m. Never stops the robot."""
    for tf, (step, keep) in LTF.items():
        for key, meta in FUTURES.items():
            if meta.get("context"):
                continue
            try:
                path = DATA / "bars" / tf / f"{key}.csv"
                old = bars.read(path)
                raw, err = fetch(meta["yahoo"], "60d" if old.empty else "5d", tf)
                if err:
                    print(f"[ltf] {key} {tf}: {err}")
                    continue
                new = bars.clean_future(bars.normalise(raw, now=now, step=step))
                merged, _, _ = bars.merge(old, new)
                merged = merged[merged["time"] >= pd.Timestamp(now).tz_convert("UTC") - pd.Timedelta(days=keep)]
                bars.write(merged.reset_index(drop=True), path)
            except Exception as e:
                print(f"[ltf error] {key} {tf}: {type(e).__name__}: {e}")


def load_calendar() -> list:
    try:
        return json.loads((DATA / "calendar.json").read_text())
    except Exception:
        return []


def write_news(now, send=telegram.send, fetch=None) -> None:
    """Stage 6: calendar (merged, 21 days) → data/calendar.json; pre-release alerts; reactions → data/news.json."""
    try:
        cal = load_calendar()
        if fetch is not None:
            try:
                cal = news.merge(cal, news.normalise(fetch()), now)
                (DATA / "calendar.json").write_text(json.dumps(cal, separators=(",", ":")))
            except Exception as e:      # a calendar outage must never stop the robot; the stored week is used
                print(f"[calendar error] {type(e).__name__}: {e}")
        state = load_state()
        lines, ids = news.pre_alerts(cal, now, state.get("news_alerted", []))
        if lines and fetch is not None and is_open(now):   # tests (no fetch) never send
            try:
                send("\n".join(lines), html=True)
            except TypeError:
                send("\n".join(lines))
        state["news_alerted"] = ids
        STATE.write_text(json.dumps(state, indent=1))
        h1 = {k: bars.read(DATA / "bars" / "1h" / f"{k}.csv") for k in FUTURES}
        reacts = news.reactions(cal, h1, now)                 # last 36 h → the brief
        week = news.reactions(cal, h1, now, hours=168)        # last 7 days → /news
        (DATA / "news.json").write_text(json.dumps({"at": str(now), "text": news.report(cal, week, now),
                                                    "brief": news.brief_lines(cal, reacts, now), "reactions": week},
                                                   separators=(",", ":"), default=str))
    except Exception as e:
        print(f"[news error] {type(e).__name__}: {e}")


def write_snapshot(summary: dict, now) -> None:
    """Market snapshot for Pipdon BOT's commands (data/snapshot.json)."""
    try:
        h1 = {k: bars.read(DATA / "bars" / "1h" / f"{k}.csv") for k in FUTURES}
        h4 = {k: bars.read(DATA / "bars" / "4h" / f"{k}.csv") for k in FUTURES}
        sp = {k: bars.read(DATA / "bars" / "1h" / f"{k}.csv") for k in SPOT}
        snap = snapshot.build({k: v for k, v in h1.items() if len(v)}, {k: v for k, v in h4.items() if len(v)},
                              {k: v for k, v in sp.items() if len(v)}, summary, now)
        (DATA / "snapshot.json").write_text(json.dumps(snap, separators=(",", ":"), default=str))
    except Exception as e:   # the snapshot must never break data collection
        print(f"[snapshot error] {type(e).__name__}: {e}")


def write_scan(send=telegram.send, now=None) -> None:
    """Stage 3: read every chart (major swings + automatic ranges + Story) → data/scan.json, and alert what changed."""
    try:
        h1 = {k: bars.read(DATA / "bars" / "1h" / f"{k}.csv") for k in FUTURES}
        h4 = {k: bars.read(DATA / "bars" / "4h" / f"{k}.csv") for k in FUTURES}
        snap = json.loads((DATA / "snapshot.json").read_text()) if (DATA / "snapshot.json").exists() else {}
        late = {k: (v.get("adr_used_pct") or 0) >= panel.LATE for k, v in (snap.get("instruments") or {}).items()}
        res = scan.scan({k: v for k, v in h1.items() if len(v)}, {k: v for k, v in h4.items() if len(v)},
                        (snap.get("strength") or {}).get("1d"), late)
        (DATA / "scan.json").write_text(json.dumps(res, separators=(",", ":"), default=str))
        try:
            pn = panel.build(snap, res, {k: v for k, v in h1.items() if len(v)}, {k: v for k, v in h4.items() if len(v)})
            (DATA / "panel.json").write_text(json.dumps(pn, separators=(",", ":"), default=str))
        except Exception as e:
            print(f"[panel error] {type(e).__name__}: {e}")
        sp = DATA / "scan_state.json"
        seen = json.loads(sp.read_text()) if sp.exists() else {}
        plans = scan.fresh_plans(res, seen)
        lines, marks = scan.news(res, seen)
        lines += [scan.plan_line(p) for p in plans if p["refused"] or p["rr_low"]]
        plans = [p for p in plans if not (p["refused"] or p["rr_low"])]
        cal = load_calendar()                    # Stage 6 blackout: no new entries ±60 min of red-folder news
        at = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        held = [(p, news.blackout(cal, p["key"], at)) for p in plans]
        lines += [f"⛔ news blackout · {p['key']} {p['tf']} {p['setup']} held back: "
                  f"{b[0]['ccy']} {b[0]['title']} at {pd.Timestamp(b[0]['t']).tz_convert('Africa/Nairobi').strftime('%H:%M')}"
                  for p, b in held if b]
        plans = [p for p, b in held if not b]
        sp.write_text(json.dumps(marks, indent=1))
        if lines:   # Robot v2: the hourly event list is no longer sent — decision changes (write_cards) replace it
            print("[scan] " + " | ".join(lines[:25]))
        for p in plans[:6]:
            inst = (snap.get("instruments") or {}).get(p["key"]) or {}
            m5 = inst.get("mt5") or {}
            txt = desk.card(p, inst.get("price"), m5.get("spot"))
            try:
                send(txt, html=True)
            except TypeError:
                send(txt)
    except Exception as e:   # the scan must never break data collection
        print(f"[scan error] {type(e).__name__}: {e}")


def write_brief(now, send=telegram.send, made=None, send_photo=telegram.send_photo) -> None:
    """Morning brief: rebuilt every run into data/brief.json (the bot's /brief); sent once a day from 08:00 UTC+3."""
    try:
        snap = json.loads((DATA / "snapshot.json").read_text())
        sc = json.loads((DATA / "scan.json").read_text())
        pn = json.loads((DATA / "panel.json").read_text()) if (DATA / "panel.json").exists() else None
        nw = json.loads((DATA / "news.json").read_text()) if (DATA / "news.json").exists() else {}
        text = brief.build(snap, sc, now, pn, nw.get("brief"))
        (DATA / "brief.json").write_text(json.dumps({"at": str(now), "text": text}))
        state = load_state()
        local = pd.Timestamp(now).tz_convert(LOCAL_TZ)
        today = local.strftime("%Y-%m-%d")
        if local.hour >= 8 and state.get("last_brief_day") != today and is_open(now):
            if made and made[1].get("board"):            # Robot v2: the board + Today
                ok = send_photo(made[1]["board"], made[0]["_today"])
            else:
                try:
                    ok = send(text, html=True)
                except TypeError:            # simple senders (tests) take the text only
                    ok = send(text)
            if ok is not False:
                state["last_brief_day"] = today
                STATE.write_text(json.dumps(state, indent=1))
    except Exception as e:   # the brief must never break data collection
        print(f"[brief error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    run(backfill=a.backfill, news_fetch=news.fetch_week, cards_on=True, ltf_fetch=yahoo.fetch)
