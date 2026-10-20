"""Hourly job: download bars, store them, build 4h, check health, alert through Pipdon BOT.

Run:  python -m jobs.download            (normal: last 5 days, merged into stored files)
      python -m jobs.download --backfill (first run: Yahoo's full ~2 years of 1h bars)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from whs import bars, brief, desk, health, panel, scan, snapshot, telegram
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


def run(backfill: bool = False, now=None, fetch=yahoo.fetch_1h, send=telegram.send) -> dict:
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
    write_scan(send)
    write_brief(now, send)
    return summary


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


def write_scan(send=telegram.send) -> None:
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
        sp.write_text(json.dumps(marks, indent=1))
        if lines:
            send("🔎 Wyckoff scan\n" + "\n".join(lines[:25]) + ("\n…" if len(lines) > 25 else ""))
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


def write_brief(now, send=telegram.send) -> None:
    """Morning brief: rebuilt every run into data/brief.json (the bot's /brief); sent once a day from 08:00 UTC+3."""
    try:
        snap = json.loads((DATA / "snapshot.json").read_text())
        sc = json.loads((DATA / "scan.json").read_text())
        pn = json.loads((DATA / "panel.json").read_text()) if (DATA / "panel.json").exists() else None
        text = brief.build(snap, sc, now, pn)
        (DATA / "brief.json").write_text(json.dumps({"at": str(now), "text": text}))
        state = load_state()
        local = pd.Timestamp(now).tz_convert(LOCAL_TZ)
        today = local.strftime("%Y-%m-%d")
        if local.hour >= 8 and state.get("last_brief_day") != today and is_open(now):
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
    run(backfill=a.backfill)
