"""Hourly job: download bars, store them, build 4h, check health, alert through Pipdon BOT.

Run:  python -m jobs.download            (normal: last 5 days, merged into stored files)
      python -m jobs.download --backfill (first run: Yahoo's full ~2 years of 1h bars)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from whs import bars, health, snapshot, telegram
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    run(backfill=a.backfill)
