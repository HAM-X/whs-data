"""Stage 6.1 + 6.5 · economic calendar, news blackout windows and the news-reaction analyser.

Fundamentals can block or downgrade a trade; they never create one.

  CALENDAR   the public weekly calendar feed (ForexFactory / faireconomy JSON), fetched every robot run and merged
             into data/calendar.json (kept 21 days, so last week's releases stay for the reaction analyser).
             Currencies → futures: USD → every future (DX included, GC through the dollar) · EUR 6E · GBP 6B ·
             AUD 6A · NZD 6N · JPY 6J · CAD 6C · CHF 6S.
  BLACKOUT   no NEW entry within ±60 min of a high-impact release on the instrument's currencies (covers both
             firms: FundedNext / FundingPips funded accounts count ±5 min). The robot's trade cards are held
             back inside a blackout and listed as "⛔ news" instead.
  PRE-ALERT  one Telegram line ≈ 1 h before each high-impact release: what, forecast / previous, which futures
             are blacked out until when; funded accounts: don't open or close within ±5 min.
  REACTION   for every high-impact release that has 3 hourly bars after it: the release bar's move from the
             previous close in ATR(14, 1h) units, its effort, and 3 bars later —
               absorbed       effort ≥ 2× and the bar closes < 0.35 ATR from where it started, range < 1.2 ATR
                              (big news, no result — Roman's effort without result: someone is taking the other side)
               followed       |move| ≥ 0.75 ATR and 3 bars later still ≥ half of it
               faded          |move| ≥ 0.75 ATR and 3 bars later ≤ a quarter of it left (often the test of a
                              news-driven climax)
               spike rejected the bar reached ≥ 1 ATR but closed < 0.4 ATR from where it started
               quiet          none of the above
             With the actual figure (when the feed carries it) the line also shows actual vs forecast.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from . import core

FEED = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
KEEP_DAYS = 21
BLACKOUT_MIN = 60
ALERT_AHEAD_MIN = 75
LOCAL = "Africa/Nairobi"
ALL = ["6E", "6B", "6A", "6N", "6J", "6C", "6S", "GC", "DX"]
CCY_KEYS = {"USD": ALL, "EUR": ["6E"], "GBP": ["6B"], "AUD": ["6A"], "NZD": ["6N"], "JPY": ["6J"],
            "CAD": ["6C"], "CHF": ["6S"]}
REACT_KEYS = {"USD": ["DX", "6E", "6J", "GC"]}          # USD releases: the dollar, the euro, the yen and gold


def fetch_week(url: str = FEED, timeout: int = 20) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "whs-robot/1.0 (+github.com/HAM-X/whs-data)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def normalise(raw: list) -> list[dict]:
    out = []
    for x in raw or []:
        try:
            ccy = (x.get("country") or "").upper()
            if ccy not in CCY_KEYS:
                continue
            t = pd.Timestamp(x["date"]).tz_convert("UTC")
            title = (x.get("title") or "").strip()
            out.append(dict(id=f"{ccy}|{title}|{t.isoformat()}", t=t.isoformat(), ccy=ccy, title=title,
                            impact=(x.get("impact") or "").capitalize(), forecast=x.get("forecast") or "",
                            previous=x.get("previous") or "", actual=x.get("actual") or ""))
        except Exception:
            continue
    return out


def merge(stored: list[dict], new: list[dict], now) -> list[dict]:
    now = pd.Timestamp(now).tz_convert("UTC")
    by = {e["id"]: e for e in stored or []}
    for e in new:
        old = by.get(e["id"], {})
        by[e["id"]] = {**old, **{k: v for k, v in e.items() if v != "" or k not in old}}
    keep = [e for e in by.values() if pd.Timestamp(e["t"]) >= now - pd.Timedelta(days=KEEP_DAYS)]
    return sorted(keep, key=lambda e: e["t"])


def high(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("impact") == "High"]


def grouped(events: list[dict]) -> list[dict]:
    """Releases of one currency at the same minute (e.g. CAD CPI m/m + median + trimmed) as ONE event:
    titles joined with ' · ', ids kept in `ids`; forecast / previous / actual from the first."""
    by = {}
    for e in sorted(events, key=lambda x: x["t"]):
        k = (e["ccy"], e["t"])
        if k not in by:
            by[k] = dict(e, ids=[e["id"]])
        else:
            by[k]["title"] += " · " + e["title"]
            by[k]["ids"].append(e["id"])
    return list(by.values())


def keys_for(ccy: str) -> list[str]:
    return CCY_KEYS.get(ccy, [])


def blackout(events: list[dict], key: str, at, minutes: int = BLACKOUT_MIN) -> list[dict]:
    """High-impact releases on `key`'s currencies within ±minutes of `at`."""
    at = pd.Timestamp(at).tz_convert("UTC")
    w = pd.Timedelta(minutes=minutes)
    return [e for e in high(events) if key in keys_for(e["ccy"]) and abs(pd.Timestamp(e["t"]) - at) <= w]


def _loc(t) -> str:
    return pd.Timestamp(t).tz_convert(LOCAL).strftime("%a %H:%M")


def _fx(e) -> str:
    bits = [f"act {e['actual']}" if e.get("actual") else "", f"fc {e['forecast']}" if e.get("forecast") else "",
            f"prev {e['previous']}" if e.get("previous") else ""]
    bits = [b for b in bits if b]
    return f" ({', '.join(bits)})" if bits else ""


def upcoming(events: list[dict], now, hours: float = 24) -> list[dict]:
    now = pd.Timestamp(now).tz_convert("UTC")
    return [e for e in high(events) if now <= pd.Timestamp(e["t"]) <= now + pd.Timedelta(hours=hours)]


def pre_alerts(events: list[dict], now, alerted: list[str]) -> tuple[list[str], list[str]]:
    """Lines for high-impact releases starting within ALERT_AHEAD_MIN that were not alerted yet."""
    now = pd.Timestamp(now).tz_convert("UTC")
    lines, ids = [], list(alerted or [])
    soon = [e for e in grouped(high(events)) if now < pd.Timestamp(e["t"]) <= now + pd.Timedelta(minutes=ALERT_AHEAD_MIN)
            and not set(e["ids"]) & set(ids)]
    for e in soon:
        t = pd.Timestamp(e["t"])
        mins = int((t - now).total_seconds() // 60)
        until = (t + pd.Timedelta(minutes=BLACKOUT_MIN)).tz_convert(LOCAL).strftime("%H:%M")
        ks = keys_for(e["ccy"])
        lines.append(f"📰 in {mins} min · {_loc(t)} <b>{e['ccy']} {e['title']}</b>{_fx(e)}\n"
                     f"   ⛔ no new entries on {'every chart' if len(ks) >= 8 else ' '.join(ks)} until {until} · "
                     f"funded accounts: don't open or close ±5 min")
        ids += e["ids"]
    cutoff = (now - pd.Timedelta(days=KEEP_DAYS)).isoformat()
    ids = [i for i in ids if i.split("|")[-1] >= cutoff]
    return lines, ids


def reaction(e: dict, d: pd.DataFrame) -> dict | None:
    """The release bar on one future's 1h bars (columns time, high, low, close, volume)."""
    if d is None or len(d) < 30:
        return None
    t = pd.Timestamp(e["t"]).tz_convert("UTC")
    times = pd.to_datetime(d["time"], utc=True)
    b = int(np.searchsorted(times.values, t.floor("h").to_datetime64(), side="left"))
    if b < 15 or b + 3 >= len(d) or times.iloc[b] != t.floor("h"):
        return None
    h, l, c = (d[x].to_numpy(float) for x in ("high", "low", "close"))
    a = core.atr(h, l, c)[b - 1]
    if not a or np.isnan(a):
        return None
    eff = core.effort(d["volume"].to_numpy(float), times, 60)[b]
    pre = c[b - 1]
    m0, m3 = (c[b] - pre) / a, (c[b + 3] - pre) / a
    up, dn = (h[b] - pre) / a, (pre - l[b]) / a
    rng = (h[b] - l[b]) / a
    if not np.isnan(eff) and eff >= 2.0 and abs(m0) < 0.35 and rng < 1.2:
        v = "absorbed — effort without result"
    elif abs(m0) >= 0.75 and m3 * np.sign(m0) >= 0.5 * abs(m0):
        v = "followed through"
    elif abs(m0) >= 0.75 and m3 * np.sign(m0) <= 0.25 * abs(m0):
        v = "faded — first move given back"
    elif max(up, dn) >= 1.0 and abs(m0) < 0.4:
        v = "spike rejected"
    else:
        v = "quiet"
    return dict(move=round(float(m0), 2), after3=round(float(m3), 2), effort=None if np.isnan(eff) else round(float(eff), 1),
                verdict=v)


def reactions(events: list[dict], h1: dict[str, pd.DataFrame], now, hours: float = 36) -> list[dict]:
    now = pd.Timestamp(now).tz_convert("UTC")
    out = []
    for e in grouped(high(events)):
        t = pd.Timestamp(e["t"])
        if not (now - pd.Timedelta(hours=hours) <= t <= now - pd.Timedelta(hours=3)):
            continue
        for k in REACT_KEYS.get(e["ccy"], keys_for(e["ccy"])):
            r = reaction(e, h1.get(k))
            if r:
                out.append(dict(event=e["id"], t=e["t"], ccy=e["ccy"], title=e["title"], key=k, **r))
    return out


def brief_lines(events: list[dict], reacts: list[dict], now) -> list[str]:
    out = []
    up = grouped(upcoming(events, now, 24))
    if up:
        out.append("<b>News next 24 h</b> (red folder · no new entries ±60 min)")
        for e in up[:10]:
            out.append(f"• {_loc(e['t'])} {e['ccy']} {e['title']}{_fx(e)}")
    else:
        out.append("News next 24 h: no high-impact releases.")
    if reacts:
        out.append("<b>How price answered the news</b> (release bar, ATR units · 3 bars later)")
        for r in reacts[-12:]:
            eff = f" on {r['effort']}×" if r.get("effort") is not None else ""
            out.append(f"• {r['ccy']} {r['title']} → {r['key']} {r['move']:+.1f}{eff} · {r['after3']:+.1f} — {r['verdict']}")
    return out


def report(events: list[dict], reacts: list[dict], now) -> str:
    """Pipdon BOT /news."""
    now = pd.Timestamp(now).tz_convert("UTC")
    out = [f"📰 News · {now.tz_convert(LOCAL).strftime('%a %d %b %H:%M')} UTC+3"]
    wk = [e for e in grouped(high(events)) if now - pd.Timedelta(hours=12) <= pd.Timestamp(e["t"]) <= now + pd.Timedelta(days=7)]
    if wk:
        out.append("<b>High impact, this week</b>")
        for e in wk[:25]:
            past = pd.Timestamp(e["t"]) < now
            out.append(f"{'✓' if past else '•'} {_loc(e['t'])} {e['ccy']} {e['title']}{_fx(e)}")
    else:
        out.append("No high-impact releases in the calendar.")
    out += [""] + [x for x in brief_lines([], reacts, now) if not x.startswith("News next")]
    out.append("<i>Blackout: no new entries ±60 min on the release's futures (USD = every chart).</i>")
    return "\n".join(out)
