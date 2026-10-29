"""One decision per chart — what Hamza does with it, in one verb, a trigger and an invalidation.

  NO NEW ENTRIES  a red-folder release on this currency within ±60 min (overrides everything)
  BUY ZONE / SELL ZONE   phase C: a confirmed Spring / Shakeout / Test (UTAD / test) — POE 1. Live for the event
                  bar + 2 bars only; `setup` = a Spring / Shakeout / UTAD (candidate or confirmed) inside that window.
                  Heal study 29 Oct 2026: no setup has a proven edge in the live-walk replay — said as such
  BUY PULLBACKS / SELL RALLIES   phase D after the SOS / SOW: LPS or BU at the old edge (information only —
                  no edge over the baseline in the replay, so never alerted or logged as a call)
  A decision whose price has CLOSED beyond its own invalidation is reported as WAIT (failed).
  WATCH           phase C forming (a test is happening, not confirmed)
  WAIT            phase A / B: the range is building cause — act only on a spring or UTAD
  TREND ↑ / ↓     no range: nothing to do at a level; the trend holds while the protected level holds
  SIDEWAYS        a climax stopped the trend, no range read yet
"""
from __future__ import annotations

COLORS = {"BUY": "#0E6E55", "SELL": "#8A3FA0", "WAIT": "#2E3A48", "WATCH": "#C99A06", "TREND_UP": "#001F3F",
          "TREND_DN": "#E74C3C", "NO": "#E08A00"}


def _p(x) -> str:
    return f"{x:.5g}" if x is not None else "—"


ENTRIES = ("BUY ZONE", "SELL ZONE", "BUY PULLBACKS", "SELL RALLIES")


def _failed(d: dict, r: dict, g: dict | None) -> dict | None:
    """Heal fix 1 · a decision is dead once price CLOSES beyond its own invalidation (audit 29 Oct: 6J 15m BUY
    PULLBACKS ✅ was alerted twice 15–36 ATR below its stop). The setup is reported as failed, never as an entry."""
    px = r.get("close")
    if px is None or not g or d["action"] not in ENTRIES:
        return None
    S, R = g["support"], g["resistance"]
    conf = [e for e in g["events"] if e[2] == "confirmed"]
    lastp = conf[-1][3] if conf else None
    long_ = d["action"] in ("BUY ZONE", "BUY PULLBACKS")
    lvl = lastp if d["action"].endswith("ZONE") else (R if long_ else S)
    if lvl is None or (px >= lvl if long_ else px <= lvl):
        return None
    word = "below" if long_ else "above"
    return dict(action="WAIT", tone="WAIT", failed=d["action"],
                do=f"{d['action'].title()} failed — price closed {word} {_p(lvl)}",
                trigger=f"long: spring below {_p(S)} closing back in · short: UTAD above {_p(R)} closing back in",
                invalid="—", watch=g["name"], short=f"failed · {word} {_p(lvl)}")


TF_MIN = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
FRESH_BARS = 2          # heal fix 2 · a phase-C entry is live on the event bar and the next 2 bars only:
                        # replayed live on 22 months of 1h, the edge was +0.61 R on the event bar, +0.52 R two bars
                        # later and +0.11 R four bars later (n 77 → 54)
REVERSAL = ("Spring", "Shakeout", "UTAD")   # phase-C reversals — the course's POE 1 (WTC S14)
PROVEN = REVERSAL                              # kept for older imports


def _age(r: dict, g: dict, tf: str | None) -> float | None:
    conf = [e for e in g["events"] if e[2] == "confirmed"]
    if not tf or tf not in TF_MIN or not conf or not r.get("last"):
        return None
    import pandas as pd
    return (pd.Timestamp(r["last"]) - pd.Timestamp(conf[-1][0])).total_seconds() / 60 / TF_MIN[tf]


def _fresh_reversal(r: dict, g: dict | None, tf: str | None) -> dict | None:
    """Heal fix 3 · the entry is keyed to the EVENT, not to the phase letter. In the replay the old rule turned only
    5 of 127 confirmed 1h Springs / Shakeouts / UTADs into a zone: the reader names them 'Spring #3' (the old test
    wanted the exact word 'Spring'), often moves the phase to D on the same bar, or confirms another event (ST, mSOW)
    on that bar. Now: the latest confirmed Spring* / Shakeout / UTAD* of the live range, known ≤ FRESH_BARS ago,
    price still on the right side of it and the range not resolved against it → BUY / SELL ZONE."""
    if not g or not tf or tf not in TF_MIN or not r.get("last"):
        return None
    import pandas as pd
    # POE 1 timing (S14: "#3 spring → the close of the bar returning into the TR"): the CANDIDATE counts. Heal replay
    # (29 Oct 2026): this rule issued 104 1h alerts → −0.12 R each, 36 on 4h → −0.40 R (adaptive stop, after costs);
    # waiting for "confirmed" was worse (−0.28 R). The Edge Lab's +1.65 R came from entering events that were only
    # confirmed LATER (+1.28 R) — hindsight. So these are EVENTS to look at, not instructions.
    rev = [e for e in g["events"] if e[2] in ("candidate", "confirmed") and str(e[1]).startswith(REVERSAL)]
    if not rev:
        return None
    t, name, _, p = rev[-1][:4]
    age = (pd.Timestamp(r["last"]) - pd.Timestamp(t)).total_seconds() / 60 / TF_MIN[tf]
    if age < 0 or age > FRESH_BARS or p is None:
        return None
    long_ = not str(name).startswith("UTAD")
    out, px = g.get("outcome", 0), r.get("close")
    if (long_ and out < 0) or (not long_ and out > 0) or px is None or (px - p) * (1 if long_ else -1) <= 0:
        return None
    S, R = g["support"], g["resistance"]
    if long_:
        return dict(action="BUY ZONE", tone="BUY", do=f"{name} at {_p(p)} — phase-C event (POE 1) — check it on your chart; no edge proven",
                    trigger=f"long while it holds above {_p(p)}; target {_p(R)} then {_p(R + (R - S))}",
                    invalid=f"close below {_p(p)}", watch=g["name"], short=f"{name} {_p(p)} · hold above", fresh=round(age, 1))
    return dict(action="SELL ZONE", tone="SELL", do=f"{name} at {_p(p)} — phase-C event (POE 1) — check it on your chart; no edge proven",
                trigger=f"short while it holds below {_p(p)}; target {_p(S)} then {_p(S - (R - S))}",
                invalid=f"close above {_p(p)}", watch=g["name"], short=f"{name} {_p(p)} · hold below", fresh=round(age, 1))


def chart(r: dict, blackout: list | None = None, local=lambda iso: iso, tf: str | None = None) -> dict:
    d = _chart(r, blackout, local)
    g0 = (r.get("ranges") or [None])[-1]
    if d["action"] != "NO NEW ENTRIES":
        fr = _fresh_reversal(r, g0, tf)
        if fr:
            d = fr
    dead = None if "fresh" in d else _failed(d, r, g0)
    if dead:
        d = dead
    d["setup"] = "fresh" in d
    if g0 and d["action"] in ("BUY ZONE", "SELL ZONE") and "fresh" not in d:
        conf = [e for e in g0["events"] if e[2] == "confirmed"]
        last = conf[-1][1] if conf else ""
        age = _age(r, g0, tf)
        if age is not None and age > FRESH_BARS:
            n = int(round(age))
            d = dict(action="WAIT", tone="WAIT", used=d["action"],
                     do=f"{last} at {_p(conf[-1][3])} was {n} bars ago — the entry window has closed",
                     trigger=f"next: a test or LPS that holds above {_p(conf[-1][3])}" if d["action"] == "BUY ZONE"
                     else f"next: a test or LPSY that holds below {_p(conf[-1][3])}",
                     invalid="—", watch=g0["name"], short=f"{last} used · {n} bars ago")
        else:
            d["setup"] = last.startswith(REVERSAL)
    d.setdefault("short", d["do"][:34])
    # numbers for the live self-grade (Edge Lab 7.4): side, stop (the invalidation), target (the measured move)
    g = (r.get("ranges") or [None])[-1]
    if g and d["action"] in ("BUY ZONE", "SELL ZONE", "BUY PULLBACKS", "SELL RALLIES"):
        S, R = g["support"], g["resistance"]
        conf = [e for e in g["events"] if e[2] == "confirmed"]
        lastp = conf[-1][3] if conf else None
        side = 1 if d["action"] in ("BUY ZONE", "BUY PULLBACKS") else -1
        if d["action"] in ("BUY ZONE", "SELL ZONE"):
            stop = lastp
            if "fresh" in d:
                rev = [e for e in g["events"] if e[2] in ("candidate", "confirmed") and str(e[1]).startswith(REVERSAL)]
                stop = rev[-1][3]
        else:
            stop = R if side == 1 else S
        d.update(side=side, stop=None if stop is None else float(stop),
                 target=float(R + (R - S)) if side == 1 else float(S - (R - S)))
    return d


def _chart(r: dict, blackout: list | None = None, local=lambda iso: iso) -> dict:
    """r = one scan.json record ('6E 4h'). Returns dict(action, tone, do, trigger, invalid, watch)."""
    m = r.get("major") or {}
    g = (r.get("ranges") or [None])[-1]
    if blackout:
        e = blackout[0]
        return dict(action="NO NEW ENTRIES", tone="NO", do=f"{e['ccy']} {e['title']} at {local(e['t'])} — wait it out",
                    trigger="re-check the chart after the release bar closes", invalid="—", watch="",
                    short=f"{e['ccy']} news {local(e['t'])}")
    if not g:
        st, prot = m.get("state"), m.get("protected")
        if st == 1:
            return dict(action="TREND ↑", tone="TREND_UP", do="No range — nothing to buy at a level yet",
                        trigger="wait for a pullback that builds structure", invalid=f"trend ends below {_p(prot)}", watch="",
                        short=f"holds above {_p(prot)}")
        if st == -1:
            return dict(action="TREND ↓", tone="TREND_DN", do="No range — nothing to sell at a level yet",
                        trigger="wait for a rally that builds structure", invalid=f"trend ends above {_p(prot)}", watch="",
                        short=f"holds below {_p(prot)}")
        return dict(action="SIDEWAYS", tone="WAIT", do="A climax stopped the trend — no range read yet",
                    trigger="let the range form (AR, then ST)", invalid="—", watch="", short="climax stopped the trend")
    S, R, ph, out = g["support"], g["resistance"], g["phase"], g.get("outcome", 0)
    conf = [e for e in g["events"] if e[2] == "confirmed"]
    last = conf[-1][1] if conf else ""
    lastp = conf[-1][3] if conf else None
    name = g["name"]
    if ph == "C" and last in ("Spring", "Shakeout", "Test") and out >= 0:
        return dict(action="BUY ZONE", tone="BUY", do=f"{last} confirmed at {_p(lastp)} — phase-C event (POE 1) — check it on your chart; no edge proven",
                    trigger=f"long while it holds above {_p(lastp)}; target {_p(R)} then {_p(R + (R - S))}",
                    invalid=f"close below {_p(lastp)}", watch=name, short=f"{last} {_p(lastp)} · hold above")
    if ph == "C" and (last.startswith("UTAD") or last == "Test of the UTAD") and out <= 0:   # same as the Pine card (roles)
        return dict(action="SELL ZONE", tone="SELL", do=f"{last} confirmed at {_p(lastp)} — phase-C event (POE 1) — check it on your chart; no edge proven",
                    trigger=f"short while it holds below {_p(lastp)}; target {_p(S)} then {_p(S - (R - S))}",
                    invalid=f"close above {_p(lastp)}", watch=name, short=f"{last} {_p(lastp)} · hold below")
    if ph == "C":
        return dict(action="WATCH", tone="WATCH", do="Phase C — a test of the range edge is happening",
                    trigger=f"long on a spring that closes back above {_p(S)} · short on a UTAD back below {_p(R)}",
                    invalid="—", watch=name, short=f"phase C · {_p(S)}–{_p(R)}")
    if ph in ("D", "E") and out == 1:
        return dict(action="BUY PULLBACKS" if ph == "D" else "TREND ↑", tone="BUY" if ph == "D" else "TREND_UP",
                    do="SOS done — demand won the range" if ph == "D" else "Price left the range up — manage, no range entries",
                    trigger=f"buy the LPS / back-up near {_p(R)}" if ph == "D" else "no new range entries",
                    invalid=f"close back below {_p(R)}", watch=name, short=f"LPS / BU near {_p(R)}")
    if ph in ("D", "E") and out == -1:
        return dict(action="SELL RALLIES" if ph == "D" else "TREND ↓", tone="SELL" if ph == "D" else "TREND_DN",
                    do="SOW done — supply won the range" if ph == "D" else "Price left the range down — manage, no range entries",
                    trigger=f"sell the LPSY / back-up near {_p(S)}" if ph == "D" else "no new range entries",
                    invalid=f"close back above {_p(S)}", watch=name, short=f"LPSY / BU near {_p(S)}")
    return dict(action="WAIT", tone="WAIT",
                do=f"Phase {ph} — the range is {'just formed' if ph == 'A' else 'building cause'}",
                trigger=f"long: spring below {_p(S)} closing back in · short: UTAD above {_p(R)} closing back in",
                invalid="—", watch=name, short=f"phase {ph} · {_p(S)}–{_p(R)}")
