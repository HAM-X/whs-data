"""One decision per chart — what Hamza does with it, in one verb, a trigger and an invalidation.

  NO NEW ENTRIES  a red-folder release on this currency within ±60 min (overrides everything)
  BUY ZONE / SELL ZONE   phase C: a confirmed Spring / Shakeout / Test (UTAD / test) — the proven setup (POE 1)
  BUY PULLBACKS / SELL RALLIES   phase D after the SOS / SOW: LPS or BU at the old edge
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


def chart(r: dict, blackout: list | None = None, local=lambda iso: iso) -> dict:
    d = _chart(r, blackout, local)
    d.setdefault("short", d["do"][:34])
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
        return dict(action="BUY ZONE", tone="BUY", do=f"{last} confirmed at {_p(lastp)} — the proven setup (POE 1)",
                    trigger=f"long while it holds above {_p(lastp)}; target {_p(R)} then {_p(R + (R - S))}",
                    invalid=f"close below {_p(lastp)}", watch=name, short=f"{last} {_p(lastp)} · hold above")
    if ph == "C" and last in ("UTAD", "Test of the UTAD", "UT") and out <= 0:
        return dict(action="SELL ZONE", tone="SELL", do=f"{last} confirmed at {_p(lastp)} — the proven setup (POE 1)",
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
