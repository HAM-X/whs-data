"""The campaign read (Hamza 30 Oct 2026) — one live hypothesis per range: is the composite operator accumulating or
distributing, how far has he shown his hand, what would confirm it, what would kill it, and what to watch next.

The read is the COURSE's logic (WTC Part 1/2, Forte's "Anatomy of a Trading Range", the Nine Buying / Selling Tests),
written down once and never fitted to the data, so the replay is a fair test of it:

  every confirmed fact of the range and of the swings inside it becomes an evidence item, + for accumulation and −
  for distribution, weighted by how much the course trusts it (phase A context 0.5–1 · phase B behaviour 1 ·
  phase C tests 2–3 · phase D signs 2–3), and GRADED BY ITS EFFORT — "heavy volume must be read by its result,
  never by its size alone"; a spring on light volume clears the way, a test on heavy volume brings supply back.

  verdict   |score| < 2 balanced · 2–4 leaning · ≥ 5 with a phase C or D item on that side = accumulation /
            distribution (the operator has shown his hand)
  levels    confirm (the SOS / SOW close) · invalidate (beyond the phase-C extreme or the far edge) · watch next
  objective point-and-figure horizontal count (Law of Cause and Effect), conservative + aggressive
  odds      the measured chance of resolving up first (R + ½ height before S − ½ height) — see ODDS_NOTE

Side convention of the range events (whs/ranges.py): s = +1 an event at the bottom edge (support side), −1 at the
top edge. Effort (ef) is the event bar's effort vs the same session slot, 1.0 = normal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The Heal study no.3 (29–30 Oct 2026, replay v2, 1D/4h/1h/15m, 476 ranges, hold-out from 1 Feb 2026): no single
# item, and no sum of them, moved the odds beyond where price sits in the range. Three effort-graded items agreed
# with the course in both halves; they are marked ✓ in the read. Everything is logged forward and re-scored weekly.
ODDS_NOTE = "from price location — in the replay no evidence item moved these odds reliably; the read is logged live"
REPLAY_OK = {"test_heavy", "sow_heavy", "bheavy_bottom"}
# A spring / UTAD counts fully only once a low-effort test or the SOS / SOW backs it. In the replay a bare phase-C
# label read the range WRONG more often than chance (−6.7 points, 112 ranges) and stood only 49 % of the time
# (77 % after the SOS) — the course says the same: wait for the test.
PROVISIONAL_C = True

PHASE_C = ("Spring", "Shakeout", "Fail")


def _fam(e: dict) -> str:
    return e.get("r") or str(e.get("n", "")).split(" #")[0]


def _ef(e: dict) -> float:
    x = e.get("ef")
    return np.nan if x is None else float(x)


def _item(side, w, text, phase, key=None):
    return dict(side=int(np.sign(side)), w=float(abs(w)), text=text, phase=phase, key=key or text,
                replay="✓" if key in REPLAY_OK else "")


def evidence(rng: dict, sw: list | None = None, verdict: str = "", major: int | None = 0, children: list | None = None) -> list:
    """Evidence items for one range. rng: S, R, kind ('SC'/'BC'), events [dict(n, r, s, st, ef, p, k)]."""
    out = []
    ev = sorted([e for e in rng.get("events") or [] if e.get("st") == "confirmed"], key=lambda e: e.get("k") or "")
    # phase A context
    if rng.get("kind") == "SC":
        out.append(_item(+1, 0.5, "a selling climax stopped a decline", "A", "ctx_sc"))
    elif rng.get("kind") == "BC":
        out.append(_item(-1, 0.5, "a buying climax stopped an advance", "A", "ctx_bc"))
    if major:
        out.append(_item(major, 1, f"major trend {'up' if major > 0 else 'down'}", "A", "ctx_major"))
    backed = {s for s in (1, -1) if any(_fam(e) == "BO" and e.get("s") == s for e in ev)}
    for e in ev:
        f, s, ef, p = _fam(e), int(e.get("s") or 0), _ef(e), e.get("p")
        nm = str(e.get("n", f)).split(" #")[0]
        at = f"{nm} {p:.5g}" if p is not None else nm
        low, heavy = ef <= 0.8, ef >= 1.2
        if f == "ST" and s:
            if low:
                out.append(_item(s, 1, f"{at}: the secondary test came on low effort ({ef:.1f}×) — "
                                       f"{'supply' if s > 0 else 'demand'} is drying up", "B", "st_low"))
            elif heavy:
                out.append(_item(-s, 1, f"{at}: the secondary test came on heavy effort ({ef:.1f}×) — "
                                        f"{'supply' if s > 0 else 'demand'} is still present", "B", "st_heavy"))
        elif f == "B_heavy" and s and heavy:
            out.append(_item(-s, 1, f"{at}: a heavy push to the {'support' if s > 0 else 'resistance'} in phase B ({ef:.1f}×) — "
                                    f"{'supply' if s > 0 else 'demand'} in control of that swing", "B",
                             "bheavy_bottom" if s > 0 else "bheavy_top"))
        elif f in PHASE_C and s:
            what = "spring / shakeout" if s > 0 else "upthrust (UTAD)"
            if PROVISIONAL_C and s not in backed:      # Forte: "until this test we can't be sure" — provisional
                out.append(_item(s, 0, f"{at}: {what}, provisional until the {'SOS' if s > 0 else 'SOW'} backs it "
                                       f"(in the replay unbacked ones led the other way more often)", "C", "c_provisional"))
            elif ef <= 0.9:
                out.append(_item(s, 3, f"{at}: {what} on light effort ({ef:.1f}×) — "
                                       f"{'no supply below' if s > 0 else 'no demand above'}, the way is clear", "C", "c_light"))
            elif ef >= 1.3:
                out.append(_item(s, 1, f"{at}: {what} on heavy effort ({ef:.1f}×) — needs a successful test", "C", "c_heavy"))
            else:
                out.append(_item(s, 2, f"{at}: {what} confirmed", "C", "c_mid"))
        elif f == "Test" and s:
            if s not in backed and ef < 1.2:
                out.append(_item(s, 0, f"{at}: the test of the {'spring' if s > 0 else 'UTAD'} held ({ef:.1f}×) — "
                                       f"provisional until the {'SOS' if s > 0 else 'SOW'}", "C", "test_provisional"))
            elif ef <= 0.9:
                out.append(_item(s, 2, f"{at}: the test of the {'spring' if s > 0 else 'UTAD'} came on low effort ({ef:.1f}×) and held", "C", "test_light"))
            elif heavy:
                out.append(_item(-s, 1, f"{at}: the test came on heavy effort ({ef:.1f}×) — "
                                        f"{'supply' if s > 0 else 'demand'} came back", "C", "test_heavy"))
            else:
                out.append(_item(s, 1, f"{at}: the test held", "C", "test_mid"))
        elif f == "LPS_C" and s:
            out.append(_item(s, 1, f"{at}: {'a higher low (LPS)' if s > 0 else 'a lower high (LPSY)'} in phase C", "C", "lps_c"))
        elif f == "BO" and s:
            if ef >= 1.1:
                out.append(_item(s, 3, f"{at}: {'SOS' if s > 0 else 'SOW'} on increasing effort ({ef:.1f}×) — "
                                       f"{'demand' if s > 0 else 'supply'} in control", "D", "sos_heavy" if s > 0 else "sow_heavy"))
            else:
                out.append(_item(s, 2, f"{at}: {'SOS' if s > 0 else 'SOW'} on thin effort ({ef:.1f}×)", "D", "bo_thin"))
        elif f in ("BU", "D_LPS", "Chain") and s:
            out.append(_item(s, 2 if ef <= 0.9 else 1, f"{at}: {'a back-up / LPS' if s > 0 else 'an LPSY'}"
                                                       f"{' on low effort' if ef <= 0.9 else ''} after the breakout", "D", "lps_d"))
    # phase B behaviour from the swings (the Nine Tests: activity, higher supports, higher tops; contracting swings)
    sw = sw or []
    up, dn = [x for x in sw if x.get("d") == 1], [x for x in sw if x.get("d") == -1]
    if up and dn:
        eu, ed = np.mean([x["eff"] for x in up]), np.mean([x["eff"] for x in dn])
        if eu > 0 and ed > 0 and abs(np.log(eu / ed)) > 0.15:
            a = np.log(eu / ed)
            out.append(_item(a, 1, f"activity {'bullish' if a > 0 else 'bearish'}: effort on {'rallies' if a > 0 else 'reactions'} "
                                   f"{np.exp(abs(a)):.1f}× the {'reactions' if a > 0 else 'rallies'}", "B", "activity"))
    if len(dn) >= 2:
        a, b = dn[-2], dn[-1]
        out.append(_item(1 if b["ep"] > a["ep"] else -1, 1, "higher supports" if b["ep"] > a["ep"] else "lower supports", "B", "supports"))
        if abs(b["ep"] - b["sp"]) < abs(a["ep"] - a["sp"]) and b["eff"] < a["eff"]:
            out.append(_item(+1, 1, "reactions contracting (shorter, on less effort) — absorption", "B", "contract_dn"))
    if len(up) >= 2:
        a, b = up[-2], up[-1]
        out.append(_item(1 if b["ep"] > a["ep"] else -1, 1, "higher tops" if b["ep"] > a["ep"] else "lower tops", "B", "tops"))
        if abs(b["ep"] - b["sp"]) < abs(a["ep"] - a["sp"]) and b["eff"] < a["eff"]:
            out.append(_item(-1, 1, "rallies contracting (shortening of the thrust up)", "B", "contract_up"))
    v = (verdict or "").upper()
    if v.startswith("DEMAND") or v.startswith("SUPPLY"):
        out.append(_item(1 if v.startswith("DEMAND") else -1, 1, f"effort/result now: {verdict.split(' —')[0].lower()}", "B", "verdict"))
    for c in children or []:        # a minor range at the edge with its own phase C (whs/hierarchy.py frames)
        if c.get("back_in"):
            z = c["zone"]
            out.append(_item(-z, 2 if c.get("own_c") else 1,
                             f"minor {'distribution at the top' if z == 1 else 'accumulation at the bottom'} on the {c['tf']}"
                             + (f", own phase C ({c['own_c']})" if c.get("own_c") else ""), "C", "child"))
    return out


def chance_up(S: float, R: float, price: float) -> float:
    """Driftless walk: P(R + ½H before S − ½H) from the price."""
    H = R - S
    pos = (price - S) / H if H > 0 else 0.5
    return float(np.clip((np.clip(pos, -0.5, 1.5) + 0.5) / 2, 0, 1))


def read(rng: dict, price: float, sw: list | None = None, verdict: str = "", major: int | None = 0,
         children: list | None = None, atr: float | None = None) -> dict:
    """The full campaign read of one range."""
    S, R = float(rng["S"]), float(rng["R"])
    H = R - S
    items = evidence(rng, sw, verdict, major, children)
    score = sum(i["side"] * i["w"] for i in items)
    up_c = any(i["side"] > 0 and i["phase"] in "CD" for i in items)
    dn_c = any(i["side"] < 0 and i["phase"] in "CD" for i in items)
    if score >= 5 and up_c:
        label, side, grade = "ACCUMULATION", 1, "shown"
    elif score <= -5 and dn_c:
        label, side, grade = "DISTRIBUTION", -1, "shown"
    elif score >= 2:
        label, side, grade = "leaning ACCUMULATION", 1, "lean"
    elif score <= -2:
        label, side, grade = "leaning DISTRIBUTION", -1, "lean"
    else:
        label, side, grade = "BALANCED", 0, "none"
    ph0 = rng.get("phase") or rng.get("ph") or "B"
    if ph0 == "E":                                  # the range has resolved: say so, no odds, no claim
        o = int(np.sign(rng.get("outcome") or score))
        mv = "MARKUP" if o > 0 else "MARKDOWN"
        return dict(label=f"{mv} under way (phase E)", side=o, grade="resolved", score=round(score, 1), items=items, phase="E",
                    S=S, R=R, inv_p=None, chance_up=None,
                    confirm=f"the range is resolved {'up' if o > 0 else 'down'} — the next structure forms after this move",
                    invalid=f"a close back inside the range ({'below' if o > 0 else 'above'} {(R if o > 0 else S):.5g})",
                    watch=("the first stopping action against the move (PSY / BC)" if o > 0 else
                           "the first stopping action against the move (PS / SC)") + ", or a re-" +
                          ("accumulation" if o > 0 else "distribution") + " range")
    ev = [e for e in rng.get("events") or [] if e.get("st") == "confirmed"]
    lows = [e["p"] for e in ev if _fam(e) in PHASE_C and e.get("s") == 1]
    highs = [e["p"] for e in ev if _fam(e) in PHASE_C and e.get("s") == -1]
    has = lambda f, s: any(_fam(e) == f and e.get("s") == s for e in ev)
    ph = rng.get("phase") or rng.get("ph") or "B"
    if side > 0:
        confirm = (f"a close above {R:.5g} (the resistance) on rising effort = SOS → phase D" if not has("BO", 1)
                   else f"the back-up holding above {R:.5g} on low effort (LPS) → markup")
        inv_p = min(lows) if lows else S - 0.15 * H
        invalid = f"a close below {inv_p:.5g} ({'the spring low' if lows else 'the support − 15 % of the height'})"
        watch = ("a spring or shakeout below the support on light effort, then a low-effort test" if not lows and not has("BO", 1) else
                 "a low-effort test of the spring that holds above it" if lows and not has("Test", 1) and not has("BO", 1) else
                 "the SOS through the resistance" if not has("BO", 1) else "an LPS pullback on low effort above the old resistance")
    elif side < 0:
        confirm = (f"a close below {S:.5g} (the support) on rising effort = SOW → phase D" if not has("BO", -1)
                   else f"the rally failing below {S:.5g} on low effort (LPSY) → markdown")
        inv_p = max(highs) if highs else R + 0.15 * H
        invalid = f"a close above {inv_p:.5g} ({'the UTAD high' if highs else 'the resistance + 15 % of the height'})"
        watch = ("an upthrust above the resistance on light effort, then a low-effort test" if not highs and not has("BO", -1) else
                 "a low-effort test of the UTAD that fails below it" if highs and not has("Test", -1) and not has("BO", -1) else
                 "the SOW through the support" if not has("BO", -1) else "an LPSY rally on low effort below the old support")
    else:
        inv_p = None
        confirm = f"a spring at {S:.5g} or an upthrust at {R:.5g} — the phase-C test decides the read"
        invalid = "—"
        watch = "which edge gets tested on light effort; until then both sides are possible"
    # where price is now changes what to watch: already through the confirmation level, or testing the far edge
    if side < 0 and not has("BO", -1):
        if price < S:
            confirm = f"price is already below the support {S:.5g} — a close below it on rising effort makes it the SOW (phase D)"
            watch = "the SOW close, then a weak rally (LPSY) that fails below the old support"
        elif price > R:
            watch = "price is above the resistance — this is where an upthrust forms: a close back inside on light effort = UTAD"
    elif side > 0 and not has("BO", 1):
        if price > R:
            confirm = f"price is already above the resistance {R:.5g} — a close above it on rising effort makes it the SOS (phase D)"
            watch = "the SOS close, then a quiet pullback (LPS / back-up) that holds above the old resistance"
        elif price < S:
            watch = "price is below the support — this is where a spring forms: a close back inside on light effort = spring"
    return dict(label=label, side=side, grade=grade, score=round(score, 1), items=items, phase=ph, S=S, R=R,
                confirm=confirm, invalid=invalid, inv_p=inv_p, watch=watch, chance_up=chance_up(S, R, price))


# ───────────────────────────── point and figure (Law of Cause and Effect) ─────────────────────────────
def pnf_columns(close: np.ndarray, box: float, rev: int = 1) -> list[dict]:
    """1-box-reversal P&F on closes (Roman's SPX example). Columns: dir, lo, hi (in boxes), i0, i1 (bar indices)."""
    if box <= 0 or len(close) < 2:
        return []
    b = np.floor(np.asarray(close, float) / box).astype(int)
    cols = [dict(dir=0, lo=b[0], hi=b[0], i0=0, i1=0)]
    for i in range(1, len(b)):
        c = cols[-1]
        if c["dir"] >= 0 and b[i] > c["hi"]:
            c.update(hi=b[i], i1=i, dir=1)
        elif c["dir"] <= 0 and b[i] < c["lo"]:
            c.update(lo=b[i], i1=i, dir=-1)
        elif c["dir"] == 1 and b[i] <= c["hi"] - rev:
            cols.append(dict(dir=-1, lo=b[i], hi=c["hi"] - 1, i0=i, i1=i))
        elif c["dir"] == -1 and b[i] >= c["lo"] + rev:
            cols.append(dict(dir=1, lo=c["lo"] + 1, hi=b[i], i0=i, i1=i))
    return cols


def pnf_count(close: np.ndarray, box: float) -> dict | None:
    """Horizontal count across the range: the row crossed by the most columns is the count line."""
    cols = pnf_columns(close, box)
    if len(cols) < 3:
        return None
    lo, hi = min(c["lo"] for c in cols), max(c["hi"] for c in cols)
    best, row = 0, None
    for r in range(lo, hi + 1):
        n = sum(1 for c in cols if c["lo"] <= r <= c["hi"])
        if n > best:
            best, row = n, r
    return dict(cols=best, line=(row + 0.5) * box, box=box, width=len(cols))


def objectives(rd: dict, close: np.ndarray, atr: float) -> dict | None:
    """Count objectives for the side of the read: conservative from the phase-C extreme (or the edge), aggressive
    from the count line. Box = ½ ATR of the range's timeframe."""
    if not rd["side"] or not atr or atr <= 0:
        return None
    pc = pnf_count(close, 0.5 * atr)
    if not pc:
        return None
    move = pc["cols"] * pc["box"]
    if rd["side"] > 0:
        base = rd["inv_p"] if rd["inv_p"] is not None and rd["inv_p"] < rd["S"] else rd["S"]
        cons, aggr = base + move, pc["line"] + move
    else:
        base = rd["inv_p"] if rd["inv_p"] is not None and rd["inv_p"] > rd["R"] else rd["R"]
        cons, aggr = base - move, pc["line"] - move
    return dict(cons=float(cons), aggr=float(aggr), cols=pc["cols"], box=pc["box"], line=pc["line"])


# ───────────────────────────── a touch, graded by what happened on the bar ─────────────────────────────
def touch_grade(o: float, h: float, l: float, c: float, eff: float, atr: float, edge: int) -> tuple[str, int]:
    """edge −1 = the bar touched the support, +1 = the resistance. Returns (grade text, sign: +1 the edge held,
    −1 it is giving way, 0 undecided). Course rules (Forte, P-S notes): at support, low effort and a narrow bar =
    no supply (test passed); heavy effort that closes off the low = absorption / stopping; heavy effort with a wide
    bar closing on the low = supply in control. Mirror at the resistance."""
    tr = (h - l) / atr if atr else np.nan
    cp = (c - l) / (h - l) if h > l else 0.5
    cp_in = cp if edge == -1 else 1 - cp           # closing away from the touched extreme
    if np.isnan(eff) or np.isnan(tr):
        return "undecided (no effort data)", 0
    if eff <= 0.9 and tr <= 1.0 and cp_in >= 0.5:
        return ("no supply — the test passed" if edge == -1 else "no demand — the test passed"), 1
    if eff >= 1.3 and cp_in >= 0.6:
        return ("absorption — heavy effort, closed off the low (stopping action)" if edge == -1 else
                "absorption — heavy effort, closed off the high (supply stopping it)"), 1
    if eff >= 1.1 and tr >= 1.1 and cp_in <= 0.35:
        return ("supply in control — wide bar on effort, closed on the low" if edge == -1 else
                "demand in control — wide bar on effort, closed on the high"), -1
    return "undecided — neither a clean test nor a break", 0


def label_text(rd: dict) -> str:
    if rd["grade"] == "resolved":
        return rd["label"]
    sc = float(np.sign(rd["score"]) * np.floor(abs(rd["score"]) + 0.5)) + 0.0   # half away from zero, as Pine prints it
    return f"{rd['label']} ({sc:+.0f})" if rd["side"] else f"BALANCED ({sc:+.0f})"


def where_text(rd: dict, price: float) -> str:
    S, R = rd["S"], rd["R"]
    H = R - S
    if price > R:
        return f"price {price:.5g} above the resistance"
    if price < S:
        return f"price {price:.5g} below the support"
    pos = (price - S) / H if H > 0 else 0.5
    return f"price {price:.5g} in the {'top' if pos >= 0.8 else 'bottom' if pos <= 0.2 else 'middle'} of the range"


def items_text(rd: dict, n: int = 4) -> str:
    """The strongest items on the side of the read (or both sides when balanced)."""
    its = sorted(rd["items"], key=lambda i: (-(i["side"] == rd["side"] or rd["side"] == 0), -i["w"]))
    keep = [i for i in its if rd["side"] == 0 or i["side"] == rd["side"]][:n]
    against = [i for i in rd["items"] if rd["side"] and i["side"] == -rd["side"]]
    s = "; ".join(i["text"] + (" ✓" if i["replay"] else "") for i in keep)
    if against:
        s += f" · against: {max(against, key=lambda i: i['w'])['text']}"
    return s
