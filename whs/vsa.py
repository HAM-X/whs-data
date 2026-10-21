"""VSA in Roman's notation — bars and swings, read the way the course writes them (WTC S7, S9–S11).

SYMBOLS (ratio vs the comparison → symbol): ≥ 1.5 '>>' · ≥ 1.1 '>' · ≤ 0.67 '<<' · ≤ 0.9 '<' · else '='
  V = volume (effort)   R = result (spread for a bar, price travel for a swing)   D = demand   S = supply

BAR (compared with the usual bar: effort vs the same session hour, spread = true range ÷ ATR14[1]):
  an UP bar (close above the previous close) —
    V>  R>  closes high      D>   demand in control (normal advance)
    V>  R<  or closes low    S>   supply met the buying: effort without result
    V<  R>  closes high      S<   ease of movement — little supply in the way
    V<  R<                   D<   no demand
    V>> R>> closes off high  climactic: D>> met by S>> (watch for a stopping action)
  a DOWN bar mirrors it (S ↔ D).

SWING (a rally or reaction vs the PREVIOUS swing of the same bias — the course's swing analysis):
  rally    V> R>  D>  demand growing          reaction  V> R>  S>  supply growing
           V> R<  S>  supply entering — effort without result (warning)
                                                       V> R<  D>  demand absorbing the selling (warning to shorts)
           V< R>  S<  ease of movement up               V< R>  D<  ease of movement down
           V< R<  D<  demand drying up                  V< R<  S<  supply drying up (exhaustion)
  and vs the previous swing of the COUNTER bias: rally effort > the reaction's → D>S, else S>D.
"""
from __future__ import annotations

import numpy as np

from . import core


def sym(x: float) -> str:
    if x is None or np.isnan(x):
        return "?"
    return ">>" if x >= 1.5 else ">" if x >= 1.1 else "<<" if x <= 0.67 else "<" if x <= 0.9 else "="


def _up(s):
    return s.startswith(">")


def _dn(s):
    return s.startswith("<")


def bar(o, h, l, c, eff, i: int) -> dict:
    """One bar in notation. Returns dict(dir, v, r, close, code, text, expr)."""
    h, l, c = (np.asarray(x, float) for x in (h, l, c))
    atr = core.atr(h, l, c)
    tr = core.true_range(h, l, c)
    a1 = atr[i - 1] if i > 0 else np.nan
    rs = tr[i] / a1 if a1 and not np.isnan(a1) else np.nan
    lo, hi = min(l[i], c[i - 1]) if i else l[i], max(h[i], c[i - 1]) if i else h[i]
    cp = (c[i] - lo) / (hi - lo) if hi > lo else 0.5
    d = 1 if i and c[i] > c[i - 1] else -1 if i and c[i] < c[i - 1] else 0
    v, r = sym(eff[i]), sym(rs)
    good_close = cp >= 0.6 if d == 1 else cp <= 0.4
    me, other = ("D", "S") if d >= 0 else ("S", "D")
    word_me = "demand" if me == "D" else "supply"
    word_ot = "supply" if other == "S" else "demand"
    if v == ">>" and r == ">>" and not good_close:
        code, text = f"{me}>> vs {other}>>", f"climactic — big {word_me} met by big {word_ot}: watch for a stopping action"
    elif _up(v) and (_up(r) or r == "=") and good_close:
        code, text = f"{me}>", f"{word_me} in control"
    elif _up(v) and (_dn(r) or not good_close):
        code, text = f"{other}>", f"{word_ot} met the {'buying' if me == 'D' else 'selling'}: effort without result"
    elif _dn(v) and _up(r) and good_close:
        code, text = f"{other}<", f"ease of movement — little {word_ot} in the way"
    elif _dn(v) and (_dn(r) or r == "="):
        code, text = f"{me}<", f"no {word_me}"
    else:
        code, text = "=", "ordinary bar"
    arrow = "▲" if d == 1 else "▼" if d == -1 else "•"
    return dict(i=int(i), dir=d, v=v, r=r, close=int(round(cp * 100)), code=code, text=text,
                expr=f"{arrow} V{v} R{r} close {int(round(cp * 100))}% → {code}")


def swing(sw, counter=None) -> dict:
    """One finished swing (swings.Swing) vs the previous swing of the same bias (+ the last counter swing)."""
    v, r = sym(sw.e_vs_prev), sym(sw.r_vs_prev)
    rally = sw.direction == 1
    if rally:
        table = {("+", "+"): ("D>", "demand growing"), ("+", "-"): ("S>", "supply entering — effort without result"),
                 ("-", "+"): ("S<", "ease of movement up"), ("-", "-"): ("D<", "demand drying up")}
    else:
        table = {("+", "+"): ("S>", "supply growing"), ("+", "-"): ("D>", "demand absorbing the selling"),
                 ("-", "+"): ("D<", "ease of movement down"), ("-", "-"): ("S<", "supply drying up")}
    kv = 1 if _up(v) else -1 if _dn(v) else 0
    kr = 1 if _up(r) else -1 if _dn(r) else 0
    if "?" in (v + r) or (kv == 0 and kr == 0):
        code, text = "=", "in line with the last one"
    elif kv > 0 and kr > 0:
        code, text = table[("+", "+")]
    elif kv < 0 and kr < 0:
        code, text = table[("-", "-")]
    elif kv >= 0 and kr <= 0:          # more (or the same) effort for less result
        code, text = table[("+", "-")]
    else:                              # less (or the same) effort for more result
        code, text = table[("-", "+")]
    dom = ""
    if counter is not None and counter.avg_effort and not np.isnan(counter.avg_effort):
        k = sw.avg_effort / counter.avg_effort
        mine, other = ("D", "S") if rally else ("S", "D")
        dom = f"{mine}>{other}" if k >= 1.1 else f"{other}>{mine}" if k <= 0.9 else f"{mine}={other}"
    name = "Rally" if rally else "Reaction"
    return dict(n=int(sw.n), dir=int(sw.direction), start=int(sw.start_i), end=int(sw.end_i), pct=float(sw.extent_pct),
                bars=int(sw.bars), effort=float(sw.avg_effort), v=v, r=r, code=code, text=text, dom=dom,
                expr=f"{name} {sw.extent_pct:+.2f}% · {sw.bars} bars · V{v} R{r} → {code}" + (f" · {dom}" if dom else ""))


def read(o, h, l, c, eff, led, n_swings: int = 4, n_bars: int = 3) -> dict:
    """The last finished swings and bars in notation + a one-line verdict of who is in control."""
    sws = [s for s in led.swings if s.confirm_i is not None and s.confirm_i < len(c)]
    out_sw = []
    for k in range(max(0, len(sws) - n_swings), len(sws)):
        prev_counter = next((s for s in reversed(sws[:k]) if s.direction == -sws[k].direction), None)
        out_sw.append(swing(sws[k], prev_counter))
    bars_ = [bar(o, h, l, c, eff, i) for i in range(len(c) - n_bars, len(c))]
    last_r = next((s for s in reversed(out_sw) if s["dir"] == 1), None)
    last_x = next((s for s in reversed(out_sw) if s["dir"] == -1), None)
    score = 0
    for s in (last_r, last_x):
        if s:
            score += {"D>": 1, "S<": 1, "D<": -1, "S>": -1}.get(s["code"], 0)
    verdict = ("DEMAND in control" if score >= 1 else "SUPPLY in control" if score <= -1 else "BALANCED — no side in control")
    return dict(swings=out_sw, bars=bars_, verdict=verdict)
