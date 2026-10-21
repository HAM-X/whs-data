"""Chart cards for Telegram — one image per chart, built to be read on a phone in two seconds.

  header     instrument · timeframe · the ACTION chip (decide.chart)
  chart      the last ~120 bars · the range box with its edges · Wyckoff events · the last swings with their VSA code
  effort     volume vs the same session hour, coloured by bar direction, darker = more effort
  decision   DO · TRIGGER · INVALID · CONTROL (who is in control, from the swing VSA) · NEWS
Palette: one palette everywhere — navy #001F3F up, red #E74C3C down, long teal #0E6E55, short purple #8A3FA0.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from .decide import COLORS  # noqa: E402

for _f in fm.findSystemFonts():
    if "Inter" in _f:
        try:
            fm.fontManager.addfont(_f)
        except Exception:
            pass
plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]

BG, PANEL, UP, DN = "#E3E5EA", "#F4F5F7", "#001F3F", "#E74C3C"
INK, QUIET, RULE = "#1C2733", "#6B7785", "#C9CDD4"
LOCAL = "Africa/Nairobi"
NAMES = {"6E": "Euro FX", "6B": "British pound", "6A": "Australian dollar", "6N": "New Zealand dollar",
         "6J": "Japanese yen", "6C": "Canadian dollar", "6S": "Swiss franc", "GC": "Gold", "DX": "Dollar index"}


def _alpha(e):
    return 0.15 if np.isnan(e) else 1.0 if e >= 1.5 else 0.7 if e >= 1.1 else 0.18 if e <= 0.67 else 0.3 if e <= 0.9 else 0.45


def _chip(fig, x, y, text, color, size=26, pad=0.012):
    t = fig.text(x, y, text, ha="right", va="center", fontsize=size, fontweight="bold", color="white",
                 bbox=dict(boxstyle=f"round,pad=0.45,rounding_size=0.6", fc=color, ec="none"))
    return t


def chart_card(key: str, tf: str, d: pd.DataFrame, eff: np.ndarray, rec: dict, dec: dict, vs: dict,
               news_line: str = "", out: str = "card.png", n: int = 120, htf_lines: list | None = None) -> str:
    t = pd.to_datetime(d["time"], utc=True).dt.tz_convert(LOCAL)
    o, h, l, c = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    i0 = max(0, len(c) - n)
    X = np.arange(len(c)) - i0
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
    # ---------- header ----------
    fig.text(0.05, 0.955, key, fontsize=44, fontweight="bold", color=INK, va="center")
    fig.text(0.05 + 0.028 * len(key) + 0.03, 0.958, f"{tf}  ·  {NAMES.get(key, '')}", fontsize=18, color=QUIET, va="center")
    _chip(fig, 0.95, 0.955, dec["action"], COLORS.get(dec["tone"], INK))
    g = (rec.get("ranges") or [None])[-1]
    sub = f"{g['name']}  ·  phase {g['phase']}" if g else {1: "Major trend up", -1: "Major trend down"}.get(
        (rec.get("major") or {}).get("state"), "Sideways")
    fig.text(0.05, 0.915, sub, fontsize=17, color=QUIET, va="center")
    fig.text(0.95, 0.915, f"{float(c[-1]):.5g}  ·  {t.iloc[-1].strftime('%a %d %b %H:%M')}", fontsize=15, color=QUIET,
             va="center", ha="right")
    # ---------- price ----------
    ax = fig.add_axes([0.05, 0.47, 0.82, 0.42], facecolor=BG)
    ae = fig.add_axes([0.05, 0.37, 0.82, 0.058], facecolor=BG, sharex=ax)
    for a in (ax, ae):
        for s in a.spines.values():
            s.set_visible(False)
        a.tick_params(colors=QUIET, labelsize=12, length=0)
        a.yaxis.tick_right()
    ax.set_yticks([])
    lo_v, hi_v = l[i0:].min(), h[i0:].max()
    if g:
        S, R = g["support"], g["resistance"]
        a1 = int(np.searchsorted(t.values, pd.Timestamp(g["id"]).tz_convert(LOCAL).to_datetime64()))
        xa = max(a1 - i0, 0)
        tone = COLORS.get(dec["tone"], INK)
        ax.add_patch(Rectangle((xa - 0.5, S), (len(c) - i0) - xa, R - S, fc=tone, alpha=0.07, ec="none"))
        for y, lab in ((R, "resistance"), (S, "support")):
            ax.plot([xa - 0.5, len(c) - i0 + 2], [y, y], color=tone, lw=1.4, alpha=0.8)
            ax.text(len(c) - i0 + 2.5, y, f"{lab}  {y:.5g}", fontsize=12, color=tone, va="center")
        lo_v, hi_v = min(lo_v, S), max(hi_v, R)
        for (iso, name, state, price) in g["events"]:
            b = int(np.searchsorted(t.values, pd.Timestamp(iso).tz_convert(LOCAL).to_datetime64()))
            if b < i0 or b >= len(c):
                continue
            y = h[b] if price >= (h[b] + l[b]) / 2 else l[b]
            above = y == h[b]
            ax.annotate(name, (b - i0, y), xytext=(0, 14 if above else -16), textcoords="offset points", ha="center",
                        va="bottom" if above else "top", fontsize=14, fontweight="bold",
                        color=INK if state == "confirmed" else QUIET)
    # higher-timeframe edges (15m / 5m cards): dashed, labelled at the left
    for lab, lv in (htf_lines or []):
        if lo_v - (hi_v - lo_v) * 0.6 <= lv <= hi_v + (hi_v - lo_v) * 0.6:
            ax.axhline(lv, color="#C99A06", lw=1.3, ls=(0, (5, 3)), alpha=0.9)
            ax.text(0, lv, f" {lab} {lv:.5g}", fontsize=12, color="#8A6A00", va="bottom", fontweight="bold")
            lo_v, hi_v = min(lo_v, lv), max(hi_v, lv)
    # candles
    for i in range(i0, len(c)):
        col = UP if c[i] >= o[i] else DN
        x = i - i0
        ax.plot([x, x], [l[i], h[i]], color=col, lw=0.9)
        ax.add_patch(Rectangle((x - 0.32, min(o[i], c[i])), 0.64, max(abs(c[i] - o[i]), (hi_v - lo_v) * 0.0015), fc=col, ec="none"))
    # swings with their VSA code
    for s in vs["swings"][-3:]:
        if s["end"] < i0:
            continue
        xs, xe = max(s["start"], i0) - i0, s["end"] - i0
        ys = l[s["start"]] if s["dir"] == 1 else h[s["start"]]
        ye = h[s["end"]] if s["dir"] == 1 else l[s["end"]]
        ax.plot([xs, xe], [ys, ye], color=QUIET, lw=1.2, alpha=0.7)
        if s["code"] != "=":
            bull = s["code"] in ("D>", "S<")
            ax.annotate(s["code"], (xe, ye), xytext=(16, 30 if s["dir"] == 1 else -34), textcoords="offset points",
                        ha="left", fontsize=13, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc=UP if bull else DN, ec="none", alpha=0.9),
                        arrowprops=dict(arrowstyle="-", color=QUIET, lw=0.8))
    pad = (hi_v - lo_v) * 0.12
    ax.set_ylim(lo_v - pad, hi_v + pad)
    ax.set_xlim(-1, len(c) - i0 + 16)
    ax.axhline(c[-1], color=INK, lw=0.6, ls=(0, (2, 3)), alpha=0.6)
    ax.text(len(c) - i0 + 2.5, c[-1], f"{c[-1]:.5g}", fontsize=12, color="white", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=INK, ec="none"))
    plt.setp(ax.get_xticklabels(), visible=False)
    # effort
    for i in range(i0, len(c)):
        e = eff[i]
        ae.bar(i - i0, 1 if np.isnan(e) else min(e, 3), width=0.75, color=UP if c[i] >= o[i] else DN, alpha=_alpha(e))
    ae.axhline(1.0, color=QUIET, lw=0.6, ls=(0, (2, 3)))
    ae.set_ylim(0, 3.1)
    ae.set_yticks([1, 2])
    ae.set_yticklabels(["1×", "2×"])
    ticks = [i for i in range(i0, len(c)) if i == i0 or t.iloc[i].date() != t.iloc[i - 1].date()]
    step = max(1, len(ticks) // 6)
    ae.set_xticks([i - i0 for i in ticks[::step]])
    ae.set_xticklabels([t.iloc[i].strftime("%d %b") for i in ticks[::step]])
    fig.text(0.05, 0.432, "effort  ·  volume vs the same hour, darker = more", fontsize=11.5, color=QUIET)
    # ---------- decision panel ----------
    fig.patches.append(FancyBboxPatch((0.04, 0.015), 0.92, 0.325, boxstyle="round,pad=0,rounding_size=0.015",
                                      transform=fig.transFigure, fc=PANEL, ec="none"))
    rows = [("DO", dec["do"])] + ([("HTF", dec["htf"])] if dec.get("htf") else []) + [("TRIGGER", dec["trigger"])]
    if dec.get("invalid") and dec["invalid"] != "—":
        rows.append(("INVALID", dec["invalid"]))
    elif g:
        rows.append(("LEVELS", f"support {g['support']:.5g}  ·  resistance {g['resistance']:.5g}"))
    last_r = next((s for s in reversed(vs["swings"]) if s["dir"] == 1), None)
    last_x = next((s for s in reversed(vs["swings"]) if s["dir"] == -1), None)
    sw = [f"{nm}  V{s['v']} R{s['r']} → {s['code']}  {s['text']}" for nm, s in (("rally", last_r), ("reaction", last_x)) if s]
    rows.append(("CONTROL", vs["verdict"] + "\n" + "\n".join(sw)))
    b = vs["bars"][-1]
    rows.append(("LAST BAR", f"{b['expr']}  {b['text']}"))
    if news_line:
        rows.append(("NEWS", news_line))
    y = 0.322
    import textwrap
    for lab, txt in rows:
        fig.text(0.065, y, lab, fontsize=13, fontweight="bold", color=QUIET, va="top")
        lines = [w for part in txt.split("\n") for w in (textwrap.wrap(part, 62) or [""])][:4]
        fig.text(0.2, y, "\n".join(lines), fontsize=17 if lab == "DO" else 15, color=INK,
                 fontweight="bold" if lab == "DO" else "normal", va="top", linespacing=1.35)
        y -= 0.0215 * max(1, len(lines)) + 0.016
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    return out


def board(rows: list[dict], when: str, out: str = "board.png") -> str:
    """The market board: one row per future — 4h and 1h action chips, who is in control, the level that matters."""
    fig = plt.figure(figsize=(10.8, 2.2 + 1.05 * len(rows)), dpi=100, facecolor=BG)
    H = fig.get_figheight()
    fig.text(0.05, 1 - 0.55 / H, "Market board", fontsize=34, fontweight="bold", color=INK, va="center")
    fig.text(0.95, 1 - 0.55 / H, when, fontsize=15, color=QUIET, va="center", ha="right")
    hy = 1 - 1.35 / H
    for x, lab in ((0.05, ""), (0.2, "4h"), (0.48, "1h"), (0.74, "who's in control (4h)")):
        fig.text(x, hy, lab, fontsize=13, color=QUIET, fontweight="bold", va="center")
    for k, r in enumerate(rows):
        y = 1 - (1.95 + 1.05 * k) / H
        fig.patches.append(FancyBboxPatch((0.035, y - 0.42 / H), 0.93, 0.84 / H, boxstyle="round,pad=0,rounding_size=0.01",
                                          transform=fig.transFigure, fc=PANEL, ec="none"))
        fig.text(0.05, y + 0.12 / H, r["key"], fontsize=24, fontweight="bold", color=INK, va="center")
        fig.text(0.05, y - 0.22 / H, NAMES.get(r["key"], ""), fontsize=11, color=QUIET, va="center")
        for x, dk in ((0.2, "d4"), (0.48, "d1")):
            dd = r[dk]
            fig.text(x, y + 0.12 / H, dd["action"], fontsize=15, fontweight="bold", color="white", va="center",
                     bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.5", fc=COLORS.get(dd["tone"], INK), ec="none"))
            fig.text(x, y - 0.25 / H, dd.get("short", ""), fontsize=11, color=QUIET, va="center")
        ctrl = r["ctrl"]
        col = UP if ctrl.startswith("DEMAND") else DN if ctrl.startswith("SUPPLY") else QUIET
        fig.text(0.74, y + 0.12 / H, ctrl.split(" ")[0], fontsize=16, fontweight="bold", color=col, va="center")
        fig.text(0.74, y - 0.25 / H, r.get("ctrl_why", ""), fontsize=11, color=QUIET, va="center")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    return out
