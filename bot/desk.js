// Pipdon BOT · Trade Desk in the bot: Manage (open trades, hourly checks) and Guard (prop / own limits).
// State lives in Cloudflare KV (binding DESK, key "desk") — private, never in the public repo.
// Everything is in % of the account and in R, so it works for any account size (prop or live).

export const RISK = { A: 0.5, B: 0.25, C: 0.125 };     // % of balance (settled 18 Oct 2026)
export const OWN = { day: 2.5, ddHalf: 6, ddStop: 8, heat: 1.5, lossesDay: 2, streak: 4 };
const MAX_JOURNAL = 300;

export const emptyState = () => ({ trades: [], journal: [], pnl: null, dd: 0, limits: { day: 5, dd: 10 }, alerts: [] });
export const localDay = (ms) => new Date(ms + 3 * 3600e3).toISOString().slice(0, 10);   // UTC+3 trading day
const fx = (v, d = 2) => (v == null || !isFinite(v) ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d));
const p6 = (v) => (v == null ? "—" : Number(v).toPrecision(6).replace(/\.?0+$/, ""));

export async function load(env) {
  if (!env.DESK) return null;
  return { ...emptyState(), ...((await env.DESK.get("desk", "json")) || {}) };
}
export async function save(env, st) {
  if (env.DESK) await env.DESK.put("desk", JSON.stringify(st));
}

// open risk of a trade in % of the account: what is lost if the stop is hit now (0 once the stop is at or past entry)
export const openRisk = (t) => Math.max(0, ((t.entry - t.stop) * t.side) / t.R) * t.risk;
export const rNow = (t, price) => ((price - t.entry) * t.side) / t.R;

export function guard(st, addRisk = 0, now = Date.now()) {
  const today = localDay(now);
  const day = st.pnl && st.pnl.day === today ? st.pnl.pct : 0;
  const open = st.trades.reduce((a, t) => a + openRisk(t), 0);
  const worst = day - open - addRisk;
  const todays = st.journal.filter((j) => j.day === today);
  const losses = todays.filter((j) => j.r < 0).length;
  let streak = 0;
  for (let i = st.journal.length - 1; i >= 0 && st.journal[i].r < 0; i--) streak++;
  const ddLoss = Math.max(0, -(st.dd || 0));
  const lim = st.limits || { day: 5, dd: 10 };
  const facts = [
    `Day: ${fx(day)}% realised · open risk ${open.toFixed(2)}%${addRisk ? ` · this trade ${addRisk}%` : ""} → worst case ${fx(worst)}% (own stop −${OWN.day}%, firm −${lim.day}%)`,
    `Account vs start: ${fx(st.dd || 0)}% · room to the firm's −${lim.dd}%: ${(lim.dd - ddLoss).toFixed(2)}%`,
    `Losses today ${losses}/${OWN.lossesDay} · losing streak ${streak}`,
  ];
  let state = "OK TO EXECUTE", factor = 1, why = [];
  const block = (w) => { state = "BLOCKED"; why.push(w); };
  if (worst <= -lim.day) block(`would breach the firm's daily −${lim.day}%`);
  if (worst <= -OWN.day) block(`would breach your own daily −${OWN.day}%`);
  if (losses >= OWN.lossesDay) block("2 losses today — done for the day");
  if (ddLoss >= OWN.ddStop) block(`account down ${ddLoss.toFixed(2)}% — stop and review (own −${OWN.ddStop}%)`);
  if (lim.dd - ddLoss <= addRisk + open) block("open risk would reach the firm's max drawdown");
  if (state !== "BLOCKED") {
    if (open + addRisk > OWN.heat) { state = "REDUCE"; why.push(`heat ${(open + addRisk).toFixed(2)}% > ${OWN.heat}%`); }
    if (ddLoss >= OWN.ddHalf) { state = "REDUCE"; factor = 0.5; why.push(`account down ${ddLoss.toFixed(2)}% → half risk`); }
    if (streak >= OWN.streak) { state = "REDUCE"; factor = 0.5; why.push(`${streak} losses in a row → half risk until a win`); }
  }
  return { state, factor, why, facts, worst, open };
}

export function guardText(st, addRisk = 0, now = Date.now()) {
  const g = guard(st, addRisk, now);
  const icon = g.state === "BLOCKED" ? "⛔" : g.state === "REDUCE" ? "🟠" : "🟢";
  return `${icon} <b>Guard: ${g.state}</b>${g.factor < 1 ? " · risk × 0.5" : ""}` + (g.why.length ? "\n" + g.why.map((w) => "• " + w).join("\n") : "") +
    "\n" + g.facts.join("\n") + "\n<i>Update with /pnl −0.4 (today realised %) and /dd 3 (account vs start %).</i>";
}

// 2 "pips" on the future: the MT5 pip scaled to the future's price (gold: 2 ticks)
export function pipOffset(inst) {
  if (!inst) return 0;
  if (inst.key === "GC") return 2 * inst.tick;
  const m = inst.mt5;
  if (!m || !m.spot || !inst.price) return 4 * inst.tick;
  return (2 * m.pip * inst.price) / m.spot;
}

export function openTrade(st, snap, args, now = Date.now()) {
  const [k, dir, e, s, g] = args;
  const inst = snap.instruments[k];
  if (!inst) return "Which instrument?";
  const side = /^(long|buy|l)$/i.test(dir) ? 1 : /^(short|sell|s)$/i.test(dir) ? -1 : 0;
  const entry = parseFloat(e), stop = parseFloat(s);
  if (!side || !isFinite(entry) || !isFinite(stop) || (entry - stop) * side <= 0)
    return "Usage: /open 6E long 1.0909 1.0889 B — future prices; stop below a long's entry, above a short's.";
  const grade = String(g || "B").toUpperCase();
  const risk = RISK[grade] ?? parseFloat(grade);
  if (!isFinite(risk) || risk <= 0) return "Grade A, B, C or a risk % like 0.25.";
  const gd = guard(st, risk, now);
  const R = Math.abs(entry - stop);
  const t = { id: now, key: k, side, entry, stop0: stop, stop, R, grade, risk: risk * gd.factor, opened: new Date(now).toISOString(),
              lastDone: null, third: false, be: false, best: 0, bars: 0, timeWarned: false };
  st.trades.push(t);
  return `📒 <b>Opened ${k} ${side === 1 ? "LONG" : "SHORT"}</b> ${p6(entry)} · stop ${p6(stop)} · 1R = ${p6(R)} · grade ${grade} ${t.risk}%\n` +
    `Plan: ⅓ off at 2R ${p6(entry + 2 * side * R)} · stop to breakeven after a close beyond +1R ${p6(entry + side * R)} · ` +
    `then the SBar staircase (each new SBar's ${side === 1 ? "low" : "high"} ∓ 2 pips) · reversal bar against → exit · 48 bars without +1R → time stop.\n` +
    guardText(st, 0, now);
}

export function closeTrade(st, snap, args, now = Date.now()) {
  const k = args[0];
  const i = st.trades.findIndex((t) => t.key === k);
  if (i < 0) return `No open ${k} trade.`;
  const t = st.trades[i];
  const px = isFinite(parseFloat(args[1])) ? parseFloat(args[1]) : snap.instruments[k]?.price;
  const r = rNow(t, px);
  st.trades.splice(i, 1);
  st.journal.push({ key: k, side: t.side, r: Math.round(r * 100) / 100, risk: t.risk, day: localDay(now), closed: new Date(now).toISOString() });
  st.journal = st.journal.slice(-MAX_JOURNAL);
  return `📕 Closed ${k} at ${p6(px)} · ${fx(r)}R (${fx(r * t.risk)}% of the account)\n` + guardText(st, 0, now);
}

export function tradesText(st, snap) {
  if (!st.trades.length) return "No open trades. /open 6E long 1.0909 1.0889 B";
  return st.trades.map((t) => {
    const px = snap.instruments[t.key]?.price;
    return `${t.key} ${t.side === 1 ? "LONG" : "SHORT"} ${p6(t.entry)} · stop ${p6(t.stop)} · now ${p6(px)} = ${fx(rNow(t, px))}R` +
      `${t.third ? " · ⅓ taken" : ""}${t.be ? " · BE" : ""} · risk left ${openRisk(t).toFixed(2)}%`;
  }).join("\n");
}

// hourly management of every open trade, bar by bar since the last check. Returns alert lines.
export function manage(st, snap, now = Date.now()) {
  const out = [];
  for (const t of [...st.trades]) {
    const inst = snap.instruments[t.key];
    const b = inst && inst.bars1h;
    if (!b) continue;
    const off = pipOffset(inst);
    for (let i = 0; i < b.t.length; i++) {
      if (Date.parse(b.t[i]) < Date.parse(t.opened)) continue;      // only full bars that started after the entry
      if (t.lastDone && b.t[i] <= t.lastDone) continue;            // already managed
      t.lastDone = b.t[i];
      t.bars += 1;
      const hi = b.h[i], lo = b.l[i], cl = b.c[i], side = t.side;
      const tag = `${t.key} ${side === 1 ? "long" : "short"}`;
      // 1) stop hit
      if ((side === 1 && lo <= t.stop) || (side === -1 && hi >= t.stop)) {
        const r = rNow(t, t.stop);
        out.push(`🛑 ${tag}: stop ${p6(t.stop)} hit (${b.t[i].slice(11, 16)} UTC) · ${fx(r)}R. Logged — /close is not needed.`);
        st.trades = st.trades.filter((x) => x !== t);
        st.journal.push({ key: t.key, side, r: Math.round(r * 100) / 100, risk: t.risk, day: localDay(now), closed: b.t[i] });
        break;
      }
      const fav = side === 1 ? hi : lo;
      t.best = Math.max(t.best, rNow(t, fav));
      // 2) ⅓ at 2R
      if (!t.third && rNow(t, fav) >= 2) {
        t.third = true;
        out.push(`💰 ${tag}: 2R reached ${p6(t.entry + 2 * side * t.R)} — take ⅓ off.`);
      }
      // 3) breakeven after a close beyond +1R
      if (!t.be && rNow(t, cl) >= 1) {
        t.be = true;
        if ((t.entry - t.stop) * side > 0) {
          t.stop = t.entry;
          out.push(`🔒 ${tag}: closed beyond +1R — stop to breakeven ${p6(t.entry)}.`);
        }
      }
      // 4) SBar staircase
      if (Math.sign(b.sb[i]) === side) {
        const ns = side === 1 ? lo - off : hi + off;
        if ((ns - t.stop) * side > 0 && (cl - ns) * side > 0) {
          t.stop = Math.round(ns / inst.tick) * inst.tick;
          out.push(`🪜 ${tag}: new SBar — stop to ${p6(t.stop)} (${fx(rNow(t, t.stop))}R locked).`);
        }
      }
      // 5) reversal bar against
      if (b.rb[i] === -side) out.push(`↩️ ${tag}: reversal bar against at ${p6(cl)} (${fx(rNow(t, cl))}R) — exit the rest at the close.`);
      // 6) climactic bar
      const a1 = i > 0 ? b.atr[i - 1] : null;
      if (b.e[i] != null && b.e[i] >= 3 && a1 && hi - lo >= 2 * a1)
        out.push(`⚡ ${tag}: climactic bar (effort ${b.e[i].toFixed(1)}×, ${((hi - lo) / a1).toFixed(1)}× ATR) — manage on the 15m swing.`);
      // 7) time stop
      if (!t.timeWarned && t.bars >= 48 && t.best < 1) {
        t.timeWarned = true;
        out.push(`⏳ ${tag}: 48 bars without +1R (best ${fx(t.best)}R) — time stop: exit or justify.`);
      }
    }
  }
  return out;
}

export function deskRoute(cmd, args, st, snap, now = Date.now()) {
  if (!st) return "Trade storage isn't set up yet (Cloudflare KV). It is created on the next bot deploy.";
  switch (cmd) {
    case "/open": return openTrade(st, snap, args, now);
    case "/close": return closeTrade(st, snap, args, now);
    case "/trades": return tradesText(st, snap);
    case "/stop": {
      const t = st.trades.find((x) => x.key === args[0]);
      const v = parseFloat(args[1]);
      if (!t || !isFinite(v)) return "Usage: /stop 6E 1.0921";
      t.stop = v;
      return `Stop for ${t.key} set to ${p6(v)} (${fx(rNow(t, v))}R).`;
    }
    case "/pnl": {
      const v = parseFloat(String(args[0] || "").replace("−", "-"));
      if (!isFinite(v)) return "Usage: /pnl -0.4 — today's realised result in % of the account.";
      st.pnl = { day: localDay(now), pct: v };
      return guardText(st, 0, now);
    }
    case "/dd": {
      const v = parseFloat(String(args[0] || "").replace("−", "-"));
      if (!isFinite(v)) return "Usage: /dd 3 — account vs its starting balance in % (+3 = up 3%).";
      st.dd = v;
      return guardText(st, 0, now);
    }
    case "/limits": {
      const d = parseFloat(args[0]), m = parseFloat(args[1]);
      if (!isFinite(d) || !isFinite(m)) return `Firm limits now: daily −${st.limits.day}% · max drawdown −${st.limits.dd}%. Change: /limits 5 10`;
      st.limits = { day: d, dd: m };
      return `Firm limits set: daily −${d}% · max drawdown −${m}%.`;
    }
    case "/guard": {
      const g = String(args[0] || "").toUpperCase();
      return guardText(st, RISK[g] ?? (parseFloat(g) || 0), now);
    }
  }
  return null;
}
export const DESK_CMDS = ["/open", "/close", "/trades", "/stop", "/pnl", "/dd", "/limits", "/guard"];
