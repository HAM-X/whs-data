// Pipdon BOT · Trade Desk in the bot: Manage (open trades, hourly checks) and Guard (prop / own limits).
// State lives in Cloudflare KV (binding DESK, key "desk") — private, never in the public repo.
// Everything is in % of the account and in R, so it works for any account size (prop or live).

export const RISK = { A: 0.5, B: 0.25, C: 0.125 };     // % of balance (settled 18 Oct 2026)
export const OWN = { day: 2.5, ddHalf: 6, ddStop: 8, heat: 1.5, theme: 0.75, lossesDay: 2, streak: 4 };

// effective heat √(Σ Σ wᵢ wⱼ ρᵢⱼ) of signed positions (+ = long the future = short USD), ρ from the Market Panel
export function effHeat(pos, rho) {
  let s = 0;
  for (const a of pos) for (const b of pos) s += a.w * b.w * (a.key === b.key ? 1 : (rho?.[a.key]?.[b.key] ?? 0));
  return Math.sqrt(Math.max(0, s));
}
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

export function guard(st, addRisk = 0, now = Date.now(), ctx = {}) {
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
  let state = "OK TO EXECUTE", factor = 1;
  const why = [];
  const block = (w) => { state = "BLOCKED"; why.push(w); };
  if (worst <= -lim.day) block(`would breach the firm's daily −${lim.day}%`);
  if (worst <= -OWN.day) block(`would breach your own daily −${OWN.day}%`);
  if (losses >= OWN.lossesDay) block("2 losses today — done for the day");
  if (ddLoss >= OWN.ddStop) block(`account down ${ddLoss.toFixed(2)}% — stop and review (own −${OWN.ddStop}%)`);
  if (lim.dd - ddLoss <= addRisk + open) block("open risk would reach the firm's max drawdown");
  // themes: correlated positions are one idea (Market Panel ρ, 60 days)
  const pos = st.trades.map((t) => ({ key: t.key, w: openRisk(t) * t.side })).filter((p) => p.w);
  if (ctx.key && addRisk) pos.push({ key: ctx.key, w: addRisk * (ctx.side || 1) });
  let heat = open + addRisk;
  if (ctx.rho && pos.length) {
    heat = effHeat(pos, ctx.rho);
    facts.push(`Effective heat ${heat.toFixed(2)}% (correlations) vs plain sum ${(open + addRisk).toFixed(2)}%`);
    if (ctx.key) {
      const bloc = (ctx.blocs || []).find((b) => b.includes(ctx.key)) || [ctx.key];
      const th = effHeat(pos.filter((p) => bloc.includes(p.key)), ctx.rho);
      if (bloc.length > 1) facts.push(`Theme ${bloc.join("/")}: ${th.toFixed(2)}% (cap ${OWN.theme}%)`);
      if (th > OWN.theme + 1e-9 && state !== "BLOCKED") {
        state = "REDUCE";
        // largest new size that keeps the theme at the cap
        let lo = 0, hi = addRisk;
        for (let i = 0; i < 30; i++) {
          const m = (lo + hi) / 2;
          const p2 = pos.slice(0, -1).filter((p) => bloc.includes(p.key)).concat([{ key: ctx.key, w: m * (ctx.side || 1) }]);
          if (effHeat(p2, ctx.rho) <= OWN.theme) lo = m; else hi = m;
        }
        why.push(`same idea as ${bloc.filter((k) => k !== ctx.key).join("/")} — max ${lo.toFixed(2)}% on this one`);
      }
    }
  }
  if (state !== "BLOCKED") {
    if (heat > OWN.heat) { state = "REDUCE"; why.push(`heat ${heat.toFixed(2)}% > ${OWN.heat}%`); }
    if (ddLoss >= OWN.ddHalf) { state = "REDUCE"; factor = 0.5; why.push(`account down ${ddLoss.toFixed(2)}% → half risk`); }
    if (streak >= OWN.streak) { state = "REDUCE"; factor = 0.5; why.push(`${streak} losses in a row → half risk until a win`); }
  }
  return { state, factor, why, facts, worst, open };
}

export function guardText(st, addRisk = 0, now = Date.now(), ctx = {}) {
  const g = guard(st, addRisk, now, ctx);
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

export function openTrade(st, snap, args, now = Date.now(), panel = null) {
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
  const pctx = panel ? { rho: panel.themes?.rho, blocs: panel.themes?.blocs, key: k, side } : {};
  const gd = guard(st, risk, now, pctx);
  if (gd.state === "BLOCKED") return guardText(st, risk, now, pctx) + "\n⛔ Not opened.";
  const R = Math.abs(entry - stop);
  const t = { id: now, key: k, side, entry, stop0: stop, stop, R, grade, risk: risk * gd.factor, opened: new Date(now).toISOString(),
              lastDone: null, third: false, be: false, best: 0, bars: 0, timeWarned: false };
  st.trades.push(t);
  return `📒 <b>Opened ${k} ${side === 1 ? "LONG" : "SHORT"}</b> ${p6(entry)} · stop ${p6(stop)} · 1R = ${p6(R)} · grade ${grade} ${t.risk}%\n` +
    `Plan: ⅓ off at 2R ${p6(entry + 2 * side * R)} · ADAPTIVE stop: each SBar commitment → its true-range ${side === 1 ? "low" : "high"}, ` +
    `after +1R a climactic bar → ½ of it · reversal bar against → exit · 48 bars without +1R → time stop.\n` +
    (gd.state === "REDUCE" ? "🟠 " + gd.why.join(" · ") + "\n" : "") + guardText(st, 0, now, panel ? { rho: panel.themes?.rho } : {});
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
      // 3) ADAPTIVE stop (the Trade Desk's, approved 20 Oct 2026): every SBar commitment in our direction → its
      //    TRUE-RANGE low/high (combo pair: the pair's); after +1R a climactic bar in our direction → ½ of that bar.
      //    Never loosens, and only when the close is beyond the new stop.
      const pc = i > 0 ? b.c[i - 1] : cl;
      const trLo = Math.min(lo, pc), trHi = Math.max(hi, pc);
      const cands = [];
      if (Math.sign(b.sb[i]) === side) {
        let lv = side === 1 ? trLo : trHi;
        if (Math.abs(b.sb[i]) === 2 && i > 0) {
          const pc2 = i > 1 ? b.c[i - 2] : b.c[i - 1];
          lv = side === 1 ? Math.min(lv, b.l[i - 1], pc2) : Math.max(lv, b.h[i - 1], pc2);
        }
        cands.push([lv, "SBar commitment → its true-range " + (side === 1 ? "low" : "high")]);
        const a1c = i > 0 ? b.atr[i - 1] : null;
        if (t.best >= 1 && b.e[i] != null && b.e[i] >= 2.5 && a1c && trHi - trLo >= 2 * a1c)
          cands.push([(trHi + trLo) / 2, "climactic bar after +1R → ½ of it"]);
      }
      for (const [lv, why] of cands) {
        const ns = Math.round(lv / inst.tick) * inst.tick;
        if ((ns - t.stop) * side > 0 && (cl - ns) * side > 0) {
          t.stop = ns;
          out.push(`🪜 ${tag}: ${why} — stop to ${p6(t.stop)} (${fx(rNow(t, t.stop))}R locked).`);
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

export function deskRoute(cmd, args, st, snap, now = Date.now(), panel = null) {
  if (!st) return "Trade storage isn't set up yet (Cloudflare KV). It is created on the next bot deploy.";
  switch (cmd) {
    case "/open": return openTrade(st, snap, args, now, panel);
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
    case "/account": {
      const sub = String(args[0] || "").toLowerCase();
      if (["pnl", "dd", "limits"].includes(sub)) {
        const out = deskRoute("/" + sub, args.slice(1), st, snap, now, panel);
        return out.startsWith("Usage") || out.startsWith("Firm limits now") ? out : accountText(st, now);
      }
      return accountText(st, now);
    }
    case "/guard": {
      // /guard B  ·  /guard 6E long B  (the second form checks the theme too)
      let key = null, side = 1, g = String(args[0] || "").toUpperCase();
      if (snap.instruments[args[0]]) { key = args[0]; side = /^(short|sell|s)$/i.test(args[1] || "") ? -1 : 1; g = String(args[2] || "B").toUpperCase(); }
      const ctx = panel ? { rho: panel.themes?.rho, blocs: panel.themes?.blocs, key, side } : {};
      return guardText(st, RISK[g] ?? (parseFloat(g) || 0), now, ctx);
    }
  }
  return null;
}
export const DESK_CMDS = ["/open", "/close", "/trades", "/stop", "/pnl", "/dd", "/limits", "/guard", "/account", "/plan"];

// /account — one screen: can I trade, today, drawdown, open risk, last trades. Setters: /account pnl -0.4 · dd 3 · limits 5 10
export function accountText(st, now = Date.now()) {
  const g = guard(st, 0, now);
  const lim = st.limits || { day: 5, dd: 10 };
  const today = localDay(now);
  const day = st.pnl && st.pnl.day === today ? st.pnl.pct : 0;
  const icon = g.state === "BLOCKED" ? "⛔" : g.state === "REDUCE" ? "🟠" : "🟢";
  const last = st.journal.slice(-5).map((j) => fx(j.r, 1) + "R");
  const sum = st.journal.slice(-5).reduce((a, j) => a + j.r, 0);
  return [`<b>Account</b>  ▸ ${icon} <b>${g.state === "OK TO EXECUTE" ? "OK TO TRADE" : g.state}</b>` + (g.factor < 1 ? " · half risk" : ""),
    ...(g.why.length ? g.why.map((w) => "• " + w) : []), "",
    `<b>Today</b>  ${fx(day)}%  (own stop −${OWN.day}% · firm −${lim.day}%)`,
    `<b>Account</b>  ${fx(st.dd || 0)}% vs start  (firm −${lim.dd}%)`,
    `<b>Open risk</b>  ${g.open.toFixed(2)}% · ${st.trades.length} trade${st.trades.length === 1 ? "" : "s"}`,
    `<b>Last trades</b>  ${last.length ? last.join(" · ") + `  (sum ${fx(sum, 1)}R)` : "none yet"}`,
    "<i>/account pnl -0.4 · /account dd 3 · /account limits 5 10</i>"].join("\n");
}
