import { deskRoute, DESK_CMDS, load as loadDesk, save as saveDesk, manage, guard, RISK } from "./desk.js";

// Pipdon BOT — Telegram command bot on Cloudflare Workers (no AI; every answer is computed).
// Reads the robot's hourly snapshot (data/snapshot.json in HAM-X/whs-data) and answers only Hamza's chat.

const RISK_GRADE = { A: 0.5, B: 0.25, C: 0.125 };           // % of balance, settled 18 Oct 2026
const TICK_VALUE = { "6E": 6.25, "6B": 6.25, "6A": 5, "6N": 5, "6J": 6.25, "6C": 5, "6S": 6.25, GC: 10 };
const HAS_MICRO = { "6E": "M6E", "6B": "M6B", "6A": "M6A", "6J": "MJY", "6C": "MCD", "6S": "MSF", GC: "MGC" };
const ALIASES = {
  "6E": ["6e", "eur", "euro", "eurusd", "fiber"],
  "6B": ["6b", "gbp", "pound", "cable", "gbpusd", "sterling"],
  "6A": ["6a", "aud", "aussie", "audusd"],
  "6N": ["6n", "nzd", "kiwi", "nzdusd"],
  "6J": ["6j", "jpy", "yen", "usdjpy"],
  "6C": ["6c", "cad", "loonie", "usdcad"],
  "6S": ["6s", "chf", "swissy", "franc", "usdchf"],
  GC: ["gc", "gold", "xau", "xauusd"],
  DX: ["dx", "dxy", "dollar", "usdx", "usd"],
};
const BAND_WORD = { "≫": "very high", ">": "high", "≈": "normal", "<": "low", "≪": "very low", "n/a": "n/a" };

export const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const decimals = (tick) => Math.max(0, Math.ceil(-Math.log10(tick) - 1e-9));
const px = (v, tick) => (v == null ? "—" : Number(v).toFixed(decimals(tick)));
const sgn = (v, d = 0) => (v == null ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d));
const spotDigits = (sym) => (sym === "USDJPY" ? 3 : sym === "XAUUSD" ? 2 : 5);

export function findSymbol(text) {
  const words = text.toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/);
  for (const [k, names] of Object.entries(ALIASES)) if (words.some((w) => names.includes(w))) return k;
  return null;
}

function ageWarning(snap, now = Date.now()) {
  const ageH = (now - Date.parse(snap.generated_utc)) / 3.6e6;
  return ageH > 2.5 ? `\n⚠️ Snapshot is ${ageH.toFixed(0)}h old (market closed, or the robot is behind).` : "";
}

function footer(snap) {
  return `\n<i>Robot data as of ${esc(snap.generated_local)} (UTC+3). TradingView is the price of record.</i>`;
}

// ---------- answers ----------
export function help() {
  return [
    "<b>Pipdon BOT</b>  — every answer ends in a decision",
    "",
    "<b>Read the market</b>",
    "/today — the board: every chart's action + what to act on, watch and avoid",
    "/refresh — a fresh read of every chart now (≈2 min) · /refresh 6E 15m for one chart",
    "/setups — what to act on now, and which charts sit at a 4h / 1h edge",
    "/chart 6E [1h] — the chart card: action, trigger, levels, who's in control",
    "/zoom 6E [5m] — the 15m / 5m structure read against the 4h / 1h range edges",
    "/vsa 6E [1h|15m|5m] — rallies, reactions and bars in VSA notation (V R D S)",
    "/why 6E [1h] — the Wyckoff story: events, phases, bias",
    "/trend 6E [tf] — major trend, protected level, swing structure · /swings 6E [tf] — every swing in notation",
    "/effort 6E [tf] — effort vs result bar by bar · /read 6E [tf] — everything at once",
    "<b>Any pair, any timeframe (5m 15m 30m 1h 4h daily weekly), any mix — fresh in ≈1 min:</b>",
    "  e.g. \"gc 15m vsa trend\" · \"yen daily why swings\" · \"euro 5m chart now\" · /ask cable 30m all",
    "/dollar — USD strong / weak / mixed and what it favours",
    "/news — red-folder releases, blackouts, how price answered",
    "",
    "<b>Trade</b>",
    "/plan 6E long 1.0909 1.0889 [A|B|C] — GO / REDUCE / NO with the MT5 order and lots",
    "/open … · /trades · /close 6E [price] — the bot manages the stop hourly",
    "/account — can I trade, today, drawdown, open risk, last trades",
    "/journal — your closed trades: win rate, average R, last 10",
    "/calls — the robot's own calls, scored live (Saturday report)",
    "",
    "<i>Plain words work too: \"euro\", \"gold vsa 15m\", \"why yen\", \"setups\", \"dollar\", \"any news\".</i>",
  ].join("\n");
}

export function status(snap, key) {
  const i = snap.instruments[key];
  if (!i || !i.available) return `No data for ${key} yet.`;
  const t = i.tick, d = i.day, l = i.last;
  const lines = [
    `<b>${key} · ${esc(i.name)}</b>  <code>${px(i.price, t)}</code>  (bar closed ${esc(l.local)})`,
    `Today ${sgn(d.change_ticks)} ticks (${sgn(d.change_pct, 2)}%) · range ${Math.round(d.range / t)} ticks = <b>${i.adr_used_pct ?? "—"}% of ADR</b> · price at ${Math.round((d.pos_in_range ?? 0) * 100)}% of the range`,
    `Day H/L <code>${px(d.high, t)}</code> / <code>${px(d.low, t)}</code>` +
      (i.prev_day ? ` · prev day H/L/C <code>${px(i.prev_day.high, t)}</code> / <code>${px(i.prev_day.low, t)}</code> / <code>${px(i.prev_day.close, t)}</code>` : ""),
    `ATR 1h ${i.atr_1h_ticks ?? "—"} ticks · ATR 4h ${i.atr_4h_ticks ?? "—"} · ADR(20) ${i.adr20_ticks ?? "—"}`,
    `Last bar: effort <b>${l.effort ?? "n/a"}×</b> (${BAND_WORD[l.effort_band]}) · spread ${l.spread_atr ?? "—"}× ATR · close at ${l.close_pos == null ? "—" : Math.round(l.close_pos * 100) + "%"} of its range`,
  ];
  if (i.mt5 && i.mt5.spot != null) lines.push(`MT5 ${i.mt5.symbol} ≈ <code>${Number(i.mt5.spot).toFixed(spotDigits(i.mt5.symbol))}</code> (${i.mt5.direction === "inverted" ? "inverted vs the future" : "same direction"})`);
  if (i.context_only) lines.push("<i>Dollar index: price context only, no exchange volume.</i>");
  if (i.roll && i.roll.in_roll) lines.push(`🔁 Roll window${i.roll.last_trade ? " (front month stops " + i.roll.last_trade + ")" : ""}: check prices on TradingView.`);
  if ((i.adr_used_pct ?? 0) >= 75) lines.push("⏱ 75%+ of the average day is used: breakout entries count as late.");
  return lines.join("\n") + ageWarning(snap) + footer(snap);
}

export function all(snap) {
  const rows = Object.entries(snap.instruments).filter(([, i]) => i.available).map(([k, i]) => {
    const e = i.last.effort == null ? " n/a" : (i.last.effort.toFixed(2) + i.last.effort_band).padStart(5);
    return `${k.padEnd(3)} ${px(i.price, i.tick).padStart(10)} ${sgn(i.day.change_pct, 2).padStart(6)}% ADR${String(i.adr_used_pct ?? "—").padStart(4)}% eff${e}`;
  });
  return `<b>All instruments</b>\n<pre>${esc(rows.join("\n"))}</pre>\n` + "day change · share of average daily range used · last bar effort" + ageWarning(snap) + footer(snap);
}

export function levels(snap, key) {
  const i = snap.instruments[key];
  if (!i || !i.available) return `No data for ${key} yet.`;
  const t = i.tick, rows = [];
  const add = (name, v) => v != null && rows.push([name, v]);
  add("Week high", i.week?.high); add("Prev week high", i.prev_week?.high); add("Prev day high", i.prev_day?.high);
  add("Day high", i.day.high); add("Day low", i.day.low); add("Prev day low", i.prev_day?.low);
  add("Prev week low", i.prev_week?.low); add("Week low", i.week?.low); add("Prev day close", i.prev_day?.close);
  for (const [s, v] of Object.entries(i.sessions || {})) { add(`${s} high`, v.high); add(`${s} low`, v.low); }
  rows.sort((a, b) => b[1] - a[1]);
  const p = i.price;
  const body = rows.map(([n, v]) => {
    const dist = Math.round((v - p) / t);
    return `${px(v, t).padStart(10)}  ${n.padEnd(15)} ${dist === 0 ? "at price" : sgn(dist) + " ticks"}`;
  });
  const priceLine = `${px(p, t).padStart(10)}  ◀ last price`;
  const idx = rows.findIndex(([, v]) => v < p);
  body.splice(idx < 0 ? body.length : idx, 0, priceLine);
  return `<b>${key} levels</b> (sorted high → low)\n<pre>${esc(body.join("\n"))}</pre>` + footer(snap);
}

export function effortReport(snap, key) {
  const i = snap.instruments[key];
  if (!i || !i.available) return `No data for ${key} yet.`;
  const rows = i.recent.map((b) => {
    const dir = b.close >= b.open ? "▲" : "▼";
    const eff = b.effort == null ? "  n/a " : `${b.effort.toFixed(2)}${b.effort_band}`.padStart(6);
    return `${b.local.padEnd(9)} ${dir} eff${eff} spr ${b.spread_atr == null ? " —  " : b.spread_atr.toFixed(2)} cls ${b.close_pos == null ? "—" : String(Math.round(b.close_pos * 100)).padStart(3) + "%"}`;
  });
  const note = i.context_only ? "\n<i>DX has no exchange volume: effort is not meaningful here.</i>" : "";
  return `<b>${key} effort, newest first</b>\n<pre>${esc(rows.join("\n"))}</pre>\n` +
    "effort = volume ÷ median of the same Chicago hour (20 sessions) · spr = true range ÷ ATR · cls = close position in the bar (0% low, 100% high)" +
    note + footer(snap);
}

export function strengthReport(snap, win = "1d") {
  const s = snap.strength?.[win];
  if (!s) return "No strength data yet.";
  const rows = s.map((c, n) => `${n + 1}. ${c.ccy}  ${sgn(c.strength_pct, 2).padStart(6)}%   vs USD ${sgn(c.vs_usd_pct, 2).padStart(6)}%`);
  const label = { "1h": "last hour", "1d": "today (CME day)", "1w": "this week" }[win];
  return `<b>Currency strength · ${label}</b>\n<pre>${esc(rows.join("\n"))}</pre>\n` +
    `Strongest vs weakest: <b>${s[0].ccy}/${s[s.length - 1].ccy}</b>. strength = average move against the other 7.` + footer(snap);
}

export function toMt5(i, level) {
  const m = i.mt5;
  if (!m || m.spot == null) return null;
  const fut = i.price;
  return m.direction === "inverted" ? (m.spot * fut) / level : (m.spot * level) / fut;
}

export function convert(snap, key, level) {
  const i = snap.instruments[key];
  if (!i || !i.mt5) return `Conversion works for 6E 6B 6A 6N 6J 6C 6S GC.`;
  if (!isFinite(level)) return `Usage: /convert ${key} <futures price>`;
  if (i.mt5.spot == null) return `No spot ${i.mt5.symbol} feed in the robot. Read the futures–spot gap off your TradingView chart.`;
  const v = toMt5(i, level);
  return `<b>${key} ${px(level, i.tick)} → ${i.mt5.symbol} <code>${v.toFixed(spotDigits(i.mt5.symbol))}</code></b>\n` +
    `Using future ${px(i.price, i.tick)} and spot ${Number(i.mt5.spot).toFixed(spotDigits(i.mt5.symbol))} (${i.mt5.direction}).` + footer(snap);
}

export function size(snap, key, entry, stop, riskArg, balance, env = {}) {
  const i = snap.instruments[key];
  if (!i || !TICK_VALUE[key]) return "Sizing works for 6E 6B 6A 6N 6J 6C 6S GC.";
  if (!isFinite(entry) || !isFinite(stop) || entry === stop) return "Usage: /size 6E <entry> <stop> <A|B|C or risk%> <balance>";
  const grade = String(riskArg || "B").toUpperCase().replace("%", "");
  const riskPct = RISK_GRADE[grade] ?? parseFloat(grade);
  const bal = balance || parseFloat(env.ACCOUNT_BALANCE || "");
  if (!isFinite(riskPct) || riskPct <= 0) return "Risk must be A, B, C or a percentage like 0.25.";
  if (!isFinite(bal) || bal <= 0) return "Add your balance at the end, e.g. /size 6E 1.0909 1.0889 B 100000";
  const riskUsd = (bal * riskPct) / 100;
  const t = i.tick;
  const ticks = Math.round((Math.abs(entry - stop) / t) * 1e6) / 1e6;   // kill float noise (39.9999 → 40)
  const contracts = Math.floor(riskUsd / (ticks * TICK_VALUE[key]));
  const micros = HAS_MICRO[key] ? Math.floor((10 * riskUsd) / (ticks * TICK_VALUE[key])) : null;
  const out = [`<b>${key} ${entry > stop ? "long" : "short"} · risk ${riskPct}% of ${bal.toLocaleString("en-US")} = $${riskUsd.toFixed(0)}</b>`,
    `Stop distance ${ticks.toFixed(1)} ticks on the future.`];
  const m = i.mt5;
  if (key === "GC") {
    const cs = parseFloat(env.XAU_CONTRACT || "100");
    const lots = Math.floor((riskUsd / (Math.abs(entry - stop) * cs)) * 100) / 100;
    out.push(`MT5 XAUUSD: <b>${lots.toFixed(2)} lots</b> (stop $${Math.abs(entry - stop).toFixed(2)}, contract ${cs} oz; set XAU_CONTRACT if your broker differs)`);
  } else if (m && m.spot != null) {
    const e5 = toMt5(i, entry), s5 = toMt5(i, stop);
    const pips = Math.abs(e5 - s5) / m.pip;
    const pipVal = m.symbol.startsWith("USD") ? (m.symbol === "USDJPY" ? 1000 / m.spot : 10 / m.spot) : 10;
    const lots = Math.floor((riskUsd / (pips * pipVal)) * 100) / 100;
    const inv = m.direction === "inverted";
    out.push(`MT5 ${m.symbol} ${inv ? (entry > stop ? "SELL" : "BUY") : entry > stop ? "BUY" : "SELL"}: entry <code>${e5.toFixed(spotDigits(m.symbol))}</code> stop <code>${s5.toFixed(spotDigits(m.symbol))}</code> · ${pips.toFixed(1)} pips · $${pipVal.toFixed(3)}/pip/lot → <b>${lots.toFixed(2)} lots</b>`);
  }
  out.push(`Futures: ${contracts} × ${key}` + (micros != null ? ` · ${micros} × ${HAS_MICRO[key]}` : ""));
  if (riskPct > 0.5) out.push("⚠️ Above your grade A risk (0.5%).");
  out.push("<i>Lots rounded down to 0.01. Check against your MT5 symbol specification.</i>");
  return out.join("\n") + footer(snap);
}

export function health(snap) {
  const p = snap.health?.problems || [];
  const syms = Object.entries(snap.health?.symbols || {}).map(([k, v]) => `${k.padEnd(7)} ${v.status.padEnd(11)} ${v.last_local || "—"}`);
  return `<b>Robot data health</b> — ${p.length ? "⚠️ " + p.length + " problem(s)" : "🟢 all good"}\n` +
    (p.length ? p.map((x) => "• " + esc(x)).join("\n") + "\n" : "") +
    `<pre>${esc(syms.join("\n"))}</pre>` + ageWarning(snap) + footer(snap);
}

// ---------- Stage 3: the robot's own Wyckoff read (data/scan.json) ----------
const MAJ = { 1: "↑ major up", "-1": "↓ major down", 0: "↔ sideways" };
const p5 = (v) => (v == null ? "—" : Number(v).toPrecision(5));
export function scanReport(sc) {
  if (!sc) return "No scan yet — the robot writes it every hour.";
  const rows = [];
  for (const [key, r] of Object.entries(sc)) {
    if (key.startsWith("__")) continue;
    const m = r.major || {};
    let line = `${key.padEnd(6)} ${MAJ[m.state] || "—"}`;
    if (m.state === 0 && m.box) line += ` ${p5(m.box[0])}–${p5(m.box[1])}`;
    else if (m.protected != null) line += ` holds ${m.state === 1 ? ">" : "<"} ${p5(m.protected)}`;
    const g = (r.ranges || []).slice(-1)[0];
    if (g) line += `\n       range ${g.id.slice(5, 10)}: phase ${g.phase} · ${g.name}`;
    rows.push(line);
  }
  return "<b>Wyckoff scan</b> (robot, Yahoo data)\n<pre>" + esc(rows.join("\n")) + "</pre>\n<i>/story 6E 4h for the full read.</i>";
}
export function storyReport(sc, key, tf) {
  const r = sc && sc[`${key} ${tf}`];
  if (!r) return `No scan for ${key} ${tf}.`;
  const g = (r.ranges || []).slice(-1)[0];
  if (!g) return `${key} ${tf}: no live range right now (${MAJ[(r.major || {}).state] || "—"}).`;
  return `<b>${key} ${tf} · range from ${esc(g.id.slice(0, 16).replace("T", " "))} UTC (${g.kind})</b>\n` +
    `${p5(g.support)} – ${p5(g.resistance)} · phase ${g.phase}\n\n` +
    g.story.map((x) => esc(x)).join("\n") + `\n\n<b>BIAS</b> ${esc(g.bias)}`;
}
export function panelReport(pn) {
  if (!pn) return "No Market Panel yet — the robot writes it every hour.";
  const o = ["<b>Market Panel</b>"];
  const top = (pn.top3 || []).map((x, i) => `${i + 1}. ${x.key} — ${x.why.join(", ")}`);
  o.push(top.length ? "Top 3:\n" + top.join("\n") : "Top 3: nothing stands out.");
  const pers = Object.entries(pn.persistence || {}).map(([c, t]) => `${c} ${t}`);
  if (pers.length) o.push("Strength: " + pers.join(" · "));
  for (const [k, lab] of [["dx_high", "DX new major high"], ["dx_low", "DX new major low"]]) {
    const s = (pn.sweet_spot || {})[k];
    if (s && s.new_extreme) {
      const hits = Object.entries(s.read || {}).filter(([, v]) => v.dir).map(([c, v]) => `${c} ${v.text.split(" — ")[0]}`);
      o.push(`${lab} (${s.at.slice(5, 10)}): ` + (hits.length ? hits.join(" · ") : "no currency held up"));
    }
  }
  const b = (pn.themes || {}).blocs || [];
  if (b.length) o.push("Themes (one idea each): " + b.map((x) => x.join("/")).join(" · "));
  const late = Object.entries(pn.late || {}).filter(([, v]) => v).map(([k]) => k);
  if (late.length) o.push("Late (≥75% ADR): " + late.join(" "));
  return o.map(esc).join("\n").replace("&lt;b&gt;Market Panel&lt;/b&gt;", "<b>Market Panel</b>");
}

export const wantsScan = (text) => /^\/(scan|story|brief|panel|news)\b|\b(scan|story|wyckoff|range|ranges|brief|morning|panel|themes|sweet|news|calendar)\b/i.test(text || "");


// ---------- Robot v2: answers built by the robot (data/decisions.json + card images in KV) ----------
const tfOf = (a, dflt = "4h") => { const m = String(a || "").toLowerCase().match(/^(4h|1h|15m|5m)$/); return m ? m[1] : dflt; };
export function todayReply(dec) {
  if (!dec || !dec._today) return "No board yet — the robot draws it every hour.";
  return { photo: "board", caption: dec._today };
}
export function chartReply(dec, k, tf) {
  if (k === "DX") return "DX is read as context only — see /today.";
  const d = dec && dec[`${k} ${tf}`];
  if (!d) return `No card for ${k} ${tf} yet — the robot draws them every hour.`;
  return { photo: `card:${k}:${tf}`, caption: d.caption };
}
export function zoomReply(dec, k, tf) {
  if (k === "DX") return "DX is read as context only — see /dollar.";
  const d = dec && dec[`${k} ${tf}`];
  if (!d) return `No ${tf} read for ${k} yet — the robot reads 15m and 5m every hour.`;
  return { photo: `card:${k}:${tf}`, caption: d.caption };
}
export function whyReply(dec, k, tf) {
  const d = dec && dec[`${k} ${tf}`];
  return d && d.why ? d.why : `No story for ${k} ${tf} yet.`;
}
const textOf = (dec, key, empty) => (dec && dec[key] ? dec[key] : empty);
export function vsaReply(dec, k, tf) {
  const d = dec && dec[`${k} ${tf}`];
  return d ? d.vsa : `No VSA read for ${k} ${tf} yet.`;
}
const CONFLICT = { 1: ["TREND ↓", "SELL ZONE", "SELL RALLIES"], "-1": ["TREND ↑", "BUY ZONE", "BUY PULLBACKS"] };
// /plan 6E long 1.0909 1.0889 [A|B|C|0.3] [balance] — one answer: GO / REDUCE / NO, the MT5 order and the lots
export function planText(snap, st, args, env = {}, dec = null, panel = null, now = Date.now()) {
  const [k, dir, e, s, g0, b0] = args;
  const i = snap.instruments[k];
  const side = /^(long|buy|l)$/i.test(dir || "") ? 1 : /^(short|sell|s)$/i.test(dir || "") ? -1 : 0;
  const entry = parseFloat(e), stop = parseFloat(s);
  if (!i || !TICK_VALUE[k] || !side || !isFinite(entry) || !isFinite(stop) || (entry - stop) * side <= 0)
    return "Usage: /plan 6E long 1.0909 1.0889 B — future prices; stop below a long's entry, above a short's. Add the balance at the end if it isn't set.";
  const grade = String(g0 || "B").toUpperCase();
  const risk = RISK[grade] ?? parseFloat(grade);
  const bal = parseFloat(b0) || parseFloat(env.ACCOUNT_BALANCE || "") || 0;
  const ctx = panel ? { rho: panel.themes?.rho, blocs: panel.themes?.blocs, key: k, side } : {};
  const gd = st ? guard(st, risk, now, ctx) : { state: "OK TO EXECUTE", why: [], factor: 1 };
  const acts = ["4h", "1h"].map((tf) => [tf, dec && dec[`${k} ${tf}`] ? dec[`${k} ${tf}`].action : "—"]);
  const against = acts.filter(([, a]) => CONFLICT[side].includes(a)).map(([tf, a]) => `${tf} says ${a}`);
  const blackout = acts.some(([, a]) => a === "NO NEW ENTRIES");
  let verdict = "GO", icon = "🟢";
  const why = [];
  if (gd.state === "BLOCKED") { verdict = "NO"; icon = "⛔"; why.push(...gd.why); }
  else if (blackout) { verdict = "NO"; icon = "⛔"; why.push("news blackout on this currency"); }
  else {
    if (gd.state === "REDUCE") { verdict = "REDUCE"; icon = "🟠"; why.push(...gd.why); }
    if (against.length) { verdict = "REDUCE"; icon = "🟠"; why.push("the chart is against you: " + against.join(", ")); }
  }
  const riskNow = risk * (gd.factor || 1) * (verdict === "REDUCE" && against.length ? 0.5 : 1);
  const R = Math.abs(entry - stop);
  const ticks = Math.round((R / i.tick) * 1e6) / 1e6;
  const usd = bal ? (bal * riskNow) / 100 : null;
  const out = [`<b>PLAN · ${k} ${side === 1 ? "LONG" : "SHORT"}</b>  ▸ ${icon} <b>${verdict}</b>`,
    `grade ${grade} · risk ${riskNow.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")}%` + (usd ? ` = $${Math.round(usd).toLocaleString("en-US")} on ${bal.toLocaleString("en-US")}` : ""),
    ...why.map((w) => "• " + esc(w)), ""];
  const m = i.mt5;
  if (k === "GC") {
    const cs = parseFloat(env.XAU_CONTRACT || "100");
    const lots = usd ? Math.floor((usd / (R * cs)) * 100) / 100 : null;
    out.push(`<b>MT5</b>  XAUUSD ${side === 1 ? "BUY" : "SELL"} · stop $${R.toFixed(2)} away` + (lots != null ? ` → <b>${lots.toFixed(2)} lots</b>` : ""));
  } else if (m && m.spot != null) {
    const e5 = toMt5(i, entry), s5 = toMt5(i, stop);
    const pips = Math.abs(e5 - s5) / m.pip;
    const pipVal = m.symbol.startsWith("USD") ? (m.symbol === "USDJPY" ? 1000 / m.spot : 10 / m.spot) : 10;
    const lots = usd ? Math.floor((usd / (pips * pipVal)) * 100) / 100 : null;
    const inv = m.direction === "inverted";
    out.push(`<b>MT5</b>  ${m.symbol} ${inv ? (side === 1 ? "SELL" : "BUY") : side === 1 ? "BUY" : "SELL"}  <code>${e5.toFixed(spotDigits(m.symbol))}</code> · stop <code>${s5.toFixed(spotDigits(m.symbol))}</code>` +
      (lots != null ? ` → <b>${lots.toFixed(2)} lots</b>` : "") + ` (${pips.toFixed(1)} pips)`);
  }
  out.push(`<b>Future</b>  ${px(entry, i.tick)} · stop ${px(stop, i.tick)} · ${ticks.toFixed(0)} ticks`);
  out.push(`<b>Take</b>  ⅓ at 2R ${px(entry + 2 * side * R, i.tick)} · then the Adaptive stop`);
  out.push(`<b>Charts</b>  ${acts.map(([tf, a]) => `${tf} ${esc(a)}`).join(" · ")}`);
  out.push(`<b>Guard</b>  ${esc(gd.state === "OK TO EXECUTE" ? "OK" : gd.state)}`);
  if (verdict !== "NO") out.push(`<i>/open ${k} ${side === 1 ? "long" : "short"} ${entry} ${stop} ${grade}</i>`);
  if (!bal) out.push("<i>Lots need your balance: add it at the end, e.g. … B 400000</i>");
  return out.join("\n");
}

// ---------- routing ----------
export function route(text, snap, env = {}, sc = null) {
  const raw = (text || "").trim();
  const [cmd0, ...args] = raw.split(/\s+/);
  const cmd = cmd0.toLowerCase().replace(/@.*$/, "");
  const sym = (s) => (s ? findSymbol(s) : null);
  const need = (k, f) => (k ? f(k) : "Which instrument? e.g. 6E, gold, yen.");
  const dec = sc && sc.__dec;
  switch (cmd) {
    case "/start": case "/help": return help();
    case "/today": case "/board": case "/brief": case "/panel": case "/scan": case "/all": case "/strength": return todayReply(dec);
    case "/chart": case "/status": case "/levels": return need(sym(args[0]), (k) => {
      const tf = tfOf(args[1]);
      return tf.endsWith("m") ? zoomReply(dec, k, tf) : chartReply(dec, k, tf);
    });
    case "/zoom": case "/ltf": return need(sym(args[0]), (k) => zoomReply(dec, k, tfOf(args[1], "15m")));
    case "/vsa": case "/effort": return need(sym(args[0]), (k) => vsaReply(dec, k, tfOf(args[1])));
    case "/why": case "/story": return need(sym(args[0]), (k) => whyReply(dec, k, tfOf(args[1])));
    case "/trend": case "/swings": case "/read": case "/ask":
      return "This one is read fresh from the market — the bot needs its GitHub token for that (see /refresh).";
    case "/setups": case "/hitlist": return textOf(dec, "_setups", "No setups list yet — the robot writes it every hour.");
    case "/dollar": case "/usd": case "/dx": return textOf(dec, "_dollar", "No dollar read yet.");
    case "/calls": case "/grade": return textOf(dec, "_calls", "No self-grade yet — the robot scores its calls every hour.");
    case "/health": return health(snap);
    case "/news": return sc && sc.__news ? sc.__news : "No calendar yet — the robot fetches it every hour.";
    case "/convert": return need(sym(args[0]), (k) => convert(snap, k, parseFloat(args[1])));
    case "/size": return need(sym(args[0]), (k) => size(snap, k, parseFloat(args[1]), parseFloat(args[2]), args[3], parseFloat(args[4]), env));
  }
  // plain words
  const t = raw.toLowerCase();
  const k = findSymbol(t);
  const tf = /\b1h\b|hour/.test(t) ? "1h" : "4h";

  if (/\b(news|calendar|cpi|nfp|release)\b/.test(t)) return sc && sc.__news ? sc.__news : "No calendar yet.";
  const tfw = /\b5m\b|5 ?min/.test(t) ? "5m" : /\b15m\b|15 ?min/.test(t) ? "15m" : tf;
  if (k && /\b(vsa|effort|volume|supply|demand)\b/.test(t)) return vsaReply(dec, k, tfw);
  if (k && /\b(why|story|wyckoff)\b/.test(t)) return whyReply(dec, k, tfw);
  if (k && /\b(zoom|ltf|15m|5m|lower|entry)\b/.test(t)) return zoomReply(dec, k, tfw.endsWith("m") ? tfw : "15m");
  if (/\b(setups?|hit ?list|opportunit\w*)\b/.test(t)) return textOf(dec, "_setups", "No setups list yet.");
  if (/\b(calls|self.?grade|track record|how (is|has) the robot)\b/.test(t)) return textOf(dec, "_calls", "No self-grade yet.");
  if (/\b(dollar|usd|dxy)\b/.test(t) && (!k || k === "DX")) return textOf(dec, "_dollar", "No dollar read yet.");
  if (/\b(health|feed|data|robot|working|broken)\b/.test(t) && !k) return health(snap);
  if (!k && /\b(today|brief|morning|board|market|overview|all|everything|dashboard|summary|scan|strength|panel)\b/.test(t)) return todayReply(dec);
  if (k) return chartReply(dec, k, tf);
  return "I didn't catch an instrument or a question.\n\n" + help();
}

async function loadSnapshot(env) {
  const url = (env.SNAPSHOT_URL || "https://raw.githubusercontent.com/HAM-X/whs-data/main/data/snapshot.json") +
    "?t=" + Math.floor(Date.now() / 120000);           // bypass stale CDN copies, 2-minute buckets
  const r = await fetch(url, { cf: { cacheTtl: 60 } });
  if (!r.ok) throw new Error(`snapshot HTTP ${r.status}`);
  return r.json();
}

async function loadScan(env) {
  const base = "https://raw.githubusercontent.com/HAM-X/whs-data/main/data/";
  const q = "?t=" + Math.floor(Date.now() / 120000);
  const [a, b, c, d, e] = await Promise.all([fetch(base + "scan.json" + q, { cf: { cacheTtl: 60 } }),
                                          fetch(base + "brief.json" + q, { cf: { cacheTtl: 60 } }),
                                          fetch(base + "panel.json" + q, { cf: { cacheTtl: 60 } }),
                                          fetch(base + "news.json" + q, { cf: { cacheTtl: 60 } }),
                                          fetch(base + "decisions.json" + q, { cf: { cacheTtl: 60 } })]);
  const sc = a.ok ? await a.json() : {};
  if (b.ok) sc.__brief = (await b.json()).text;
  if (c.ok) sc.__panel = await c.json();
  if (d.ok) sc.__news = (await d.json()).text;
  const gh = e.ok ? await e.json().catch(() => null) : null;
  const kv = env.DESK ? await env.DESK.get("dec:latest", { type: "json" }).catch(() => null) : null;
  sc.__dec = kv && (!gh || String(kv._at) >= String(gh._at)) ? kv : gh;   // /refresh writes KV first
  return sc;
}

// ---------- fresh answers: any pair, any timeframe, any analysis (robot-now → jobs/ask.py, ≈ 1 min) ----------
const PRE_TF = ["4h", "1h", "15m", "5m"];                 // what the hourly robot precomputes
const ANALYSIS = { "/chart": "chart", "/zoom": "chart", "/ltf": "chart", "/status": "chart", "/levels": "chart", "/vsa": "vsa",
  "/why": "why", "/story": "why", "/effort": "effort", "/trend": "trend", "/swings": "swings", "/read": "all", "/ask": "" };
export function tfIn(text) {
  const m = String(text).toLowerCase().match(/\b(5m|15m|30m|1h|60m|4h|1d|daily|day|1w|weekly|week)\b/);
  if (!m) return null;
  return { "60m": "1h", daily: "1d", day: "1d", weekly: "1w", week: "1w" }[m[1]] || m[1];
}
// Is this question answered from the hourly read (instant) or fresh (robot-now)? Returns null (instant) or the ask text.
export function freshAsk(text, dec, now = Date.now()) {
  const raw = String(text || "").trim();
  const [c0, ...rest] = raw.split(/\s+/);
  const cmd = c0.toLowerCase().replace(/@.*$/, "");
  const k = findSymbol(raw);
  const isCmd = cmd in ANALYSIS;
  const plain = !cmd.startsWith("/") && k && /\b(vsa|effort|result|volume|trend|structure|swings?|why|story|wyckoff|chart|zoom|ltf|now|fresh|live|\d+m|\d[hdw]|daily|weekly)\b/i.test(raw);
  if (!k || (!isCmd && !plain)) return null;
  const want = isCmd ? ANALYSIS[cmd] : "";
  const tf = tfIn(raw) || (cmd === "/zoom" || cmd === "/ltf" ? "15m" : "4h");
  const forced = /\b(now|fresh|live)\b/i.test(raw) || cmd === "/ask" || cmd === "/read";
  const words = (want ? want + " " : "") + (isCmd ? rest.join(" ") : raw);
  const ask = words.replace(/\b(now|fresh|live)\b/gi, "").trim() + (tfIn(words) ? "" : " " + tf);
  if (forced || !PRE_TF.includes(tf) || ["effort", "trend", "swings", "all"].includes(want)) return ask;
  if (!isCmd && /\b(effort|result|trend|structure|swings?)\b/i.test(raw)) return ask;
  const d = dec && dec[`${k} ${tf}`];
  const ageMin = dec && dec._at ? (now - Date.parse(dec._at)) / 60000 : 1e9;
  if (!d || ageMin > (tf.endsWith("m") ? 20 : 75)) return ask;
  return null;
}
export async function askFresh(env, ask) {
  if (!env.GH_TOKEN) return null;
  const r = await fetch("https://api.github.com/repos/HAM-X/whs-data/actions/workflows/robot-now.yml/dispatches", {
    method: "POST", body: JSON.stringify({ ref: "main", inputs: { what: "", ask } }),
    headers: { Authorization: `Bearer ${env.GH_TOKEN}`, Accept: "application/vnd.github+json", "User-Agent": "pipdon-bot",
               "content-type": "application/json" } });
  return r.ok;
}

// ---------- dead-man switch: the robot must have run in the last 100 minutes while CME FX trades ----------
export function marketOpen(ms = Date.now()) {
  const d = new Date(ms), wd = d.getUTCDay(), h = d.getUTCHours();
  if (wd === 6) return false;                       // Saturday
  if (wd === 5 && h >= 21) return false;            // Friday after the close
  if (wd === 0 && h < 23) return false;             // Sunday before the open
  return true;
}
export function deadmanCheck(lastAt, flag, now = Date.now()) {
  const age = lastAt ? (now - Date.parse(lastAt)) / 60000 : 1e9;
  if (age > 100 && marketOpen(now) && marketOpen(now - 100 * 60000)) return flag ? "still" : "down";
  if (age <= 100 && flag) return "back";
  return "ok";
}
async function deadman(env) {
  if (!env.DESK) return;
  const dec = await env.DESK.get("dec:latest", { type: "json" }).catch(() => null);
  const flag = await env.DESK.get("deadman");
  const state = deadmanCheck(dec && dec._at, flag);
  const last = dec && dec._at ? new Date(Date.parse(dec._at) + 3 * 3600e3).toISOString().slice(11, 16) : "—";
  if (state === "down") {
    let restarted = false;
    if (env.GH_TOKEN) {
      const r = await fetch("https://api.github.com/repos/HAM-X/whs-data/actions/workflows/robot-hourly.yml/dispatches", {
        method: "POST", body: JSON.stringify({ ref: "main" }),
        headers: { Authorization: `Bearer ${env.GH_TOKEN}`, Accept: "application/vnd.github+json", "User-Agent": "pipdon-bot",
                   "content-type": "application/json" } }).catch(() => null);
      restarted = !!(r && r.ok);
    }
    await env.DESK.put("deadman", String(Date.now()));
    await reply(env, env.TELEGRAM_CHAT_ID, `⚠️ <b>The robot has not run since ${last} UTC+3.</b> ` +
      (restarted ? "I restarted it — a fresh board should follow within ~10 minutes." : "Could not restart it — trust TradingView until it is back."));
  } else if (state === "back") {
    await env.DESK.delete("deadman");
    await reply(env, env.TELEGRAM_CHAT_ID, `✅ The robot is back (last run ${last} UTC+3).`);
  }
}

// /refresh — start the robot-now workflow (needs the GH_TOKEN secret: a fine-grained token with Actions read & write)
export async function refresh(env, args = []) {
  if (!env.GH_TOKEN) return "⚠️ /refresh needs a GitHub token on the bot (secret GH_DISPATCH_TOKEN). Until then /today shows the latest hourly board.";
  const k = args[0] ? findSymbol(args[0]) : null;
  const what = k && k !== "DX" ? `${k} ${tfOf(args[1])}` : "";
  const r = await fetch("https://api.github.com/repos/HAM-X/whs-data/actions/workflows/robot-now.yml/dispatches", {
    method: "POST", body: JSON.stringify({ ref: "main", inputs: { what } }),
    headers: { Authorization: `Bearer ${env.GH_TOKEN}`, Accept: "application/vnd.github+json", "User-Agent": "pipdon-bot",
               "content-type": "application/json" } });
  return r.ok ? `⏳ Fresh read started${what ? " for " + what : ""} — ${what ? "the card" : "the board + Today"} arrives in about 2 minutes.`
              : `⚠️ Could not start the fresh read (GitHub ${r.status}). /today shows the latest hourly board.`;
}

async function replyPhoto(env, chatId, name, caption) {
  let img = null;
  if (env.DESK) {
    const bundle = await env.DESK.get("img:bundle", { type: "json" }).catch(() => null);
    if (bundle && bundle[name]) img = Uint8Array.from(atob(bundle[name]), (ch) => ch.charCodeAt(0));
    else img = await env.DESK.get("img:" + name, { type: "arrayBuffer" });
  }
  if (!img) return reply(env, chatId, caption);
  const fd = new FormData();
  fd.append("chat_id", String(chatId));
  fd.append("caption", caption.slice(0, 1024));
  fd.append("parse_mode", "HTML");
  fd.append("photo", new Blob([img], { type: "image/png" }), name.replace(/:/g, "_") + ".png");
  const r = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_TOKEN}/sendPhoto`, { method: "POST", body: fd });
  if (!r.ok) await reply(env, chatId, caption);
}

async function answer(env, chatId, out) {
  if (out && typeof out === "object" && out.photo) return replyPhoto(env, chatId, out.photo, out.caption);
  return reply(env, chatId, String(out));
}

async function reply(env, chatId, text) {
  const body = { chat_id: chatId, text: text.slice(0, 4000), parse_mode: "HTML", disable_web_page_preview: true };
  await fetch(`https://api.telegram.org/bot${env.TELEGRAM_TOKEN}/sendMessage`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
}

// TradingView alert → Telegram (WHS Range / Engine alerts). The webhook URL carries a private key: /tv/<TV_KEY>
export function tvMessage(body) {
  let t = String(body || "").trim();
  try {
    const j = JSON.parse(t);
    if (j && typeof j === "object") t = j.text || j.message || JSON.stringify(j);
  } catch (_) { /* plain text alert */ }
  t = t.slice(0, 1500);
  const flag = /left the range/.test(t) ? "⚑" : /still going on/.test(t) ? "↺" : /confirmed/.test(t) ? "●" : "📐";
  return `${flag} <b>TradingView</b> · ${esc(t)}`;
}

export default {
  // hourly, after the robot's hh:12 run has committed fresh bars: manage open trades
  async scheduled(event, env, ctx) {
    ctx.waitUntil((async () => {
      try { await deadman(env); } catch (e) { /* never let the watchdog break trade management */ }
      const st = await loadDesk(env);
      if (!st || !st.trades.length) return;
      const lines = manage(st, await loadSnapshot(env));
      await saveDesk(env, st);
      if (lines.length) await reply(env, env.TELEGRAM_CHAT_ID, "<b>Trade Desk</b>\n" + lines.map(esc).join("\n"));
    })());
  },
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    if (request.method === "POST" && path.startsWith("/tv/")) {
      if (!env.TV_KEY || path.slice(4) !== env.TV_KEY) return new Response("forbidden", { status: 403 });
      await reply(env, env.TELEGRAM_CHAT_ID, tvMessage(await request.text()));
      return new Response("ok");
    }
    if (request.method !== "POST") {
      const b = env.DESK ? await env.DESK.get("img:bundle", { type: "json" }).catch(() => null) : null;
      const n = b ? Object.keys(b).length : 0;
      return new Response(`Pipdon BOT is running. Chart cards stored: ${n}.`, { status: 200 });
    }
    if (request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.WEBHOOK_SECRET) return new Response("forbidden", { status: 403 });
    const update = await request.json().catch(() => ({}));
    const msg = update.message || update.edited_message;
    if (!msg || !msg.text) return new Response("ok");
    if (String(msg.chat.id) !== String(env.TELEGRAM_CHAT_ID)) return new Response("ok");   // only Hamza
    let text;
    const [c0, ...a0] = msg.text.trim().split(/\s+/);
    const dcmd = c0.toLowerCase().replace(/@.*$/, "");
    if (dcmd === "/refresh" || /^(refresh|update now|fresh read)\b/i.test(msg.text.trim())) {
      await reply(env, msg.chat.id, await refresh(env, a0));
      return new Response("ok");
    }
    if (DESK_CMDS.includes(dcmd)) {
      try {
        const snap = await loadSnapshot(env);
        const st = await loadDesk(env);
        const args = a0.slice();
        if (args[0]) { const k = findSymbol(args[0]); if (k) args[0] = k; }
        const pr = await fetch("https://raw.githubusercontent.com/HAM-X/whs-data/main/data/panel.json?t=" + Math.floor(Date.now() / 120000),
                               { cf: { cacheTtl: 60 } }).catch(() => null);
        const panel = pr && pr.ok ? await pr.json().catch(() => null) : null;
        if (dcmd === "/plan") {
          const sc = await loadScan(env).catch(() => null);
          text = planText(snap, st, args, env, sc && sc.__dec, panel);
        } else text = deskRoute(dcmd, args, st, snap, Date.now(), panel);
        if (st) await saveDesk(env, st);
      } catch (e) {
        text = `⚠️ Trade Desk error: ${esc(e.message)}`;
      }
      await reply(env, msg.chat.id, text);
      return new Response("ok");
    }
    try {
      const sc = wantsScan(msg.text) ? await loadScan(env).catch(() => null) : null;
      const ask = freshAsk(msg.text, sc && sc.__dec);
      if (ask) {
        const ok = await askFresh(env, ask);
        if (ok) { await reply(env, msg.chat.id, `⚡ Reading <b>${esc(ask)}</b> fresh from the market — about a minute.`); return new Response("ok"); }
        if (ok === false) text = "⚠️ Could not start the fresh read — showing the hourly one.\n\n";
      }
      const out = route(msg.text, await loadSnapshot(env), env, sc);
      text = text && typeof out === "string" ? text + out : out;
    } catch (e) {
      text = `⚠️ Could not read the robot's snapshot (${esc(e.message)}). Try again in a minute.`;
    }
    await answer(env, msg.chat.id, text);
    return new Response("ok");
  },
};
