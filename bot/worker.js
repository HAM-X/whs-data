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
    "<b>Pipdon BOT</b> — ask in commands or plain words (\"how's gold\", \"euro levels\").",
    "",
    "/status 6E — price, day change, ranges, ATR, ADR used, last bar effort, MT5 price",
    "/all — every instrument on one screen",
    "/levels GC — previous day/week highs and lows, session ranges",
    "/effort 6J — last 6 bars: effort vs the same hour, spread, close position",
    "/strength — currency strength ranking (add 1h, 1d or 1w)",
    "/convert 6J 0.0067100 — futures level → MT5 price",
    "/size 6E 1.0909 1.0889 B 100000 — lots for a trade (risk as A/B/C or %, then balance)",
    "/scan — every chart: major swing and live Wyckoff ranges (read by the robot, no clicks)",
    "/story 6E 4h — the Wyckoff Story of the latest range on that chart (1h or 4h)",
    "/health — is the robot's data fresh?",
    "",
    "Symbols: 6E 6B 6A 6N 6J 6C 6S GC DX (or euro, pound, aussie, kiwi, yen, cad, franc, gold, dollar).",
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
export const wantsScan = (text) => /^\/(scan|story)\b|\b(scan|story|wyckoff|range|ranges)\b/i.test(text || "");

// ---------- routing ----------
export function route(text, snap, env = {}, sc = null) {
  const raw = (text || "").trim();
  const [cmd0, ...args] = raw.split(/\s+/);
  const cmd = cmd0.toLowerCase().replace(/@.*$/, "");
  const sym = (s) => (s ? findSymbol(s) : null);
  const need = (k, f) => (k ? f(k) : "Which instrument? e.g. 6E, gold, yen.");
  switch (cmd) {
    case "/start": case "/help": return help();
    case "/status": return need(sym(args[0]), (k) => status(snap, k));
    case "/all": return all(snap);
    case "/levels": return need(sym(args[0]), (k) => levels(snap, k));
    case "/effort": return need(sym(args[0]), (k) => effortReport(snap, k));
    case "/strength": return strengthReport(snap, ["1h", "1d", "1w"].includes(args[0]) ? args[0] : "1d");
    case "/health": return health(snap);
    case "/scan": return scanReport(sc);
    case "/story": return need(sym(args[0]), (k) => storyReport(sc, k, args[1] === "1h" ? "1h" : "4h"));
    case "/convert": return need(sym(args[0]), (k) => convert(snap, k, parseFloat(args[1])));
    case "/size": return need(sym(args[0]), (k) => size(snap, k, parseFloat(args[1]), parseFloat(args[2]), args[3], parseFloat(args[4]), env));
  }
  // plain words
  const t = raw.toLowerCase();
  const k = findSymbol(t);
  if (/\b(strength|strong|strongest|weak|weakest|rank|ranking)\b/.test(t)) {
    const win = /\bweek|weekly|1w\b/.test(t) ? "1w" : /\bhour|1h\b/.test(t) ? "1h" : "1d";
    return strengthReport(snap, win);
  }
  if (/\b(story|wyckoff)\b/.test(t) && k) return storyReport(sc, k, /\b1h\b/.test(t) ? "1h" : "4h");
  if (/\b(scan|ranges)\b/.test(t) && !k) return scanReport(sc);
  if (/\b(health|feed|data|robot|working|broken)\b/.test(t) && !k) return health(snap);
  if (/\b(all|overview|market|everything|dashboard|summary)\b/.test(t) && !k) return all(snap);
  if (k && /\b(effort|volume|vol|supply|demand)\b/.test(t)) return effortReport(snap, k);
  if (k && /\b(level|levels|high|highs|low|lows|support|resistance|pdh|pdl)\b/.test(t)) return levels(snap, k);
  if (k) return status(snap, k);
  return "I didn't catch an instrument or a question I know.\n\n" + help();
}

async function loadSnapshot(env) {
  const url = (env.SNAPSHOT_URL || "https://raw.githubusercontent.com/HAM-X/whs-data/main/data/snapshot.json") +
    "?t=" + Math.floor(Date.now() / 120000);           // bypass stale CDN copies, 2-minute buckets
  const r = await fetch(url, { cf: { cacheTtl: 60 } });
  if (!r.ok) throw new Error(`snapshot HTTP ${r.status}`);
  return r.json();
}

async function loadScan(env) {
  const url = "https://raw.githubusercontent.com/HAM-X/whs-data/main/data/scan.json?t=" + Math.floor(Date.now() / 120000);
  const r = await fetch(url, { cf: { cacheTtl: 60 } });
  return r.ok ? r.json() : null;
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
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    if (request.method === "POST" && path.startsWith("/tv/")) {
      if (!env.TV_KEY || path.slice(4) !== env.TV_KEY) return new Response("forbidden", { status: 403 });
      await reply(env, env.TELEGRAM_CHAT_ID, tvMessage(await request.text()));
      return new Response("ok");
    }
    if (request.method !== "POST") return new Response("Pipdon BOT is running.", { status: 200 });
    if (request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.WEBHOOK_SECRET) return new Response("forbidden", { status: 403 });
    const update = await request.json().catch(() => ({}));
    const msg = update.message || update.edited_message;
    if (!msg || !msg.text) return new Response("ok");
    if (String(msg.chat.id) !== String(env.TELEGRAM_CHAT_ID)) return new Response("ok");   // only Hamza
    let text;
    try {
      const sc = wantsScan(msg.text) ? await loadScan(env).catch(() => null) : null;
      text = route(msg.text, await loadSnapshot(env), env, sc);
    } catch (e) {
      text = `⚠️ Could not read the robot's snapshot (${esc(e.message)}). Try again in a minute.`;
    }
    await reply(env, msg.chat.id, text);
    return new Response("ok");
  },
};
