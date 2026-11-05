// Pipdon BOT · the self-scoring journal (R1.4, 5 Nov 2026).
// Every trade opened with /open is tagged with the WHS R1 reading at entry (the robot's snapshot r1 block: the last
// closed 1h bar + the 4h commitment). /score groups the closed trades by each tag and asks one question per group:
// does it pay more than chance?  Chance = 0R: random entries under ANY fixed management rule average 0R before costs
// (optional stopping on a fair price path). A group "pays" only when its 95 % range sits above 0R AND both halves of
// its trades (older / newer) are positive — the same in-sample / hold-out test the robot's replay uses.

export const MIN_N = 10;

// UTC+3 helpers (Hamza's clock) and the CME session of a moment (Chicago clock)
const ctHour = (ms) => {
  const s = new Date(ms).toLocaleString("en-US", { timeZone: "America/Chicago", hour: "numeric", hour12: false });
  return parseInt(s, 10) % 24;
};
export function sessionOf(ms) {
  const h = ctHour(ms);
  return h >= 17 || h < 2 ? "Asia" : h < 7 ? "London" : h < 16 ? "New York" : "late";
}

const band = (x, hi = 1.1, lo = 0.9) => (x == null || !isFinite(x) ? null : x >= hi ? "more" : x <= lo ? "less" : "even");

// the tags of a new trade, from the snapshot's R1 reading of its instrument
export function entryTags(snap, key, side, now = Date.now(), setup = null, grade = null) {
  const r = snap?.instruments?.[key]?.r1 || null;
  const tags = { setup: setup ? String(setup).toLowerCase().slice(0, 24) : null, grade, key, session: sessionOf(now) };
  if (!r) return { ...tags, r1: false };
  const age = (now - Date.parse(r.bar)) / 3600e3;        // hours since the bar OPENED (a 1h bar is complete at 1)
  const rel = (s) => (s === side ? "with" : s === -side ? "against" : "none");
  const it = r.intent * side;                            // intention in the trade's direction
  tags.r1 = true;
  tags.bar = r.bar;
  tags.stale = age > 3;
  tags.with1h = rel(r.state);
  tags.with4h = r.parent ? rel(r.parent.state) : "none";
  tags.eff = r.band || "n/a";
  tags.effX = r.effort;
  tags.intent = it === 2 ? "yes" : it === 1 ? "failed" : it < 0 ? "against" : "inside";
  tags.step = r.state === side ? (r.step >= 3 ? "3+" : String(Math.max(1, r.step))) : "none";
  tags.wave = r.wave && r.wave.dE != null ? band(r.wave.dE) : "first";
  tags.waveLight = r.wave?.light || "";
  tags.slAtr = r.sl_atr;
  return tags;
}

// one line for the /open reply: what R1 said at entry
export function tagsLine(tg, side) {
  if (!tg || !tg.r1) return "R1 at entry: not available (the robot's snapshot has no reading yet) — tagged by setup and session only.";
  const arrow = (w) => (w === "with" ? "with you" : w === "against" ? "against you" : "none");
  const it = { yes: "tried your way → Yes", failed: "tried your way → No", against: "tried against you", inside: "inside bar" }[tg.intent];
  const t = new Date(Date.parse(tg.bar) + 3 * 3600e3).toISOString().slice(11, 16);
  return `🏷 <b>R1 at entry</b> (${t} bar): 1h commitment ${arrow(tg.with1h)}${tg.step !== "none" ? ` · SBar ${tg.step}` : ""} · 4h ${arrow(tg.with4h)} · ` +
    `effort ${tg.effX != null ? tg.effX.toFixed(2) + "× " : ""}${tg.eff} · ${it} · wave effort ${tg.wave === "first" ? "first wave" : tg.wave}` +
    `${tg.setup ? ` · setup “${tg.setup}”` : ""} · ${tg.session}${tg.stale ? " · ⚠️ reading is old (market was closed?)" : ""}`;
}

// R of a closed trade: ⅓ taken at 2R counts (the plan), the rest at the exit
export const blendedR = (third, rExit) => (third ? (2 + 2 * rExit) / 3 : rExit);

// deterministic bootstrap of the mean (same answer every time for the same trades)
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function stats(rs) {
  const n = rs.length;
  if (!n) return null;
  const mean = rs.reduce((a, x) => a + x, 0) / n;
  const win = rs.filter((x) => x > 0).length / n;
  let lo = null, hi = null, verdict = "few";
  if (n >= MIN_N) {
    const rand = rng(1000 + n), ms = [];
    for (let b = 0; b < 2000; b++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += rs[Math.floor(rand() * n)];
      ms.push(s / n);
    }
    ms.sort((a, b) => a - b);
    lo = ms[Math.floor(0.025 * ms.length)];
    hi = ms[Math.floor(0.975 * ms.length) - 1];
    const h = Math.floor(n / 2);
    const m1 = rs.slice(0, h).reduce((a, x) => a + x, 0) / h, m2 = rs.slice(h).reduce((a, x) => a + x, 0) / (n - h);
    verdict = lo > 0 && m1 > 0 && m2 > 0 ? "pays" : hi < 0 && m1 < 0 && m2 < 0 ? "costs" : "open";
  }
  return { n, mean, win, lo, hi, verdict };
}

export const DIMS = [
  ["setup", "Your setup", (j) => j.tags?.setup || "untagged"],
  ["with1h", "With the 1h commitment", (j) => j.tags?.r1 ? j.tags.with1h : null],
  ["with4h", "With the 4h commitment", (j) => j.tags?.r1 ? j.tags.with4h : null],
  ["both", "1h and 4h together", (j) => (j.tags?.r1 ? (j.tags.with1h === "with" && j.tags.with4h === "with" ? "both with you"
    : j.tags.with1h === "with" || j.tags.with4h === "with" ? "one with you" : "neither") : null)],
  ["step", "SBar step of the 1h commitment", (j) => j.tags?.r1 ? (j.tags.step === "none" ? "not with you" : "SBar " + j.tags.step) : null],
  ["eff", "Entry bar effort", (j) => j.tags?.r1 ? j.tags.eff : null],
  ["intent", "Entry bar intention", (j) => j.tags?.r1 ? { yes: "your way → Yes", failed: "your way → No", against: "against you", inside: "inside bar" }[j.tags.intent] : null],
  ["wave", "Wave effort vs the last same-side wave", (j) => j.tags?.r1 ? (j.tags.wave === "first" ? "first wave" : j.tags.wave + " effort") : null],
  ["session", "Session", (j) => j.tags?.session || null],
  ["grade", "Grade", (j) => j.tags?.grade || null],
  ["key", "Instrument", (j) => j.key],
];
const ORDER = { "≫": 0, ">": 1, "≈": 2, "<": 3, "≪": 4, with: 0, against: 1, none: 2, "both with you": 0, "one with you": 1, neither: 2,
  "SBar 1": 0, "SBar 2": 1, "SBar 3+": 2, "not with you": 3, "more effort": 0, "even effort": 1, "less effort": 2, "first wave": 3,
  Asia: 0, London: 1, "New York": 2, late: 3, A: 0, B: 1, C: 2,
  "your way → Yes": 0, "your way → No": 1, "against you": 2, "inside bar": 3 };

const fx = (v, d = 2) => (v == null || !isFinite(v) ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d));
const MARK = { pays: "✅", costs: "⛔", open: "•", few: "○" };

function rowText(name, s) {
  const range = s.lo != null ? ` · 95% ${fx(s.lo)}…${fx(s.hi)}` : "";
  return `${MARK[s.verdict]} ${name} · ${s.n} · ${fx(s.mean)}R · ${Math.round(100 * s.win)}%${range}`;
}

// /score — the closed trades, grouped by every tag, against chance (0R)
export function scoreText(journal, title = "Score", from = null) {
  const J = (journal || []).filter((j) => !from || String(j.closed) >= from);
  if (!J.length) return `<b>${title}</b>  no closed trades yet. Every /open is tagged with the R1 reading at entry; /close or a hit stop logs it.`;
  const all = stats(J.map((j) => j.r));
  const tagged = J.filter((j) => j.tags).length;
  const out = [`<b>${title}</b>  ${rowText("all trades", all)}`,
    `<i>Chance = 0R: random entries average 0R before costs. ✅ pays · ⛔ costs = the 95% range clears 0R and the older and newer halves agree · • not proven · ○ under ${MIN_N} trades.</i>`];
  if (tagged < J.length) out.push(`<i>${J.length - tagged} older trade${J.length - tagged === 1 ? "" : "s"} before tagging count in "all" only.</i>`);
  const proven = [];
  for (const [k, label, f] of DIMS) {
    const groups = {};
    for (const j of J) {
      const g = f(j);
      if (g == null) continue;
      (groups[g] = groups[g] || []).push(j.r);
    }
    const names = Object.keys(groups);
    if (!names.length || (names.length === 1 && k !== "setup")) continue;
    names.sort((a, b) => (ORDER[a] ?? 9) - (ORDER[b] ?? 9) || groups[b].length - groups[a].length);
    out.push("", `<b>${label}</b>`);
    for (const g of names) {
      const s = stats(groups[g]);
      out.push(rowText(g, s));
      if (s.verdict === "pays" || s.verdict === "costs") proven.push(`${MARK[s.verdict]} ${label.toLowerCase()}: ${g} (${fx(s.mean)}R, n ${s.n})`);
    }
  }
  out.splice(2, 0, proven.length ? "\n<b>Proven by tag</b>\n" + proven.join("\n") : "\n<b>Proven by tag</b>: nothing yet — a row needs " + MIN_N + "+ trades, a 95% range clear of 0R and both halves agreeing.");
  out.push("", `<i>About 40 rows are tested, so 1–2 can look proven by luck: trust a ✅ that stays ✅ for two monthly reports.</i>`);
  return out.join("\n");
}

// the monthly report: last month's trades, then everything
export function monthlyText(journal, now = Date.now()) {
  const d = new Date(now + 3 * 3600e3);
  const first = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() - 1, 1)), end = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1));
  const name = first.toLocaleString("en-GB", { month: "long", timeZone: "UTC" });
  const M = (journal || []).filter((j) => String(j.closed) >= first.toISOString() && String(j.closed) < end.toISOString());
  const s = stats(M.map((j) => j.r));
  const head = s ? `📊 <b>${name}</b>: ${s.n} trade${s.n === 1 ? "" : "s"} · ${fx(s.mean)}R average · ${Math.round(100 * s.win)}% won · total ${fx(s.mean * s.n)}R`
    : `📊 <b>${name}</b>: no closed trades.`;
  return head + "\n\n" + scoreText(journal, "All trades so far");
}
export const monthKey = (now = Date.now()) => new Date(now + 3 * 3600e3).toISOString().slice(0, 7);
