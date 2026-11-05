import test from "node:test";
import assert from "node:assert/strict";
import { entryTags, stats, scoreText, blendedR, sessionOf, monthlyText, MIN_N } from "./score.js";
import { emptyState, openTrade, closeTrade, manage, deskRoute } from "./desk.js";
import { chunks } from "./worker.js";

const T0 = Date.parse("2026-11-05T12:30:00Z");            // Thu 15:30 UTC+3 = 06:30 CT → London
const r1 = (o = {}) => ({ bar: "2026-11-05T11:00:00+00:00", tf: "1h", effort: 1.62, band: "≫", eff_vs_prev: 1.3, side_vs_prev: 1.4, spread: 1.4,
  close_pct: 82, c2c: 1, intent: 2, sb: 1, rb: 0, state: 1, sl: 1.1590, step: 2, since: "2026-11-04T20:00:00+00:00", sl_atr: 1.2,
  wave: { bars: 4, effort: 3.1, travel: 0.6, dE: 0.3, dR: 0.2, cost: 1.5, cost_band: "dearer" }, last_wave: { side: -1, bars: 11, effort: 14.2, travel: 1.4, dE: 1.35, dR: 0.72, light: "red" }, parent: { tf: "4h", state: -1, sl: 1.1650, step: 1 }, ...o });
const snapOf = (o) => ({ instruments: { "6E": { key: "6E", tick: 0.00005, price: 1.1620, mt5: { pip: 0.0001, spot: 1.162 }, bars1h: null, r1: r1(o) } } });

test("sessions on the Chicago clock", () => {
  assert.equal(sessionOf(Date.parse("2026-11-05T14:00:00Z")), "New York");   // 08:00 CT (CST = UTC−6 after 1 Nov)
  assert.equal(sessionOf(Date.parse("2026-11-05T12:30:00Z")), "London");     // 06:30 CT
  assert.equal(sessionOf(Date.parse("2026-11-05T09:00:00Z")), "London");     // 03:00 CT
  assert.equal(sessionOf(Date.parse("2026-11-05T01:00:00Z")), "Asia");       // 19:00 CT
  assert.equal(sessionOf(Date.parse("2026-11-05T22:30:00Z")), "late");       // 16:30 CT
});

test("entry tags read R1 relative to the trade's side", () => {
  const L = entryTags(snapOf(), "6E", 1, T0, "Spring", "A");
  assert.equal(L.with1h, "with"); assert.equal(L.with4h, "against"); assert.equal(L.intent, "yes");
  assert.equal(L.step, "2"); assert.equal(L.wave, "your wave dearer"); assert.equal(L.eff, "≫"); assert.equal(L.setup, "spring"); assert.equal(L.stale, false);
  const S = entryTags(snapOf(), "6E", -1, T0);
  assert.equal(S.with1h, "against"); assert.equal(S.with4h, "with"); assert.equal(S.intent, "against"); assert.equal(S.step, "none");
  assert.equal(S.wave, "their wave dearer");
  assert.equal(entryTags(snapOf({ wave: { bars: 2 } }), "6E", 1, T0).wave, "young");
  assert.equal(entryTags(snapOf({ step: 0 }), "6E", 1, T0).step, "0");                 // just flipped: RBar, no SBar yet
  assert.equal(entryTags({ instruments: { "6E": {} } }, "6E", 1, T0).r1, false);
  assert.equal(entryTags(snapOf(), "6E", 1, T0 + 10 * 3600e3).stale, true);
});

test("⅓ at 2R counts in the trade's R", () => {
  assert.equal(blendedR(false, -1), -1);
  assert.ok(Math.abs(blendedR(true, 0) - 2 / 3) < 1e-12);               // ⅓ at +2R, the rest at breakeven
  assert.ok(Math.abs(blendedR(true, 3) - 8 / 3) < 1e-12);
});

test("stats: too few, then a 95% range and the two-halves check", () => {
  assert.equal(stats([1, -1, 2]).verdict, "few");
  const pay = Array.from({ length: 30 }, (_, i) => (i % 3 === 0 ? -1 : 1.5));
  const s = stats(pay);
  assert.equal(s.verdict, "pays"); assert.ok(s.lo > 0 && s.hi > s.lo);
  assert.deepEqual(stats(pay), s);                                          // deterministic
  const coin = Array.from({ length: 40 }, (_, i) => (i % 2 ? 1 : -1));
  assert.equal(stats(coin).verdict, "open");
  // a good average that only came from the older half is not "pays"
  const decay = [...Array(15).fill(2), ...Array(15).fill(-0.4)];
  assert.notEqual(stats(decay).verdict, "pays");
  assert.equal(stats(Array(20).fill(-1).map((x, i) => (i % 5 ? x : 0.5))).verdict, "costs");
});

test("/open tags the trade, /close logs R + tags, /score groups them", () => {
  const st = emptyState();
  const snap = snapOf();
  const out = openTrade(st, snap, ["6E", "long", "1.1620", "1.1600", "B", "test"], T0);
  assert.ok(out.includes("R1 at entry") && out.includes("with you") && out.includes("“test”"), out);
  assert.equal(st.trades[0].tags.setup, "test");
  // grade left out: the word is the setup, grade B
  const st2 = emptyState();
  openTrade(st2, snap, ["6E", "long", "1.1620", "1.1600", "spring"], T0);
  assert.equal(st2.trades[0].grade, "B"); assert.equal(st2.trades[0].tags.setup, "spring");
  st.trades[0].third = true; st.trades[0].best = 2.4;
  const c = closeTrade(st, snap, ["6E", "1.1620"], T0 + 5 * 3600e3);
  assert.ok(c.includes("+0.67R") && c.includes("⅓ at 2R counted"), c);
  const j = st.journal[0];
  assert.equal(j.r, 0.67); assert.equal(j.rx, 0); assert.equal(j.best, 2.4); assert.equal(j.tags.with1h, "with");
  for (let i = 0; i < 14; i++) {
    openTrade(st, snap, ["6E", i % 2 ? "short" : "long", "1.1620", i % 2 ? "1.1640" : "1.1600", "C", i % 2 ? "sos" : "test"], T0 + i * 864e5);
    closeTrade(st, snap, ["6E", i % 2 ? "1.1630" : "1.1650"], T0 + i * 864e5 + 3600e3);
  }
  const sc = deskRoute("/score", [], st, snap, T0);
  for (const h of ["Your setup", "With the 1h commitment", "With the 4h commitment", "Entry bar intention", "Proven by tag", "test", "sos"])
    assert.ok(sc.includes(h), h + "\n" + sc);
  assert.ok(sc.length < 3900 * 2);
});

test("a stop the bot manages is logged with its tags", () => {
  const st = emptyState();
  const snap = snapOf();
  openTrade(st, snap, ["6E", "long", "1.1620", "1.1600", "B", "lps"], T0);
  snap.instruments["6E"].bars1h = { t: ["2026-11-05T13:00:00.000Z"], o: [1.1615], h: [1.1618], l: [1.1595], c: [1.1598], e: [1], sb: [0], rb: [0], atr: [0.0012] };
  const out = manage(st, snap, T0 + 2 * 3600e3);
  assert.ok(out[0].includes("-1.00R"), out[0]);
  assert.equal(st.journal[0].tags.setup, "lps");
});

test("/tag names the newest trade, open or closed", () => {
  const st = emptyState();
  const snap = snapOf();
  openTrade(st, snap, ["6E", "long", "1.1620", "1.1600"], T0);
  assert.ok(deskRoute("/tag", ["spring"], st, snap, T0).includes("“spring”"));
  assert.equal(st.trades[0].tags.setup, "spring");
  closeTrade(st, snap, ["6E", "1.1630"], T0 + 3600e3);
  deskRoute("/tag", ["shakeout"], st, snap, T0);
  assert.equal(st.journal[0].tags.setup, "shakeout");
});

test("monthly report: last month's trades first", () => {
  const J = [{ key: "6E", r: 1, closed: "2026-10-12T10:00:00Z", tags: { setup: "test" } }, { key: "6E", r: -1, closed: "2026-11-01T10:00:00Z" }];
  const m = monthlyText(J, Date.parse("2026-11-01T07:00:00Z"));
  assert.ok(m.startsWith("📊 <b>October</b>: 1 trade · +1.00R"), m);
});

test("long replies split on line breaks, never inside a tag", () => {
  const txt = Array.from({ length: 300 }, (_, i) => `<b>row ${i}</b> · some text that is long enough`).join("\n");
  const parts = chunks(txt);
  assert.ok(parts.length >= 3 && parts.every((p) => p.length <= 3900));
  assert.ok(parts.every((p) => (p.match(/<b>/g) || []).length === (p.match(/<\/b>/g) || []).length));
  assert.equal(parts.join("\n"), txt);
  assert.deepEqual(chunks("short"), ["short"]);
});
