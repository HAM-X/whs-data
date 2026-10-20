import test from "node:test";
import assert from "node:assert/strict";
import { emptyState, openTrade, manage, guard, closeTrade, pipOffset } from "./desk.js";

const T0 = Date.parse("2026-08-03T12:00:00Z");   // Mon 03 Aug 15:00 UTC+3
const inst = (bars) => ({ key: "6E", tick: 0.00005, price: 1.0909, mt5: { pip: 0.0001, spot: 1.0909 }, bars1h: bars });
const B = (rows) => ({ t: rows.map((r) => r[0]), o: rows.map((r) => r[1]), h: rows.map((r) => r[2]), l: rows.map((r) => r[3]),
  c: rows.map((r) => r[4]), e: rows.map((r) => r[5] ?? 1), sb: rows.map((r) => r[6] ?? 0), rb: rows.map((r) => r[7] ?? 0),
  atr: rows.map(() => 0.0013) });

test("Adaptive stop: SBar commitment → its true-range low, then 2R", () => {
  const st = emptyState();
  const snap = { instruments: { "6E": inst(B([])) } };
  openTrade(st, snap, ["6E", "long", "1.0909", "1.0889", "B"], T0);
  snap.instruments["6E"].bars1h = B([
    ["2026-08-03T12:00:00.000Z", 1.0909, 1.0915, 1.0905, 1.0912],
    ["2026-08-03T16:00:00.000Z", 1.0915, 1.0935, 1.0923, 1.0933, 2.0, 1],   // SBar: true-range low = prev close 1.0912 → stop
    ["2026-08-03T18:00:00.000Z", 1.0933, 1.0949, 1.0930, 1.0945],          // high 1.0949 = 2R
  ]);
  const out = manage(st, snap, T0 + 7 * 3600e3);
  assert.ok(out.some((x) => x.includes("stop to 1.0912")), out.join("\n"));
  assert.ok(out.some((x) => x.includes("2R reached")));
  assert.ok(out.some((x) => x.includes("breakeven")) || st.trades[0].stop > 1.0909);
  assert.equal(manage(st, snap, T0 + 8 * 3600e3).length, 0);           // nothing twice
});

test("stop hit is logged with its R", () => {
  const st = emptyState();
  const snap = { instruments: { "6E": inst(B([])) } };
  openTrade(st, snap, ["6E", "long", "1.0909", "1.0889", "B"], T0);
  snap.instruments["6E"].bars1h = B([["2026-08-03T13:00:00.000Z", 1.0905, 1.0906, 1.0885, 1.0890]]);
  const out = manage(st, snap, T0 + 2 * 3600e3);
  assert.ok(out[0].includes("stop 1.0889 hit") && out[0].includes("-1.00R"));
  assert.equal(st.trades.length, 0);
  assert.equal(st.journal[0].r, -1);
});

test("Guard: −1.90% day + 0.60% open + 0.5% new = −3.00% → BLOCKED (own 2.5%)", () => {
  const st = emptyState();
  st.pnl = { day: new Date(T0 + 3 * 3600e3).toISOString().slice(0, 10), pct: -1.9 };
  st.trades.push({ key: "6B", side: 1, entry: 1.3, stop: 1.299, R: 0.001, risk: 0.6 });
  const g = guard(st, 0.5, T0);
  assert.equal(g.state, "BLOCKED");
  assert.ok(Math.abs(g.worst + 3.0) < 1e-9);
  st.pnl.pct = 0;
  assert.equal(guard(st, 0.25, T0).state, "OK TO EXECUTE");
});

test("Guard: 4 losses in a row halves risk; 2 losses today blocks", () => {
  const st = emptyState();
  st.journal = [-1, -1, -1, -1].map((r) => ({ r, day: "2026-08-01" }));
  assert.equal(guard(st, 0.25, T0).factor, 0.5);
  st.journal.push({ r: -1, day: "2026-08-03" }, { r: -1, day: "2026-08-03" });
  assert.equal(guard(st, 0.25, T0).state, "BLOCKED");
});

test("2 pips on the future and close", () => {
  assert.ok(Math.abs(pipOffset(inst(null)) - 0.0002) < 1e-12);
  const st = emptyState();
  const snap = { instruments: { "6E": inst(B([])) } };
  openTrade(st, snap, ["6E", "long", "1.0909", "1.0889", "B"], T0);
  assert.ok(closeTrade(st, snap, ["6E", "1.0949"], T0).includes("+2.00R"));
});

test("themes: Roman's heat example — 6E, 6B, 6C at 0.5% → effective 1.29%; theme cap trims the third", async () => {
  const { effHeat, guard } = await import("./desk.js");
  const rho = { "6E": { "6B": 0.78, "6C": 0.55 }, "6B": { "6E": 0.78, "6C": 0.5 }, "6C": { "6E": 0.55, "6B": 0.5 } };
  const pos = [{ key: "6E", w: 0.5 }, { key: "6B", w: 0.5 }, { key: "6C", w: 0.5 }];
  assert.ok(Math.abs(effHeat(pos, rho) - 1.29) < 0.01);
  assert.ok(Math.abs(effHeat(pos.slice(0, 2), rho) - 0.94) < 0.01);
  const st = emptyState();
  st.trades.push({ key: "6E", side: 1, entry: 1.1, stop: 1.099, R: 0.001, risk: 0.5 }, { key: "6B", side: 1, entry: 1.3, stop: 1.299, R: 0.001, risk: 0.5 });
  const g = guard(st, 0.5, T0, { rho, blocs: [["6E", "6B", "6C"]], key: "6C", side: 1 });
  assert.equal(g.state, "REDUCE");
  assert.ok(g.why.some((w) => w.includes("same idea")));
});
