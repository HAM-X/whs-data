// Offline tests: node --test bot/worker.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { route, size, findSymbol, toMt5, tvMessage } from "./worker.js";

const snap = JSON.parse(fs.readFileSync(new URL("./fixtures/snapshot.sample.json", import.meta.url)));
const strip = (s) => s.replace(/<[^>]+>/g, "");

test("aliases", () => {
  assert.equal(findSymbol("how's gold doing?"), "GC");
  assert.equal(findSymbol("yen effort"), "6J");
  assert.equal(findSymbol("EURUSD levels"), "6E");
  assert.equal(findSymbol("nothing here"), null);
});

const dec = {
  _today: "<b>Today · Wed 21 Oct</b>\n<b>Act on</b>  nothing", _at: "x", _setups: "<b>Setups</b>", _dollar: "<b>Dollar</b>",
  "6E 15m": { action: "BUY ZONE", caption: "<b>6E · 15m</b>", vsa: "v15", why: "w15" },
  "6E 5m": { action: "WAIT", caption: "<b>6E · 5m</b>", vsa: "v5", why: "w5" },
  "6E 4h": { action: "WAIT", caption: "<b>6E · 4h</b>  ▸ <b>WAIT</b>", vsa: "<b>6E · 4h · VSA</b>", why: "<b>story</b>" },
  "6E 1h": { action: "BUY ZONE", caption: "<b>6E · 1h</b>  ▸ <b>BUY ZONE</b>", vsa: "<b>6E · 1h · VSA</b>" },
  "GC 4h": { action: "TREND ↓", caption: "<b>GC · 4h</b>", vsa: "<b>GC · 4h · VSA</b>" },
  "GC 1h": { action: "TREND ↓", caption: "<b>GC · 1h</b>", vsa: "x" },
};
const sc = { __dec: dec, __news: "📰 News · test" };

test("v2 commands: photos for /today and /chart, text for /vsa, old commands map to new ones", () => {
  assert.deepEqual(route("/today", snap, {}, sc), { photo: "board", caption: dec._today });
  assert.deepEqual(route("/chart euro", snap, {}, sc), { photo: "card:6E:4h", caption: dec["6E 4h"].caption });
  assert.equal(route("/chart 6E 1h", snap, {}, sc).photo, "card:6E:1h");
  assert.equal(route("/vsa 6E", snap, {}, sc), dec["6E 4h"].vsa);
  assert.equal(route("/status 6E", snap, {}, sc).photo, "card:6E:4h");      // old → /chart
  assert.equal(route("/effort 6E 1h", snap, {}, sc), dec["6E 1h"].vsa);     // old → /vsa
  assert.equal(route("/brief", snap, {}, sc).photo, "board");               // old → /today
  assert.equal(route("how's gold", snap, {}, sc).photo, "card:GC:4h");
  assert.equal(route("gold vsa 1h", snap, {}, sc), "x");
  assert.equal(route("today", snap, {}, sc).photo, "board");
  for (const q of ["/help", "/health", "/convert 6J 0.0067", "/size 6E 1.1200 1.1180 B 100000", "random words", "/chart dx"]) {
    const out = route(q, snap, {}, sc);
    assert.ok(typeof out === "string" && out.length > 20 && out.length < 4000, q);
  }
  assert.match(route("/today", snap, {}, null), /No board yet/);
});

test("lower timeframes, setups, story, dollar", () => {
  assert.deepEqual(route("/zoom 6E", snap, {}, sc), { photo: "card:6E:15m", caption: "<b>6E · 15m</b>" });
  assert.equal(route("/zoom euro 5m", snap, {}, sc).photo, "card:6E:5m");
  assert.equal(route("/chart 6E 5m", snap, {}, sc).photo, "card:6E:5m");
  assert.equal(route("/vsa 6E 15m", snap, {}, sc), "v15");
  assert.equal(route("euro 5m vsa", snap, {}, sc), "v5");
  assert.equal(route("/why 6E", snap, {}, sc), "<b>story</b>");
  assert.equal(route("why euro 15m", snap, {}, sc), "w15");
  assert.equal(route("/setups", snap, {}, sc), "<b>Setups</b>");
  assert.equal(route("any setups?", snap, {}, sc), "<b>Setups</b>");
  assert.equal(route("/dollar", snap, {}, sc), "<b>Dollar</b>");
  assert.equal(route("how is the dollar", snap, {}, sc), "<b>Dollar</b>");
  assert.match(route("/zoom gc", snap, {}, sc), /No 15m read/);
});

test("/plan: GO with MT5 lots; REDUCE when the chart is against; NO in a blackout", async () => {
  const { planText } = await import("./worker.js");
  const st = { trades: [], journal: [], pnl: null, dd: 0, limits: { day: 5, dd: 10 } };
  const go = strip(planText(snap, st, ["6E", "long", "1.1200", "1.1180", "B", "400000"], {}, dec));
  assert.match(go, /GO/); assert.match(go, /lots/); assert.match(go, /1h BUY ZONE/);
  const red = strip(planText(snap, st, ["GC", "long", String(snap.instruments.GC.price), String(snap.instruments.GC.price - 10), "B", "400000"], {}, dec));
  assert.match(red, /REDUCE/); assert.match(red, /against you/);
  const d2 = { ...dec, "6E 4h": { action: "NO NEW ENTRIES" } };
  assert.match(strip(planText(snap, st, ["6E", "long", "1.1200", "1.1180", "B"], {}, d2)), /NO/);
  assert.match(planText(snap, st, ["6E", "long", "1.1180", "1.1200"], {}, dec), /Usage/);
});

// Stack v2 worked examples (risk $500 = 0.25% of 200,000)
const fake = (key, fut, spot, sym, dir, tick, pip) => ({ instruments: { [key]: { available: true, tick, price: fut,
  mt5: { symbol: sym, direction: dir, spot, pip } } }, generated_local: "x", generated_utc: new Date().toISOString() });

test("example A: 6E -> EURUSD 2.50 lots", () => {
  const s = fake("6E", 1.0909, 1.0874, "EURUSD", "same", 0.00005, 0.0001);
  assert.equal(toMt5(s.instruments["6E"], 1.0889).toFixed(5), "1.08541");
  const out = strip(size(s, "6E", 1.0909, 1.0889, "0.25", 200000));
  assert.match(out, /2\.50 lots/); assert.match(out, /2 × 6E/); assert.match(out, /20 × M6E/);
});

test("example B: 6J long -> USDJPY SELL 1.67 lots", () => {
  const s = fake("6J", 0.00671, 149.60, "USDJPY", "inverted", 0.0000005, 0.01);
  assert.equal(toMt5(s.instruments["6J"], 0.00669).toFixed(3), "150.047");
  const out = strip(size(s, "6J", 0.00671, 0.00669, "B", 200000));
  assert.match(out, /SELL/); assert.match(out, /1\.67 lots/); assert.match(out, /2 × 6J/); assert.match(out, /20 × MJY/);
});

test("example C: GC short -> 0.53 lots XAUUSD", () => {
  const s = fake("GC", 3412.0, null, "XAUUSD", "same", 0.1, 0.01);
  const out = strip(size(s, "GC", 3412.0, 3421.4, "B", 200000));
  assert.match(out, /0\.53 lots/); assert.match(out, /0 × GC/); assert.match(out, /5 × MGC/);
});

test("grade A above 0.5 warns, C is 0.125", () => {
  const s = fake("6E", 1.0909, 1.0874, "EURUSD", "same", 0.00005, 0.0001);
  assert.match(strip(size(s, "6E", 1.0909, 1.0889, "1", 100000)), /Above your grade A/);
  assert.match(strip(size(s, "6E", 1.0909, 1.0889, "C", 100000)), /risk 0\.125%/);
});

test("TradingView alerts become Telegram messages with a flag", () => {
  assert.match(tvMessage("6E1! 240 · SOS confirmed — bullish: Re-accumulation"), /^● <b>TradingView<\/b> · 6E1! 240 · SOS confirmed/);
  assert.match(tvMessage("6E1! 60 · SOS failed — price is back in the range, it is still going on (SOS → UA)"), /^↺/);
  assert.match(tvMessage("GC1! 240 · left the range — Distribution confirmed"), /^⚑/);
  assert.match(tvMessage('{"text":"6B1! 60 · Spring #3"}'), /Spring #3/);
  assert.ok(!tvMessage("<script>").includes("<script>"));
});

// Stage 3 scan commands
{
  const { route, scanReport, storyReport } = await import("./worker.js");
  const sc = { "6E 4h": { major: { state: 0, box: [1.1142, 1.1309], protected: null },
    ranges: [{ id: "2026-10-05T02:00:00+00:00", kind: "SC", phase: "B", name: "Accumulation / Distribution", support: 1.1142, resistance: 1.1309,
      story: ["#1 A: SC on 6.1×"], bias: "OPEN · Accumulation 1 vs Distribution 0" }] } };
  const a = scanReport(sc);
  if (!a.includes("sideways") || !a.includes("phase B")) throw new Error("scan report");
  const b = storyReport(sc, "6E", "4h");
  if (!b.includes("#1 A") || !b.includes("BIAS")) throw new Error("story report");
  console.log("ok scan commands");
}

{
  const { panelReport } = await import("./worker.js");
  const t = panelReport({ top3: [{ key: "6J", why: ["JPY higher low vs DX (bullish)"] }], persistence: { CAD: "strong (all 3 windows)" },
    sweet_spot: { dx_high: { at: "2026-10-14T14:00", new_extreme: true, read: { JPY: { dir: 1, text: "higher low — x" } } } },
    themes: { blocs: [["6E", "6B"]] }, late: { "6E": true } });
  if (!t.includes("<b>Market Panel</b>") || !t.includes("6E/6B") || !t.includes("JPY higher low")) throw new Error("panel " + t);
  console.log("ok panel");
}


test("/news and plain 'any news today' answer from news.json", async () => {
  assert.equal(route("/news", {}, {}, sc), "📰 News · test");
  assert.equal(route("any news today?", {}, {}, sc), "📰 News · test");
});


test("fresh or instant: any pair, timeframe and analysis", async () => {
  const { freshAsk } = await import("./worker.js");
  const now = Date.parse("2026-10-21T09:20:00Z");
  const d = { _at: "2026-10-21T09:12:00Z", "6E 4h": { action: "WAIT" }, "6E 15m": { action: "WAIT" }, "GC 15m": { action: "WAIT" } };
  assert.equal(freshAsk("/vsa 6E", d, now), null);                       // hourly read is 8 min old → instant
  assert.equal(freshAsk("/vsa gc 15m", d, now), null);
  assert.equal(freshAsk("/vsa gc 15m", d, Date.parse("2026-10-21T09:40:00Z")), "vsa gc 15m");   // 28 min old on 15m → fresh
  assert.equal(freshAsk("/vsa 6B 15m", d, now), "vsa 6B 15m");           // not precomputed → fresh
  assert.equal(freshAsk("/chart 6E daily", d, now), "chart 6E daily");    // daily is always fresh
  assert.equal(freshAsk("gc 15m vsa trend", d, now), "gc 15m vsa trend");
  assert.equal(freshAsk("/trend euro", d, now), "trend euro 4h");
  assert.equal(freshAsk("euro 4h vsa now", d, now), "euro 4h vsa");
  assert.equal(freshAsk("/read cable 30m", d, now), "all cable 30m");
  assert.equal(freshAsk("/today", d, now), null);
  assert.equal(freshAsk("how is the dollar", d, now), null);
});
