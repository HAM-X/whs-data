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

test("every command answers without throwing", () => {
  for (const q of ["/help", "/status 6E", "/status gold", "/all", "/levels 6B", "/effort 6J", "/effort dx",
    "/strength", "/strength 1w", "/health", "/convert 6J 0.0067", "/size 6E 1.1200 1.1180 B 100000",
    "how's the euro", "kiwi levels", "aussie volume", "who is strongest this week", "is the data ok", "market overview",
    "random words"]) {
    const out = route(q, snap, {});
    assert.ok(out.length > 20 && out.length < 4000, q);
  }
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
