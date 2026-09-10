import assert from "node:assert/strict";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import puppeteer from "puppeteer";

const base = process.env.AUDIT_URL ?? "http://127.0.0.1:5187";
const out = resolve("../audit");
mkdirSync(out, { recursive: true });
const browser = await puppeteer.launch({ headless: true, ...(existsSync("C:/Program Files/Google/Chrome/Application/chrome.exe") ? { executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe" } : {}) });
const checks = [];
const errors = [];
const record = (name) => { checks.push(name); console.log(`PASS ${name}`); };
const tick = (page) => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const signals = (n) => Array.from({ length: n }, (_, layer) => ({ layer, cosine_similarity: 0.5, residual_variance: 1.25, residual_variance_last: 1.10, residual_delta_variance: 0.85, top_tokens: Array.from({ length: 5 }, (_, i) => ({ token: `BASE${i}`, prob: 0.5 / (i + 1) })), last_top_tokens: Array.from({ length: 5 }, (_, i) => ({ token: `LAST${i}`, prob: 0.5 / (i + 1) })) }));
const metrics = (score) => ({ ES: score, PS: score, NS: score, S: score });
let holdEdit = false, failEdit = false;
const pending = [];
const sent = [];

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.setRequestInterception(true);
  page.on("request", async (request) => {
    const path = new URL(request.url()).pathname;
    if (!["/health", "/probe", "/edit", "/compare", "/generate"].includes(path)) return request.continue();
    if (request.method() === "OPTIONS") return request.respond({ status: 204, headers: { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*", "Access-Control-Allow-Methods": "GET, POST, OPTIONS" } });
    const body = request.postData() ? JSON.parse(request.postData()) : {};
    sent.push({ path, body });
    const model = body.model ?? new URL(request.url()).searchParams.get("model") ?? "gpt2-xl";
    const n = model === "gpt2-xl" ? 48 : 28;
    const layerSignals = signals(n);
    const neighborhood = [{ prompt: "Measured neighborhood", pre_text: "ACTUAL unchanged", post_text: "ACTUAL unchanged", kl_divergence: 0 }];
    const weight_drift = { total_absolute_frobenius: 0.42, total_relative_frobenius: 0.0035, mean_layer_relative_frobenius: 0.0035, per_layer: {} };
    let data;
    if (path === "/health") data = { status: "ok", model, n_layers: n, methods: ["memit", "rome"] };
    if (path === "/generate") data = { model, prompt: "fixture prompt", generation: "MEASURED_GENERATION" };
    if (path === "/probe") data = { rewrite_prompt: "probe", layer_signals: layerSignals };
    if (path === "/edit") data = { method: body.method, edited_layers: body.layers, weight_drift, pre_edit: { layer_signals: layerSignals, generations: ["BEFORE"], metrics: metrics(0) }, post_edit: { layer_signals: layerSignals, generations: ["EDIT_RESPONSE"], metrics: metrics(0) }, damage: { kl_divergence: 0, n_prompts: 1, note: "fixture" }, neighborhood };
    if (path === "/compare") data = { method: body.method, baseline: { layer_signals: layerSignals, generation: "BASELINE", metrics: metrics(0) }, schemes: body.schemes.map((layers, index) => ({ layers, weight_drift, metrics: metrics(index === 1 ? 1 : 0), generation: `SCHEME_${layers.join("-")}`, damage: { kl_divergence: index / 100, n_prompts: 1, note: "fixture" }, neighborhood })) };
    if (path === "/edit" && holdEdit) await new Promise((resolve) => pending.push(resolve));
    try {
      await request.respond({ status: path === "/edit" && failEdit ? 500 : 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*" }, body: JSON.stringify(path === "/edit" && failEdit ? { detail: "REAL_BACKEND_FAILURE" } : data) });
    } catch (e) { if (!page.isClosed()) throw e; }
  });
  await page.goto(base, { waitUntil: "networkidle0" });
  await page.click(".btn-primary");
  await page.waitForFunction(() => document.querySelector(".diff-body")?.textContent?.includes("EDIT_RESPONSE"));
  assert.equal(await page.$$eval(".lens-pre .token-bubble", (els) => els.length), 48 * 5);
  const bubble = await page.$('.lens-pre .token-bubble[data-layer="0"][data-rank="1"]');
  await bubble.hover();
  await page.waitForFunction(() => [...document.querySelectorAll(".lens-pre .token-path")].some((path) => path.getAttribute("stroke-width") === "2"));
  assert.match(await page.$eval(".lens-pre .token-mini-detail", (el) => el.textContent), /LAST0/);
  await page.mouse.move(1, 1);
  await page.click(".chat-box-content button");
  await page.waitForFunction(() => document.querySelector(".chat-target")?.textContent === "MEASURED_GENERATION");
  record("all five ranks render and hover highlights token paths; chat uses API output");
  assert.equal(await page.$eval(".col-efficacy .eval-pill", (el) => el.classList.contains("eval-fail")), true);
  assert.match(await page.$eval(".drift-view-panel", (el) => el.textContent), /Reference KL: 0\.000e\+0/);
  assert.equal(await page.$$eval(".drift-svg circle", (els) => els.length), 0);
  record("zero efficacy fails; zero KL preserved; no fabricated projection");

  assert.equal(sent.filter((r) => r.path === "/edit").at(-1).body.optimization, "context");
  await page.select('[aria-label="MEMIT objective"]', "standard");
  assert.equal(await page.$(".diff-body"), null);
  await page.click(".btn-primary");
  await page.waitForFunction(() => document.querySelector(".diff-body")?.textContent?.includes("EDIT_RESPONSE"));
  assert.equal(sent.filter((r) => r.path === "/edit").at(-1).body.optimization, "standard");
  holdEdit = true;
  await page.click(".btn-primary");
  await page.waitForFunction(() => Boolean(document.querySelector(".telemetry-bar")));
  await page.select('[aria-label="MEMIT objective"]', "context");
  pending.splice(0).forEach((resolve) => resolve());
  await page.waitForNetworkIdle();
  assert.equal(await page.$(".diff-body"), null);
  record("MEMIT profile is sent to API and profile changes discard stale responses");

  holdEdit = true;
  await page.click(".btn-primary");
  await page.waitForFunction(() => Boolean(document.querySelector(".telemetry-bar")));
  await page.click(".method-pill-group button:nth-child(2)");
  await tick(page);
  assert.equal(await page.$(".telemetry-bar"), null);
  assert.equal(await page.$(".diff-body"), null);
  await page.waitForFunction(() => document.querySelectorAll(".layer-chip.active").length === 1);
  pending.splice(0).forEach((resolve) => resolve());
  await page.waitForNetworkIdle();
  assert.equal(await page.$(".diff-body"), null);
  record("method switch invalidates in-flight edit and collapses ROME selection");

  await page.click(".btn-primary");
  await page.waitForFunction(() => Boolean(document.querySelector(".telemetry-bar")));
  await page.select(".model-dropdown-select", "EleutherAI/gpt-j-6B");
  await page.waitForFunction(() => document.querySelectorAll(".axis-tick").length === 28);
  pending.splice(0).forEach((resolve) => resolve());
  await page.waitForNetworkIdle();
  assert.equal(await page.$(".telemetry-bar"), null);
  assert.equal(await page.$(".diff-body"), null);
  assert.equal(await page.$$eval(".token-ranking-box svg rect", (els) => els.length), 0);
  for (let i = 0; i < 4; i++) await page.select(".model-dropdown-select", i % 2 ? "EleutherAI/gpt-j-6B" : "gpt2-xl");
  await page.waitForNetworkIdle();
  assert.equal(await page.$$eval(".axis-tick", (els) => els.length), 28);
  record("rapid 48/28 model switches discard stale results and release loading");

  holdEdit = false;
  await page.click(".btn-action:nth-child(2)");
  await page.waitForSelector(".scheme-row");
  const request = sent.filter((r) => r.path === "/compare").at(-1);
  assert(request.body.schemes.every((layers) => layers.length === 1));
  assert.equal(new Set(request.body.schemes.map(String)).size, request.body.schemes.length);
  record("ROME comparisons submit unique singleton schemes");

  const checkAnchors = async () => {
    await tick(page);
    return page.evaluate(() => Array.from(document.querySelectorAll("[data-wire-key]")).map((group) => {
      const path = group.querySelector(".scheme-connector");
      const p = path.getPointAtLength(path.getTotalLength()).matrixTransform(path.getScreenCTM());
      const row = Array.from(document.querySelectorAll("[data-scheme-key]")).find((row) => row.dataset.schemeKey === group.dataset.wireKey).getBoundingClientRect();
      return Math.max(Math.abs(p.x - row.left), Math.abs(p.y - (row.top + row.height / 2)));
    }));
  };
  let anchors = await checkAnchors();
  assert.equal(anchors.length, request.body.schemes.length);
  assert(anchors.every((e) => e < 1), JSON.stringify(anchors));
  await page.evaluate(() => document.querySelector(".scheme-row td").style.height = "95px");
  await tick(page);
  anchors = await checkAnchors();
  assert(anchors.every((e) => e < 1), JSON.stringify(anchors));
  await page.setViewport({ width: 1366, height: 900 });
  anchors = await checkAnchors();
  assert(anchors.every((e) => e < 1), JSON.stringify(anchors));
  record("sorted wire endpoints match dynamic table rows within one screen pixel");

  const key = await page.$eval(".scheme-row:last-child", (el) => el.dataset.schemeKey);
  await page.click(".scheme-row:last-child");
  await page.waitForFunction((key) => document.querySelector(".col-generation")?.textContent?.includes(`SCHEME_${key}`), {}, key);
  assert.equal(await page.$eval(".selected-layers-pill .pill-val", (el) => el.textContent), key);
  assert.match(await page.$eval(".diff-body", (el) => el.textContent), new RegExp(`SCHEME_${key}`));
  const wirePoint = await page.$eval("[data-wire-key]:first-child .scheme-connector", (path) => {
    const p = path.getPointAtLength(path.getTotalLength() * 0.9).matrixTransform(path.getScreenCTM());
    return { x: p.x, y: p.y };
  });
  await page.mouse.click(wirePoint.x, wirePoint.y);
  await tick(page);
  const firstKey = await page.$eval("[data-wire-key]", (el) => el.dataset.wireKey);
  assert.equal(await page.$eval(".scheme-row.selected", (el) => el.dataset.schemeKey), firstKey);
  record("row and wire selection synchronize layers, generation, metrics and diff");
  await page.setViewport({ width: 1920, height: 1080 });
  await page.screenshot({ path: resolve(out, "audit-desktop.png"), fullPage: true });

  failEdit = true;
  await page.click(".btn-primary");
  await page.waitForFunction(() => document.querySelector(".error-banner")?.textContent === "REAL_BACKEND_FAILURE");
  assert.equal(await page.$(".lens-post .token-ranking-box"), null);
  record("HTTP failures remain errors instead of synthetic successful edits");
  await page.setViewport({ width: 390, height: 844 });
  await tick(page);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  record("mobile layout contains comparison scrolling without page overflow");
  await page.screenshot({ path: resolve(out, "audit-mobile.png"), fullPage: true });

  await page.evaluate(async () => { window.audit = await import("/tests/component_harness.tsx"); });
  const diffs = await page.evaluate(() => {
    const samples = [["", ""], ["", "new"], ["old", ""], ["Hello, world!\nNext paragraph.", "Hello world?\nAnother paragraph."], ["x ".repeat(100000) + "OLD_TAIL", "x ".repeat(100000) + "NEW_TAIL"]];
    return samples.map(([oldText, newText]) => {
      const start = performance.now();
      const chunks = window.audit.computeWordDiff(oldText, newText);
      return { old: chunks.filter((c) => c.type !== "ins").map((c) => c.text).join("") === oldText, next: chunks.filter((c) => c.type !== "del").map((c) => c.text).join("") === newText, ms: performance.now() - start };
    });
  });
  assert(diffs.every((d) => d.old && d.next));
  assert(diffs.every((d) => d.ms < 500), JSON.stringify(diffs));
  record(`diff preserves punctuation, whitespace, empty strings and 200KB tails (max ${Math.max(...diffs.map((d) => d.ms)).toFixed(1)}ms)`);

  await page.evaluate(() => window.audit.renderPrompts("missing"));
  await tick(page);
  assert.equal(await page.$$eval(".eval-pass", (els) => els.length), 0);
  assert.match(await page.$eval(".col-neighborhood", (el) => el.textContent), /London/);
  await page.evaluate(() => window.audit.renderPrompts("mixed"));
  await tick(page);
  assert.equal(await page.$$eval(".col-paraphrase .eval-pill", (els) => els.map((el) => el.classList.contains("eval-pass")).join(",")), "true,false");
  record("missing metrics stay unknown and mixed prompt outcomes use per-prompt evidence");

  await page.evaluate(() => window.audit.renderCharts("valid"));
  await tick(page);
  assert.equal(await page.$$eval("rect.post", (els) => els.length), 1);
  await page.evaluate(() => window.audit.renderCharts("disjoint"));
  await tick(page);
  assert.equal(await page.$$eval("rect.post", (els) => els.length), 0);
  await page.evaluate(() => window.audit.renderCharts("empty"));
  await tick(page);
  assert.equal(await page.$$eval(".token-ranking-box svg *", (els) => els.length), 0);
  record("disjoint and empty D3 data clear old geometry without NaN rectangles");

  await page.setViewport({ width: 1000, height: 800 });
  await page.evaluate(() => window.audit.renderDrift());
  await tick(page);
  await page.evaluate(() => { const svg = document.querySelector(".drift-svg"); svg.style.width = "200px"; svg.style.height = "300px"; });
  const region = await page.evaluate(() => {
    const svg = document.querySelector(".drift-svg");
    const ctm = svg.getScreenCTM();
    const ps = ["0-pre", "0-post"].map((id) => { const el = document.querySelector(`[data-point-id="${id}"]`); return { x: +el.getAttribute("cx"), y: +el.getAttribute("cy") }; });
    const a = new DOMPoint(Math.min(...ps.map((p) => p.x)) - 5, Math.min(...ps.map((p) => p.y)) - 5).matrixTransform(ctm);
    const b = new DOMPoint(Math.max(...ps.map((p) => p.x)) + 5, Math.max(...ps.map((p) => p.y)) + 5).matrixTransform(ctm);
    return { a: { x: a.x, y: a.y }, b: { x: b.x, y: b.y } };
  });
  await page.mouse.move(region.a.x, region.a.y);
  await page.mouse.down();
  await page.mouse.move(region.b.x, region.b.y);
  await page.mouse.up();
  await tick(page);
  assert.equal(await page.$eval('[data-drift-row="0"]', (el) => el.classList.contains("row-selected")), true);
  assert.equal(await page.$eval('[data-drift-row="1"]', (el) => el.classList.contains("row-selected")), false);
  await page.click('.drift-action-buttons button:nth-child(2)');
  await tick(page);
  await page.mouse.move(region.a.x, region.a.y);
  await page.mouse.down();
  await page.mouse.move(region.b.x, region.b.y);
  await page.mouse.up();
  await tick(page);
  assert.equal(await page.$$eval('.drift-svg circle[stroke="#0284C7"]', (els) => els.map((el) => el.dataset.pointId).join(",")), "0-post");
  record("letterboxed SVG marquee uses CTM and selects only visible points");
  assert.deepEqual(errors, []);
  record("no browser runtime exceptions");
  writeFileSync(resolve(out, "frontend-results.json"), JSON.stringify({ checks, diffs, errors }, null, 2));
} finally {
  pending.splice(0).forEach((resolve) => resolve());
  await browser.close();
}
