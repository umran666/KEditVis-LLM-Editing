import assert from "node:assert/strict";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const base = process.env.AUDIT_URL ?? "http://127.0.0.1:5187";
// Fixture output goes to a gitignored directory by default. This suite
// regenerates screenshots and millisecond timings on every run, which would
// otherwise dirty tracked evidence files (audit/audit-desktop.png,
// audit/audit-mobile.png, audit/frontend-results.json) on every invocation.
// Set AUDIT_OUT=../audit to deliberately refresh that committed evidence.
const out = resolve(process.env.AUDIT_OUT ?? fileURLToPath(new URL("../../audit/run/", import.meta.url)));
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

  await page.click(".chat-mode-toggle button:nth-child(2)");
  const setNeighborAnswers = async (text) => {
    await page.click('[aria-label="Neighborhood answers"]');
    await page.keyboard.down("Control");
    await page.keyboard.press("KeyA");
    await page.keyboard.up("Control");
    await page.keyboard.press("Backspace");
    if (text) await page.type('[aria-label="Neighborhood answers"]', text);
  };
  await setNeighborAnswers("Paris\nLondon");
  await page.click(".btn-primary");
  await page.waitForFunction(() => document.querySelector(".diff-body")?.textContent?.includes("EDIT_RESPONSE"));
  assert.deepEqual(sent.filter((r) => r.path === "/edit").at(-1).body.neighborhood_targets, ["Paris", "London"]);
  assert.deepEqual(await page.$$eval(".col-neighborhood .eval-pill strong", (els) => els.map((el) => el.textContent)), ["Paris", "London"]);
  await setNeighborAnswers("Paris");
  const editCount = sent.filter((r) => r.path === "/edit").length;
  const compareCount = sent.filter((r) => r.path === "/compare").length;
  await page.click(".btn-primary");
  await page.waitForFunction(() => document.querySelector(".error-banner")?.textContent?.includes("one non-empty original answer"));
  assert.equal(sent.filter((r) => r.path === "/edit").length, editCount);
  await page.click(".action-buttons-group button:nth-child(2)");
  await tick(page);
  assert.equal(sent.filter((r) => r.path === "/compare").length, compareCount);
  await setNeighborAnswers("");
  await page.click(".btn-primary");
  await page.waitForFunction(() => document.querySelector(".diff-body")?.textContent?.includes("EDIT_RESPONSE"));
  assert.equal(Object.hasOwn(sent.filter((r) => r.path === "/edit").at(-1).body, "neighborhood_targets"), false);
  assert.deepEqual(await page.$$eval(".col-neighborhood .eval-pill strong", (els) => els.map((el) => el.textContent)), ["Paris", "Paris"]);
  await page.click(".facts-card .btn-tiny");
  record("per-neighborhood answers are submitted and displayed, mismatches reject locally, and empty input restores default answers");

  assert.equal(sent.filter((r) => r.path === "/edit").at(-1).body.optimization, "context");
  await page.select('[aria-label="MEMIT objective"]', "standard");
  assert.equal(await page.$(".diff-body"), null);
  await page.click(".btn-primary");
  await page.waitForFunction(() => document.querySelector(".diff-body")?.textContent?.includes("EDIT_RESPONSE"));
  assert.equal(sent.filter((r) => r.path === "/edit").at(-1).body.optimization, "standard");
  holdEdit = true;
  await page.click(".btn-primary");
  await page.waitForFunction(() => Boolean(document.querySelector(".telemetry-bar")));
  const beforePendingLayers = await page.$eval(".selected-layers-pill .pill-val", (el) => el.textContent);
  assert.equal(await page.$(".diff-body"), null);
  assert.equal(await page.$$eval(".layer-selector button", (els) => els.every((el) => el.disabled)), true);
  await page.click('.lens-pre .token-bubble[data-layer="0"][data-rank="1"]');
  await tick(page);
  assert.equal(await page.$eval(".selected-layers-pill .pill-val", (el) => el.textContent), beforePendingLayers);
  record("pending edits clear prior results and prevent layer changes");
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
  await page.click(".method-pill-group button:nth-child(1)");
  assert.equal(await page.$$eval(".layer-chip.active", (els) => els.length), 6);
  await page.click(".action-buttons-group .btn-action:first-child");
  await page.waitForFunction(() => document.querySelector(".selected-layers-pill .pill-val")?.textContent === "0-5");
  assert.match(await page.$eval(".layer-actions button:first-child", (el) => el.textContent), /0, 1, 2, 3, 4, 5/);
  await page.click(".layer-actions button:nth-child(2)");
  await page.click(".layer-actions button:first-child");
  assert.equal(await page.$eval(".selected-layers-pill .pill-val", (el) => el.textContent), "0-5");
  record("GPT-J method switching and both recommendation controls select six layers");
  await page.click(".method-pill-group button:nth-child(2)");
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

  await page.evaluate(() => window.audit.renderInvalidPromptMetrics());
  await tick(page);
  assert.equal(await page.$$eval(".eval-fail, .eval-pass", (els) => els.length), 1); // aggregate ES alone is known
  assert.equal(await page.$$eval(".eval-pill.eval-unknown", (els) => els.length), 3);
  record("partial and non-finite per-prompt metrics stay unknown without throwing");

  await page.evaluate(() => window.audit.renderDuplicatePromptMetrics());
  await tick(page);
  assert.deepEqual(await page.$$eval(".col-neighborhood .eval-pill", (els) => els.map((el) => el.classList.contains("eval-pass"))), [true, false]);
  assert.deepEqual(await page.$$eval(".col-neighborhood .eval-pill strong", (els) => els.map((el) => el.textContent)), ["London", "Paris"]);
  assert.deepEqual(await page.$$eval(".col-paraphrase .eval-pill", (els) => els.map((el) => el.classList.contains("eval-unknown"))), [false, true]);
  await page.evaluate(() => window.audit.renderDuplicatePromptMetrics(true));
  await tick(page);
  assert.deepEqual(await page.$$eval(".col-neighborhood .eval-pill", (els) => els.map((el) => el.classList.contains("eval-unknown"))), [true, false]);
  record("duplicate prompt rows use their own ordered result and missing or mismatched evidence stays unknown");

  await page.evaluate(() => window.audit.renderSelector("complete"));
  await tick(page);
  assert.match(await page.$eval(".layer-actions button:first-child", (el) => el.textContent), /6, 7, 8, 9, 10, 11/);
  await page.evaluate(() => window.audit.renderSelector("partial"));
  await tick(page);
  assert.equal(await page.$eval(".layer-actions button:first-child", (el) => el.disabled), true);
  const recommendations = await page.evaluate(() => {
    const full = Array.from({ length: 48 }, (_, layer) => ({ layer, cosine_similarity: layer >= 10 && layer <= 14 ? 0 : 0.9 }));
    const invalid = [full.slice(1), [...full.slice(1), full[1]], full.map((s) => s.layer === 0 ? { ...s, cosine_similarity: NaN } : s), full.map((s) => s.layer === 0 ? { ...s, layer: 49 } : s), full.map((s) => s.layer === 0 ? { ...s, cosine_similarity: 1.00001 } : s)];
    const nearUnit = full.map((s) => s.layer < 2 ? { ...s, cosine_similarity: s.layer === 0 ? 1.0000003576278687 : -1.0000003576278687 } : s);
    return { layers: window.audit.recommendLayers(full.reverse(), "memit", 48), nearUnit: window.audit.recommendLayers(nearUnit, "memit", 48), rejected: invalid.map((signals) => { try { window.audit.recommendLayers(signals, "memit", 48); return false; } catch { return true; } }) };
  });
  assert.deepEqual(recommendations.layers, [10, 11, 12, 13, 14]);
  assert.deepEqual(recommendations.nearUnit, [10, 11, 12, 13, 14]);
  assert(recommendations.rejected.every(Boolean));
  record("recommendations use complete finite layer telemetry and reject missing or duplicate data");

  await page.evaluate(() => window.audit.renderMissingSignals());
  await tick(page);
  await page.evaluate(() => [...document.querySelectorAll("button")].find((el) => el.textContent.trim() === "Residual Var").click());
  await tick(page);
  assert.deepEqual(await page.$$eval("rect.bar", (els) => els.map((el) => [el.dataset.layer, +el.getAttribute("width")])), [["1", 196], ["2", 0]]);
  await page.evaluate(() => [...document.querySelectorAll("button")].find((el) => el.textContent.trim() === "Variance").click());
  await tick(page);
  assert.equal(await page.$$eval("rect.post", (els) => els.length), 1);
  assert.equal(await page.$eval("rect.post", (el) => +el.getAttribute("width")), 0);
  await page.evaluate(() => [...document.querySelectorAll("button")].find((el) => el.textContent.trim() === "Δ Var").click());
  await tick(page);
  assert.deepEqual(await page.$$eval("rect.bar", (els) => els.map((el) => el.dataset.layer)), ["1", "2"]);
  record("missing residual measurements are omitted and measured zero variance retains zero geometry");

  await page.evaluate(() => window.audit.renderMissingSignals(true));
  await tick(page);
  await page.evaluate(() => [...document.querySelectorAll("button")].filter((el) => ["Cosine", "Cosine Activity"].includes(el.textContent.trim())).forEach((el) => el.click()));
  await tick(page);
  assert.equal(await page.$$eval("rect.bar", (els) => els.length), 3);
  assert.equal(await page.$eval('rect.bar[data-layer="2"]', (el) => +el.getAttribute("width")), 0);
  assert.equal(await page.$$eval("rect.post", (els) => els.length), 3);
  assert.equal(await page.$eval("rect.post:last-of-type", (el) => +el.getAttribute("width")), 0);
  record("float32 near-unit cosine roundoff remains valid telemetry and renders zero activity");

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
