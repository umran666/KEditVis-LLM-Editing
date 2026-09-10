import assert from "node:assert/strict";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import puppeteer from "puppeteer";

const out = resolve("../audit/live/browser");
mkdirSync(out, { recursive: true });
const browser = await puppeteer.launch({ headless: true, ...(existsSync("C:/Program Files/Google/Chrome/Application/chrome.exe") ? { executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe" } : {}) });
const checks = [], errors = [];
const record = (name) => { checks.push(name); console.log(`PASS ${name}`); };
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });
  page.on("pageerror", (error) => errors.push(String(error)));
  async function operation(name, path, action) {
    const pending = page.waitForResponse((response) => new URL(response.url()).pathname === path && response.request().method() !== "OPTIONS", { timeout: 1800000 });
    await action();
    const response = await pending;
    const data = await response.json();
    writeFileSync(resolve(out, `${name}.json`), JSON.stringify({ url: response.url(), status: response.status(), data }, null, 2));
    assert.equal(response.status(), 200);
    await page.waitForFunction(() => !document.querySelector(".telemetry-bar"), { timeout: 1800000 });
    assert.equal(await page.$(".error-banner"), null);
    return data;
  }
  const health = await operation("health", "/health", () => page.goto(process.env.AUDIT_URL ?? "http://127.0.0.1:5187", { waitUntil: "domcontentloaded" }));
  assert.equal(health.model, "gpt2-xl");
  const generated = await operation("generate", "/generate", () => page.click(".chat-box-content button"));
  assert.equal(await page.$eval(".chat-target", (el) => el.textContent), generated.generation);
  record("real generation reaches chat through the configured Modal API");
  const baseline = await operation("recommend-probe", "/probe", () => page.click(".action-buttons-group button:nth-child(1)"));
  assert.equal(await page.$$eval(".lens-pre .token-bubble", (els) => els.length), 240);
  const scores = baseline.layer_signals.slice(0, -4).map((_, i) => baseline.layer_signals.slice(i, i + 5).reduce((sum, signal) => sum + Math.abs(signal.cosine_similarity), 0));
  const best = scores.indexOf(Math.min(...scores));
  assert.equal(await page.$eval(".pill-val", (el) => el.textContent), `${best}-${best + 4}`);
  record("one Recommend click probes the model and selects a layer window");
  await page.click(".method-pill-group button:nth-child(2)");
  const schemes = await page.$('textarea[aria-label="Comparison schemes"]');
  await schemes.click({ clickCount: 3 });
  await page.keyboard.down("Control");
  await page.keyboard.press("A");
  await page.keyboard.up("Control");
  await page.keyboard.type("17\n16");
  const comparison = await operation("compare", "/compare", () => page.click(".action-buttons-group button:nth-child(2)"));
  assert.equal(comparison.schemes.length, 2);
  assert.equal(await page.$$eval(".lens-post .token-bubble", (els) => els.length), 240);
  assert.equal(await page.$$eval(".drift-svg circle", (els) => els.length), 4);
  record("two real ROME schemes populate scores, post-edit signals and measured drift");
  const edited = await operation("edit", "/edit", () => page.click(".btn-primary"));
  assert.equal(edited.edited_layers.length, 1);
  assert.match(await page.$eval(".diff-body", (el) => el.textContent), /Eiffel/);
  await page.screenshot({ path: resolve(out, "desktop.png"), fullPage: true });
  await page.setViewport({ width: 390, height: 844 });
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true);
  await page.screenshot({ path: resolve(out, "mobile.png"), fullPage: true });
  record("real edit results render on desktop and mobile without page overflow");
  const restored = await operation("revert", "/probe", () => page.click(".action-buttons-group button:nth-child(3)"));
  assert.deepEqual(restored.layer_signals, baseline.layer_signals);
  assert.equal(await page.$(".diff-body"), null);
  record("Revert restores the baseline view and reproduces the original layer signals");
  assert.deepEqual(errors, []);
  record("no browser runtime exceptions during the live workflow");
} finally {
  writeFileSync(resolve(out, "results.json"), JSON.stringify({ checks, errors }, null, 2));
  await browser.close();
}
