import assert from "node:assert/strict";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import puppeteer from "puppeteer";

const out = resolve("../audit/optimization/browser");
mkdirSync(out, { recursive: true });
const browser = await puppeteer.launch({ headless: true, ...(existsSync("C:/Program Files/Google/Chrome/Application/chrome.exe") ? { executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe" } : {}) });
const checks = [], errors = [];
const record = (name) => { checks.push(name); console.log(`PASS ${name}`); };
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1000 });
  page.on("pageerror", (error) => errors.push(String(error)));
  async function operation(name, path, action) {
    const pending = page.waitForResponse((r) => new URL(r.url()).pathname === path && r.request().method() !== "OPTIONS", { timeout: 1800000 });
    await action();
    const response = await pending;
    const data = await response.json();
    const request = response.request().postData();
    writeFileSync(resolve(out, `${name}.json`), JSON.stringify({ status: response.status(), request: request ? JSON.parse(request) : null, data }, null, 2));
    assert.equal(response.status(), 200);
    await page.waitForFunction(() => !document.querySelector(".telemetry-bar"), { timeout: 1800000 });
    assert.equal(await page.$(".error-banner"), null);
    return data;
  }
  await operation("health", "/health", () => page.goto(process.env.AUDIT_URL ?? "http://127.0.0.1:5187", { waitUntil: "domcontentloaded" }));
  assert.equal(await page.$eval('[aria-label="MEMIT objective"]', (el) => el.value), "context");
  const baseline = await operation("baseline", "/probe", () => page.click(".action-buttons-group button:nth-child(3)"));
  const edited = await operation("context-edit", "/edit", () => page.click(".btn-primary"));
  assert.equal(edited.optimization, "context");
  assert.equal(edited.optimization_config.revision, "context-v3");
  assert.equal(edited.post_edit.metrics.PS, 1);
  assert.equal(await page.$$eval(".col-paraphrase .eval-pass", (els) => els.length), 3);
  record("default context profile passes both dashboard paraphrases using real GPU results");
  await page.screenshot({ path: resolve(out, "desktop.png"), fullPage: true });
  await page.setViewport({ width: 390, height: 844 });
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.screenshot({ path: resolve(out, "mobile.png"), fullPage: true });
  record("context selector and real edit results fit desktop and mobile");
  await page.setViewport({ width: 1440, height: 1000 });
  const schemes = await page.$('textarea[aria-label="Comparison schemes"]');
  await schemes.click({ clickCount: 3 });
  await page.keyboard.down("Control");
  await page.keyboard.press("A");
  await page.keyboard.up("Control");
  await page.keyboard.type("13-17\n12-16");
  const comparison = await operation("context-compare", "/compare", () => page.click(".action-buttons-group button:nth-child(2)"));
  assert.equal(comparison.optimization, "context");
  assert.equal(comparison.schemes.length, 2);
  assert.equal(comparison.schemes.find((s) => s.layers.join() === "13,14,15,16,17").metrics.PS, 1);
  record("comparison uses the context profile independently for both schemes");
  const restored = await operation("restored", "/probe", () => page.click(".action-buttons-group button:nth-child(3)"));
  assert.deepEqual(restored, baseline);
  await page.select('[aria-label="MEMIT objective"]', "standard");
  // Select the original layer window after comparison may select another row.
  await page.reload({ waitUntil: "networkidle0" });
  await page.select('[aria-label="MEMIT objective"]', "standard");
  const standard = await operation("standard-edit", "/edit", () => page.click(".btn-primary"));
  assert.equal(standard.optimization, "standard");
  const failed = standard.post_edit.metrics.details.paraphrase.find((p) => p.prefix.startsWith("You can find"));
  assert(failed.target_new_nll > failed.target_true_nll);
  const displayed = await page.$$eval(".col-paraphrase .eval-pill", (els) => els.map((el) => ({ text: el.textContent, failed: el.classList.contains("eval-fail") })));
  for (const result of standard.post_edit.metrics.details.paraphrase) {
    const row = displayed.find((item) => item.text.includes(result.prefix));
    assert(row);
    assert.equal(row.failed, result.target_new_nll >= result.target_true_nll);
  }
  record("standard MEMIT remains available and its known failure is displayed honestly");
  const finalBaseline = await operation("final-restored", "/probe", () => page.click(".action-buttons-group button:nth-child(3)"));
  assert.deepEqual(finalBaseline, baseline);
  await page.select('[aria-label="MEMIT objective"]', "context");
  assert.deepEqual(errors, []);
  record("original model is restored after edits and comparisons; no browser exceptions");
} finally {
  writeFileSync(resolve(out, "results.json"), JSON.stringify({ checks, errors }, null, 2));
  await browser.close();
}
