import assert from "node:assert/strict";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const checks = [];
const read = (path) => JSON.parse(readFileSync(resolve(root, path), "utf8"));
const record = (name) => { checks.push(name); console.log(`PASS ${name}`); };
const labels = ["frozen-gpt2-context-extra", "frozen-gpt2-standard-extra", "frozen-gpt2-context-bigben", "frozen-gpt2-standard-bigben", "frozen-gptj-context-extra", "frozen-gptj-standard-extra"];
const artifacts = Object.fromEntries(labels.map((name) => [name, read(`audit/optimization/${name}/result.json`)]));
for (const [name, artifact] of Object.entries(artifacts)) {
  assert.equal(artifact.status, 200, name);
  assert.equal(artifact.restored, true, name);
  const metrics = artifact.response.post_edit.metrics;
  for (const [category, key] of [["efficacy", "ES"], ["paraphrase", "PS"], ["neighborhood", "NS"]]) {
    const rows = metrics.details[category];
    assert(rows.length > 0);
    const expected = rows.filter((r) => category === "neighborhood" ? r.target_true_nll < r.target_new_nll : r.target_new_nll < r.target_true_nll).length / rows.length;
    assert.equal(metrics[key], expected, `${name} ${key}`);
    assert(rows.every((r) => Number.isFinite(r.target_new_nll) && Number.isFinite(r.target_true_nll)));
  }
  assert.equal(metrics.ES, 1, name);
  assert.equal(metrics.NS, 1, name);
  assert(Number.isFinite(artifact.response.damage.kl_divergence));
}
record("all six frozen GPU results have finite measured scores, efficacy and neighborhood success, and exact restoration");
const score = (name) => artifacts[name].response.post_edit.metrics.PS;
assert.equal(score("frozen-gpt2-standard-extra"), .2);
assert.equal(score("frozen-gpt2-context-extra"), .8);
assert.equal(score("frozen-gpt2-context-bigben"), 1);
assert.equal(score("frozen-gpt2-standard-bigben"), 1);
assert.equal(score("frozen-gptj-context-extra"), 1);
assert.equal(score("frozen-gptj-standard-extra"), 1);
record("expanded GPT-2 paraphrase success rises from 1/5 to 4/5; Big Ben and GPT-J retain their pass rates");
const findReported = (name) => artifacts[name].response.post_edit.metrics.details.paraphrase.find((r) => r.prefix.startsWith("You can find"));
const before = findReported("frozen-gpt2-standard-extra"), after = findReported("frozen-gpt2-context-extra");
assert(before.target_new_nll > before.target_true_nll);
assert(after.target_new_nll < after.target_true_nll);
record("the reported GPT-2 paraphrase changes from failure to success under the unchanged likelihood rule");
const browser = read("audit/optimization/browser/results.json");
assert.equal(browser.checks.length, 5);
assert.deepEqual(browser.errors, []);
for (const viewport of ["desktop", "mobile"]) assert(existsSync(resolve(root, `audit/optimization/browser/${viewport}.png`)));
const browserEdit = read("audit/optimization/browser/context-edit.json");
assert.equal(browserEdit.data.optimization_config.revision, "context-v3");
assert.equal(browserEdit.request.optimization, "context");
record("five real browser checks pass and the deployed API identifies the frozen context profile");
const files = ["modal_app.py", "editing_optimizations.py", "test_editing_optimizations.py", "test_audit_backend.py", "test_optimization_live.py", "frontend/src/App.tsx", "frontend/src/App.css", "frontend/src/types.ts", "frontend/src/api/client.ts", "frontend/tests/audit.mjs", "frontend/tests/optimization-live.mjs"];
const sha256 = Object.fromEntries(files.map((file) => [file, createHash("sha256").update(readFileSync(resolve(root, file))).digest("hex")]));
writeFileSync(resolve(root, "audit/optimization/verification.json"), JSON.stringify({ verified_at: new Date().toISOString(), checks, sha256, limitation: "Small regression set; one of five expanded GPT-2 Eiffel phrasings still fails. Scores compare candidate likelihoods, not greedy generation." }, null, 2));
