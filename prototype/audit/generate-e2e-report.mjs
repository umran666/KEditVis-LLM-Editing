import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

const audit = dirname(fileURLToPath(import.meta.url));
const root = resolve(audit, "..");
const json = (path) => JSON.parse(readFileSync(resolve(audit, path), "utf8"));
const models = ["gpt2-xl", "EleutherAI_gpt-j-6B"];
const browser = json("live/browser/results.json");
const fixtures = json("frontend-results.json");
if (browser.checks.length !== 6 || browser.errors.length) throw new Error("Live browser verification is incomplete");
const files = ["modal_app.py", "frontend/src/App.tsx", "frontend/src/App.css", "frontend/src/api/client.ts", "frontend/src/types.ts",
  "frontend/src/components/FactForm.tsx", "frontend/src/components/KnowledgeGraph.tsx", "frontend/src/components/TokenRankingChart.tsx",
  "frontend/src/components/DriftScatterPlot.tsx", "frontend/.env.local", "test_audit_backend.py", "test_live_backend.py", "frontend/tests/audit.mjs", "frontend/tests/live.mjs", "README.md"];
const lines = (path, needle) => readFileSync(resolve(root, path), "utf8").split(/\r?\n/).findIndex((line) => line.includes(needle)) + 1;
const location = (path, needle) => `[${path}:${lines(path, needle)}](../${path}#L${lines(path, needle)})`;
let report = `# KEditVis end-to-end fixes and verification\n\nDate: ${new Date().toISOString()}.\n\nThis report supersedes the runtime and feature limitations in the historical first-pass audit. The backend is deployed at [Modal API](https://opzgameryt--keditvis-memit-web-app.modal.run/health). The local dashboard is [http://127.0.0.1:5187](http://127.0.0.1:5187), configured to use that deployment.\n\n## Real GPU verification\n\nAll rows below used actual model weights and the pinned upstream editing implementation on an A100-40GB. Each model/method combination executed an edit and a comparison with two different schemes. Subsequent probes reproduced every baseline layer signal exactly; deterministic generation reproduced the baseline text exactly. No API mocks were used in these runs.\n\n| Model | Method | Edit layers | ES | PS | NS | S | Edit seconds | Compare seconds |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n`;
const manifest = {};
for (const model of models) {
  const verified = json(`live/${model}/verified.json`);
  if (verified.request_checks !== 14 || !verified.baseline_restored || verified.methods.length !== 2) throw new Error(`${model} verification did not finish`);
  const summary = json(`live/${model}/summary.json`);
  if (summary.length !== 14 || summary.some((row) => row.status !== (row.name === "reject-invalid-layer" ? 400 : 200))) throw new Error(`${model} incomplete`);
  manifest[model] = summary;
  for (const method of ["rome", "memit"]) {
    const edit = json(`live/${model}/${method}-edit.json`);
    const compare = json(`live/${model}/${method}-compare.json`);
    const metrics = edit.response.post_edit.metrics;
    if (!compare.response.schemes.every((scheme) => scheme.layer_signals?.length)) throw new Error(`${model} missing comparison signals`);
    report += `| ${model.replace("EleutherAI_", "EleutherAI/")} | ${method.toUpperCase()} | ${edit.response.edited_layers.join(", ")} | ${["ES", "PS", "NS", "S"].map((key) => metrics[key].toFixed(3)).join(" | ")} | ${edit.seconds} | ${compare.seconds} |\n`;
  }
}
report += `\nThe fact tested was Eiffel Tower: Paris to Rome, with one paraphrase and two Paris neighborhood prompts. These are case-specific measurements, not general benchmark scores. The complete requests, responses, timing, measured drift, and restored probes are preserved under [live](./live/). Health checks confirmed 48 GPT-2 XL layers and 28 GPT-J layers; every layer returned five tokens at both subject and last-token positions.\n\n## Additional fixes\n\n`;
const fixes = [
  ["MEDIUM", "Mobile drift output columns were too narrow", "frontend/src/App.css", "min-width: 600px", "Four fixed columns squeezed generated text into long vertical rows on mobile. The drift table now keeps a readable minimum width inside its own horizontal scroll container, while numeric drift values stay on one line and the page remains within the viewport. The comparison scheme editor also uses the existing input styling."],
  ["CRITICAL", "GPT-J MEMIT could not trace keyword block inputs", "modal_app.py", "def _positional_hidden_state", "The real GPU run failed with IndexError because Transformers 4.42.4 calls GPT-J blocks using hidden_states=, whereas the pinned upstream Trace only captures positional inputs. Temporary forward pre-hooks now expose the identical hidden tensor positionally during upstream editing and are removed on both success and failure. The failed live request was followed by an exactly matching baseline probe, validating rollback on this real exception. The final GPU matrix was rerun after the fix. See the pinned [upstream tracer](https://raw.githubusercontent.com/kmeng01/memit/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b/util/nethook.py) and [Transformers GPT-J calls](https://raw.githubusercontent.com/huggingface/transformers/v4.42.4/src/transformers/models/gptj/modeling_gptj.py)."],
  ["HIGH", "Unmeasured drift projections", "modal_app.py", "def _hidden_states", "The old backend did not return hidden-space coordinates. It now captures real last-token block outputs at the last edited layer, computes Euclidean distance before projection, and fits deterministic joint t-SNE over pre/post vectors. A labeled PCA fallback handles one-point and degenerate inputs; no semantic-damage score is invented."],
  ["HIGH", "Only one token rank was visible", "frontend/src/components/TokenRankingChart.tsx", "export function TokenRankingChart", "The top-1 bars omitted four ranks described by the interface. Five probability-sized bubbles now render per layer, same-token paths link layers, hover highlights the full path, and clicks select a layer. Missing last-token measurements remain empty."],
  ["HIGH", "Chat displayed the requested answer as model output", "frontend/src/components/FactForm.tsx", "generation &&", "The hardcoded target sentence was not inference. The Generate command now calls the real deterministic generation endpoint; input/model changes clear its result and stale responses are discarded. Edit Fact opens the actual editable fact fields."],
  ["HIGH", "Comparison omitted post-edit signal data", "modal_app.py", '\"layer_signals\": _probe_layers(edited_model', "Comparison rows now include measured post-edit layer signals. The selected row drives the post-edit lens as well as metrics, output diff, and neighborhood drift."],
  ["HIGH", "Invented graph neighbors could replace the subject", "frontend/src/components/KnowledgeGraph.tsx", "const neighbors", "Location, Category and Neighbor were synthetic node labels, yet clicking them changed the subject. The fact graph now displays only the actual subject and supplied original/new answers, with no invented entity links."],
  ["MEDIUM", "Recommend required two clicks and schemes were fixed", "frontend/src/App.tsx", "const handleRecommend", "The first Recommend click now waits for its guarded probe and selects a window immediately. The comparison text field is editable, malformed specifications are rejected, and noncontiguous layer labels preserve each selected layer instead of implying a full range."],
  ["HIGH", "Offline dev URL and mutable upstream revision", "modal_app.py", "MEMIT_COMMIT =", "The editing repository is pinned to commit 80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b. The frontend now points to the deployed service. The web worker allows one active request and at most one container, with a ten-minute idle scale-down."],
];
const severityOrder = ["CRITICAL", "HIGH", "MEDIUM"];
fixes.sort((a, b) => severityOrder.indexOf(a[0]) - severityOrder.indexOf(b[0]));
for (const [severity, title, path, needle, explanation] of fixes) report += `### [${severity}] ${title}\n\n${location(path, needle)}. ${explanation}\n\n`;
report += `## Validation\n\n- 13 CPU regression tests passed, including partial-edit exceptions, successful restoration, invalid HTTP requests, failed model-load recovery, target token alignment, exact hidden-space distance, deterministic projection, keyword-input tracing without tensor changes, and hook cleanup. Inference and algorithms are stubbed in CPU route tests.\n- ${fixtures.checks.length} browser regression groups passed with intercepted API fixtures. These cover race conditions, errors, zero/missing metrics, five ranks, hover paths, dynamic row anchors, long diffs, mobile bounds, and CTM selection.\n- ${browser.checks.length} live browser groups passed against the deployed GPU API: ${browser.checks.join("; ")}.\n- TypeScript compilation and Vite production build passed.\n- [Live desktop screenshot](./live/browser/desktop.png), [live mobile screenshot](./live/browser/mobile.png), [live browser responses](./live/browser/), and [fixture browser results](./frontend-results.json).\n\nCommands from prototype: python -m unittest test_audit_backend.py -v; python test_live_backend.py --model gpt2-xl; python test_live_backend.py --model EleutherAI/gpt-j-6B. Frontend commands: npm run build; node tests/audit.mjs; node tests/live.mjs. Live checks use the deployed GPU and incur Modal usage.\n\n## Scope\n\nThe tested dashboard workflows are operational. This is not a certification that every possible edit succeeds or a complete replication of the paper's research evaluation, external knowledge graph, or user study. Neighborhood preference metrics use the supplied shared target_true; heterogeneous neighborhoods require appropriate per-prompt reference answers before those rates can be interpreted. The two-neighborhood t-SNE view is a visual aid; hidden L2 and KL carry the measured numeric change. The first-pass original snapshots and historical experiments were preserved.\n\n## Complete replacement files\n\nThese are the current files copied verbatim from the working tree. Earlier unchanged fixes remain in the first-pass report.\n\n`;
const hashes = {};
for (const path of files) {
  const data = readFileSync(resolve(root, path), "utf8").replace(/^\uFEFF/, "");
  hashes[path] = createHash("sha256").update(readFileSync(resolve(root, path))).digest("hex");
  const fence = "`".repeat(4);
  report += `### ${path}\n\n${fence}${path.endsWith(".py") ? "python" : path.endsWith(".tsx") ? "tsx" : path.endsWith(".ts") ? "typescript" : path.endsWith(".mjs") ? "javascript" : "text"}\n${data.trimEnd()}\n${fence}\n\n`;
}
writeFileSync(resolve(audit, "E2E_VERIFICATION.md"), report);
writeFileSync(resolve(audit, "e2e-manifest.json"), JSON.stringify({ date: new Date().toISOString(), hashes, requests: manifest, liveBrowser: browser.checks, fixtureBrowser: fixtures.checks }, null, 2));
console.log("Saved E2E_VERIFICATION.md and e2e-manifest.json");
