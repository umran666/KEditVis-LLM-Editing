import { readFileSync, writeFileSync, readdirSync, existsSync } from "node:fs";
import { resolve, relative, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

const directory = dirname(fileURLToPath(import.meta.url));
const prototype = resolve(directory, "..");
const read = (path) => readFileSync(path, "utf8").replace(/^\uFEFF/, "").replaceAll("\r\n", "\n");
const f = (name) => `frontend/src/${name}`;
const c = (name) => f(`components/${name}.tsx`);
const issues = [
  {
    severity: "CRITICAL", title: "An exception inside MEMIT or ROME can contaminate the cached baseline",
    locations: [["modal_app.py", 1247, 1264], ["modal_app.py", 1322, 1345], ["modal_app.py", 559, 580], ["modal_app.py", 679, 703], ["modal_app.py", 802, 816]],
    cause: "Every caller enters its try/finally only after the upstream apply function returns. MEMIT temporarily writes each layer during its solve, and both algorithms insert updates before returning original weights. A later solve or insertion failure bypasses the caller's restoration completely. return_orig_weights=True cannot return anything after an exception.",
    fix: "Snapshot each configured rewrite matrix to independent CPU storage before entering upstream code. Restore on BaseException inside _apply_with_rollback, and return those snapshots to the existing evaluation finally blocks. All five application paths now use the wrapper.",
    files: ["modal_app.py"], test: "CPU tensors: inject partial mutations followed by RuntimeError and KeyboardInterrupt; assert every matrix equals its baseline. Real FastAPI routes with mocked inference also verify application failures and post-evaluation failures, followed by a successful request.",
    source: "[Upstream MEMIT application and solve](https://raw.githubusercontent.com/kmeng01/memit/main/memit/memit_main.py), [upstream ROME application](https://raw.githubusercontent.com/kmeng01/rome/main/rome/rome_main.py)."
  },
  {
    severity: "CRITICAL", title: "API failures silently become invented successful experiments",
    locations: [[f("api/client.ts"), 62, 191]],
    cause: "Every exported API function catches all errors, including HTTP 400/500, network failures and invalid JSON, then returns synthetic results without a provenance flag. Failed editing is therefore displayed as a measured success. Mock S values are also hardcoded independently of ES/PS/NS.",
    fix: "Remove unconditional synthetic fallbacks. Preserve HTTP validation messages and propagate network/JSON errors. Offline fixtures are isolated in browser tests and cannot become application results.",
    files: [f("api/client.ts")], test: "Intercept /edit with HTTP 500 and REAL_BACKEND_FAILURE; assert the exact error appears and no post-edit lens is created. No real API traffic is used in browser tests."
  },
  {
    severity: "CRITICAL", title: "Context changes admit stale results or permanently retain loading state",
    locations: [[f("App.tsx"), 37, 87], [f("App.tsx"), 119, 175]],
    cause: "Changing method clears results but does not invalidate activeRequestId; an old MEMIT response can repopulate ROME state. Model changes increment the same counter via health, so an older request's finally cannot clear loading and health never clears it either. nLayers remains the previous model's value until health completes. Fact changes do not clear evidence for the previous fact.",
    fix: "Invalidate operation IDs synchronously in model, method and fact handlers; clear loading and results; derive the two supported model dimensions immediately; give health its own counter. Tie elapsed-time cleanup to loading. Preserve operation/model guards on all responses.",
    files: [f("App.tsx")], test: "Browser tests hold an edit response, switch method/model, release it, and verify no result reappears, loading ends, and ROME has one selected layer. Four rapid model switches leave exactly 28 layer ticks."
  },
  {
    severity: "CRITICAL", title: "GPT-J cannot fit the configured float32 model on a T4",
    locations: [["modal_app.py", 1065, 1074], ["modal_app.py", 1113, 1115]],
    cause: "from_pretrained uses the float32 weights and then moves the entire GPT-J model to CUDA. Roughly 6 billion parameters require about 24 GB before activations and editing workspaces, exceeding the configured 16 GB T4. Changing only the model to float16 is not a valid repair: upstream ROME uses float32 inverse covariance matrices with model representations.",
    fix: "Keep explicit float32 and configure MODEL_GPU = A100-40GB for the four Modal workers that accept either model. This changes future deployment costs, including GPT-2 runs. No worker was launched or deployed during the audit.",
    files: ["modal_app.py"], test: "Verified source configuration and float32 selection in CPU tests. Full model loading, peak VRAM and editing on the A100 remain UNRUN.",
    source: "[ROME float32 covariance code](https://raw.githubusercontent.com/kmeng01/memit/main/rome/compute_u.py), [Modal GPU configuration](https://modal.com/docs/guide/gpu)."
  },
  {
    severity: "CRITICAL", title: "GPT-J hyperparameter filenames do not match upstream",
    locations: [["modal_app.py", 1169, 1180], ["modal_app.py", 1239, 1245], ["modal_app.py", 540, 540], ["modal_app.py", 668, 668], ["modal_app.py", 768, 768]],
    cause: "Interpolating EleutherAI/gpt-j-6B creates an extra directory in hparams/ROME/EleutherAI/gpt-j-6B.json. The checked upstream files are named EleutherAI_gpt-j-6B.json. The same path defect affects MEMIT and CLI entry points.",
    fix: "Use _hparams_path to replace model-name slashes with underscores at every load. ROME continues to receive a one-element list, not an integer; both official dataclasses define layers as List[int].",
    files: ["modal_app.py"], test: "CPU path tests cover both methods, and the real FastAPI dispatcher with stubbed loaders requests the exact GPT-J ROME path.",
    source: "[ROME GPT-J parameters](https://raw.githubusercontent.com/kmeng01/memit/main/hparams/ROME/EleutherAI_gpt-j-6B.json), [MEMIT GPT-J parameters](https://raw.githubusercontent.com/kmeng01/memit/main/hparams/MEMIT/EleutherAI_gpt-j-6B.json), [ROME schema](https://raw.githubusercontent.com/kmeng01/rome/main/rome/rome_hparams.py)."
  },
  {
    severity: "CRITICAL", title: "A failed model switch leaves the cache dictionary unusable",
    locations: [["modal_app.py", 1105, 1118]],
    cause: "The loader deletes model and tokenizer keys while retaining the old model_name. If loading the replacement fails, selecting the old model takes the cache-hit branch and indexes missing keys; selecting another model also accesses a deleted key. Subsequent requests fail until the worker restarts.",
    fix: "Reset model_name, model and tok to None together before releasing GPU allocations. Populate all three only after a successful load. Health can recover an empty cache using the requested or default model.",
    files: ["modal_app.py"], test: "Inject a GPT-J loader failure after GPT-2 was cached, then successfully reload GPT-2 and edit with GPT-J using mocked model loading."
  },
  {
    severity: "CRITICAL", title: "Unvalidated requests reach crashing or incorrectly labeled algorithm paths",
    locations: [["modal_app.py", 1124, 1181]],
    cause: "Empty MEMIT layer lists reach upstream layers[-1], empty new targets reach target_new['str'][0], malformed Python format fields raise during prompt formatting, and arbitrary method strings silently fall through to MEMIT while retaining the incorrect response label. ROME comparison expansion also repeats overlapping layers.",
    fix: "Validate model/method literals, strict integer layers and nonempty schemes, nonblank subjects/new targets, and exactly one plain subject placeholder. Normalize plain prompts consistently with the frontend. Preserve optional target_true by converting blank values to None. Reject multi-layer ROME edits before inference and deduplicate comparison schemes.",
    files: ["modal_app.py", c("FactForm"), c("PromptDetailCards"), f("schemes.ts"), f("App.tsx")], test: "FastAPI tests reject empty, negative, boolean and fractional layers, unknown methods, blank new targets and malformed placeholders without entering the edit algorithm. Overlapping ROME ranges execute once per distinct layer."
  },
  {
    severity: "HIGH", title: "Reported success metrics use accuracy, and incomplete scores look complete",
    locations: [["modal_app.py", 335, 339], ["modal_app.py", 365, 403]],
    cause: "ES/PS average target_new_correct and NS averages target_true_correct, which are greedy token accuracy checks. Upstream CounterFact success instead compares the two targets' NLLs; accuracy is reported separately. A third token can beat both answers even when the desired target beats the old target. _harmonic_mean silently drops missing metrics, so S=1 can appear when generalization/locality were not evaluated.",
    fix: "Compute ES/PS with new NLL < true NLL and NS with true NLL < new NLL. Keep greedy correctness in details. Return null for S unless all three rates are available; preserve zero if any measured rate is zero. Update shared types to allow null S.",
    files: ["modal_app.py", f("types.ts")], test: "A controlled case where neither target is greedy-correct but the intended likelihood comparisons pass now yields ES=PS=NS=S=1. Tests cover zero, null and the exact three-rate harmonic formula.",
    source: "[Official MEMIT success versus accuracy aggregation](https://raw.githubusercontent.com/kmeng01/memit/main/experiments/summarize.py)."
  },
  {
    severity: "HIGH", title: "Whitespace and token-boundary differences misalign target scoring",
    locations: [["modal_app.py", 290, 322]],
    cause: "The combined prompt uses raw target strings while independently tokenized target IDs use stripped strings. Prefix lengths are also computed separately, although tokenization can change at the concatenation boundary. Leading target spaces or trailing prefix spaces can therefore score different positions/tokens than the actual model input.",
    fix: "Normalize prefix/target spacing, tokenize each combined string once with offsets, and score the actual input IDs whose spans overlap the target. Retain attention masks and float32 log-softmax for stable scoring.",
    files: ["modal_app.py"], test: "A CPU predictor and tokenizer with actual character offsets exercise trailing/leading whitespace, unequal padded sequences and multiple-token targets; both targets are reconstructed and scored correctly. Full Hugging Face model integration remains unrun."
  },
  {
    severity: "HIGH", title: "ROME context templates leak across edits and model switches",
    locations: [["modal_app.py", 98, 113], ["modal_app.py", 1169, 1173]],
    cause: "ROME calls _seed_memit_rng, but that helper clears only MEMIT's global context cache. ROME keeps its own process-global cache, not keyed by model; a warm request can reuse templates sampled for another model and defeat per-edit reproducibility.",
    fix: "Clear both algorithm-specific context-template caches whenever editing is reseeded. Covariance caches retain their upstream model/layer keys.",
    files: ["modal_app.py"], test: "Seed both caches with old values, call the production helper, and assert both are None.",
    source: "[ROME context template cache](https://raw.githubusercontent.com/kmeng01/memit/main/rome/rome_main.py)."
  },
  {
    severity: "HIGH", title: "Sorted comparison rows, wires and diagnostics refer to different schemes",
    locations: [[c("WireframeLinker"), 56, 90], [c("SchemeComparisonTable"), 21, 22], [f("App.tsx"), 89, 96], [f("App.tsx"), 394, 397], [f("App.tsx"), 453, 468]],
    cause: "The linker uses the original request order with a hardcoded 36-pixel row height, omitting table headings, the baseline row and SVG scale factors. The table sorts by S. App's table selection changes only the key, while diagnostics always read schemes[0] and can combine its metrics with baseline generation. ROME responses contain singleton schemes while the linker still shows original multi-layer ranges. The damage fallback uses ||, which discards valid zero KL.",
    fix: "Share canonical layer keys and sorting, submit unique singleton ROME comparisons, build links from returned schemes, and measure actual keyed row centers plus layer ticks through screen CTMs. Resize/scroll observers remeasure geometry. Both selection directions update layers, and diagnostics read the selected scheme with explicit result ownership instead of truthiness fallbacks.",
    files: [f("App.tsx"), f("schemes.ts"), c("WireframeLinker"), c("SchemeComparisonTable")], test: "Browser tests deliberately reorder scores, increase a row to 95px, resize the viewport, and verify each connector endpoint within one screen pixel. Clicking rows and actual curve points synchronizes table selection, layers, generations and diffs. Zero KL is retained."
  },
  {
    severity: "HIGH", title: "Prompt cards report false passes and fabricated text",
    locations: [[c("PromptDetailCards"), 28, 35], [c("PromptDetailCards"), 45, 76]],
    cause: "The truthiness check maps ES=0 to the default true branch. Missing metrics also default to passes. Category-average PS/NS is applied to every prompt, hiding mixed outcomes. Neighborhood targets are hardcoded to Paris and missing generations are replaced by invented text.",
    fix: "Use explicit numeric checks and >0.5 only for measured category rates, show unknown when absent, derive each prompt's outcome from its own NLL detail, use targetTrue, and render only actual generation text. Category headers summarize aggregate rates without assigning that rate to every prompt.",
    files: [c("PromptDetailCards"), f("types.ts"), f("App.css")], test: "Browser checks cover ES=0, missing metrics, PS=0.5 with one passing and one failing prompt, London as the original target, and unavailable generation text."
  },
  {
    severity: "HIGH", title: "Drift evidence is invented, and marquee selection can include hidden points",
    locations: [[c("DriftScatterPlot"), 49, 96], [c("DriftScatterPlot"), 112, 159], [c("DriftScatterPlot"), 216, 253]],
    cause: "Neighborhood output changes, six-digit drift scores and scatter coordinates are constants. A threshold on aggregate neutral-corpus KL fabricates semantic bleed in the first neighborhood, which that scalar cannot establish. Pointer-up hit-tests all points, including hidden ones, and uses the previous React selectionBox rather than final pointer coordinates. A zero-hit selection is treated as no filter.",
    fix: "Return deterministic actual pre/post neighborhood generations and per-prompt KL from the backend. Remove invented outputs, scores and coordinates. Render optional externally supplied measured projections; show Projection unavailable when absent. Use pointer capture, final event coordinates transformed by inverse screen CTM, visible-point hit testing, and distinct null versus empty selections.",
    files: [c("DriftScatterPlot"), "modal_app.py", f("types.ts"), f("App.tsx")], test: "CPU tests preserve actual unchanged neighborhood text and compute known KL values. Browser tests inject clearly identified projection fixtures into a 200x130 viewBox displayed in a 200x300 element, select the correct linked row, and verify only visible post points are selected. Production backend projection generation is still unavailable."
  },
  {
    severity: "HIGH", title: "The diff silently drops long-output changes and collapses paragraph whitespace",
    locations: [[c("DiffViewer"), 20, 23], [f("App.css"), 800, 809]],
    cause: "Both token arrays are sliced to 300 entries and their tails are discarded. Differences after the cap disappear entirely. Captured whitespace is rendered under normal CSS whitespace handling, collapsing line breaks and repeated spaces; punctuation remains attached to words.",
    fix: "Tokenize words, whitespace and punctuation while bounding the LCS matrix to 300 tokens per side. Preserve the unprocessed tails as a coarse complete replacement or unchanged tail. Render with pre-wrap and overflow-wrap. This bounds quadratic work without claiming sub-millisecond timing.",
    files: [c("DiffViewer"), f("App.css")], test: "Reconstruct both original strings exactly from diff chunks for empty, one-sided, punctuation, multiline and 200KB-tail cases. Browser regression timing is recorded in frontend-results.json, with a 500ms regression ceiling."
  },
  {
    severity: "HIGH", title: "Empty or disjoint D3 data leaves previous-model geometry visible",
    locations: [[c("TokenRankingChart"), 27, 39], [c("CosineSimilarityCompareChart"), 23, 45]],
    cause: "TokenRankingChart exits before clearing its persistent SVG on an empty signal set. The compare chart also exits when nonempty pre/post arrays have no shared layer IDs, leaving old bars on screen. The existing empty-domain guards already prevent the specifically alleged scaleBand([]) NaN scenario; stale geometry is the demonstrated defect.",
    fix: "Clear SVG children before early returns, filter nonfinite paired values, and bound the cosine activity to [0,1]. Preserve normal empty-state handling.",
    files: [c("TokenRankingChart"), c("CosineSimilarityCompareChart")], test: "Render valid data, then disjoint layer sets, then empty data; assert old bars and token-chart SVG children are gone. Rapid model-switch browser tests also confirm old token bars disappear."
  },
  {
    severity: "MEDIUM", title: "Narrow layouts overflow and route connectors across table text",
    locations: [[f("App.css"), 41, 54], [f("App.css"), 719, 723], [f("App.css"), 764, 769], [f("App.css"), 1031, 1041]],
    cause: "A fixed-height nonwrapping toolbar extends past narrow viewports. The four prompt columns and two diagnostic columns remain compressed. The <=1100px breakpoint stacks linker and table vertically, invalidating the intended side-by-side relationship and making actual connector paths cross table text.",
    fix: "Wrap toolbar controls, allow constrained grid children to shrink, place the comparison workspace in its own horizontal scroll area with adjacent linker/table columns, and stack prompt and diagnostic panels on small screens.",
    files: [f("App.css")], test: "Reviewed desktop and 390px-wide screenshots; browser assertion verifies document width never exceeds the mobile viewport."
  }
];

function walk(path) {
  return readdirSync(path, { withFileTypes: true }).flatMap((entry) => entry.isDirectory() ? walk(resolve(path, entry.name)) : [resolve(path, entry.name)]);
}
const all = [resolve(prototype, "modal_app.py"), ...walk(resolve(prototype, "frontend/src"))];
const changed = all.filter((path) => {
  const previous = resolve(directory, "before", relative(prototype, path));
  return !existsSync(previous) || read(path) !== read(previous);
});
const changedNames = changed.map((path) => relative(prototype, path).replaceAll("\\", "/"));
const codeAnchors = new Map(changedNames.map((name, index) => [name, `replacement-${index + 1}`]));
const counts = issues.reduce((counts, issue) => ({ ...counts, [issue.severity]: (counts[issue.severity] ?? 0) + 1 }), {});
const frontend = JSON.parse(read(resolve(directory, "frontend-results.json")));
let report = `# KEditVis audit and applied fixes\n\nDate: 2026-09-09. Workspace: C:/Users/shaik/Research/LLM Editing.\n\n${issues.length} verified source findings: ${counts.CRITICAL} CRITICAL, ${counts.HIGH} HIGH, ${counts.MEDIUM} MEDIUM. Fixes are applied locally across ${changed.length} production files. The complete replacement code appears in the appendix, copied directly from the tested working files.\n\nOriginal locations below refer to the preserved pre-audit files under [before](./before/), not the shifted line numbers after fixes. Each link opens the original snapshot at its first cited line.\n\n`;
for (const [index, issue] of issues.entries()) {
  report += `## F${String(index + 1).padStart(2, "0")} [${issue.severity}] ${issue.title}\n\n`;
  for (const [file, start, end] of issue.locations) {
    const path = resolve(directory, "before", file).replaceAll("\\", "/");
    const lines = read(path).trimEnd().split("\n");
    if (start < 1 || end > lines.length || !lines[start - 1].trim()) throw new Error(`Invalid location ${file}:${start}-${end}`);
    report += `- [${file}](${encodeURI(path)}:${start}), lines ${start}-${end}.\n`;
  }
  report += `\n**Root cause:** ${issue.cause}\n\n**Applied fix:** ${issue.fix}\n\n**Verification:** ${issue.test}\n\n**Complete replacement code:** ${issue.files.map((file) => `[${file}](#${codeAnchors.get(file)})`).join(", ")}. Shared dependent files are included in the same appendix.\n\n`;
  if (issue.source) report += `**Primary-source check:** ${issue.source}\n\n`;
}
report += `## Verified boundaries and remaining limitations\n\n- The original inverse-screen-CTM formula was correct for SVG letterboxing. The changes address event finalization, visibility filtering and data provenance, not a replacement affine formula.\n- No reproducible out-of-bounds D3 crash was found for the supported 48/28-layer models. State invalidation, stale SVGs and delayed layer metadata were the concrete failures.\n- Existing post-evaluation finally blocks and _restore_weights perform in-place matrix restoration correctly after a successful apply. The missing protection was before apply returned. The Modal max_inputs=1 decorator remains present; distributed GPU concurrency was not executed locally.\n- Both upstream algorithms use list-valued layers and select matrix orientation based on actual weight shape. ROME receives one element under this application's policy. Upstream MEMIT can process noncontiguous lists; this audit does not turn the local contiguous-window heuristic into a mathematical requirement.\n- ES/PS/NS now use CounterFact target preference. They do not prove the desired answer wins over every vocabulary item. S is null if any category is missing. The current single target_true applies to every neighborhood prompt, so arbitrary heterogeneous neighborhoods still require a richer reference-answer schema.\n- The drift backend returns measured text and KL, but does not produce embedding projections. The component supports supplied coordinates and shows an unavailable state otherwise. No semantic-bleed label is inferred from neutral-reference KL.\n- The existing token ranking component is a top-1 bar chart, not the five-rank bubble plot described in the brief. The existing knowledge graph and chat preview remain prototype features. This audit does not certify complete paper replication.\n- GPU load/edit runs, Modal image construction, full GPT-J/GPT-2 inference and numerical edit quality were not executed. CPU tests use real tensors and actual route/function code with model/algorithm stubs. The changed GPU configuration increases future deployment cost. Nothing was deployed.\n- MEMIT_COMMIT remains main; this report verified current upstream source contracts, not a fixed immutable upstream build. The saved historical JSON experiments were not rewritten or treated as fresh verification.\n\n## Validation\n\n- Backend: 11 unittest cases on CPU, including actual FastAPI route validation with mocked model/algorithm dependencies. Command: python -m unittest prototype/test_audit_backend.py -v.\n- Frontend: ${frontend.checks.length} browser regression checks against Vite with intercepted API fixtures. Command from prototype/frontend: node tests/audit.mjs.\n- Build: npm run build (TypeScript and Vite).\n- Syntax: python -m py_compile prototype/modal_app.py.\n- [Browser evidence](./frontend-results.json), [backend output](./backend-results.txt), [build output](./build-results.txt), [desktop screenshot](./audit-desktop.png), [mobile screenshot](./audit-mobile.png). Screenshots contain labeled test fixture values, not real model outputs.\n- Local dev server: [http://127.0.0.1:5187](http://127.0.0.1:5187). It uses the existing API configuration; the backend source changes are not deployed to that API.\n\n## Complete replacement files\n\nThese blocks are full working files, including imports and supporting definitions. Replace the corresponding file as a unit; the shared types and scheme helpers are part of the same change. Original snapshots remain available under before/.\n\n`;
for (const [index, path] of changed.entries()) {
  const name = changedNames[index];
  const content = read(path);
  const language = name.endsWith(".py") ? "python" : name.endsWith(".tsx") ? "tsx" : name.endsWith(".css") ? "css" : "typescript";
  report += `<a id="${codeAnchors.get(name)}"></a>\n\n### ${name}\n\nCurrent file: [${name}](${encodeURI(path.replaceAll("\\", "/"))}).\n\nSHA-256: ${createHash("sha256").update(readFileSync(path)).digest("hex")}\n\n\`\`\`${language}\n${content.trimEnd()}\n\`\`\`\n\n`;
}
writeFileSync(resolve(directory, "KEDITVIS_AUDIT.md"), report);
writeFileSync(resolve(directory, "manifest.json"), JSON.stringify({ counts, changedFiles: changedNames, findings: issues.map(({ title, severity, locations }) => ({ title, severity, locations })) }, null, 2));
console.log(JSON.stringify({ findings: issues.length, counts, changedFiles: changed.length, reportBytes: Buffer.byteLength(report) }));
