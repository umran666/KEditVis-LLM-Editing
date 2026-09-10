# KEditVis audit and applied fixes

Historical first-pass snapshot. The later [end-to-end verification report](./E2E_VERIFICATION.md) supersedes the deployment, GPU, projection, token-chart, and chat limitations below and contains the current replacement files.

Date: 2026-09-09. Workspace: C:/Users/shaik/Research/LLM Editing.

16 verified source findings: 7 CRITICAL, 8 HIGH, 1 MEDIUM. Fixes are applied locally across 14 production files. The complete replacement code appears in the appendix, copied directly from the tested working files.

Original locations below refer to the preserved pre-audit files under [before](./before/), not the shifted line numbers after fixes. Each link opens the original snapshot at its first cited line.

## F01 [CRITICAL] An exception inside MEMIT or ROME can contaminate the cached baseline

- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:1247), lines 1247-1264.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:1322), lines 1322-1345.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:559), lines 559-580.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:679), lines 679-703.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:802), lines 802-816.

**Root cause:** Every caller enters its try/finally only after the upstream apply function returns. MEMIT temporarily writes each layer during its solve, and both algorithms insert updates before returning original weights. A later solve or insertion failure bypasses the caller's restoration completely. return_orig_weights=True cannot return anything after an exception.

**Applied fix:** Snapshot each configured rewrite matrix to independent CPU storage before entering upstream code. Restore on BaseException inside _apply_with_rollback, and return those snapshots to the existing evaluation finally blocks. All five application paths now use the wrapper.

**Verification:** CPU tensors: inject partial mutations followed by RuntimeError and KeyboardInterrupt; assert every matrix equals its baseline. Real FastAPI routes with mocked inference also verify application failures and post-evaluation failures, followed by a successful request.

**Complete replacement code:** [modal_app.py](#replacement-1). Shared dependent files are included in the same appendix.

**Primary-source check:** [Upstream MEMIT application and solve](https://raw.githubusercontent.com/kmeng01/memit/main/memit/memit_main.py), [upstream ROME application](https://raw.githubusercontent.com/kmeng01/rome/main/rome/rome_main.py).

## F02 [CRITICAL] API failures silently become invented successful experiments

- [frontend/src/api/client.ts](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/api/client.ts:62), lines 62-191.

**Root cause:** Every exported API function catches all errors, including HTTP 400/500, network failures and invalid JSON, then returns synthetic results without a provenance flag. Failed editing is therefore displayed as a measured success. Mock S values are also hardcoded independently of ES/PS/NS.

**Applied fix:** Remove unconditional synthetic fallbacks. Preserve HTTP validation messages and propagate network/JSON errors. Offline fixtures are isolated in browser tests and cannot become application results.

**Verification:** Intercept /edit with HTTP 500 and REAL_BACKEND_FAILURE; assert the exact error appears and no post-edit lens is created. No real API traffic is used in browser tests.

**Complete replacement code:** [frontend/src/api/client.ts](#replacement-2). Shared dependent files are included in the same appendix.

## F03 [CRITICAL] Context changes admit stale results or permanently retain loading state

- [frontend/src/App.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/App.tsx:37), lines 37-87.
- [frontend/src/App.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/App.tsx:119), lines 119-175.

**Root cause:** Changing method clears results but does not invalidate activeRequestId; an old MEMIT response can repopulate ROME state. Model changes increment the same counter via health, so an older request's finally cannot clear loading and health never clears it either. nLayers remains the previous model's value until health completes. Fact changes do not clear evidence for the previous fact.

**Applied fix:** Invalidate operation IDs synchronously in model, method and fact handlers; clear loading and results; derive the two supported model dimensions immediately; give health its own counter. Tie elapsed-time cleanup to loading. Preserve operation/model guards on all responses.

**Verification:** Browser tests hold an edit response, switch method/model, release it, and verify no result reappears, loading ends, and ROME has one selected layer. Four rapid model switches leave exactly 28 layer ticks.

**Complete replacement code:** [frontend/src/App.tsx](#replacement-4). Shared dependent files are included in the same appendix.

## F04 [CRITICAL] GPT-J cannot fit the configured float32 model on a T4

- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:1065), lines 1065-1074.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:1113), lines 1113-1115.

**Root cause:** from_pretrained uses the float32 weights and then moves the entire GPT-J model to CUDA. Roughly 6 billion parameters require about 24 GB before activations and editing workspaces, exceeding the configured 16 GB T4. Changing only the model to float16 is not a valid repair: upstream ROME uses float32 inverse covariance matrices with model representations.

**Applied fix:** Keep explicit float32 and configure MODEL_GPU = A100-40GB for the four Modal workers that accept either model. This changes future deployment costs, including GPT-2 runs. No worker was launched or deployed during the audit.

**Verification:** Verified source configuration and float32 selection in CPU tests. Full model loading, peak VRAM and editing on the A100 remain UNRUN.

**Complete replacement code:** [modal_app.py](#replacement-1). Shared dependent files are included in the same appendix.

**Primary-source check:** [ROME float32 covariance code](https://raw.githubusercontent.com/kmeng01/memit/main/rome/compute_u.py), [Modal GPU configuration](https://modal.com/docs/guide/gpu).

## F05 [CRITICAL] GPT-J hyperparameter filenames do not match upstream

- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:1169), lines 1169-1180.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:1239), lines 1239-1245.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:540), lines 540-540.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:668), lines 668-668.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:768), lines 768-768.

**Root cause:** Interpolating EleutherAI/gpt-j-6B creates an extra directory in hparams/ROME/EleutherAI/gpt-j-6B.json. The checked upstream files are named EleutherAI_gpt-j-6B.json. The same path defect affects MEMIT and CLI entry points.

**Applied fix:** Use _hparams_path to replace model-name slashes with underscores at every load. ROME continues to receive a one-element list, not an integer; both official dataclasses define layers as List[int].

**Verification:** CPU path tests cover both methods, and the real FastAPI dispatcher with stubbed loaders requests the exact GPT-J ROME path.

**Complete replacement code:** [modal_app.py](#replacement-1). Shared dependent files are included in the same appendix.

**Primary-source check:** [ROME GPT-J parameters](https://raw.githubusercontent.com/kmeng01/memit/main/hparams/ROME/EleutherAI_gpt-j-6B.json), [MEMIT GPT-J parameters](https://raw.githubusercontent.com/kmeng01/memit/main/hparams/MEMIT/EleutherAI_gpt-j-6B.json), [ROME schema](https://raw.githubusercontent.com/kmeng01/rome/main/rome/rome_hparams.py).

## F06 [CRITICAL] A failed model switch leaves the cache dictionary unusable

- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:1105), lines 1105-1118.

**Root cause:** The loader deletes model and tokenizer keys while retaining the old model_name. If loading the replacement fails, selecting the old model takes the cache-hit branch and indexes missing keys; selecting another model also accesses a deleted key. Subsequent requests fail until the worker restarts.

**Applied fix:** Reset model_name, model and tok to None together before releasing GPU allocations. Populate all three only after a successful load. Health can recover an empty cache using the requested or default model.

**Verification:** Inject a GPT-J loader failure after GPT-2 was cached, then successfully reload GPT-2 and edit with GPT-J using mocked model loading.

**Complete replacement code:** [modal_app.py](#replacement-1). Shared dependent files are included in the same appendix.

## F07 [CRITICAL] Unvalidated requests reach crashing or incorrectly labeled algorithm paths

- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:1124), lines 1124-1181.

**Root cause:** Empty MEMIT layer lists reach upstream layers[-1], empty new targets reach target_new['str'][0], malformed Python format fields raise during prompt formatting, and arbitrary method strings silently fall through to MEMIT while retaining the incorrect response label. ROME comparison expansion also repeats overlapping layers.

**Applied fix:** Validate model/method literals, strict integer layers and nonempty schemes, nonblank subjects/new targets, and exactly one plain subject placeholder. Normalize plain prompts consistently with the frontend. Preserve optional target_true by converting blank values to None. Reject multi-layer ROME edits before inference and deduplicate comparison schemes.

**Verification:** FastAPI tests reject empty, negative, boolean and fractional layers, unknown methods, blank new targets and malformed placeholders without entering the edit algorithm. Overlapping ROME ranges execute once per distinct layer.

**Complete replacement code:** [modal_app.py](#replacement-1), [frontend/src/components/FactForm.tsx](#replacement-8), [frontend/src/components/PromptDetailCards.tsx](#replacement-9), [frontend/src/schemes.ts](#replacement-13), [frontend/src/App.tsx](#replacement-4). Shared dependent files are included in the same appendix.

## F08 [HIGH] Reported success metrics use accuracy, and incomplete scores look complete

- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:335), lines 335-339.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:365), lines 365-403.

**Root cause:** ES/PS average target_new_correct and NS averages target_true_correct, which are greedy token accuracy checks. Upstream CounterFact success instead compares the two targets' NLLs; accuracy is reported separately. A third token can beat both answers even when the desired target beats the old target. _harmonic_mean silently drops missing metrics, so S=1 can appear when generalization/locality were not evaluated.

**Applied fix:** Compute ES/PS with new NLL < true NLL and NS with true NLL < new NLL. Keep greedy correctness in details. Return null for S unless all three rates are available; preserve zero if any measured rate is zero. Update shared types to allow null S.

**Verification:** A controlled case where neither target is greedy-correct but the intended likelihood comparisons pass now yields ES=PS=NS=S=1. Tests cover zero, null and the exact three-rate harmonic formula.

**Complete replacement code:** [modal_app.py](#replacement-1), [frontend/src/types.ts](#replacement-14). Shared dependent files are included in the same appendix.

**Primary-source check:** [Official MEMIT success versus accuracy aggregation](https://raw.githubusercontent.com/kmeng01/memit/main/experiments/summarize.py).

## F09 [HIGH] Whitespace and token-boundary differences misalign target scoring

- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:290), lines 290-322.

**Root cause:** The combined prompt uses raw target strings while independently tokenized target IDs use stripped strings. Prefix lengths are also computed separately, although tokenization can change at the concatenation boundary. Leading target spaces or trailing prefix spaces can therefore score different positions/tokens than the actual model input.

**Applied fix:** Normalize prefix/target spacing, tokenize each combined string once with offsets, and score the actual input IDs whose spans overlap the target. Retain attention masks and float32 log-softmax for stable scoring.

**Verification:** A CPU predictor and tokenizer with actual character offsets exercise trailing/leading whitespace, unequal padded sequences and multiple-token targets; both targets are reconstructed and scored correctly. Full Hugging Face model integration remains unrun.

**Complete replacement code:** [modal_app.py](#replacement-1). Shared dependent files are included in the same appendix.

## F10 [HIGH] ROME context templates leak across edits and model switches

- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:98), lines 98-113.
- [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/modal_app.py:1169), lines 1169-1173.

**Root cause:** ROME calls _seed_memit_rng, but that helper clears only MEMIT's global context cache. ROME keeps its own process-global cache, not keyed by model; a warm request can reuse templates sampled for another model and defeat per-edit reproducibility.

**Applied fix:** Clear both algorithm-specific context-template caches whenever editing is reseeded. Covariance caches retain their upstream model/layer keys.

**Verification:** Seed both caches with old values, call the production helper, and assert both are None.

**Complete replacement code:** [modal_app.py](#replacement-1). Shared dependent files are included in the same appendix.

**Primary-source check:** [ROME context template cache](https://raw.githubusercontent.com/kmeng01/memit/main/rome/rome_main.py).

## F11 [HIGH] Sorted comparison rows, wires and diagnostics refer to different schemes

- [frontend/src/components/WireframeLinker.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/components/WireframeLinker.tsx:56), lines 56-90.
- [frontend/src/components/SchemeComparisonTable.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/components/SchemeComparisonTable.tsx:21), lines 21-22.
- [frontend/src/App.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/App.tsx:89), lines 89-96.
- [frontend/src/App.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/App.tsx:394), lines 394-397.
- [frontend/src/App.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/App.tsx:453), lines 453-468.

**Root cause:** The linker uses the original request order with a hardcoded 36-pixel row height, omitting table headings, the baseline row and SVG scale factors. The table sorts by S. App's table selection changes only the key, while diagnostics always read schemes[0] and can combine its metrics with baseline generation. ROME responses contain singleton schemes while the linker still shows original multi-layer ranges. The damage fallback uses ||, which discards valid zero KL.

**Applied fix:** Share canonical layer keys and sorting, submit unique singleton ROME comparisons, build links from returned schemes, and measure actual keyed row centers plus layer ticks through screen CTMs. Resize/scroll observers remeasure geometry. Both selection directions update layers, and diagnostics read the selected scheme with explicit result ownership instead of truthiness fallbacks.

**Verification:** Browser tests deliberately reorder scores, increase a row to 95px, resize the viewport, and verify each connector endpoint within one screen pixel. Clicking rows and actual curve points synchronizes table selection, layers, generations and diffs. Zero KL is retained.

**Complete replacement code:** [frontend/src/App.tsx](#replacement-4), [frontend/src/schemes.ts](#replacement-13), [frontend/src/components/WireframeLinker.tsx](#replacement-12), [frontend/src/components/SchemeComparisonTable.tsx](#replacement-10). Shared dependent files are included in the same appendix.

## F12 [HIGH] Prompt cards report false passes and fabricated text

- [frontend/src/components/PromptDetailCards.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/components/PromptDetailCards.tsx:28), lines 28-35.
- [frontend/src/components/PromptDetailCards.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/components/PromptDetailCards.tsx:45), lines 45-76.

**Root cause:** The truthiness check maps ES=0 to the default true branch. Missing metrics also default to passes. Category-average PS/NS is applied to every prompt, hiding mixed outcomes. Neighborhood targets are hardcoded to Paris and missing generations are replaced by invented text.

**Applied fix:** Use explicit numeric checks and >0.5 only for measured category rates, show unknown when absent, derive each prompt's outcome from its own NLL detail, use targetTrue, and render only actual generation text. Category headers summarize aggregate rates without assigning that rate to every prompt.

**Verification:** Browser checks cover ES=0, missing metrics, PS=0.5 with one passing and one failing prompt, London as the original target, and unavailable generation text.

**Complete replacement code:** [frontend/src/components/PromptDetailCards.tsx](#replacement-9), [frontend/src/types.ts](#replacement-14), [frontend/src/App.css](#replacement-3). Shared dependent files are included in the same appendix.

## F13 [HIGH] Drift evidence is invented, and marquee selection can include hidden points

- [frontend/src/components/DriftScatterPlot.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/components/DriftScatterPlot.tsx:49), lines 49-96.
- [frontend/src/components/DriftScatterPlot.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/components/DriftScatterPlot.tsx:112), lines 112-159.
- [frontend/src/components/DriftScatterPlot.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/components/DriftScatterPlot.tsx:216), lines 216-253.

**Root cause:** Neighborhood output changes, six-digit drift scores and scatter coordinates are constants. A threshold on aggregate neutral-corpus KL fabricates semantic bleed in the first neighborhood, which that scalar cannot establish. Pointer-up hit-tests all points, including hidden ones, and uses the previous React selectionBox rather than final pointer coordinates. A zero-hit selection is treated as no filter.

**Applied fix:** Return deterministic actual pre/post neighborhood generations and per-prompt KL from the backend. Remove invented outputs, scores and coordinates. Render optional externally supplied measured projections; show Projection unavailable when absent. Use pointer capture, final event coordinates transformed by inverse screen CTM, visible-point hit testing, and distinct null versus empty selections.

**Verification:** CPU tests preserve actual unchanged neighborhood text and compute known KL values. Browser tests inject clearly identified projection fixtures into a 200x130 viewBox displayed in a 200x300 element, select the correct linked row, and verify only visible post points are selected. Production backend projection generation is still unavailable.

**Complete replacement code:** [frontend/src/components/DriftScatterPlot.tsx](#replacement-7), [modal_app.py](#replacement-1), [frontend/src/types.ts](#replacement-14), [frontend/src/App.tsx](#replacement-4). Shared dependent files are included in the same appendix.

## F14 [HIGH] The diff silently drops long-output changes and collapses paragraph whitespace

- [frontend/src/components/DiffViewer.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/components/DiffViewer.tsx:20), lines 20-23.
- [frontend/src/App.css](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/App.css:800), lines 800-809.

**Root cause:** Both token arrays are sliced to 300 entries and their tails are discarded. Differences after the cap disappear entirely. Captured whitespace is rendered under normal CSS whitespace handling, collapsing line breaks and repeated spaces; punctuation remains attached to words.

**Applied fix:** Tokenize words, whitespace and punctuation while bounding the LCS matrix to 300 tokens per side. Preserve the unprocessed tails as a coarse complete replacement or unchanged tail. Render with pre-wrap and overflow-wrap. This bounds quadratic work without claiming sub-millisecond timing.

**Verification:** Reconstruct both original strings exactly from diff chunks for empty, one-sided, punctuation, multiline and 200KB-tail cases. Browser regression timing is recorded in frontend-results.json, with a 500ms regression ceiling.

**Complete replacement code:** [frontend/src/components/DiffViewer.tsx](#replacement-6), [frontend/src/App.css](#replacement-3). Shared dependent files are included in the same appendix.

## F15 [HIGH] Empty or disjoint D3 data leaves previous-model geometry visible

- [frontend/src/components/TokenRankingChart.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/components/TokenRankingChart.tsx:27), lines 27-39.
- [frontend/src/components/CosineSimilarityCompareChart.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/components/CosineSimilarityCompareChart.tsx:23), lines 23-45.

**Root cause:** TokenRankingChart exits before clearing its persistent SVG on an empty signal set. The compare chart also exits when nonempty pre/post arrays have no shared layer IDs, leaving old bars on screen. The existing empty-domain guards already prevent the specifically alleged scaleBand([]) NaN scenario; stale geometry is the demonstrated defect.

**Applied fix:** Clear SVG children before early returns, filter nonfinite paired values, and bound the cosine activity to [0,1]. Preserve normal empty-state handling.

**Verification:** Render valid data, then disjoint layer sets, then empty data; assert old bars and token-chart SVG children are gone. Rapid model-switch browser tests also confirm old token bars disappear.

**Complete replacement code:** [frontend/src/components/TokenRankingChart.tsx](#replacement-11), [frontend/src/components/CosineSimilarityCompareChart.tsx](#replacement-5). Shared dependent files are included in the same appendix.

## F16 [MEDIUM] Narrow layouts overflow and route connectors across table text

- [frontend/src/App.css](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/App.css:41), lines 41-54.
- [frontend/src/App.css](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/App.css:719), lines 719-723.
- [frontend/src/App.css](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/App.css:764), lines 764-769.
- [frontend/src/App.css](C:/Users/shaik/Research/LLM%20Editing/prototype/audit/before/frontend/src/App.css:1031), lines 1031-1041.

**Root cause:** A fixed-height nonwrapping toolbar extends past narrow viewports. The four prompt columns and two diagnostic columns remain compressed. The <=1100px breakpoint stacks linker and table vertically, invalidating the intended side-by-side relationship and making actual connector paths cross table text.

**Applied fix:** Wrap toolbar controls, allow constrained grid children to shrink, place the comparison workspace in its own horizontal scroll area with adjacent linker/table columns, and stack prompt and diagnostic panels on small screens.

**Verification:** Reviewed desktop and 390px-wide screenshots; browser assertion verifies document width never exceeds the mobile viewport.

**Complete replacement code:** [frontend/src/App.css](#replacement-3). Shared dependent files are included in the same appendix.

## Verified boundaries and remaining limitations

- The original inverse-screen-CTM formula was correct for SVG letterboxing. The changes address event finalization, visibility filtering and data provenance, not a replacement affine formula.
- No reproducible out-of-bounds D3 crash was found for the supported 48/28-layer models. State invalidation, stale SVGs and delayed layer metadata were the concrete failures.
- Existing post-evaluation finally blocks and _restore_weights perform in-place matrix restoration correctly after a successful apply. The missing protection was before apply returned. The Modal max_inputs=1 decorator remains present; distributed GPU concurrency was not executed locally.
- Both upstream algorithms use list-valued layers and select matrix orientation based on actual weight shape. ROME receives one element under this application's policy. Upstream MEMIT can process noncontiguous lists; this audit does not turn the local contiguous-window heuristic into a mathematical requirement.
- ES/PS/NS now use CounterFact target preference. They do not prove the desired answer wins over every vocabulary item. S is null if any category is missing. The current single target_true applies to every neighborhood prompt, so arbitrary heterogeneous neighborhoods still require a richer reference-answer schema.
- The drift backend returns measured text and KL, but does not produce embedding projections. The component supports supplied coordinates and shows an unavailable state otherwise. No semantic-bleed label is inferred from neutral-reference KL.
- The existing token ranking component is a top-1 bar chart, not the five-rank bubble plot described in the brief. The existing knowledge graph and chat preview remain prototype features. This audit does not certify complete paper replication.
- GPU load/edit runs, Modal image construction, full GPT-J/GPT-2 inference and numerical edit quality were not executed. CPU tests use real tensors and actual route/function code with model/algorithm stubs. The changed GPU configuration increases future deployment cost. Nothing was deployed.
- MEMIT_COMMIT remains main; this report verified current upstream source contracts, not a fixed immutable upstream build. The saved historical JSON experiments were not rewritten or treated as fresh verification.

## Validation

- Backend: 11 unittest cases on CPU, including actual FastAPI route validation with mocked model/algorithm dependencies. Command: python -m unittest prototype/test_audit_backend.py -v.
- Frontend: 13 browser regression checks against Vite with intercepted API fixtures. Command from prototype/frontend: node tests/audit.mjs.
- Build: npm run build (TypeScript and Vite).
- Syntax: python -m py_compile prototype/modal_app.py.
- [Browser evidence](./frontend-results.json), [backend output](./backend-results.txt), [build output](./build-results.txt), [desktop screenshot](./audit-desktop.png), [mobile screenshot](./audit-mobile.png). Screenshots contain labeled test fixture values, not real model outputs.
- Local dev server: [http://127.0.0.1:5187](http://127.0.0.1:5187). It uses the existing API configuration; the backend source changes are not deployed to that API.

## Complete replacement files

These blocks are full working files, including imports and supporting definitions. Replace the corresponding file as a unit; the shared types and scheme helpers are part of the same change. Original snapshots remain available under before/.

<a id="replacement-1"></a>

### modal_app.py

Current file: [modal_app.py](C:/Users/shaik/Research/LLM%20Editing/prototype/modal_app.py).

SHA-256: b160f5b0401eeb305b76a153a789ec26e3b2ee7afd996244307ee52b84ec3149

```python
"""
Runs a real MEMIT edit on GPT-2 XL on a Modal GPU, and extracts the same
KEditVis-style layer signals (cosine similarity + logit-lens token ranking)
before and after the edit, plus before/after text generations.

This wraps the original kmeng01/memit implementation unmodified rather than
reimplementing the MEMIT math, since its hyperparameters and precomputed
covariance statistics for gpt2-xl are already published and tested.

Setup (one-time):
    pip install modal
    modal setup                    # authenticates this machine with Modal

Run (uses MEMIT's default hardcoded layer preset for gpt2-xl, [13-17]):
    modal run modal_app.py \
        --prompt "{} is located in the city of" \
        --subject "Eiffel Tower" \
        --target "Rome"

Run with a user-specified layer range (the actual capstone contribution --
human-in-the-loop layer selection instead of MEMIT's static preset):
    modal run modal_app.py \
        --prompt "{} is located in the city of" \
        --subject "Eiffel Tower" \
        --target "Rome" \
        --layers "8-12"

    # or an explicit, non-contiguous list:
    modal run modal_app.py ... --layers "6,9,12"

Run with ES/PS/NS/S metric computation (quantitative scheme comparison,
adapted from MEMIT's own CounterFact evaluation formulas). Requires
--target_true (the original correct answer) so locality/neighborhood
checks have something to compare against:
    modal run modal_app.py \
        --prompt "{} is located in the city of" \
        --subject "Eiffel Tower" \
        --target "Rome" \
        --layers "8-12" \
        --target_true "Paris" \
        --paraphrase_prompts "The Eiffel Tower is located in the city of;You can find the Eiffel Tower in the city of" \
        --neighborhood_prompts "The Louvre Museum is located in the city of;Notre-Dame Cathedral is located in the city of"

    (--target_true, --paraphrase_prompts, --neighborhood_prompts already
    default to the values above, so metrics are computed automatically
    for the default Eiffel Tower/Rome demo with no extra flags needed.)

Cost note: the first run per model will download gpt2-xl (~6GB) and MEMIT's
precomputed covariance statistics for the edited layers into persistent
Modal Volumes, so subsequent runs are much faster/cheaper. A single edit
plus signal extraction on a T4 typically takes 2-5 minutes.
"""

import json

import modal

app = modal.App("keditvis-memit")

hf_cache_vol = modal.Volume.from_name("keditvis-hf-cache", create_if_missing=True)
memit_data_vol = modal.Volume.from_name("keditvis-memit-data", create_if_missing=True)

MEMIT_COMMIT = "main"  # pin to a specific commit SHA once you've validated the setup

# One warm worker serves both models. GPT-J in the upstream float32 editing
# routines needs more than a T4's 16 GB; this also accommodates covariance solves.
MODEL_GPU = "A100-40GB"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.42.4",
        "datasets==2.20.0",
        "numpy==1.26.4",
        "pyyaml==6.0.1",
        "tqdm==4.66.4",
        "matplotlib==3.9.0",  # unused import in rome/compute_v.py, but required at import time
        "fastapi[standard]==0.115.6",
    )
    .run_commands(
        f"git clone --depth 1 --branch {MEMIT_COMMIT} "
        "https://github.com/kmeng01/memit /root/memit || "
        "git clone https://github.com/kmeng01/memit /root/memit",
        # NOTE: do not pre-create /root/memit/data here -- it's mounted as a
        # Modal Volume below, and Modal refuses to mount a Volume onto a
        # non-empty path baked into the image.
    )
)

HF_CACHE_PATH = "/root/.cache/huggingface"
MEMIT_DATA_PATH = "/root/memit/data"


def _normalize_scheme(scheme: list[int]) -> list[int]:
    """Dedupes + sorts a layer scheme. Duplicates would make MEMIT solve and
    apply its rank-one update twice to the same weight matrix."""
    if not scheme or any(type(layer) is not int or layer < 0 for layer in scheme):
        raise ValueError("A scheme must contain non-negative integer layers.")
    return sorted(set(scheme))


def _seed_memit_rng():
    # MEMIT's get_context_templates() samples context templates via
    # torch.multinomial on its FIRST call and caches them for the process
    # lifetime. Without reseeding, the templates (and hence the edit itself)
    # depend on how much RNG prior forward passes / generations consumed, so
    # the same layers can give different ES results across runs. Reseeding
    # alone is not enough once the cache is populated, so also drop the cache
    # and let it rebuild from the fixed seed; every edit then sees the same
    # canonical templates as a fresh container.
    import torch

    import memit.memit_main as _memit_main
    import rome.rome_main as _rome_main

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    _memit_main.CONTEXT_TEMPLATES_CACHE = None
    _rome_main.CONTEXT_TEMPLATES_CACHE = None


def _restore_weights(model, orig_weights):
    import torch

    from util import nethook

    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v


def _apply_with_rollback(apply_fn, model, tok, requests, hparams):
    """Snapshot before entering upstream code, which can fail mid-update."""
    from util import nethook

    layers = _normalize_scheme(hparams.layers)
    if any(layer >= model.config.n_layer for layer in layers):
        raise ValueError("Editing layer is outside the model.")
    hparams.layers = layers
    originals = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight":
        nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(layer)}.weight").detach().cpu().clone()
        for layer in layers
    }
    try:
        edited, _ = apply_fn(model, tok, requests, hparams, return_orig_weights=False)
        return edited, originals
    except BaseException:
        _restore_weights(model, originals)
        raise


def _hparams_path(method, model_name):
    return f"hparams/{method.upper()}/{model_name.replace('/', '_')}.json"


def _model_dtype(model_name):
    import torch

    # Upstream ROME multiplies float32 covariance matrices by model activations.
    return torch.float32


def _parse_layers(spec: str) -> list[int]:
    """
    Parses a user-facing layer-range spec into a list of ints.

    Supports:
      "8-12"    -> [8, 9, 10, 11, 12]   (inclusive range, KEditVis-style scheme)
      "6,9,12"  -> [6, 9, 12]           (explicit, possibly non-contiguous list)
      "7"       -> [7]                  (single layer)

    Output is always deduped + sorted (see _normalize_scheme).
    """
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        start, end = spec.split("-")
        start, end = int(start.strip()), int(end.strip())
        if start > end:
            raise ValueError(f"Invalid layer range {spec!r}: start > end")
        return list(range(start, end + 1))
    return _normalize_scheme(
        [int(x.strip()) for x in spec.split(",") if x.strip()]
    )


def _parse_schemes(spec: str) -> list:
    """
    Parses a user-facing multi-scheme spec (schemes separated by '|', each
    scheme parsed the same way as _parse_layers) into a list of layer lists.

    Example: "13-17|8-12|6-8|20-21" ->
        [[13,14,15,16,17], [8,9,10,11,12], [6,7,8], [20,21]]
    """
    return [_parse_layers(part) for part in spec.split("|") if part.strip()]


def _get_layer_names(model):
    # GPT-2 and GPT-J share `transformer.h.{i}.mlp` for MLP blocks.
    # Extend this if supporting non-GPT architectures (e.g. LLaMA uses model.layers.{i}.mlp).
    n_layers = model.config.n_layer
    return [f"transformer.h.{i}.mlp" for i in range(n_layers)]


def _find_subject_token_index(tok, prompt_filled, subject):
    char_start = prompt_filled.rindex(subject)
    char_end = char_start + len(subject)
    enc = tok(prompt_filled, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    last_tok_idx = None
    for i, (s, e) in enumerate(offsets):
        if s < char_end and e > char_start:
            last_tok_idx = i
    if last_tok_idx is None:
        raise ValueError(f"Could not locate subject {subject!r} in {prompt_filled!r}")
    return last_tok_idx


def _probe_layers(model, tok, prompt_filled, subject, top_k=5):
    """Same signal extraction as local_probe.py, duplicated here so this
    file has no import-time dependency on running inside the memit repo.

    Per layer, records (at the subject position) the MLP cosine similarity
    and the subject-position logit-lens top-k, plus the *last-token*
    logit-lens top-k. The last-token view is where the prompt's object
    ("Paris", "Microsoft", ..) surfaces as a next-token prediction, so it
    is the view KEditVis Sec 4.2.2's "two peak layers" rule operates on;
    the subject view alone never ranks the object (verified on gpt2).
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    subj_idx = _find_subject_token_index(tok, prompt_filled, subject)

    mlp_names = _get_layer_names(model)
    block_names = [f"transformer.h.{i}" for i in range(model.config.n_layer)]

    captured, residuals = {}, {}

    def make_mlp_hook(name):
        def hook(module, inputs, output):
            captured[name] = {"input": inputs[0].detach(), "output": output.detach()}
        return hook

    def make_block_hook(name):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            residuals[name] = hs.detach()
        return hook

    mods = dict(model.named_modules())
    handles = [mods[n].register_forward_hook(make_mlp_hook(n)) for n in mlp_names]
    handles += [mods[n].register_forward_hook(make_block_hook(n)) for n in block_names]

    try:
        with torch.no_grad():
            inputs = tok(prompt_filled, return_tensors="pt").to(device)
            last_idx = inputs["input_ids"].shape[1] - 1
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.transformer.wte

    results = []
    with torch.no_grad():
        for i, (mlp_name, block_name) in enumerate(zip(mlp_names, block_names)):
            mlp_in = captured[mlp_name]["input"][0, subj_idx, :]
            mlp_out = captured[mlp_name]["output"][0, subj_idx, :]
            cos_sim = F.cosine_similarity(
                mlp_in.unsqueeze(0), mlp_out.unsqueeze(0)
            ).item()

            resid_subj = residuals[block_name][0, subj_idx, :]
            logits_subj = lm_head(ln_f(resid_subj.unsqueeze(0))).squeeze(0)
            probs_subj = F.softmax(logits_subj, dim=-1)
            top_probs_s, top_ids_s = torch.topk(probs_subj, top_k)
            top_tokens = [
                {"token": tok.decode([tid]), "prob": p.item()}
                for tid, p in zip(top_ids_s.tolist(), top_probs_s)
            ]

            resid_last = residuals[block_name][0, last_idx, :]
            logits_last = lm_head(ln_f(resid_last.unsqueeze(0))).squeeze(0)
            probs_last = F.softmax(logits_last, dim=-1)
            top_probs_l, top_ids_l = torch.topk(probs_last, top_k)
            last_top_tokens = [
                {"token": tok.decode([tid]), "prob": p.item()}
                for tid, p in zip(top_ids_l.tolist(), top_probs_l)
            ]

            results.append(
                {
                    "layer": i,
                    "cosine_similarity": cos_sim,
                    "top_tokens": top_tokens,
                    "last_top_tokens": last_top_tokens,
                }
            )

    return results


def _parse_prompt_list(spec: str) -> list:
    """Splits a semicolon-separated string of full prompts into a list,
    dropping empty entries. Returns [] for an empty/blank spec."""
    return [p.strip() for p in spec.split(";") if p.strip()]


def _eval_prefix_targets(model, tok, prefixes, target_new, target_true):
    """
    For each prefix (a fully filled-in prompt string), computes the average
    per-token negative log-likelihood of continuing with target_new and with
    target_true, plus whether greedy decoding at each target position would
    exactly reproduce that target string.

    Adapted from kmeng01/memit's experiments/py/eval_utils_counterfact.py
    `test_batch_prediction`, simplified to always score both targets for
    every prefix instead of picking one "correct" side up front.
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    if not prefixes:
        return []
    clean_prefixes = [prefix.rstrip() for prefix in prefixes]
    targets = [target_new.strip(), target_true.strip()]
    if any(not prefix for prefix in clean_prefixes) or any(not target for target in targets):
        raise ValueError("Scoring requires non-empty prefixes and targets.")

    combined = [
        f"{prefix} {suffix}"
        for prefix in clean_prefixes
        for suffix in targets
    ]
    batch = tok(combined, padding=True, return_offsets_mapping=True, return_tensors="pt").to(device)
    offsets = batch.pop("offset_mapping").cpu().tolist()

    with torch.no_grad():
        logits = model(**batch).logits

    def score(row):
        nll = 0.0
        correct = True
        target_start = len(clean_prefixes[row // 2]) + 1
        positions = [i for i, (_, end) in enumerate(offsets[row]) if end > target_start and batch["attention_mask"][row, i]]
        if not positions or positions[0] == 0:
            raise ValueError("Could not align target tokens with a preceding prefix.")
        for position in positions:
            token_id = batch["input_ids"][row, position]
            dist = F.log_softmax(logits[row, position - 1, :].float(), dim=0)
            nll += -dist[token_id].item()
            if logits[row, position - 1, :].argmax().item() != token_id.item():
                correct = False
        return nll / len(positions), correct

    results = []
    for i in range(0, logits.size(0), 2):
        prefix_idx = i // 2
        new_nll, new_correct = score(i)
        true_nll, true_correct = score(i + 1)
        results.append(
            {
                "prefix": prefixes[prefix_idx],
                "target_new_nll": new_nll,
                "target_true_nll": true_nll,
                "target_new_correct": new_correct,
                "target_true_correct": true_correct,
            }
        )
    return results


def _harmonic_mean(vals):
    if any(v is None for v in vals):
        return None
    if not vals or any(v <= 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def _evaluate_edit(
    model,
    tok,
    prompt,
    subject,
    target_new,
    target_true,
    paraphrase_prompts,
    neighborhood_prompts,
):
    """
    Computes KEditVis-style ES / PS / NS / S metrics for the current model
    state against a single requested edit:

      ES (Efficacy Success):    does the exact rewrite prompt now produce
                                 target_new?
      PS (Paraphrase Success):  do paraphrased versions of the prompt also
                                 produce target_new (generalization)?
      NS (Neighborhood Success): do unrelated/nearby prompts still produce
                                 target_true, i.e. did the edit avoid
                                 corrupting nearby knowledge (locality)?
      S: harmonic mean of all three metrics; unavailable if any is missing.
    """
    rewrite_prompt = prompt.format(subject)
    es_results = _eval_prefix_targets(model, tok, [rewrite_prompt], target_new, target_true)
    es_rate = sum(r["target_new_nll"] < r["target_true_nll"] for r in es_results) / len(es_results)

    ps_results = (
        _eval_prefix_targets(model, tok, paraphrase_prompts, target_new, target_true)
        if paraphrase_prompts
        else []
    )
    ps_rate = (
        sum(r["target_new_nll"] < r["target_true_nll"] for r in ps_results) / len(ps_results)
        if ps_results
        else None
    )

    ns_results = (
        _eval_prefix_targets(model, tok, neighborhood_prompts, target_new, target_true)
        if neighborhood_prompts
        else []
    )
    ns_rate = (
        sum(r["target_true_nll"] < r["target_new_nll"] for r in ns_results) / len(ns_results)
        if ns_results
        else None
    )

    s = _harmonic_mean([es_rate, ps_rate, ns_rate])

    return {
        "ES": es_rate,
        "PS": ps_rate,
        "NS": ns_rate,
        "S": s,
        "details": {
            "efficacy": es_results,
            "paraphrase": ps_results,
            "neighborhood": ns_results,
        },
    }


# ---------------------------------------------------------------------------
# KL-divergence "damage" metric
#
# ES/PS/NS only look at a handful of target tokens. To quantify how much an
# edit perturbs the model's *general* behavior, we score a fixed neutral
# reference corpus with both the original and the edited model and report
# mean per-token KL(P_edited || P_orig) in nats:
#   ~0.00   -> edit left unrelated behavior untouched (clean edit)
#   higher  -> behavioral drift / collateral damage
# The corpus avoids every entity in the demo facts so it stays neutral for
# the default runs. Keep it FIXED across schemes so values are comparable.
# ---------------------------------------------------------------------------

DAMAGE_REF_PROMPTS = [
    "The sun rises in the east and sets in the west.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The library was quiet, and the students studied at long wooden tables.",
    "She poured a cup of coffee and watched the rain through the kitchen window.",
    "The train arrived at the station exactly on time this morning.",
    "Bread is baked in an oven from flour, water, salt, and yeast.",
    "The old bridge was closed for repairs after engineers found cracks in it.",
    "Birds migrate south when the weather turns cold in autumn.",
    "He locked the front door and walked down the empty street.",
    "The garden grew tomatoes, carrots, and peppers during the summer.",
    "A good night of sleep helps the body recover after a long day.",
    "The orchestra played the final movement while the audience listened in silence.",
]


def _neighborhood_snapshot(model, tok, prompts):
    """Measure actual neighborhood text and distributions, without inferred drift."""
    import torch

    texts = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for prompt in prompts:
            encoded = tok(prompt, return_tensors="pt").to(device)
            output = model.generate(
                **encoded, do_sample=False, max_new_tokens=40,
                pad_token_id=tok.eos_token_id,
            )
            texts.append(tok.decode(output[0], skip_special_tokens=True))
    return texts, _damage_logprobs(model, tok, prompts)


def _neighborhood_report(prompts, before, after):
    pre_text, pre_lp = before
    post_text, post_lp = after
    if not all(len(values) == len(prompts) for values in [pre_text, post_text, pre_lp, post_lp]):
        raise ValueError("Neighborhood measurements must align with prompts.")
    return [
        {"prompt": prompt, "pre_text": pre_text[i], "post_text": post_text[i],
         "kl_divergence": _kl_divergence([post_lp[i]], [pre_lp[i]])}
        for i, prompt in enumerate(prompts)
    ]


def _damage_logprobs(model, tok, prompts, batch_size=4):
    """Full-vocab next-token log-probabilities at every position.

    Returns a list of float32 CPU tensors, one per prompt, shaped
    (n_positions, vocab_size), where position t holds log P(token | prefix)
    for the tokens that follow. Padding positions are excluded via the
    attention mask; left-padding is never produced because the GPT-2
    tokenizer defaults to right-padding.
    """
    import torch

    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            logits = model(**enc).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            attn = enc["attention_mask"]
            # shift: logits[:, t] predicts the token at position t+1
            valid = attn[:, 1:].bool() & attn[:, :-1].bool()
            for b in range(len(batch)):
                out.append(logprobs[b, :-1][valid[b]].cpu())
    return out


def _kl_divergence(lp_edited, lp_orig):
    """Mean per-token KL(P_edited || P_orig) in nats, computed exactly.

    Both args come from `_damage_logprobs` on the same prompts: aligned
    lists of (n_positions, vocab_size) log-probability tensors.
    KL = sum_v p_e(v) * (log p_e(v) - log p_o(v)). Individual summands
    may be negative; the full distribution's divergence is non-negative.
    """
    assert len(lp_edited) == len(lp_orig), "edited/orig logprob lists must align"
    total, count = 0.0, 0
    for e, o in zip(lp_edited, lp_orig):
        assert e.shape == o.shape, "edited/orig logprob shapes must align"
        kl_per_pos = ((e.exp() * (e - o)).sum(dim=-1)).double()
        total += float(kl_per_pos.sum())
        count += int(kl_per_pos.numel())
    return total / count if count else 0.0


def _damage_report(kl, prompts):
    return {
        "kl_divergence": kl,
        "n_prompts": len(prompts),
        "note": "mean per-token KL(P_edited || P_orig) over a neutral "
        "reference corpus, in nats; ~0 means the edit left unrelated "
        "behavior untouched",
    }


@app.function(
    image=image,
    gpu=MODEL_GPU,
    timeout=30 * 60,
    volumes={HF_CACHE_PATH: hf_cache_vol, MEMIT_DATA_PATH: memit_data_vol},
)
def run_memit_edit(
    prompt: str,
    subject: str,
    target_new: str,
    generation_prompts: list[str],
    model_name: str = "gpt2-xl",
    layers: list[int] | None = None,
    target_true: str | None = None,
    paraphrase_prompts: list[str] | None = None,
    neighborhood_prompts: list[str] | None = None,
):
    import os
    import sys

    os.environ["HF_HOME"] = HF_CACHE_PATH
    os.chdir("/root/memit")
    sys.path.insert(0, "/root/memit")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memit import MEMITHyperParams, apply_memit_to_model
    from util.generate import generate_fast

    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=_model_dtype(model_name)).to("cuda").eval()

    prompt_filled = prompt.format(subject)

    print("Extracting pre-edit layer signals...")
    pre_signals = _probe_layers(model, tok, prompt_filled, subject)

    print("Generating pre-edit text...")
    pre_text = generate_fast(model, tok, generation_prompts, max_out_len=80)

    pre_metrics = None
    if target_true:
        print("Computing pre-edit ES/PS/NS/S metrics...")
        pre_metrics = _evaluate_edit(
            model, tok, prompt, subject, target_new, target_true,
            paraphrase_prompts, neighborhood_prompts,
        )

    hparams = MEMITHyperParams.from_json(_hparams_path("memit", model_name))
    request = [
        {"prompt": prompt, "subject": subject, "target_new": {"str": target_new}}
    ]

    if layers is not None:
        layers = _normalize_scheme(layers)
        n_layers = model.config.n_layer
        bad = [l for l in layers if l < 0 or l >= n_layers]
        if bad:
            raise ValueError(
                f"Layer(s) {bad} out of range for {model_name} (has {n_layers} layers, "
                f"valid range 0-{n_layers - 1})."
            )
        print(f"Overriding MEMIT's default preset {hparams.layers} with user-selected layers {layers}")
        hparams.layers = layers

    print(f"Applying MEMIT to layers {hparams.layers}...")
    _seed_memit_rng()
    model, orig_weights = _apply_with_rollback(
        apply_memit_to_model, model, tok, request, hparams
    )

    try:
        print("Extracting post-edit layer signals...")
        post_signals = _probe_layers(model, tok, prompt_filled, subject)

        print("Generating post-edit text...")
        post_text = generate_fast(model, tok, generation_prompts, max_out_len=80)

        post_metrics = None
        if target_true:
            print("Computing post-edit ES/PS/NS/S metrics...")
            post_metrics = _evaluate_edit(
                model, tok, prompt, subject, target_new, target_true,
                paraphrase_prompts, neighborhood_prompts,
            )
    finally:
        # Always revert to original weights, even if extraction above fails,
        # so a failed run can't leave edited weights behind.
        _restore_weights(model, orig_weights)

    hf_cache_vol.commit()
    memit_data_vol.commit()

    return {
        "request": {"prompt": prompt, "subject": subject, "target_new": target_new},
        "edited_layers": hparams.layers,
        "pre_edit": {
            "layer_signals": pre_signals,
            "generations": pre_text,
            "metrics": pre_metrics,
        },
        "post_edit": {
            "layer_signals": post_signals,
            "generations": post_text,
            "metrics": post_metrics,
        },
    }


@app.function(
    image=image,
    gpu=MODEL_GPU,
    timeout=60 * 60,
    volumes={HF_CACHE_PATH: hf_cache_vol, MEMIT_DATA_PATH: memit_data_vol},
)
def compare_layer_schemes(
    prompt: str,
    subject: str,
    target_new: str,
    schemes: list,
    target_true: str | None = None,
    paraphrase_prompts: list | None = None,
    neighborhood_prompts: list | None = None,
    model_name: str = "gpt2-xl",
):
    """
    Compares multiple candidate layer-selection schemes for the SAME fact,
    loading the model only once and restoring original weights between
    schemes -- this is KEditVis's "Compare" button / multi-metric ranking
    table (Sec 4.3), reusing the ES/PS/NS/S metrics from _evaluate_edit.

    Each scheme is edited and evaluated independently, starting from the
    same unedited base model, so results are directly comparable and don't
    depend on the order schemes are listed in.
    """
    import os
    import sys

    os.environ["HF_HOME"] = HF_CACHE_PATH
    os.chdir("/root/memit")
    sys.path.insert(0, "/root/memit")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memit import MEMITHyperParams, apply_memit_to_model
    from util.generate import generate_fast

    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=_model_dtype(model_name)).to("cuda").eval()

    n_layers = model.config.n_layer
    schemes = [_normalize_scheme(s) for s in schemes]
    for scheme in schemes:
        bad = [l for l in scheme if l < 0 or l >= n_layers]
        if bad:
            raise ValueError(
                f"Layer(s) {bad} out of range for {model_name} (has {n_layers} layers, "
                f"valid range 0-{n_layers - 1})."
            )

    rewrite_prompt = prompt.format(subject)

    print("Extracting baseline (pre-edit) layer signals...")
    baseline_signals = _probe_layers(model, tok, rewrite_prompt, subject)

    baseline_metrics = None
    if target_true:
        print("Computing baseline (pre-edit) ES/PS/NS/S metrics...")
        baseline_metrics = _evaluate_edit(
            model, tok, prompt, subject, target_new, target_true,
            paraphrase_prompts, neighborhood_prompts,
        )
    baseline_generation = generate_fast(model, tok, [rewrite_prompt], max_out_len=60)[0]

    hparams = MEMITHyperParams.from_json(_hparams_path("memit", model_name))
    request = [
        {"prompt": prompt, "subject": subject, "target_new": {"str": target_new}}
    ]

    results = []
    for scheme in schemes:
        print(f"\n--- Scheme {scheme} ---")
        hparams.layers = scheme

        _seed_memit_rng()
        edited_model, orig_weights = _apply_with_rollback(
            apply_memit_to_model, model, tok, request, hparams
        )

        try:
            metrics = None
            if target_true:
                metrics = _evaluate_edit(
                    edited_model, tok, prompt, subject, target_new, target_true,
                    paraphrase_prompts, neighborhood_prompts,
                )
            generation = generate_fast(edited_model, tok, [rewrite_prompt], max_out_len=60)[0]

            results.append(
                {
                    "layers": scheme,
                    "metrics": metrics,
                    "generation": generation,
                }
            )
        finally:
            # Restore original weights before trying the next scheme, so every
            # scheme is evaluated against the same clean baseline model -- even
            # if evaluation of this scheme throws.
            _restore_weights(edited_model, orig_weights)

    hf_cache_vol.commit()
    memit_data_vol.commit()

    return {
        "request": {"prompt": prompt, "subject": subject, "target_new": target_new},
        "baseline": {
            "metrics": baseline_metrics,
            "generation": baseline_generation,
            "layer_signals": baseline_signals,
        },
        "schemes": results,
    }


@app.function(
    image=image,
    gpu=MODEL_GPU,
    timeout=2 * 60 * 60,
    volumes={HF_CACHE_PATH: hf_cache_vol, MEMIT_DATA_PATH: memit_data_vol},
)
def batch_compare_schemes(
    facts: list,
    schemes: list,
    model_name: str = "gpt2-xl",
):
    """
    Sweeps multiple facts x multiple layer schemes in a SINGLE Modal
    container invocation (one model load, not one per fact), so the
    cosine-similarity-vs-edit-success analysis in analyze_schemes.py can be
    run over many more (fact, scheme) data points instead of just one
    fact's worth.

    `facts` is a list of dicts, each with keys: prompt, subject, target_new,
    target_true (optional -- metrics skipped if absent), paraphrase_prompts
    (optional list), neighborhood_prompts (optional list). See facts.json
    for the reference format.
    """
    import os
    import sys

    os.environ["HF_HOME"] = HF_CACHE_PATH
    os.chdir("/root/memit")
    sys.path.insert(0, "/root/memit")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memit import MEMITHyperParams, apply_memit_to_model

    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=_model_dtype(model_name)).to("cuda").eval()

    n_layers = model.config.n_layer
    schemes = [_normalize_scheme(s) for s in schemes]
    for scheme in schemes:
        bad = [l for l in scheme if l < 0 or l >= n_layers]
        if bad:
            raise ValueError(
                f"Layer(s) {bad} out of range for {model_name} (has {n_layers} layers, "
                f"valid range 0-{n_layers - 1})."
            )

    hparams = MEMITHyperParams.from_json(_hparams_path("memit", model_name))

    fact_results = []
    for fi, fact in enumerate(facts):
        prompt = fact["prompt"]
        subject = fact["subject"]
        target_new = fact["target_new"]
        target_true = fact.get("target_true")
        paraphrase_prompts = fact.get("paraphrase_prompts") or []
        neighborhood_prompts = fact.get("neighborhood_prompts") or []
        rewrite_prompt = prompt.format(subject)

        print(f"\n=== Fact {fi + 1}/{len(facts)}: {rewrite_prompt} -> {target_new} ===")

        print("Extracting baseline layer signals...")
        baseline_signals = _probe_layers(model, tok, rewrite_prompt, subject)

        baseline_metrics = None
        if target_true:
            baseline_metrics = _evaluate_edit(
                model, tok, prompt, subject, target_new, target_true,
                paraphrase_prompts, neighborhood_prompts,
            )

        request = [
            {"prompt": prompt, "subject": subject, "target_new": {"str": target_new}}
        ]

        scheme_results = []
        for scheme in schemes:
            print(f"  scheme {scheme}...")
            hparams.layers = scheme

            _seed_memit_rng()
            edited_model, orig_weights = _apply_with_rollback(
                apply_memit_to_model, model, tok, request, hparams
            )

            try:
                metrics = None
                if target_true:
                    metrics = _evaluate_edit(
                        edited_model, tok, prompt, subject, target_new, target_true,
                        paraphrase_prompts, neighborhood_prompts,
                    )

                scheme_results.append({"layers": scheme, "metrics": metrics})
            finally:
                _restore_weights(edited_model, orig_weights)

        fact_results.append(
            {
                "fact": {
                    "prompt": prompt,
                    "subject": subject,
                    "target_new": target_new,
                    "target_true": target_true,
                },
                "baseline": {
                    "metrics": baseline_metrics,
                    "layer_signals": baseline_signals,
                },
                "schemes": scheme_results,
            }
        )

        hf_cache_vol.commit()
        memit_data_vol.commit()

    return {"facts": fact_results}


@app.local_entrypoint()
def batch(
    facts_file: str = "facts.json",
    schemes: str = "13-17|8-12|6-8|20-21",
    model_name: str = "gpt2-xl",
    out: str = "batch_comparison.json",
):
    """
    Sweeps every fact in `facts_file` (see facts.json for the format)
    across every scheme in `schemes`, in a single Modal container.

    Usage:
        modal run modal_app.py::batch --schemes "13-17|8-12|6-8|20-21"
    """
    with open(facts_file) as f:
        facts = json.load(f)

    parsed_schemes = _parse_schemes(schemes)
    if not parsed_schemes:
        raise ValueError(f"No valid schemes parsed from {schemes!r}")

    print(f"Sweeping {len(facts)} facts x {len(parsed_schemes)} schemes "
          f"= {len(facts) * len(parsed_schemes)} edits total")

    result = batch_compare_schemes.remote(
        facts=facts, schemes=parsed_schemes, model_name=model_name,
    )

    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved full batch results to {out}")

    def f(x):
        return f"{x:.2f}" if x is not None else "n/a"

    for fact_result in result["facts"]:
        fact = fact_result["fact"]
        print(f"\n=== {fact['prompt'].format(fact['subject'])} -> {fact['target_new']} ===")
        for s in fact_result["schemes"]:
            m = s["metrics"] or {}
            print(
                f"  layers={s['layers']!s:<20} ES={f(m.get('ES'))} "
                f"PS={f(m.get('PS'))} NS={f(m.get('NS'))} S={f(m.get('S'))}"
            )


@app.local_entrypoint()
def main(
    prompt: str = "{} is located in the city of",
    subject: str = "Eiffel Tower",
    target: str = "Rome",
    model_name: str = "gpt2-xl",
    layers: str = "",
    target_true: str = "Paris",
    paraphrase_prompts: str = (
        "The Eiffel Tower is located in the city of;"
        "You can find the Eiffel Tower in the city of"
    ),
    neighborhood_prompts: str = (
        "The Louvre Museum is located in the city of;"
        "Notre-Dame Cathedral is located in the city of"
    ),
    out: str = "memit_result.json",
):
    generation_prompts = [
        prompt.format(subject),
        f"Tell me about {subject}.",
        f"{subject} is famous for",
    ]

    parsed_layers = _parse_layers(layers) if layers else None
    if parsed_layers is not None:
        print(f"Using user-specified layers: {parsed_layers}")
    else:
        print("No --layers given; using MEMIT's default hardcoded preset for this model.")

    parsed_paraphrase = _parse_prompt_list(paraphrase_prompts)
    parsed_neighborhood = _parse_prompt_list(neighborhood_prompts)
    if target_true:
        print(f"Will compute ES/PS/NS/S metrics against target_true={target_true!r}")
    else:
        print("No --target_true given; skipping ES/PS/NS/S metric computation.")

    result = run_memit_edit.remote(
        prompt=prompt,
        subject=subject,
        target_new=target,
        generation_prompts=generation_prompts,
        model_name=model_name,
        layers=parsed_layers,
        target_true=target_true or None,
        paraphrase_prompts=parsed_paraphrase,
        neighborhood_prompts=parsed_neighborhood,
    )

    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved full result to {out}\n")
    print("=== Pre-edit generations ===")
    for g in result["pre_edit"]["generations"]:
        print(f"  {g}")
    print("\n=== Post-edit generations ===")
    for g in result["post_edit"]["generations"]:
        print(f"  {g}")
    print(f"\nEdited layers: {result['edited_layers']}")

    def fmt_metrics(m):
        if m is None:
            return "  (skipped -- no --target_true given)"
        def f(x):
            return f"{x:.2f}" if x is not None else "n/a"
        return (
            f"  ES={f(m['ES'])}  PS={f(m['PS'])}  NS={f(m['NS'])}  S={f(m['S'])}"
        )

    print("\n=== Metrics (before edit) ===")
    print(fmt_metrics(result["pre_edit"]["metrics"]))
    print("=== Metrics (after edit) ===")
    print(fmt_metrics(result["post_edit"]["metrics"]))


@app.local_entrypoint()
def compare(
    prompt: str = "{} is located in the city of",
    subject: str = "Eiffel Tower",
    target: str = "Rome",
    model_name: str = "gpt2-xl",
    schemes: str = "13-17|8-12|6-8|20-21",
    target_true: str = "Paris",
    paraphrase_prompts: str = (
        "The Eiffel Tower is located in the city of;"
        "You can find the Eiffel Tower in the city of"
    ),
    neighborhood_prompts: str = (
        "The Louvre Museum is located in the city of;"
        "Notre-Dame Cathedral is located in the city of"
    ),
    out: str = "scheme_comparison.json",
):
    """
    Compares multiple layer-selection schemes for the same fact in a single
    Modal container invocation (loads the 6GB model only once).

    Usage:
        modal run modal_app.py::compare --schemes "13-17|8-12|6-8|20-21"
    """
    parsed_schemes = _parse_schemes(schemes)
    if not parsed_schemes:
        raise ValueError(f"No valid schemes parsed from {schemes!r}")
    print(f"Comparing {len(parsed_schemes)} schemes: {parsed_schemes}")

    parsed_paraphrase = _parse_prompt_list(paraphrase_prompts)
    parsed_neighborhood = _parse_prompt_list(neighborhood_prompts)

    result = compare_layer_schemes.remote(
        prompt=prompt,
        subject=subject,
        target_new=target,
        schemes=parsed_schemes,
        target_true=target_true or None,
        paraphrase_prompts=parsed_paraphrase,
        neighborhood_prompts=parsed_neighborhood,
        model_name=model_name,
    )

    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved full comparison to {out}\n")

    def f(x):
        return f"{x:.2f}" if x is not None else "n/a"

    baseline_m = result["baseline"]["metrics"]
    print("=== Baseline (unedited model) ===")
    if baseline_m:
        print(
            f"  ES={f(baseline_m['ES'])}  PS={f(baseline_m['PS'])}  "
            f"NS={f(baseline_m['NS'])}  S={f(baseline_m['S'])}"
        )
    print(f"  generation: {result['baseline']['generation']}\n")

    header = f"{'Layers':<20} | {'ES':>5} | {'PS':>5} | {'NS':>5} | {'S':>5}"
    print("=== Scheme comparison (sorted by S, descending) ===")
    print(header)
    print("-" * len(header))

    def sort_key(r):
        m = r["metrics"]
        return m["S"] if m and m["S"] is not None else -1

    for r in sorted(result["schemes"], key=sort_key, reverse=True):
        m = r["metrics"]
        layers_str = str(r["layers"])
        if m:
            print(
                f"{layers_str:<20} | {f(m['ES']):>5} | {f(m['PS']):>5} | "
                f"{f(m['NS']):>5} | {f(m['S']):>5}"
            )
        else:
            print(f"{layers_str:<20} | (no metrics -- pass --target_true)")

    print("\n=== Generations per scheme ===")
    for r in result["schemes"]:
        print(f"  layers={r['layers']}: {r['generation']}")


# ---------------------------------------------------------------------------
# Interactive dashboard backend (FastAPI, served/deployed via Modal)
#
# Dev:    modal serve modal_app.py    (temporary URL, live-reloads on save)
# Deploy: modal deploy modal_app.py   (persistent URL)
#
# The model is loaded ONCE per container, in the body of `web_app()` below,
# which Modal calls a single time when a container starts -- NOT once per
# HTTP request. Subsequent requests are routed to the already-warm FastAPI
# app's routes, avoiding a ~20-30s model reload on every click in the UI.
#
# `@modal.concurrent(max_inputs=1)` ensures only one request is in flight per
# container at a time, since /edit and /compare temporarily mutate the
# shared model's weights before restoring them -- concurrent requests to the
# same container could otherwise corrupt each other's edits.
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=MODEL_GPU,
    timeout=30 * 60,
    scaledown_window=10 * 60,
    volumes={HF_CACHE_PATH: hf_cache_vol, MEMIT_DATA_PATH: memit_data_vol},
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def web_app():
    import os
    import sys

    os.environ["HF_HOME"] = HF_CACHE_PATH
    os.chdir("/root/memit")
    sys.path.insert(0, "/root/memit")

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator
    from typing import Literal
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memit import MEMITHyperParams, apply_memit_to_model
    from rome import ROMEHyperParams, apply_rome_to_model
    from util.generate import generate_fast

    import torch

    SUPPORTED_MODELS = ["gpt2-xl", "EleutherAI/gpt-j-6B"]
    SUPPORTED_METHODS = ["memit", "rome"]

    # Lazy model state: keep one model resident at a time.
    _state = {"model_name": None, "model": None, "tok": None}

    def _ensure_model(model_name: str):
        if model_name not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model {model_name!r}. Supported: {SUPPORTED_MODELS}",
            )
        if _state["model_name"] == model_name:
            return _state["model"], _state["tok"]
        if _state["model"] is not None:
            print(f"[web_app] Unloading {_state['model_name']}...")
            _state.update(model_name=None, model=None, tok=None)
            torch.cuda.empty_cache()
        print(f"[web_app] Loading {model_name}...")
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=_model_dtype(model_name)).to("cuda").eval()
        _state["model_name"] = model_name
        _state["model"] = model
        _state["tok"] = tok
        print(f"[web_app] {model_name} loaded ({model.config.n_layer} layers).")
        return model, tok

    _ensure_model("gpt2-xl")

    class ProbeRequest(BaseModel):
        prompt: str = Field(min_length=1)
        subject: str = Field(min_length=1)
        target: str | None = None
        model: Literal["gpt2-xl", "EleutherAI/gpt-j-6B"] = "gpt2-xl"

        @field_validator("prompt", "subject")
        @classmethod
        def nonblank(cls, value):
            if not value.strip():
                raise ValueError("Prompt and subject must not be blank.")
            return value.strip()

        @model_validator(mode="after")
        def prompt_template(self):
            from string import Formatter

            fields = [(name, spec, conversion) for _, name, spec, conversion in Formatter().parse(self.prompt) if name is not None]
            if fields and fields != [("", "", None)]:
                raise ValueError("Use exactly one plain {} subject placeholder.")
            if not fields:
                if "{" in self.prompt or "}" in self.prompt:
                    raise ValueError("Literal braces are not supported in prompts.")
                self.prompt = self.prompt.replace(self.subject, "{}", 1) if self.subject in self.prompt else "{} " + self.prompt
            return self

    class FactRequest(ProbeRequest):
        target_new: str = Field(min_length=1)
        target_true: str | None = None
        paraphrase_prompts: list[str] = []
        neighborhood_prompts: list[str] = []
        generation_prompts: list[str] | None = None
        damage_prompts: list[str] | None = None
        method: Literal["memit", "rome"] = "memit"

        @field_validator("target_new")
        @classmethod
        def targets(cls, value):
            if not value.strip():
                raise ValueError("The new target must not be blank.")
            return value.strip()

        @field_validator("target_true")
        @classmethod
        def original_target(cls, value):
            return value.strip() or None if value is not None else None

        @field_validator("paraphrase_prompts", "neighborhood_prompts", "generation_prompts", "damage_prompts")
        @classmethod
        def prompts(cls, values):
            if values is not None and any(not value.strip() for value in values):
                raise ValueError("Evaluation prompts must not be blank.")
            return [value.strip() for value in values] if values is not None else None

    class EditRequest(FactRequest):
        layers: list[StrictInt] | None = Field(default=None, min_length=1)

    class CompareRequest(FactRequest):
        schemes: list[list[StrictInt]] = Field(min_length=1)

        @field_validator("schemes")
        @classmethod
        def nonempty_schemes(cls, schemes):
            if any(not scheme for scheme in schemes):
                raise ValueError("Each scheme must contain at least one layer.")
            return schemes

    def validate_layers(layers, model):
        if layers is None:
            return
        n_layers = model.config.n_layer
        bad = [l for l in layers if l < 0 or l >= n_layers]
        if bad:
            raise HTTPException(
                status_code=400,
                detail=f"Layer(s) {bad} out of range "
                f"(has {n_layers} layers, valid range 0-{n_layers - 1}).",
            )

    def _apply_edit(method, model, tok, request_, layers, model_name):
        """Dispatch to MEMIT or ROME based on method string."""
        if method not in SUPPORTED_METHODS:
            raise HTTPException(status_code=400, detail="Unsupported editing method.")
        if method == "rome":
            if len(layers) != 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"ROME edits exactly 1 layer, got {len(layers)}. "
                    "Use MEMIT for multi-layer edits.",
                )
            hparams = ROMEHyperParams.from_json(_hparams_path("rome", model_name))
            hparams.layers = layers
            _seed_memit_rng()
            return _apply_with_rollback(
                apply_rome_to_model, model, tok, request_, hparams
            )
        else:
            hparams = MEMITHyperParams.from_json(_hparams_path("memit", model_name))
            hparams.layers = layers
            _seed_memit_rng()
            return _apply_with_rollback(
                apply_memit_to_model, model, tok, request_, hparams
            )

    web = FastAPI(title="KEditVis API")
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web.get("/health")
    def health(model: str = Query(default=None)):
        _ensure_model(model or _state["model_name"] or "gpt2-xl")
        m = _state["model"]
        return {
            "status": "ok",
            "model": _state["model_name"],
            "n_layers": m.config.n_layer,
            "methods": SUPPORTED_METHODS,
        }

    @web.post("/probe")
    def probe(body: ProbeRequest):
        model, tok = _ensure_model(body.model)
        rewrite_prompt = body.prompt.format(body.subject)
        try:
            signals = _probe_layers(model, tok, rewrite_prompt, body.subject)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"rewrite_prompt": rewrite_prompt, "layer_signals": signals}

    @web.post("/edit")
    def edit(body: EditRequest):
        model, tok = _ensure_model(body.model)
        model_name = _state["model_name"]
        validate_layers(body.layers, model)
        layers = _normalize_scheme(body.layers) if body.layers is not None else None
        if body.method == "rome" and layers is not None and len(layers) != 1:
            raise HTTPException(status_code=400, detail="ROME edits exactly one layer.")
        rewrite_prompt = body.prompt.format(body.subject)
        generation_prompts = body.generation_prompts or [rewrite_prompt]
        damage_prompts = body.damage_prompts or DAMAGE_REF_PROMPTS

        pre_signals = _probe_layers(model, tok, rewrite_prompt, body.subject)
        pre_text = generate_fast(model, tok, generation_prompts, max_out_len=80)
        pre_metrics = (
            _evaluate_edit(
                model, tok, body.prompt, body.subject, body.target_new,
                body.target_true, body.paraphrase_prompts, body.neighborhood_prompts,
            )
            if body.target_true else None
        )
        orig_damage_lp = _damage_logprobs(model, tok, damage_prompts)
        pre_neighborhood = _neighborhood_snapshot(model, tok, body.neighborhood_prompts)

        request_ = [
            {"prompt": body.prompt, "subject": body.subject,
             "target_new": {"str": body.target_new}}
        ]

        if layers is None:
            if body.method == "rome":
                hparams = ROMEHyperParams.from_json(_hparams_path("rome", model_name))
                layers = hparams.layers if isinstance(hparams.layers, list) else [hparams.layers]
            else:
                hparams = MEMITHyperParams.from_json(_hparams_path("memit", model_name))
                layers = hparams.layers

        edited_model, orig_weights = _apply_edit(
            body.method, model, tok, request_, layers, model_name
        )

        try:
            post_signals = _probe_layers(edited_model, tok, rewrite_prompt, body.subject)
            post_text = generate_fast(edited_model, tok, generation_prompts, max_out_len=80)
            post_metrics = (
                _evaluate_edit(
                    edited_model, tok, body.prompt, body.subject, body.target_new,
                    body.target_true, body.paraphrase_prompts, body.neighborhood_prompts,
                )
                if body.target_true else None
            )
            edited_damage_lp = _damage_logprobs(edited_model, tok, damage_prompts)
            damage_kl = _kl_divergence(edited_damage_lp, orig_damage_lp)
            neighborhood = _neighborhood_report(
                body.neighborhood_prompts, pre_neighborhood,
                _neighborhood_snapshot(edited_model, tok, body.neighborhood_prompts),
            )
        finally:
            _restore_weights(edited_model, orig_weights)

        return {
            "method": body.method,
            "edited_layers": layers,
            "neighborhood": neighborhood,
            "pre_edit": {
                "layer_signals": pre_signals,
                "generations": pre_text,
                "metrics": pre_metrics,
            },
            "post_edit": {
                "layer_signals": post_signals,
                "generations": post_text,
                "metrics": post_metrics,
            },
            "damage": _damage_report(damage_kl, damage_prompts),
        }

    @web.post("/compare")
    def compare(body: CompareRequest):
        model, tok = _ensure_model(body.model)
        model_name = _state["model_name"]
        if not body.schemes:
            raise HTTPException(status_code=400, detail="schemes must be non-empty")
        for scheme in body.schemes:
            validate_layers(scheme, model)
        schemes = [_normalize_scheme(s) for s in body.schemes]

        # ROME auto-split: multi-layer schemes become individual single-layer comparisons
        if body.method == "rome":
            flat = []
            for s in schemes:
                for layer in s:
                    flat.append([layer])
            schemes = flat

        schemes = [list(s) for s in dict.fromkeys(tuple(s) for s in schemes)]

        for scheme in schemes:
            validate_layers(scheme, model)

        rewrite_prompt = body.prompt.format(body.subject)
        damage_prompts = body.damage_prompts or DAMAGE_REF_PROMPTS

        baseline_signals = _probe_layers(model, tok, rewrite_prompt, body.subject)
        baseline_metrics = (
            _evaluate_edit(
                model, tok, body.prompt, body.subject, body.target_new,
                body.target_true, body.paraphrase_prompts, body.neighborhood_prompts,
            )
            if body.target_true else None
        )
        baseline_generation = generate_fast(model, tok, [rewrite_prompt], max_out_len=60)[0]
        orig_damage_lp = _damage_logprobs(model, tok, damage_prompts)
        pre_neighborhood = _neighborhood_snapshot(model, tok, body.neighborhood_prompts)

        request_ = [
            {"prompt": body.prompt, "subject": body.subject,
             "target_new": {"str": body.target_new}}
        ]

        results = []
        for scheme in schemes:
            edited_model, orig_weights = _apply_edit(
                body.method, model, tok, request_, scheme, model_name
            )

            try:
                metrics = (
                    _evaluate_edit(
                        edited_model, tok, body.prompt, body.subject, body.target_new,
                        body.target_true, body.paraphrase_prompts, body.neighborhood_prompts,
                    )
                    if body.target_true else None
                )
                generation = generate_fast(edited_model, tok, [rewrite_prompt], max_out_len=60)[0]
                edited_damage_lp = _damage_logprobs(edited_model, tok, damage_prompts)
                damage_kl = _kl_divergence(edited_damage_lp, orig_damage_lp)
                neighborhood = _neighborhood_report(
                    body.neighborhood_prompts, pre_neighborhood,
                    _neighborhood_snapshot(edited_model, tok, body.neighborhood_prompts),
                )

                results.append({
                    "layers": scheme,
                    "metrics": metrics,
                    "generation": generation,
                    "damage": _damage_report(damage_kl, damage_prompts),
                    "neighborhood": neighborhood,
                })
            finally:
                _restore_weights(edited_model, orig_weights)

        return {
            "method": body.method,
            "baseline": {
                "metrics": baseline_metrics,
                "generation": baseline_generation,
                "layer_signals": baseline_signals,
                "damage": _damage_report(0.0, damage_prompts),
            },
            "schemes": results,
        }

    return web
```

<a id="replacement-2"></a>

### frontend/src/api/client.ts

Current file: [frontend/src/api/client.ts](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/api/client.ts).

SHA-256: da7305afdeaedf4586666705bc799a730473a537a0afcddd1746cd61dc98ce12

```typescript
import type { CompareResponse, EditResponse, FactInput, HealthResponse, ProbeResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (typeof data?.detail === "string") message = data.detail;
      else if (Array.isArray(data?.detail)) {
        message = data.detail.map((item: { msg?: string }) => item.msg ?? "Invalid request").join("; ");
      }
    } catch {
      // Preserve the HTTP error when the response body is not JSON.
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export function getApiBase(): string {
  return API_BASE;
}

export function health(model?: string): Promise<HealthResponse> {
  return request(model ? `/health?model=${encodeURIComponent(model)}` : "/health");
}

export function probe(body: Pick<FactInput, "prompt" | "subject"> & { model?: string }): Promise<ProbeResponse> {
  return request("/probe", { method: "POST", body: JSON.stringify(body) });
}

export function edit(body: FactInput & { layers: number[] | null; method?: string; model?: string }): Promise<EditResponse> {
  return request("/edit", { method: "POST", body: JSON.stringify(body) });
}

export function compare(body: FactInput & { schemes: number[][]; method?: string; model?: string }): Promise<CompareResponse> {
  return request("/compare", { method: "POST", body: JSON.stringify(body) });
}
```

<a id="replacement-3"></a>

### frontend/src/App.css

Current file: [frontend/src/App.css](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/App.css).

SHA-256: 00ee5d731e7ec7587514c04e8ff03a3a7d9b2396a7a4c38d2bb2d2111185e68d

```css
:root {
  --bg-page: #F1F5F9;
  --bg-canvas: #F8FAFC;
  --bg-card: #FFFFFF;
  --bg-card-subtle: #F8FAFC;
  
  --border: #E2E8F0;
  --border-strong: #CBD5E1;
  --border-subtle: #F1F5F9;
  --border-active: #3B82F6;
  
  --text-main: #0F172A;
  --text-muted: #475569;
  --text-faint: #64748B;
  --text-light: #94A3B8;
  
  --nav-bg: #1E293B;
  --nav-text: #FFFFFF;
  
  --accent-blue: #3B82F6;
  --accent-light-blue: #60A5FA;
  --accent-bar: #7DD3FC;
  --accent-bar-fill: #93C5FD;
  --accent-green: #10B981;
  --accent-amber: #F59E0B;
  --accent-rose: #F43F5E;
  --accent-red: #EF4444;
  --accent-purple: #8B5CF6;

  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;

  font-family: var(--font-sans);
  color: var(--text-main);
  background-color: var(--bg-page);
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg-page); color: var(--text-main); }

/* Top Navigation Bar */
.top-navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.55rem 1.4rem;
  background: var(--nav-bg);
  color: var(--nav-text);
  border-bottom: 1px solid #0F172A;
  height: 50px;
  position: sticky;
  top: 0;
  z-index: 100;
}

.nav-left, .nav-right {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}

.logo-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.logo-title {
  font-family: var(--font-mono);
  font-weight: 800;
  font-size: 1.1rem;
  letter-spacing: -0.03em;
  color: #FFFFFF;
}

/* Model Dropdown Selector */
.model-dropdown-select {
  padding: 0.28rem 0.6rem;
  border-radius: 4px;
  border: 1px solid #334155;
  background: #0F172A;
  color: #E2E8F0;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
}

.model-dropdown-select:focus { outline: none; border-color: var(--accent-blue); }

/* Method Pill Group (MEMIT / ROME) */
.method-pill-group {
  display: flex;
  background: #0F172A;
  border: 1px solid #334155;
  border-radius: 4px;
  padding: 2px;
}

.method-pill-group button {
  padding: 0.2rem 0.55rem;
  border: none;
  background: transparent;
  color: #94A3B8;
  font-family: var(--font-mono);
  font-size: 0.70rem;
  font-weight: 600;
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.method-pill-group button.active {
  background: var(--accent-blue);
  color: #FFFFFF;
}

/* Selected Layers Indicator */
.selected-layers-pill {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  background: #0F172A;
  border: 1px solid #334155;
  padding: 0.25rem 0.6rem;
  border-radius: 4px;
}

.pill-lbl {
  font-size: 0.62rem;
  font-family: var(--font-mono);
  color: #64748B;
  font-weight: 700;
}

.pill-val {
  font-size: 0.74rem;
  font-family: var(--font-mono);
  color: #38BDF8;
  font-weight: 800;
}

/* Action Buttons Group */
.action-buttons-group {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}

.btn-action {
  padding: 0.32rem 0.75rem;
  border: 1px solid #334155;
  background: #1E293B;
  color: #F8FAFC;
  border-radius: 4px;
  font-size: 0.76rem;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  transition: all 0.15s ease;
}

.btn-action:hover:not(:disabled) {
  background: #334155;
  border-color: #475569;
}

.btn-action:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-action.btn-primary {
  background: var(--accent-blue);
  border-color: var(--accent-blue);
  color: #FFFFFF;
}

.btn-action.btn-primary:hover:not(:disabled) {
  background: #1D4ED8;
}

/* Telemetry & Error Banners */
.telemetry-bar {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  background: #EFF6FF;
  border-bottom: 1px solid #BFDBFE;
  color: #1E40AF;
  padding: 0.45rem 1.4rem;
  font-size: 0.78rem;
  font-family: var(--font-mono);
}

.error-banner {
  background: #FEF2F2;
  border-bottom: 1px solid #FCA5A5;
  color: #B91C1C;
  padding: 0.5rem 1.4rem;
  font-size: 0.82rem;
}

/* Authentic Layout Grid (Figure 3 & Figure 4) */
.main-layout-grid {
  display: grid;
  grid-template-columns: 270px 1fr;
  gap: 0.85rem;
  padding: 0.85rem;
  max-width: 1850px;
  margin: 0 auto;
}

/* Left Sidebar (A1 -> A2 -> A3) */
.left-sidebar {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}


.fact-manager-panel {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.chat-card, .facts-card, .prompts-card, .knowledge-graph-box {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.75rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.card-header, .panel-header-sub {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.55rem;
}

.title-with-badge {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

/* Cyan/Teal Badge Tags (A1, A2, A3, B1, B2, B3, B4, C, D) */
.badge-tag {
  background: #F0F9FF;
  color: #0284C7;
  border: 1px solid #7DD3FC;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 700;
  padding: 1px 4px;
  border-radius: 3px;
}

.card-header h3, .panel-header-sub h4 {
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--text-main);
}

.chat-mode-toggle {
  display: flex;
  background: #F1F5F9;
  border: 1px solid var(--border);
  border-radius: 4px;
}

.chat-mode-toggle button {
  background: transparent;
  border: none;
  color: var(--text-faint);
  font-size: 0.62rem;
  font-weight: 700;
  padding: 2px 6px;
  cursor: pointer;
}

.chat-mode-toggle button.active {
  background: #0284C7;
  color: #FFFFFF;
  border-radius: 3px;
}

.chat-box-content {
  background: #F8FAFC;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 0.6rem;
  font-size: 0.78rem;
  line-height: 1.45;
}

.chat-prompt { color: var(--text-muted); margin-bottom: 0.3rem; }
.chat-target { color: var(--text-main); }

.fact-item {
  background: #F8FAFC;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 0.55rem;
}

.fact-main-text {
  font-size: 0.78rem;
  color: var(--text-main);
  margin-bottom: 0.35rem;
}

.fact-orig { color: var(--text-faint); font-size: 0.72rem; }

.fact-meta-tags {
  display: flex;
  gap: 0.4rem;
}

.meta-tag {
  background: #E2E8F0;
  color: var(--text-muted);
  font-size: 0.62rem;
  font-family: var(--font-mono);
  padding: 1px 4px;
  border-radius: 3px;
}

.btn-tiny {
  background: #F1F5F9;
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 0.7rem;
  padding: 1px 5px;
  border-radius: 3px;
  cursor: pointer;
}

.fact-edit-inputs {
  margin-top: 0.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.fact-edit-inputs input, .fact-edit-inputs textarea {
  background: #FFFFFF;
  border: 1px solid var(--border-strong);
  color: var(--text-main);
  font-size: 0.75rem;
  padding: 0.3rem 0.5rem;
  border-radius: 4px;
  width: 100%;
}

.row-inputs {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.3rem;
}

/* Prompt Badges (A3) */
.prompt-badge-list {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.prompt-badge-item {
  display: flex;
  border-radius: 5px;
  overflow: hidden;
  border: 1px solid var(--border);
  background: #FFFFFF;
}

.badge-bar { width: 4px; flex-shrink: 0; }
.badge-efficacy { background: #F5F3FF; border-color: #DDD6FE; }
.badge-efficacy .badge-bar { background: var(--accent-purple); }

.badge-paraphrase { background: #FFFBEB; border-color: #FDE68A; }
.badge-paraphrase .badge-bar { background: var(--accent-amber); }

.badge-neighborhood { background: #FFF1F2; border-color: #FECDD3; }
.badge-neighborhood .badge-bar { background: var(--accent-rose); }

.badge-generation { background: #F8FAFC; border-color: #E2E8F0; }
.badge-generation .badge-bar { background: var(--accent-green); }

.badge-content { padding: 0.4rem 0.55rem; width: 100%; }
.badge-text { font-size: 0.74rem; color: var(--text-main); line-height: 1.3; font-weight: 500; }
.badge-meta { font-size: 0.6rem; font-family: var(--font-mono); color: var(--text-faint); margin-top: 2px; }

/* Knowledge Graph Canvas */
.kg-canvas-wrapper {
  background: #F8FAFC;
  border: 1px solid var(--border);
  border-radius: 5px;
  overflow: hidden;
}
.kg-svg { width: 100%; height: 115px; }

/* Center Canvas Column */
.center-canvas {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.edit-view-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.85rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.edit-view-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.45rem;
}

.edit-view-header h3 { font-size: 0.92rem; font-weight: 800; color: var(--text-main); }

.lens-view-toggle {
  display: flex;
  background: #F1F5F9;
  border: 1px solid var(--border);
  border-radius: 4px;
}

.lens-view-toggle button {
  background: transparent;
  border: none;
  color: var(--text-faint);
  font-size: 0.68rem;
  font-weight: 700;
  padding: 3px 8px;
  cursor: pointer;
}

.lens-view-toggle button.active {
  background: #0284C7;
  color: #FFFFFF;
  border-radius: 3px;
}

/* Edit Signals Row (B1, B2, B3, B4) */
.edit-signals-row {
  display: grid;
  grid-template-columns: 270px 130px 1fr 270px;
  gap: 0.55rem;
  align-items: start;
  min-height: 460px;
  overflow-x: auto;
}

.center-canvas, .edit-view-panel, .table-column, .drift-right-col {
  min-width: 0;
}

.table-scroll-container { overflow-x: auto; }
.eval-pill { overflow-wrap: anywhere; }

.lens-column {
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 0.5rem;
}

.version-tag {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--text-muted);
  margin-bottom: 0.35rem;
  padding-bottom: 0.25rem;
  border-bottom: 1px solid var(--border);
}

.charts-pair {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

/* Wireframe Linker (B2) */
.wireframe-column {
  display: flex;
  flex-direction: column;
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 0.4rem;
}

.wireframe-container {
  display: flex;
  position: relative;
}

.layer-axis-column {
  display: flex;
  flex-direction: column;
  width: 22px;
  flex-shrink: 0;
}

.axis-header {
  font-size: 0.65rem;
  font-family: var(--font-mono);
  color: var(--text-faint);
  text-align: center;
  height: 20px;
}

.axis-ticks {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.axis-tick {
  font-size: 0.58rem;
  font-family: var(--font-mono);
  color: var(--text-faint);
  border-left: 2px solid #CBD5E1;
  padding-left: 2px;
  line-height: 1;
}

.axis-tick.active {
  border-left-color: var(--accent-red);
  color: var(--accent-red);
  font-weight: 700;
}

.wireframe-svg { flex-grow: 1; }

/* Scheme Table (B3) */
.table-column {
  display: flex;
  flex-direction: column;
}

.scheme-table-wrap {
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 5px;
  overflow: hidden;
}

.table-header-bar {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.45rem 0.65rem;
  background: #F8FAFC;
  border-bottom: 1px solid var(--border);
}

.table-header-bar h4 { font-size: 0.78rem; font-weight: 700; color: var(--text-main); }

.scheme-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.75rem;
}

.scheme-table th {
  background: #F8FAFC;
  color: var(--text-faint);
  font-family: var(--font-mono);
  font-size: 0.68rem;
  padding: 0.35rem 0.5rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}

.scheme-table td {
  padding: 0.4rem 0.5rem;
  border-bottom: 1px solid var(--border);
}

.scheme-row { cursor: pointer; transition: background 0.15s; }
.scheme-row:hover { background: #F0F9FF; }
.scheme-row.selected { background: #E0F2FE; }

.scheme-chip {
  background: #F1F5F9;
  border: 1px solid var(--border-strong);
  color: var(--text-main);
  font-family: var(--font-mono);
  font-size: 0.68rem;
  padding: 2px 5px;
  border-radius: 3px;
  font-weight: 600;
}

.scheme-chip.base { background: #F8FAFC; color: var(--text-faint); }

.metric-bar-cell {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  width: 100%;
}

.metric-bar-track {
  background: #E2E8F0;
  border-radius: 3px;
  height: 8px;
  flex-grow: 1;
  overflow: hidden;
}

.metric-bar-fill {
  height: 100%;
  border-radius: 3px;
  background-color: var(--accent-light-blue);
  transition: width 0.3s ease;
}

.metric-bar-text {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  color: var(--text-main);
  width: 28px;
  text-align: right;
  font-weight: 600;
}

.version-cell {
  font-family: var(--font-mono);
  color: var(--text-faint);
  font-size: 0.68rem;
}

.table-placeholder {
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 5px;
}

.placeholder-content {
  padding: 1.1rem;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.8rem;
}

/* Layer Selector */
.layer-selector h2 { font-size: 0.85rem; font-weight: 700; margin-bottom: 0.3rem; }
.layer-selector .hint { font-size: 0.72rem; color: var(--text-faint); margin-bottom: 0.6rem; }
.layer-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.6rem;
}
.layer-actions button {
  background: #F1F5F9;
  border: 1px solid var(--border-strong);
  color: var(--text-main);
  padding: 0.25rem 0.55rem;
  border-radius: 4px;
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
}
.layer-actions button:hover:not(:disabled) { background: #E2E8F0; }
.layer-actions button:disabled { opacity: 0.5; cursor: not-allowed; }
.selected-label { font-size: 0.72rem; color: var(--text-muted); }

.layer-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 3px;
  justify-content: center;
  max-width: 480px;
  margin: 0 auto;
}

.layer-chip {
  background: #F1F5F9;
  border: 1px solid var(--border-strong);
  color: var(--text-main);
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 600;
  width: 26px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.12s;
}

.layer-chip:hover { border-color: var(--accent-blue); }
.layer-chip.active {
  background: var(--accent-blue) !important;
  color: #FFFFFF !important;
  border-color: var(--accent-blue) !important;
  font-weight: 700;
}

/* Prompt Detail Cards (B3 Bottom) */
.prompt-detail-section {
  margin-top: 0.75rem;
  border-top: 1px solid var(--border);
  padding-top: 0.75rem;
}

.category-columns-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.55rem;
}

.category-card {
  border-radius: 5px;
  padding: 0.5rem;
  border: 1px solid var(--border);
}

.category-card.col-efficacy { background: #DCFCE7; border-color: #86EFAC; }
.category-card.col-paraphrase { background: #FEF3C7; border-color: #FDE68A; }
.category-card.col-neighborhood { background: #FFEDD5; border-color: #FED7AA; }
.category-card.col-generation { background: #F8FAFC; border-color: #E2E8F0; }

.cat-header {
  font-size: 0.72rem;
  font-weight: 800;
  margin-bottom: 0.35rem;
}

.cat-eff { color: #15803D; }
.cat-para { color: #B45309; }
.cat-neigh { color: #C2410C; }
.cat-gen { color: #475569; }

.eval-pill {
  background: #FFFFFF;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 4px;
  padding: 0.35rem 0.5rem;
  font-size: 0.72rem;
  line-height: 1.35;
  margin-bottom: 0.25rem;
  color: #1E293B;
}

.eval-pass { border-left: 3px solid #16A34A; }
.eval-unknown { border-left: 3px solid #94A3B8; }
.eval-neigh { border-left: 3px solid #EA580C; }
.eval-neutral { border-left: 3px solid #D97706; }
.eval-gen { border-left: 3px solid #64748B; }

/* Bottom Row Diagnostics: Diff & Drift */
.bottom-diagnostics-row {
  display: grid;
  grid-template-columns: 1fr 1.35fr;
  gap: 0.85rem;
  margin-top: 0.85rem;
}

.diff-viewer, .drift-view-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.75rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.chart-legend-row {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin-top: 0.25rem;
  margin-bottom: 0.35rem;
}

.legend-chip {
  font-size: 0.65rem;
  padding: 1px 6px;
  border-radius: 3px;
  font-family: var(--font-mono);
  font-weight: 600;
}

.legend-chip.pre { background: #CBD5E1; color: #1E293B; }
.legend-chip.post { background: #3B82F6; color: #FFFFFF; }
.legend-chip.edited { background: #EF4444; color: #FFFFFF; }


.diff-body {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #F8FAFC;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 0.7rem;
  font-size: 0.8rem;
  line-height: 1.6;
  color: var(--text-main);
}

.diff-del {
  background: #FEE2E2;
  color: #DC2626;
  text-decoration: line-through;
  padding: 0 3px;
  border-radius: 2px;
}

.diff-ins {
  background: #DCFCE7;
  color: #16A34A;
  text-decoration: underline;
  font-weight: 600;
  padding: 0 3px;
  border-radius: 2px;
}

/* Drift View Panel Layout matching Fig 3 D */
.drift-view-panel {
  display: grid;
  grid-template-columns: 210px 1fr;
  gap: 0.75rem;
  align-items: start;
}

.drift-left-col {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.drift-controls-header {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.drift-action-buttons {
  display: flex;
  gap: 0.35rem;
}

.btn-tool-icon {
  background: #1E293B;
  color: #FFFFFF;
  border: 1px solid #334155;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s;
}

.btn-tool-icon:hover { background: #334155; }
.btn-tool-icon.active { background: #0284C7; border-color: #0284C7; }

.drift-legend-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--text-muted);
  margin-top: 0.2rem;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.dot-pre {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background-color: #FB7185;
}

.dot-post {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background-color: #A5B4FC;
}

.drift-scatter-box {
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.2rem;
}

.drift-svg { width: 100%; height: 115px; }

.drift-right-col {
  overflow-x: auto;
}

.drift-detail-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.72rem;
}

.drift-detail-table th {
  color: var(--text-faint);
  font-size: 0.68rem;
  font-weight: 600;
  padding: 0.35rem 0.5rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
  background: #F8FAFC;
}

.drift-detail-table td {
  padding: 0.45rem 0.5rem;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
  line-height: 1.45;
}

.prompt-cell {
  font-weight: 600;
  color: var(--text-main);
}

.output-change-cell {
  color: var(--text-muted);
}

.diff-del-inline {
  color: #F87171;
  text-decoration: line-through;
  margin-right: 3px;
}

.diff-ins-inline {
  color: #16A34A;
  text-decoration: underline;
  font-weight: 600;
  margin-right: 3px;
}

.drift-score-cell {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-main);
  text-align: right;
}

.legend-item.dimmed {
  opacity: 0.35;
}

.drift-detail-table tr.row-selected {
  background-color: #E0F2FE;
}

/* Right Diagnostics Column (C: DiffViewer + D: DriftScatterPlot) */
.right-diagnostics-column {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

/* Evaluation Status Badges */
.eval-fail {
  border-left: 3px solid #EF4444 !important;
  background: #FEF2F2 !important;
  color: #991B1B !important;
}

/* General UI & Hint Utilities */
.hint {
  font-size: 0.74rem;
  color: #64748B;
  line-height: 1.4;
}

.chart-card {
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 0.6rem;
}

.chart-card.empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 160px;
}

.spinner {
  width: 14px;
  height: 14px;
  border: 2px solid #BFDBFE;
  border-top-color: #2563EB;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Responsive Breakpoints */
@media (max-width: 1440px) {
  .main-layout-grid {
    grid-template-columns: 270px minmax(0, 1fr);
  }
  .right-diagnostics-column {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 1100px) {
  .top-navbar {
    height: auto;
    min-height: 50px;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding: 0.6rem;
  }
  .nav-left, .nav-right, .action-buttons-group { flex-wrap: wrap; }
  .main-layout-grid {
    grid-template-columns: minmax(0, 1fr);
  }
  .right-diagnostics-column {
    grid-template-columns: 1fr;
  }
  .edit-signals-row {
    grid-template-columns: 240px 110px 480px 240px;
  }
}

@media (max-width: 700px) {
  .category-columns-grid, .bottom-diagnostics-row, .drift-view-panel {
    grid-template-columns: minmax(0, 1fr);
  }
  .nav-left, .nav-right { gap: 0.4rem; }
  .layer-actions { flex-wrap: wrap; }
  .drift-detail-table { table-layout: fixed; }
  .drift-detail-table td { overflow-wrap: anywhere; }
}
```

<a id="replacement-4"></a>

### frontend/src/App.tsx

Current file: [frontend/src/App.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/App.tsx).

SHA-256: 522b4c030b612049a8af8e1b35590d435f3d475b3ca68b7132954d3429ca1c93

```tsx
import { useCallback, useEffect, useState, useMemo, useRef } from "react";
import * as api from "./api/client";
import { CosineSimilarityChart } from "./components/CosineSimilarityChart";
import { CosineSimilarityCompareChart } from "./components/CosineSimilarityCompareChart";
import { DiffViewer } from "./components/DiffViewer";
import { DriftScatterPlot } from "./components/DriftScatterPlot";
import { DEFAULT_FACT, DEFAULT_SCHEMES, FactForm, parseSchemesText } from "./components/FactForm";
import { KnowledgeGraph } from "./components/KnowledgeGraph";
import { LayerSelector } from "./components/LayerSelector";
import { PromptDetailCards } from "./components/PromptDetailCards";
import { SchemeComparisonTable } from "./components/SchemeComparisonTable";
import { TokenRankingChart } from "./components/TokenRankingChart";
import { WireframeLinker } from "./components/WireframeLinker";
import type { CompareResponse, EditResponse, LayerSignal } from "./types";
import { comparisonSchemes, schemeKey, sortSchemes } from "./schemes";
import "./App.css";

export default function App() {
  const [fact, setFact] = useState(DEFAULT_FACT);
  const [schemesText] = useState(DEFAULT_SCHEMES);
  const [selectedLayers, setSelectedLayers] = useState<number[]>([13, 14, 15, 16, 17]);
  const [signals, setSignals] = useState<LayerSignal[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editResult, setEditResult] = useState<EditResponse | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [lensView, setLensView] = useState<"subject" | "last">("last");
  const [modelName, setModelName] = useState("gpt2-xl");
  const nLayers = modelName === "gpt2-xl" ? 48 : 28;
  const MODELS = ["gpt2-xl", "EleutherAI/gpt-j-6B"];
  const [method, setMethod] = useState<"memit" | "rome">("memit");
  const [elapsed, setElapsed] = useState(0);
  
  // Scheme selection keyed by layer string (e.g. "13-14-15-16-17"), NOT array index
  const [selectedSchemeKey, setSelectedSchemeKey] = useState<string | null>(null);

  // Request ID / model-stamp guard to eliminate async race conditions
  const activeRequestId = useRef(0);
  const activeModelRef = useRef(modelName);
  const healthRequestId = useRef(0);

  const invalidate = useCallback(() => {
    activeRequestId.current++;
    setLoading(false);
    setElapsed(0);
    setError(null);
    setSignals([]);
    setEditResult(null);
    setCompareResult(null);
    setSelectedSchemeKey(null);
  }, []);

  const changeModel = (next: string) => {
    if (next === modelName) return;
    invalidate();
    activeModelRef.current = next;
    setModelName(next);
    const layers = next === "gpt2-xl" ? [13, 14, 15, 16, 17] : [3, 4, 5, 6, 7, 8];
    setSelectedLayers(method === "rome" ? [layers[0]] : layers);
  };

  const changeMethod = (next: "memit" | "rome") => {
    if (next === method) return;
    invalidate();
    setMethod(next);
    setSelectedLayers((prev) => {
      const start = prev[0] ?? (modelName === "gpt2-xl" ? 13 : 5);
      return next === "rome" ? [start] : Array.from({ length: Math.min(5, nLayers - start) }, (_, i) => start + i);
    });
  };

  const changeFact = (next: typeof fact) => {
    invalidate();
    setFact(next);
  };

  // Single guarded checkHealth
  const checkHealth = useCallback(async (targetModel: string) => {
    const reqId = ++healthRequestId.current;
    const operationId = activeRequestId.current;
    try {
      const h = await api.health(targetModel);
      if (h.model !== targetModel || h.n_layers !== (targetModel === "gpt2-xl" ? 48 : 28)) {
        throw new Error("API model metadata does not match the selected model.");
      }
    } catch (e) {
      if (reqId === healthRequestId.current && operationId === activeRequestId.current && activeModelRef.current === targetModel) {
        setError(e instanceof Error ? e.message : "Cannot reach API");
      }
    }
  }, []);

  useEffect(() => {
    checkHealth(modelName);
    return () => { healthRequestId.current++; };
  }, [modelName, checkHealth]);

  useEffect(() => {
    if (!loading) return;
    const timer = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(timer);
  }, [loading]);

  useEffect(() => () => { activeRequestId.current++; }, []);

  const parsedSchemes = useMemo(() => {
    const layersList = compareResult
      ? sortSchemes(compareResult.schemes).map((s) => s.layers)
      : comparisonSchemes(parseSchemesText(schemesText), method, nLayers);
    return layersList.map((layers, idx) => ({
      key: schemeKey(layers),
      layers,
      label: `Scheme ${idx + 1}`,
    }));
  }, [schemesText, method, nLayers, compareResult]);

  const selectedScheme = compareResult?.schemes.find((s) => schemeKey(s.layers) === selectedSchemeKey);
  const selectScheme = (key: string, layers: number[]) => {
    setSelectedSchemeKey(key);
    setSelectedLayers(method === "rome" ? layers.slice(0, 1) : layers);
  };

  const runProbe = useCallback(async () => {
    const targetModel = activeModelRef.current;
    const reqId = ++activeRequestId.current;
    setLoading(true);
    setError(null);
    setElapsed(0);
    try {
      const res = await api.probe({ prompt: fact.prompt, subject: fact.subject, model: targetModel });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) {
        setSignals(res.layer_signals);
      }
    } catch (e) {
      if (reqId === activeRequestId.current) {
        setError(e instanceof Error ? e.message : "Probe failed");
      }
    } finally {
      if (reqId === activeRequestId.current) {
        setLoading(false);
      }
    }
  }, [fact.prompt, fact.subject]);

  const runEdit = useCallback(async () => {
    if (selectedLayers.length === 0) {
      setError("Select at least one layer before editing.");
      return;
    }
    const targetModel = activeModelRef.current;
    const reqId = ++activeRequestId.current;
    setLoading(true);
    setError(null);
    setElapsed(0);
    try {
      const res = await api.edit({ ...fact, layers: selectedLayers, method, model: targetModel });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) {
        setEditResult(res);
        setCompareResult(null);
        setSelectedSchemeKey(null);
        setSignals(res.pre_edit.layer_signals);
      }
    } catch (e) {
      if (reqId === activeRequestId.current) {
        setError(e instanceof Error ? e.message : "Edit failed");
      }
    } finally {
      if (reqId === activeRequestId.current) {
        setLoading(false);
      }
    }
  }, [fact, selectedLayers, method]);

  const runCompare = useCallback(async () => {
    const targetModel = activeModelRef.current;
    const reqId = ++activeRequestId.current;
    setLoading(true);
    setError(null);
    setElapsed(0);
    try {
      const schemes = comparisonSchemes(parseSchemesText(schemesText), method, nLayers);
      if (schemes.length === 0) throw new Error("Add at least one scheme.");
      const res = await api.compare({ ...fact, schemes, method, model: targetModel });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) {
        setCompareResult(res);
        setEditResult(null);
        const first = sortSchemes(res.schemes)[0];
        setSelectedSchemeKey(first ? schemeKey(first.layers) : null);
        if (first) setSelectedLayers(first.layers);
        setSignals(res.baseline.layer_signals);
      }
    } catch (e) {
      if (reqId === activeRequestId.current) {
        setError(e instanceof Error ? e.message : "Compare failed");
      }
    } finally {
      if (reqId === activeRequestId.current) {
        setLoading(false);
      }
    }
  }, [fact, schemesText, method, nLayers]);

  const handleRecommend = useCallback(() => {
    if (signals.length === 0) {
      runProbe();
      return;
    }
    if (method === "rome") {
      let minL = 0;
      let minVal = Infinity;
      signals.forEach((s) => {
        const v = Math.abs(s.cosine_similarity);
        if (v < minVal) {
          minVal = v;
          minL = s.layer;
        }
      });
      setSelectedLayers([minL]);
      return;
    }
    const windowSize = 5;
    let bestStart = 0;
    let minScore = Infinity;
    for (let i = 0; i <= nLayers - windowSize; i++) {
      let sum = 0;
      for (let j = 0; j < windowSize; j++) {
        sum += Math.abs(signals[i + j]?.cosine_similarity ?? 1);
      }
      if (sum < minScore) {
        minScore = sum;
        bestStart = i;
      }
    }
    setSelectedLayers(Array.from({ length: windowSize }, (_, k) => bestStart + k));
  }, [signals, method, nLayers, runProbe]);

  // Memoized layer click handler
  const handleLayerClick = useCallback((layer: number) => {
    if (!Number.isInteger(layer) || layer < 0 || layer >= nLayers) return;
    setSelectedLayers((prev) => {
      if (method === "rome") {
        return prev.includes(layer) ? [] : [layer];
      }
      return prev.includes(layer)
        ? prev.filter((l) => l !== layer)
        : [...prev, layer].sort((a, b) => a - b);
    });
  }, [method, nLayers]);

  const selectedLayerLabel = useMemo(() => {
    if (selectedLayers.length === 0) return "None";
    const minL = Math.min(...selectedLayers);
    const maxL = Math.max(...selectedLayers);
    return minL === maxL ? `${minL}` : `${minL}-${maxL}`;
  }, [selectedLayers]);

  return (
    <div className="keditvis-app">
      {/* Top Navigation Bar */}
      <header className="top-navbar">
        <div className="nav-left">
          <div className="logo-badge">
            <span className="logo-title">KEditVis</span>
          </div>
          <select
            className="model-dropdown-select"
            value={modelName}
            onChange={(e) => changeModel(e.target.value)}
          >
            {MODELS.map((m) => (
              <option key={m} value={m}>
                {m === "gpt2-xl" ? "GPT2-XL / 48" : "GPT-J-6B / 28"}
              </option>
            ))}
          </select>
          <div className="method-pill-group">
            <button
              type="button"
              className={method === "memit" ? "active" : ""}
              onClick={() => changeMethod("memit")}
              title="Mass-Editing Memory in a Transformer (contiguous multi-layer range)"
            >
              MEMIT
            </button>
            <button
              type="button"
              className={method === "rome" ? "active" : ""}
              onClick={() => changeMethod("rome")}
              title="Rank-One Model Editing (single-layer critical point)"
            >
              ROME
            </button>
          </div>
        </div>

        <div className="nav-right">
          <div className="selected-layers-pill">
            <span className="pill-lbl">SELECTED LAYERS</span>
            <span className="pill-val">{selectedLayerLabel}</span>
          </div>

          <div className="action-buttons-group">
            <button type="button" className="btn-action" onClick={handleRecommend} disabled={loading}>
              🪄 Recommend
            </button>
            <button type="button" className="btn-action" onClick={runCompare} disabled={loading}>
              📊 Compare
            </button>
            <button
              type="button"
              className="btn-action"
              onClick={() => {
                setEditResult(null);
                setCompareResult(null);
                runProbe();
              }}
              disabled={loading}
            >
              ↺ Revert
            </button>
            <button type="button" className="btn-action btn-primary" onClick={runEdit} disabled={loading}>
              ✏️ Edit
            </button>
          </div>
        </div>
      </header>

      {/* Loading telemetry banner */}
      {loading && (
        <div className="telemetry-bar">
          <div className="spinner" />
          <span>Executing {method.toUpperCase()} on {modelName}… ({elapsed}s elapsed)</span>
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {/* Main Layout Grid: Authentic Paper Architecture (Left: Input A1-A3, Right: Main Workspace B1-B4 + C & D) */}
      <main className="main-layout-grid">
        {/* Left Sidebar: Input & Entity Panel (A1 -> A2 -> A3) */}
        <aside className="left-sidebar">
          <FactForm
            value={fact}
            onChange={changeFact}
            disabled={loading}
            knowledgeGraphSlot={
              <KnowledgeGraph
                subject={fact.subject}
                target={fact.target_new}
                onSelectSubject={(subj) => changeFact({ ...fact, subject: subj })}
              />
            }
          />
        </aside>

        {/* Center/Main Workspace: Edit View (B1, B2, B3, B4) + Prompt Detail Cards + Diagnostics (C, D) */}
        <section className="center-canvas">
          <div className="edit-view-panel">
            <div className="edit-view-header">
              <div className="view-title">
                <h3>Edit View</h3>
              </div>
              <div className="lens-view-toggle">
                <button
                  type="button"
                  className={lensView === "subject" ? "active" : ""}
                  onClick={() => setLensView("subject")}
                >
                  SUBJECT TOKEN
                </button>
                <button
                  type="button"
                  className={lensView === "last" ? "active" : ""}
                  onClick={() => setLensView("last")}
                >
                  LAST TOKEN
                </button>
              </div>
            </div>

            {/* Central Signal & Comparison Grid */}
            <div className="edit-signals-row">
              {/* Left Lens: Initial Model (B1) */}
              <div className="lens-column lens-pre">
                <div className="version-tag">
                  <span className="badge-tag">B1</span>
                  <span>Version 0 (Initial Model)</span>
                </div>
                <div className="charts-pair">
                  <TokenRankingChart signals={signals} view={lensView} />
                  <CosineSimilarityChart
                    signals={signals}
                    selectedLayers={selectedLayers}
                    onSelectLayer={handleLayerClick}
                  />
                </div>
              </div>

              {/* Center: Wireframe Set Linker (B2) */}
              <div className="wireframe-column">
                <div className="version-tag">
                  <span className="badge-tag">B2</span>
                </div>
                <WireframeLinker
                  nLayers={nLayers}
                  schemes={parsedSchemes}
                  selectedSchemeKey={selectedSchemeKey}
                  onSelectSchemeKey={selectScheme}
                  selectedLayers={selectedLayers}
                />
              </div>

              {/* Comparison Table (B3) */}
              <div className="table-column">
                {compareResult ? (
                  <SchemeComparisonTable
                    data={compareResult}
                    selectedSchemeKey={selectedSchemeKey}
                    onSelectSchemeKey={selectScheme}
                  />
                ) : (
                  <div className="table-placeholder">
                    <div className="table-header-bar">
                      <span className="badge-tag">B3</span>
                      <h4>Editing results preview for different schemes</h4>
                    </div>
                    <div className="placeholder-content">
                      <p style={{ marginBottom: "0.8rem" }}>Click <strong>Compare</strong> in the top bar to preview multi-scheme metrics.</p>
                      <LayerSelector
                        nLayers={nLayers}
                        selected={selectedLayers}
                        onChange={setSelectedLayers}
                        signals={signals}
                        method={method}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Right Lens: Edited Model (B4) */}
              <div className="lens-column lens-post">
                <div className="version-tag">
                  <span className="badge-tag">B4</span>
                  <span>Version 1 (Model after edit)</span>
                </div>
                {editResult ? (
                  <div className="charts-pair">
                    <TokenRankingChart
                      signals={editResult.post_edit.layer_signals}
                      view={lensView}
                    />
                    <CosineSimilarityCompareChart
                      preSignals={editResult.pre_edit.layer_signals}
                      postSignals={editResult.post_edit.layer_signals}
                      editedLayers={editResult.edited_layers}
                    />
                  </div>
                ) : (
                  <div className="empty-post-lens">
                    <p className="hint" style={{ padding: "1rem", textAlign: "center", color: "var(--text-faint)" }}>
                      Apply an edit to view post-edit token trajectories and layer shifts.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Detailed Prompt Evaluation Cards with Real Pass/Fail Metrics */}
            <PromptDetailCards
              prompt={fact.prompt}
              subject={fact.subject}
              targetNew={fact.target_new}
              targetTrue={fact.target_true}
              paraphrasePrompts={fact.paraphrase_prompts}
              neighborhoodPrompts={fact.neighborhood_prompts}
              generations={editResult ? editResult.post_edit.generations : selectedScheme ? [selectedScheme.generation] : []}
              metrics={editResult ? editResult.post_edit.metrics : selectedScheme?.metrics}
            />
          </div>

          {/* Full-Width Bottom Diagnostics Row: Output Comparison (C) & Drift View (D) */}
          <div className="bottom-diagnostics-row">
            <DiffViewer
              preText={editResult ? editResult.pre_edit.generations?.[0] : compareResult?.baseline.generation}
              postText={editResult ? editResult.post_edit.generations?.[0] : selectedScheme?.generation}
            />
            <DriftScatterPlot
              damageScore={editResult ? editResult.damage?.kl_divergence : selectedScheme?.damage?.kl_divergence}
              rows={editResult ? editResult.neighborhood : selectedScheme?.neighborhood}
            />
          </div>
        </section>
      </main>
    </div>
  );
}
```

<a id="replacement-5"></a>

### frontend/src/components/CosineSimilarityCompareChart.tsx

Current file: [frontend/src/components/CosineSimilarityCompareChart.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/components/CosineSimilarityCompareChart.tsx).

SHA-256: 66c3012899a102f23d398fb848b28751374c1235aa26ca6404eacae33b6ccfc8

```tsx
import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { LayerSignal } from "../types";

interface Props {
  preSignals: LayerSignal[];
  postSignals: LayerSignal[];
  editedLayers?: number[];
}

function activity(cos: number): number {
  return Math.max(0, Math.min(1, 1 - Math.abs(cos)));
}

/** Paired pre/post bars per layer — KEditVis before/after editing view. */
export function CosineSimilarityCompareChart({
  preSignals,
  postSignals,
  editedLayers = [],
}: Props) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    d3.select(ref.current).selectAll("*").remove();
    if (preSignals.length === 0 || postSignals.length === 0) return;

    const postByLayer = new Map(postSignals.map((s) => [s.layer, s]));
    const rows = preSignals
      .map((pre) => ({
        layer: pre.layer,
        pre: activity(pre.cosine_similarity),
        post: activity(postByLayer.get(pre.layer)?.cosine_similarity ?? pre.cosine_similarity),
      }))
      .filter((r) => postByLayer.has(r.layer) && Number.isFinite(r.pre) && Number.isFinite(r.post));

    if (rows.length === 0) return;

    const edited = new Set(editedLayers);
    const width = 260;
    const height = Math.max(260, rows.length * 12 + 40);
    const margin = { top: 22, right: 12, bottom: 20, left: 30 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const y = d3
      .scaleBand<number>()
      .domain(rows.map((d) => d.layer))
      .range([0, innerH])
      .padding(0.2);

    const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
    const barH = y.bandwidth() / 2 - 1;

    g.selectAll("rect.pre")
      .data(rows)
      .join("rect")
      .attr("class", "pre")
      .attr("y", (d) => y(d.layer)!)
      .attr("x", 0)
      .attr("height", barH)
      .attr("width", (d) => x(d.pre))
      .attr("fill", "#CBD5E1")
      .attr("rx", 1.5);

    g.selectAll("rect.post")
      .data(rows)
      .join("rect")
      .attr("class", "post")
      .attr("y", (d) => y(d.layer)! + barH + 2)
      .attr("x", 0)
      .attr("height", barH)
      .attr("width", (d) => x(d.post))
      .attr("fill", (d) => (edited.has(d.layer) ? "#EF4444" : "#3B82F6"))
      .attr("rx", 1.5);

    g.selectAll("text.label")
      .data(rows)
      .join("text")
      .attr("class", "label")
      .attr("x", -4)
      .attr("y", (d) => y(d.layer)! + y.bandwidth() / 2)
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "middle")
      .attr("font-size", 8)
      .attr("font-family", "var(--font-mono)")
      .attr("fill", (d) => (edited.has(d.layer) ? "#EF4444" : "#64748B"))
      .attr("font-weight", (d) => (edited.has(d.layer) ? 700 : 400))
      .text((d) => (d.layer % 4 === 0 || d.layer === rows.length - 1 ? d.layer : ""));
  }, [preSignals, postSignals, editedLayers]);

  if (preSignals.length === 0 || postSignals.length === 0) {
    return (
      <div className="chart-card empty">
        <h4>Cosine shift comparison</h4>
        <p className="hint">No comparison data</p>
      </div>
    );
  }

  return (
    <div className="chart-card">
      <div className="chart-header">
        <h4>Cosine shift (Pre vs Post)</h4>
        <div className="chart-legend-row">
          <span className="legend-chip pre">Pre</span>
          <span className="legend-chip post">Post</span>
          <span className="legend-chip edited">Edited</span>
        </div>
      </div>
      <div className="chart-svg-wrap" style={{ maxHeight: "380px", overflowY: "auto" }}>
        <svg ref={ref} style={{ width: "100%", height: "auto" }} />
      </div>
    </div>
  );
}
```

<a id="replacement-6"></a>

### frontend/src/components/DiffViewer.tsx

Current file: [frontend/src/components/DiffViewer.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/components/DiffViewer.tsx).

SHA-256: dde107b39f75bd94478a51b4c20691e07dc7bc79267fab703e228a6cc822bb7d

```tsx
import React, { useMemo } from "react";

interface Props {
  preText?: string | null;
  postText?: string | null;
  title?: string;
}

interface DiffChunk {
  type: "same" | "del" | "ins";
  text: string;
}

export function computeWordDiff(oldStr: string, newStr: string): DiffChunk[] {
  if (!oldStr && !newStr) return [];
  if (!oldStr) return [{ type: "ins", text: newStr }];
  if (!newStr) return [{ type: "del", text: oldStr }];
  if (oldStr === newStr) return [{ type: "same", text: oldStr }];

  // Bound the LCS matrix; preserve the remainder as a coarse replacement.
  const maxTokens = 300;
  const tokenize = (text: string) => {
    const pattern = /\s+|[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu;
    const words: string[] = [];
    let end = 0;
    let match: RegExpExecArray | null;
    while (words.length < maxTokens && (match = pattern.exec(text))) {
      words.push(match[0]);
      end = pattern.lastIndex;
    }
    return { words, tail: text.slice(end) };
  };
  const oldTokens = tokenize(oldStr);
  const newTokens = tokenize(newStr);
  const oldWords = oldTokens.words;
  const newWords = newTokens.words;

  const dp: number[][] = Array(oldWords.length + 1)
    .fill(0)
    .map(() => Array(newWords.length + 1).fill(0));

  for (let i = 0; i < oldWords.length; i++) {
    for (let j = 0; j < newWords.length; j++) {
      if (oldWords[i] === newWords[j]) {
        dp[i + 1][j + 1] = dp[i][j] + 1;
      } else {
        dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }

  let i = oldWords.length;
  let j = newWords.length;
  const chunks: DiffChunk[] = [];

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldWords[i - 1] === newWords[j - 1]) {
      chunks.unshift({ type: "same", text: oldWords[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      chunks.unshift({ type: "ins", text: newWords[j - 1] });
      j--;
    } else if (i > 0) {
      chunks.unshift({ type: "del", text: oldWords[i - 1] });
      i--;
    }
  }

  if (oldTokens.tail === newTokens.tail) {
    if (oldTokens.tail) chunks.push({ type: "same", text: oldTokens.tail });
  } else {
    if (oldTokens.tail) chunks.push({ type: "del", text: oldTokens.tail });
    if (newTokens.tail) chunks.push({ type: "ins", text: newTokens.tail });
  }

  // Merge adjacent same-type chunks
  const merged: DiffChunk[] = [];
  for (const c of chunks) {
    if (merged.length > 0 && merged[merged.length - 1].type === c.type) {
      merged[merged.length - 1].text += c.text;
    } else {
      merged.push({ ...c });
    }
  }
  return merged;
}

export const DiffViewer: React.FC<Props> = ({
  preText,
  postText,
  title = "Output Comparison",
}) => {
  const chunks = useMemo(() => {
    return computeWordDiff(preText || "", postText || "");
  }, [preText, postText]);

  if (!preText && !postText) {
    return (
      <div className="diff-viewer empty">
        <div className="panel-header-sub">
          <span className="badge-tag">C</span>
          <h4>{title}</h4>
        </div>
        <p className="hint" style={{ padding: "0.8rem" }}>Run an edit to inspect token-level generation differences.</p>
      </div>
    );
  }

  return (
    <div className="diff-viewer">
      <div className="panel-header-sub">
        <span className="badge-tag">C</span>
        <h4>{title}</h4>
      </div>
      <div className="diff-body">
        {chunks.map((c, idx) => {
          if (c.type === "del") {
            return (
              <span key={idx} className="diff-del" title="Pre-edit text (removed)">
                {c.text}
              </span>
            );
          }
          if (c.type === "ins") {
            return (
              <span key={idx} className="diff-ins" title="Post-edit text (added)">
                {c.text}
              </span>
            );
          }
          return <span key={idx}>{c.text}</span>;
        })}
      </div>
    </div>
  );
};
```

<a id="replacement-7"></a>

### frontend/src/components/DriftScatterPlot.tsx

Current file: [frontend/src/components/DriftScatterPlot.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/components/DriftScatterPlot.tsx).

SHA-256: b1548c5858e689af105e7ad7c87e708aed7fd41a5a96e5eb2800273a5e55d4a7

```tsx
import React, { useMemo, useState, useRef, useEffect } from "react";
import type { NeighborhoodResult } from "../types";
import { computeWordDiff } from "./DiffViewer";

interface Props {
  damageScore?: number | null;
  rows?: NeighborhoodResult[];
}

type Point = { id: string; row: number; type: "pre" | "post"; x: number; y: number };
type Box = { x1: number; y1: number; x2: number; y2: number };
const EMPTY_ROWS: NeighborhoodResult[] = [];

export const DriftScatterPlot: React.FC<Props> = ({ damageScore, rows = EMPTY_ROWS }) => {
  const [visibility, setVisibility] = useState<"all" | "pre" | "post">("all");
  const [selected, setSelected] = useState<string[] | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number; pointer: number } | null>(null);
  const points = useMemo<Point[]>(() => rows.flatMap((row, index) => {
    if (!row.projection) return [];
    return (["pre", "post"] as const).flatMap((type) => {
      const [x, y] = row.projection![type];
      return Number.isFinite(x) && Number.isFinite(y) ? [{ id: `${index}-${type}`, row: index, type, x, y }] : [];
    });
  }), [rows]);
  const domain = (values: number[]): [number, number] => {
    if (!values.length) return [0, 1];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const pad = (max - min || 1) * 0.1;
    return [min - pad, max + pad];
  };
  const [minX, maxX] = domain(points.map((p) => p.x));
  const [minY, maxY] = domain(points.map((p) => p.y));
  const scaleX = (x: number) => 30 + (x - minX) / (maxX - minX) * 162;
  const scaleY = (y: number) => 110 - (y - minY) / (maxY - minY) * 100;
  const visible = points.filter((p) => visibility === "all" || p.type === visibility);
  const selectedRows = new Set(points.filter((p) => selected?.includes(p.id)).map((p) => p.row));

  useEffect(() => {
    setSelected(null);
    setBox(null);
    drag.current = null;
  }, [rows, visibility]);

  const coordinates = (clientX: number, clientY: number) => {
    const ctm = svgRef.current?.getScreenCTM();
    if (!ctm) return null;
    const p = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse());
    return Number.isFinite(p.x) && Number.isFinite(p.y) ? { x: p.x, y: p.y } : null;
  };
  const clear = () => { drag.current = null; setBox(null); };
  const down = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0 || !points.length) return;
    const p = coordinates(e.clientX, e.clientY);
    if (!p) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { ...p, pointer: e.pointerId };
    setBox({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
  };
  const move = (e: React.PointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    const p = coordinates(e.clientX, e.clientY);
    if (!start || start.pointer !== e.pointerId || !p) return;
    setBox({ x1: start.x, y1: start.y, x2: p.x, y2: p.y });
  };
  const up = (e: React.PointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    const p = coordinates(e.clientX, e.clientY);
    if (!start || start.pointer !== e.pointerId) return;
    if (p) {
      const x1 = Math.min(start.x, p.x), x2 = Math.max(start.x, p.x);
      const y1 = Math.min(start.y, p.y), y2 = Math.max(start.y, p.y);
      setSelected(x2 - x1 < 3 && y2 - y1 < 3 ? null : visible.filter((point) =>
        scaleX(point.x) >= x1 && scaleX(point.x) <= x2 &&
        scaleY(point.y) >= y1 && scaleY(point.y) <= y2
      ).map((point) => point.id));
    }
    clear();
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };

  return (
    <div className="drift-view-panel">
      <div className="drift-left-col">
        <div className="drift-controls-header">
          <div className="title-with-badge"><h4>Drift View</h4><span className="badge-tag">D</span></div>
          <div className="drift-action-buttons">
            <button type="button" className="btn-tool-icon" title="Clear region selection" onClick={() => { setSelected(null); clear(); }} disabled={!points.length}>{"\u2922"}</button>
            <button type="button" className="btn-tool-icon" title={`Toggle points visibility (current: ${visibility})`} onClick={() => setVisibility((v) => v === "all" ? "post" : v === "post" ? "pre" : "all")} disabled={!points.length}>{"\u25c9"}</button>
          </div>
          <div className="drift-legend-group"><span className="dot-pre" />Pre-edit <span className="dot-post" />Post-edit</div>
          <div className="mono-cell">Reference KL: {damageScore == null ? "Unavailable" : damageScore.toExponential(3)}</div>
        </div>
        <div className="drift-scatter-box">
          {points.length ? (
            <svg ref={svgRef} viewBox="0 0 200 130" className="drift-svg" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={clear} onLostPointerCapture={clear} style={{ touchAction: "none", cursor: "crosshair" }}>
              <path d="M30 10V110H192" fill="none" stroke="#94A3B8" />
              {visibility === "all" && rows.map((row, i) => row.projection && row.projection.pre.every(Number.isFinite) && row.projection.post.every(Number.isFinite) ? (
                <line key={i} x1={scaleX(row.projection.pre[0])} y1={scaleY(row.projection.pre[1])} x2={scaleX(row.projection.post[0])} y2={scaleY(row.projection.post[1])} stroke="#CBD5E1" />
              ) : null)}
              {visible.map((p) => (
                <circle key={p.id} data-point-id={p.id} cx={scaleX(p.x)} cy={scaleY(p.y)} r={selected?.includes(p.id) ? 3.4 : 2.8} fill={p.type === "pre" ? "#FB7185" : "#A5B4FC"} stroke={selected?.includes(p.id) ? "#0284C7" : "#64748B"} opacity={selected !== null && !selected.includes(p.id) ? 0.3 : 1} onPointerDown={(e) => e.stopPropagation()} onClick={() => setSelected([p.id])}><title>{rows[p.row].prompt}: {p.type}</title></circle>
              ))}
              {box && <rect x={Math.min(box.x1, box.x2)} y={Math.min(box.y1, box.y2)} width={Math.abs(box.x2 - box.x1)} height={Math.abs(box.y2 - box.y1)} fill="rgba(2,132,199,0.12)" stroke="#0284C7" pointerEvents="none" />}
            </svg>
          ) : <p className="hint">Projection unavailable</p>}
        </div>
      </div>
      <div className="drift-right-col">
        <table className="drift-detail-table">
          <thead><tr><th>Prompt</th><th>Output Change</th><th title="Per-token KL on this neighborhood prompt">KL Drift</th></tr></thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} data-drift-row={i} className={selected !== null && selectedRows.has(i) ? "row-selected" : ""} style={{ opacity: selected === null || selectedRows.has(i) ? 1 : 0.35 }}>
                <td className="prompt-cell">{row.prompt}</td>
                <td className="output-change-cell" style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
                  {computeWordDiff(row.pre_text, row.post_text).map((chunk, j) => <span key={j} className={chunk.type === "del" ? "diff-del-inline" : chunk.type === "ins" ? "diff-ins-inline" : undefined}>{chunk.text}</span>)}
                </td>
                <td className="drift-score-cell">{row.kl_divergence.toExponential(3)}</td>
              </tr>
            ))}
            {!rows.length && <tr><td colSpan={3}>No neighborhood measurements</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
};
```

<a id="replacement-8"></a>

### frontend/src/components/FactForm.tsx

Current file: [frontend/src/components/FactForm.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/components/FactForm.tsx).

SHA-256: e4fca961ab7c820355c7c98750bd66cfceb327ae10e942aa32c32e1e67035e6f

```tsx
import React, { useState } from "react";
import type { FactInput } from "../types";

interface Props {
  value: FactInput;
  onChange: (next: FactInput) => void;
  disabled?: boolean;
  knowledgeGraphSlot?: React.ReactNode;
}

export function splitLines(text: string): string[] {
  return text
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

export function joinLines(items: string[]): string {
  return items.join("\n");
}

export const FactForm: React.FC<Props> = ({
  value,
  onChange,
  disabled,
  knowledgeGraphSlot,
}) => {
  const [mode, setMode] = useState<"completion" | "rewrite">("completion");
  const [isEditingPrompts, setIsEditingPrompts] = useState(false);

  const set = <K extends keyof FactInput>(key: K, v: FactInput[K]) =>
    onChange({ ...value, [key]: v });

  const [paraphraseText, setParaphraseText] = useState(() =>
    joinLines(value.paraphrase_prompts),
  );
  const [neighborhoodText, setNeighborhoodText] = useState(() =>
    joinLines(value.neighborhood_prompts),
  );

  const filledPrompt = value.prompt.includes("{}")
    ? value.prompt.replace("{}", value.subject)
    : value.prompt.includes(value.subject) ? value.prompt : `${value.subject} ${value.prompt}`;

  return (
    <div className="fact-manager-panel">
      {/* Panel A1: LLM Chat Header */}
      <div className="chat-card">
        <div className="card-header">
          <div className="title-with-badge">
            <span className="badge-tag">A1</span>
            <h3>LLM Chat</h3>
          </div>
          <div className="chat-mode-toggle">
            <button
              type="button"
              className={mode === "completion" ? "active" : ""}
              onClick={() => setMode("completion")}
            >
              COMPLETION
            </button>
            <button
              type="button"
              className={mode === "rewrite" ? "active" : ""}
              onClick={() => setMode("rewrite")}
            >
              REWRITE
            </button>
          </div>
        </div>
        <div className="chat-box-content">
          <p className="chat-prompt">{filledPrompt}</p>
          <p className="chat-target">
            <strong>{value.target_new}</strong>, and it is the target destination.
          </p>
        </div>
      </div>

      {/* Panel A2: Knowledge Graph Slot (Placed between A1 and A3) */}
      {knowledgeGraphSlot}

      {/* Panel A3: Facts for Editing */}
      <div className="facts-card">
        <div className="card-header">
          <div className="title-with-badge">
            <span className="badge-tag">A3</span>
            <h3>Facts for Editing</h3>
          </div>
          <button
            type="button"
            className="btn-tiny"
            title="Edit entity parameters"
            onClick={() => setIsEditingPrompts((p) => !p)}
          >
            {isEditingPrompts ? "Done" : "+"}
          </button>
        </div>

        <div className="fact-item">
          <div className="fact-main-text">
            <span>{value.subject}</span> → <strong style={{ color: "#10B981" }}>{value.target_new}</strong>
            <span className="fact-orig"> (was {value.target_true || "original"})</span>
          </div>
          <div className="fact-meta-tags">
            <span className="meta-tag">ITERATIONS: X1</span>
            <span className="meta-tag">QUANTITY: X1</span>
          </div>
        </div>

        {isEditingPrompts && (
          <div className="fact-edit-inputs">
            <label>
              Prompt
              <input
                value={value.prompt}
                disabled={disabled}
                onChange={(e) => set("prompt", e.target.value)}
              />
            </label>
            <div className="row-inputs">
              <input
                placeholder="Subject"
                value={value.subject}
                disabled={disabled}
                onChange={(e) => set("subject", e.target.value)}
              />
              <input
                placeholder="Target (new)"
                value={value.target_new}
                disabled={disabled}
                onChange={(e) => set("target_new", e.target.value)}
              />
              <input
                placeholder="Target (true)"
                value={value.target_true || ""}
                disabled={disabled}
                onChange={(e) => set("target_true", e.target.value)}
              />
            </div>
            <label>
              Paraphrases
              <textarea
                rows={2}
                value={paraphraseText}
                onChange={(e) => {
                  setParaphraseText(e.target.value);
                  set("paraphrase_prompts", splitLines(e.target.value));
                }}
              />
            </label>
            <label>
              Neighborhoods
              <textarea
                rows={2}
                value={neighborhoodText}
                onChange={(e) => {
                  setNeighborhoodText(e.target.value);
                  set("neighborhood_prompts", splitLines(e.target.value));
                }}
              />
            </label>
          </div>
        )}
      </div>

      {/* Panel A3: Prompts for Testing */}
      <div className="prompts-card">
        <div className="card-header">
          <div className="title-with-badge">
            <span className="badge-tag">A3</span>
            <h3>Prompts for Testing</h3>
          </div>
        </div>

        <div className="prompt-badge-list">
          {/* Efficacy prompt badge */}
          <div className="prompt-badge-item badge-efficacy">
            <div className="badge-bar" />
            <div className="badge-content">
              <div className="badge-text">{filledPrompt}</div>
              <div className="badge-meta">TYPE: EFFICACY / QUANTITY: X1</div>
            </div>
          </div>

          {/* Paraphrase prompt badge */}
          <div className="prompt-badge-item badge-paraphrase">
            <div className="badge-bar" />
            <div className="badge-content">
              <div className="badge-text">
                {value.paraphrase_prompts[0] || `The location of ${value.subject} is`}
              </div>
              <div className="badge-meta">TYPE: PARAPHRASE / QUANTITY: X{value.paraphrase_prompts.length || 1}</div>
            </div>
          </div>

          {/* Neighborhood prompt badge */}
          <div className="prompt-badge-item badge-neighborhood">
            <div className="badge-bar" />
            <div className="badge-content">
              <div className="badge-text">
                {value.neighborhood_prompts[0] || "The Louvre Museum is located in"}
              </div>
              <div className="badge-meta">TYPE: NEIGHBORHOOD / QUANTITY: X{value.neighborhood_prompts.length || 1}</div>
            </div>
          </div>

          {/* Generation prompt badge */}
          <div className="prompt-badge-item badge-generation">
            <div className="badge-bar" />
            <div className="badge-content">
              <div className="badge-text">list some facts about {value.subject}</div>
              <div className="badge-meta">TYPE: GENERATION / QUANTITY: X1</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export const DEFAULT_FACT: FactInput = {
  prompt: "{} is located in the city of",
  subject: "Eiffel Tower",
  target_new: "Rome",
  target_true: "Paris",
  paraphrase_prompts: [
    "The Eiffel Tower is located in the city of",
    "You can find the Eiffel Tower in the city of",
  ],
  neighborhood_prompts: [
    "The Louvre Museum is located in the city of",
    "Notre-Dame Cathedral is located in the city of",
  ],
};

export const DEFAULT_SCHEMES = "13-17\n8-12\n6-8\n20-21";

export function parseLayerSpec(spec: string): number[] {
  const trimmed = spec.trim();
  if (!trimmed) return [];
  if (trimmed.includes("-") && !trimmed.includes(",")) {
    const [start, end] = trimmed.split("-").map((s) => parseInt(s.trim(), 10));
    if (Number.isNaN(start) || Number.isNaN(end) || start > end) {
      throw new Error(`Invalid range: ${spec}`);
    }
    return Array.from({ length: end - start + 1 }, (_, i) => start + i);
  }
  return trimmed.split(",").map((s) => parseInt(s.trim(), 10));
}

export function parseSchemesText(text: string): number[][] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map(parseLayerSpec);
}
```

<a id="replacement-9"></a>

### frontend/src/components/PromptDetailCards.tsx

Current file: [frontend/src/components/PromptDetailCards.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/components/PromptDetailCards.tsx).

SHA-256: 6b7495e29a36e6128732fb22be68b25579899683d7ecef46922d4900221f5dc7

```tsx
import React from "react";
import type { Metrics } from "../types";

interface Props {
  prompt: string;
  subject: string;
  targetNew: string;
  targetTrue?: string;
  paraphrasePrompts: string[];
  neighborhoodPrompts: string[];
  generations?: string[];
  metrics?: Metrics | null;
}

export const PromptDetailCards: React.FC<Props> = ({
  prompt,
  subject,
  targetNew,
  targetTrue,
  paraphrasePrompts,
  neighborhoodPrompts,
  generations = [],
  metrics,
}) => {
  const filledPrompt = prompt.includes("{}")
    ? prompt.replace("{}", subject)
    : prompt.includes(subject) ? prompt : `${subject} ${prompt}`;

  // Evaluate real pass/fail state based on metrics if available
  const evaluationClass = (value: number | null | undefined) =>
    value == null || !Number.isFinite(value) ? "eval-unknown" : value > 0.5 ? "eval-pass" : "eval-fail";
  const promptClass = (category: "efficacy" | "paraphrase" | "neighborhood", prefix: string) => {
    const result = metrics?.details?.[category].find((r) => r.prefix === prefix);
    if (!result) return "eval-unknown";
    const passed = category === "neighborhood"
      ? result.target_true_nll < result.target_new_nll
      : result.target_new_nll < result.target_true_nll;
    return evaluationClass(Number(passed));
  };

  return (
    <div className="category-columns-grid">
      {/* 1. Efficacy Column */}
      <div className="category-card col-efficacy">
        <div className={`cat-header cat-eff ${evaluationClass(metrics?.ES)}`}>Efficacy</div>
        <div className={`eval-pill ${metrics?.details ? promptClass("efficacy", filledPrompt) : evaluationClass(metrics?.ES)}`}>
          <div>{filledPrompt} <strong>{targetNew}</strong></div>
        </div>
      </div>

      {/* 2. Paraphrase Column */}
      <div className="category-card col-paraphrase">
        <div className={`cat-header cat-para ${evaluationClass(metrics?.PS)}`}>Paraphrase</div>
        {paraphrasePrompts.map((p, idx) => (
          <div key={idx} className={`eval-pill ${promptClass("paraphrase", p)}`}>
            <div>{p} <strong>{targetNew}</strong></div>
          </div>
        ))}
      </div>

      {/* 3. Neighborhood Column */}
      <div className="category-card col-neighborhood">
        <div className={`cat-header cat-neigh ${evaluationClass(metrics?.NS)}`}>Neighborhood</div>
        {neighborhoodPrompts.map((p, idx) => (
          <div key={idx} className={`eval-pill ${promptClass("neighborhood", p)}`}>
            <div>{p} <strong>{targetTrue ?? ""}</strong></div>
          </div>
        ))}
      </div>

      {/* 4. Generation Column */}
      <div className="category-card col-generation">
        <div className="cat-header cat-gen">Generation</div>
        <div className="eval-pill eval-gen">
          <div style={{ fontSize: "0.68rem", color: "#334155" }}>
            {generations[0] != null ? (
              <span>{generations[0]}</span>
            ) : (
              <span>No generation data</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
```

<a id="replacement-10"></a>

### frontend/src/components/SchemeComparisonTable.tsx

Current file: [frontend/src/components/SchemeComparisonTable.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/components/SchemeComparisonTable.tsx).

SHA-256: ad7fba4819af238be41921da3ffb7ae312853cd20ee808d997ac06b062f1aaed

```tsx
import type { CompareResponse, DamageReport } from "../types";
import { MetricBarGauge } from "./MetricBarGauge";
import { schemeKey as getSchemeKey, sortSchemes } from "../schemes";

interface Props {
  data: CompareResponse;
  selectedSchemeKey?: string | null;
  onSelectSchemeKey?: (schemeKey: string, layers: number[]) => void;
}

export function fmtKL(d: DamageReport | undefined): string {
  if (!d) return "—";
  const kl = d.kl_divergence;
  return Math.abs(kl) < 0.001 ? kl.toExponential(2) : kl.toFixed(4);
}

export function SchemeComparisonTable({
  data,
  selectedSchemeKey,
  onSelectSchemeKey,
}: Props) {
  // Sort descending by harmonic score S
  const sorted = sortSchemes(data.schemes);

  return (
    <div className="scheme-table-wrap">
      <div className="table-header-bar">
        <span className="badge-tag">B3</span>
        <h4>Editing results preview for different schemes</h4>
      </div>

      <div className="table-scroll-container">
        <table className="scheme-table">
          <thead>
            <tr>
              <th style={{ minWidth: "75px" }}>Scheme</th>
              <th title="Efficacy Success (0-1)">ES</th>
              <th title="Paraphrase Success (0-1)">PS</th>
              <th title="Neighborhood Success (0-1)">NS</th>
              <th title="Harmonic Mean Score (0-1)">S</th>
              <th title="Representation Stability / Damage KL">KL Drift</th>
              <th style={{ width: "55px" }}>Version</th>
            </tr>
          </thead>
          <tbody>
            {/* Baseline Row */}
            <tr className="baseline-row">
              <td>
                <span className="scheme-chip base">Base</span>
              </td>
              <td><MetricBarGauge value={data.baseline.metrics?.ES} color="#94A3B8" /></td>
              <td><MetricBarGauge value={data.baseline.metrics?.PS} color="#94A3B8" /></td>
              <td><MetricBarGauge value={data.baseline.metrics?.NS} color="#94A3B8" /></td>
              <td><MetricBarGauge value={data.baseline.metrics?.S} color="#94A3B8" /></td>
              <td className="mono-cell">0.0</td>
              <td className="version-cell">v0</td>
            </tr>

            {/* Scheme Rows */}
            {sorted.map((s) => {
              const schemeKey = getSchemeKey(s.layers);
              const isSelected = selectedSchemeKey === schemeKey;
              const minLayer = Math.min(...s.layers);
              const maxLayer = Math.max(...s.layers);
              const label = minLayer === maxLayer ? `${minLayer}` : `${minLayer}-${maxLayer}`;

              return (
                <tr
                  key={schemeKey}
                  data-scheme-key={schemeKey}
                  className={`scheme-row ${isSelected ? "selected" : ""}`}
                  onClick={() => onSelectSchemeKey?.(schemeKey, s.layers)}
                >
                  <td>
                    <span className="scheme-chip" title={`Layers: [${s.layers.join(", ")}]`}>
                      {label}
                    </span>
                  </td>
                  <td><MetricBarGauge value={s.metrics?.ES} color="#818CF8" /></td>
                  <td><MetricBarGauge value={s.metrics?.PS} color="#60A5FA" /></td>
                  <td><MetricBarGauge value={s.metrics?.NS} color="#34D399" /></td>
                  <td><MetricBarGauge value={s.metrics?.S} color="#A78BFA" /></td>
                  <td className="mono-cell">{fmtKL(s.damage)}</td>
                  <td className="version-cell">v1</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

<a id="replacement-11"></a>

### frontend/src/components/TokenRankingChart.tsx

Current file: [frontend/src/components/TokenRankingChart.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/components/TokenRankingChart.tsx).

SHA-256: 1f4b16ac82abec49f5400bcd6f08780534d044689c48088824be4d2c7644c71e

```tsx
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { LayerSignal } from "../types";

interface Props {
  signals: LayerSignal[];
  selectedLayer?: number;
  view?: "subject" | "last";
}

export function TokenRankingChart({
  signals,
  selectedLayer,
  view = "subject",
}: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hoverLayer, setHoverLayer] = useState<number | null>(null);

  const hasLast = signals.some((s) => s.last_top_tokens && s.last_top_tokens.length > 0);
  const effectiveView: "subject" | "last" =
    view === "last" && hasLast ? "last" : "subject";
  const pick = (s: LayerSignal) =>
    effectiveView === "last" && s.last_top_tokens ? s.last_top_tokens : s.top_tokens;

  const activeLayer = hoverLayer ?? selectedLayer ?? 0;

  useEffect(() => {
    if (!ref.current) return;
    d3.select(ref.current).selectAll("*").remove();
    if (signals.length === 0) return;

    const width = 260;
    const height = 200;
    const margin = { top: 22, right: 12, bottom: 24, left: 32 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand<number>()
      .domain(signals.map((d) => d.layer))
      .range([0, innerW])
      .padding(0.1);

    const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    const xAxis = d3.axisBottom(x).tickValues(x.domain().filter((_, i) => i % 6 === 0));
    const yAxis = d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%"));

    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(xAxis)
      .attr("font-size", 8)
      .attr("color", "#64748B");

    g.append("g")
      .call(yAxis)
      .attr("font-size", 8)
      .attr("color", "#64748B");

    g.selectAll("rect")
      .data(signals)
      .join("rect")
      .attr("x", (d) => x(d.layer)!)
      .attr("y", (d) => y(pick(d)[0]?.prob ?? 0))
      .attr("width", x.bandwidth())
      .attr("height", (d) => innerH - y(pick(d)[0]?.prob ?? 0))
      .attr("fill", (d) => (d.layer === activeLayer ? "#3B82F6" : "#93C5FD"))
      .attr("rx", 1.5)
      .style("cursor", "pointer")
      .on("mouseenter", (_, d) => setHoverLayer(d.layer))
      .on("mouseleave", () => setHoverLayer(null));

    g.append("text")
      .attr("x", 0)
      .attr("y", -8)
      .attr("font-size", 9)
      .attr("font-weight", 700)
      .attr("fill", "#0F172A")
      .text(
        effectiveView === "last"
          ? "Top-1 logit lens (last token)"
          : "Top-1 logit lens (subject token)",
      );
  }, [signals, activeLayer, effectiveView]);

  const layer = signals.find((s) => s.layer === activeLayer);
  const detailTokens = layer ? pick(layer) : [];

  return (
    <div className="token-ranking-box">
      <svg ref={ref} style={{ width: "100%", height: "auto" }} />
      {layer && (
        <div className="token-mini-detail" style={{ fontSize: "0.72rem", padding: "0.3rem 0.5rem", background: "#F8FAFC", borderRadius: "4px", border: "1px solid #E2E8F0", marginTop: "4px" }}>
          <strong>L{layer.layer}: </strong>
          <span style={{ color: "#3B82F6", fontWeight: 600 }}>{detailTokens[0]?.token || "—"}</span>
          <span style={{ color: "#64748B", marginLeft: "4px" }}>({((detailTokens[0]?.prob ?? 0) * 100).toFixed(1)}%)</span>
        </div>
      )}
    </div>
  );
}
```

<a id="replacement-12"></a>

### frontend/src/components/WireframeLinker.tsx

Current file: [frontend/src/components/WireframeLinker.tsx](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/components/WireframeLinker.tsx).

SHA-256: 80e27a00c68f473f8cd2fd9ee8b23352645b9bec5e3bb892dd1b3718ca005fdc

```tsx
import React, { useLayoutEffect, useRef, useState } from "react";

interface Scheme {
  key: string;
  layers: number[];
  label?: string;
}
interface Props {
  nLayers: number;
  schemes: Scheme[];
  selectedSchemeKey?: string | null;
  onSelectSchemeKey?: (key: string, layers: number[]) => void;
  selectedLayers: number[];
}
interface Anchor { start: number; end: number; targetX: number; targetY: number }

export const WireframeLinker: React.FC<Props> = ({ nLayers, schemes, selectedSchemeKey, onSelectSchemeKey, selectedLayers }) => {
  const ref = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [anchors, setAnchors] = useState<Record<string, Anchor>>({});
  const height = 440;
  const width = 110;
  const count = Math.max(0, Math.floor(nLayers));

  useLayoutEffect(() => {
    const container = ref.current;
    const svg = svgRef.current;
    const workspace = container?.closest(".edit-signals-row");
    if (!container || !svg || !workspace) return;
    const measure = () => {
      const ctm = svg.getScreenCTM();
      if (!ctm) return;
      const inverse = ctm.inverse();
      const tableRows = Array.from(workspace.querySelectorAll<HTMLElement>("[data-scheme-key]"));
      const ticks = container.querySelectorAll<HTMLElement>(".axis-tick");
      const next: Record<string, Anchor> = {};
      for (const scheme of schemes) {
        const layers = scheme.layers.filter((l) => Number.isInteger(l) && l >= 0 && l < count);
        const row = tableRows.find((r) => r.dataset.schemeKey === scheme.key);
        if (!row || !layers.length) continue;
        const first = ticks[Math.min(...layers)]?.getBoundingClientRect();
        const last = ticks[Math.max(...layers)]?.getBoundingClientRect();
        if (!first || !last) continue;
        const rect = row.getBoundingClientRect();
        const start = new DOMPoint(first.right, first.top + first.height / 2).matrixTransform(inverse).y;
        const end = new DOMPoint(last.right, last.top + last.height / 2).matrixTransform(inverse).y;
        const target = new DOMPoint(rect.left, rect.top + rect.height / 2).matrixTransform(inverse);
        if ([start, end, target.x, target.y].every(Number.isFinite)) next[scheme.key] = { start, end, targetX: target.x, targetY: target.y };
      }
      setAnchors(next);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(workspace);
    observer.observe(container);
    workspace.querySelectorAll(".scheme-table, .scheme-table tr, .table-header-bar").forEach((node) => observer.observe(node));
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [schemes, count]);

  return (
    <div className="wireframe-container" ref={ref}>
      <div className="layer-axis-column">
        <div className="axis-header">Layer</div>
        <div className="axis-ticks" style={{ height }}>
          {Array.from({ length: count }, (_, i) => <div key={i} className={`axis-tick ${selectedLayers.includes(i) ? "active" : ""}`} style={{ height: height / count, flexShrink: 0 }} title={`Layer ${i}`}>{i % 4 === 0 || i === count - 1 ? <span>{i}</span> : null}</div>)}
        </div>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${width} ${height + 20}`} preserveAspectRatio="none" className="wireframe-svg" style={{ overflow: "visible", width: "calc(100% - 22px)", height: height + 20 }}>
        {schemes.map((scheme) => {
          const anchor = anchors[scheme.key];
          if (!anchor) return null;
          const { start, end, targetX, targetY } = anchor;
          const mid = (start + end) / 2;
          const active = selectedSchemeKey === scheme.key;
          const stroke = active ? "#2563EB" : "#94A3B8";
          return (
            <g key={scheme.key} data-wire-key={scheme.key} className={`scheme-wireframe-group ${active ? "active" : ""}`} onClick={() => onSelectSchemeKey?.(scheme.key, scheme.layers)} style={{ cursor: "pointer" }}>
              <title>{`Layers: ${scheme.layers.join(", ")}`}</title>
              <path d={`M2 ${start}H16V${end}H2`} fill="none" stroke={stroke} strokeWidth={active ? 2.2 : 1.2} />
              <path className="scheme-connector" d={`M16 ${mid}C45 ${mid},65 ${targetY},${targetX} ${targetY}`} fill="none" stroke={stroke} strokeWidth={active ? 2.2 : 1.2} strokeDasharray={active ? undefined : "3 2"} />
            </g>
          );
        })}
      </svg>
    </div>
  );
};
```

<a id="replacement-13"></a>

### frontend/src/schemes.ts

Current file: [frontend/src/schemes.ts](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/schemes.ts).

SHA-256: 797189cbab7c068122f5c9b27105322ba65f1f750cc32c0cf26e4ecefcb0e2f0

```typescript
import type { SchemeResult } from "./types";

export function schemeKey(layers: number[]): string {
  return [...new Set(layers)].sort((a, b) => a - b).join("-");
}

export function comparisonSchemes(schemes: number[][], method: "memit" | "rome", nLayers: number): number[][] {
  const normalized = schemes.map((layers) => [...new Set(layers)].sort((a, b) => a - b));
  if (normalized.some((layers) => !layers.length || layers.some((l) => !Number.isInteger(l) || l < 0 || l >= nLayers))) {
    throw new Error(`Schemes must contain layers from 0 to ${nLayers - 1}.`);
  }
  const candidates = method === "rome" ? normalized.flatMap((layers) => layers.map((l) => [l])) : normalized;
  return [...new Map(candidates.map((layers) => [schemeKey(layers), layers])).values()];
}

export function sortSchemes(schemes: SchemeResult[]): SchemeResult[] {
  return [...schemes].sort((a, b) => (b.metrics?.S ?? -1) - (a.metrics?.S ?? -1));
}
```

<a id="replacement-14"></a>

### frontend/src/types.ts

Current file: [frontend/src/types.ts](C:/Users/shaik/Research/LLM%20Editing/prototype/frontend/src/types.ts).

SHA-256: 720ffe9ab85903226a9fc84beebefa146f9a0b8931441d5cdbd6cf1b2d7d753f

```typescript
export interface TopToken {
  token: string;
  prob: number;
}

export interface LayerSignal {
  layer: number;
  cosine_similarity: number;
  top_tokens: TopToken[];
  last_top_tokens?: TopToken[];
  fact_top_tokens?: TopToken[];
}

export interface Metrics {
  ES: number;
  PS: number | null;
  NS: number | null;
  S: number | null;
  details?: {
    efficacy: PromptEvaluation[];
    paraphrase: PromptEvaluation[];
    neighborhood: PromptEvaluation[];
  };
}

export interface PromptEvaluation {
  prefix: string;
  target_new_nll: number;
  target_true_nll: number;
  target_new_correct: boolean;
  target_true_correct: boolean;
}

export interface NeighborhoodResult {
  prompt: string;
  pre_text: string;
  post_text: string;
  kl_divergence: number;
  projection?: { pre: [number, number]; post: [number, number] };
}

export interface DamageReport {
  kl_divergence: number;
  n_prompts: number;
  note: string;
}

export interface FactInput {
  prompt: string;
  subject: string;
  target_new: string;
  target_true: string;
  paraphrase_prompts: string[];
  neighborhood_prompts: string[];
}

export interface ProbeResponse {
  rewrite_prompt: string;
  layer_signals: LayerSignal[];
}

export interface EditResponse {
  neighborhood?: NeighborhoodResult[];
  method?: string;
  edited_layers: number[];
  pre_edit: {
    layer_signals: LayerSignal[];
    generations: string[];
    metrics: Metrics | null;
  };
  post_edit: {
    layer_signals: LayerSignal[];
    generations: string[];
    metrics: Metrics | null;
  };
  damage?: DamageReport;
}

export interface SchemeResult {
  neighborhood?: NeighborhoodResult[];
  layers: number[];
  metrics: Metrics | null;
  generation: string;
  damage?: DamageReport;
}

export interface CompareResponse {
  method?: string;
  baseline: {
    metrics: Metrics | null;
    generation: string;
    layer_signals: LayerSignal[];
    damage?: DamageReport;
  };
  schemes: SchemeResult[];
}

export interface HealthResponse {
  status: string;
  model: string;
  n_layers: number;
  methods: string[];
}
```
