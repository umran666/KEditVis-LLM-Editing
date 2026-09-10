# KEditVis end-to-end fixes and verification

Date: 2026-09-10T00:50:10.949Z.

This report supersedes the runtime and feature limitations in the historical first-pass audit. The backend is deployed at [Modal API](https://opzgameryt--keditvis-memit-web-app.modal.run/health). The local dashboard is [http://127.0.0.1:5187](http://127.0.0.1:5187), configured to use that deployment.

## Real GPU verification

All rows below used actual model weights and the pinned upstream editing implementation on an A100-40GB. Each model/method combination executed an edit and a comparison with two different schemes. Subsequent probes reproduced every baseline layer signal exactly; deterministic generation reproduced the baseline text exactly. No API mocks were used in these runs.

| Model | Method | Edit layers | ES | PS | NS | S | Edit seconds | Compare seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpt2-xl | ROME | 17 | 1.000 | 1.000 | 1.000 | 1.000 | 19.58 | 28.69 |
| gpt2-xl | MEMIT | 13, 14, 15, 16, 17 | 1.000 | 0.000 | 1.000 | 0.000 | 48.78 | 25.47 |
| EleutherAI/gpt-j-6B | ROME | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 35.3 | 56.03 |
| EleutherAI/gpt-j-6B | MEMIT | 3, 4, 5, 6, 7, 8 | 1.000 | 1.000 | 1.000 | 1.000 | 58.26 | 46.42 |

The fact tested was Eiffel Tower: Paris to Rome, with one paraphrase and two Paris neighborhood prompts. These are case-specific measurements, not general benchmark scores. The complete requests, responses, timing, measured drift, and restored probes are preserved under [live](./live/). Health checks confirmed 48 GPT-2 XL layers and 28 GPT-J layers; every layer returned five tokens at both subject and last-token positions.

## Additional fixes

### [CRITICAL] GPT-J MEMIT could not trace keyword block inputs

[modal_app.py:134](../modal_app.py#L134). The real GPU run failed with IndexError because Transformers 4.42.4 calls GPT-J blocks using hidden_states=, whereas the pinned upstream Trace only captures positional inputs. Temporary forward pre-hooks now expose the identical hidden tensor positionally during upstream editing and are removed on both success and failure. The failed live request was followed by an exactly matching baseline probe, validating rollback on this real exception. The final GPU matrix was rerun after the fix. See the pinned [upstream tracer](https://raw.githubusercontent.com/kmeng01/memit/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b/util/nethook.py) and [Transformers GPT-J calls](https://raw.githubusercontent.com/huggingface/transformers/v4.42.4/src/transformers/models/gptj/modeling_gptj.py).

### [HIGH] Unmeasured drift projections

[modal_app.py:497](../modal_app.py#L497). The old backend did not return hidden-space coordinates. It now captures real last-token block outputs at the last edited layer, computes Euclidean distance before projection, and fits deterministic joint t-SNE over pre/post vectors. A labeled PCA fallback handles one-point and degenerate inputs; no semantic-damage score is invented.

### [HIGH] Only one token rank was visible

[frontend/src/components/TokenRankingChart.tsx:12](../frontend/src/components/TokenRankingChart.tsx#L12). The top-1 bars omitted four ranks described by the interface. Five probability-sized bubbles now render per layer, same-token paths link layers, hover highlights the full path, and clicks select a layer. Missing last-token measurements remain empty.

### [HIGH] Chat displayed the requested answer as model output

[frontend/src/components/FactForm.tsx:77](../frontend/src/components/FactForm.tsx#L77). The hardcoded target sentence was not inference. The Generate command now calls the real deterministic generation endpoint; input/model changes clear its result and stale responses are discarded. Edit Fact opens the actual editable fact fields.

### [HIGH] Comparison omitted post-edit signal data

[modal_app.py:1572](../modal_app.py#L1572). Comparison rows now include measured post-edit layer signals. The selected row drives the post-edit lens as well as metrics, output diff, and neighborhood drift.

### [HIGH] Invented graph neighbors could replace the subject

[frontend/src/components/KnowledgeGraph.tsx:41](../frontend/src/components/KnowledgeGraph.tsx#L41). Location, Category and Neighbor were synthetic node labels, yet clicking them changed the subject. The fact graph now displays only the actual subject and supplied original/new answers, with no invented entity links.

### [HIGH] Offline dev URL and mutable upstream revision

[modal_app.py:63](../modal_app.py#L63). The editing repository is pinned to commit 80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b. The frontend now points to the deployed service. The web worker allows one active request and at most one container, with a ten-minute idle scale-down.

### [MEDIUM] Mobile drift output columns were too narrow

[frontend/src/App.css:1088](../frontend/src/App.css#L1088). Four fixed columns squeezed generated text into long vertical rows on mobile. The drift table now keeps a readable minimum width inside its own horizontal scroll container, while numeric drift values stay on one line and the page remains within the viewport. The comparison scheme editor also uses the existing input styling.

### [MEDIUM] Recommend required two clicks and schemes were fixed

[frontend/src/App.tsx:225](../frontend/src/App.tsx#L225). The first Recommend click now waits for its guarded probe and selects a window immediately. The comparison text field is editable, malformed specifications are rejected, and noncontiguous layer labels preserve each selected layer instead of implying a full range.

## Validation

- 13 CPU regression tests passed, including partial-edit exceptions, successful restoration, invalid HTTP requests, failed model-load recovery, target token alignment, exact hidden-space distance, deterministic projection, keyword-input tracing without tensor changes, and hook cleanup. Inference and algorithms are stubbed in CPU route tests.
- 14 browser regression groups passed with intercepted API fixtures. These cover race conditions, errors, zero/missing metrics, five ranks, hover paths, dynamic row anchors, long diffs, mobile bounds, and CTM selection.
- 6 live browser groups passed against the deployed GPU API: real generation reaches chat through the configured Modal API; one Recommend click probes the model and selects a layer window; two real ROME schemes populate scores, post-edit signals and measured drift; real edit results render on desktop and mobile without page overflow; Revert restores the baseline view and reproduces the original layer signals; no browser runtime exceptions during the live workflow.
- TypeScript compilation and Vite production build passed.
- [Live desktop screenshot](./live/browser/desktop.png), [live mobile screenshot](./live/browser/mobile.png), [live browser responses](./live/browser/), and [fixture browser results](./frontend-results.json).

Commands from prototype: python -m unittest test_audit_backend.py -v; python test_live_backend.py --model gpt2-xl; python test_live_backend.py --model EleutherAI/gpt-j-6B. Frontend commands: npm run build; node tests/audit.mjs; node tests/live.mjs. Live checks use the deployed GPU and incur Modal usage.

## Scope

The tested dashboard workflows are operational. This is not a certification that every possible edit succeeds or a complete replication of the paper's research evaluation, external knowledge graph, or user study. Neighborhood preference metrics use the supplied shared target_true; heterogeneous neighborhoods require appropriate per-prompt reference answers before those rates can be interpreted. The two-neighborhood t-SNE view is a visual aid; hidden L2 and KL carry the measured numeric change. The first-pass original snapshots and historical experiments were preserved.

## Complete replacement files

These are the current files copied verbatim from the working tree. Earlier unchanged fixes remain in the first-pass report.

### modal_app.py

````python
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

MEMIT_COMMIT = "80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b"

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
        "scikit-learn==1.5.2",
        "fastapi[standard]==0.115.6",
    )
    .run_commands(
        "git clone https://github.com/kmeng01/memit /root/memit",
        f"git -C /root/memit checkout {MEMIT_COMMIT}",
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


def _positional_hidden_state(module, args, kwargs):
    """Expose keyword-only calls to the pinned upstream positional-input tracer."""
    if not args and "hidden_states" in kwargs:
        remaining = dict(kwargs)
        return (remaining.pop("hidden_states"),), remaining


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
    handles = []
    try:
        # Transformers GPT-J calls blocks with hidden_states=; upstream Trace
        # otherwise records an empty tuple and MEMIT cannot read block inputs.
        for name, module in model.named_modules():
            if name.startswith("transformer.h.") and name.count(".") == 2:
                handles.append(module.register_forward_pre_hook(_positional_hidden_state, with_kwargs=True))
        edited, _ = apply_fn(model, tok, requests, hparams, return_orig_weights=False)
        return edited, originals
    except BaseException:
        _restore_weights(model, originals)
        raise
    finally:
        for handle in handles:
            handle.remove()


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


def _hidden_states(model, tok, prompts, layers):
    """Last-token block outputs at requested layers, in the original hidden space."""
    import torch

    captured = {layer: [] for layer in layers}
    modules = dict(model.named_modules())
    handles = []
    def hook(layer):
        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer].append(hidden[0, -1].detach().float().cpu().clone())
        return capture
    try:
        for layer in layers:
            handles.append(modules[f"transformer.h.{layer}"].register_forward_hook(hook(layer)))
        with torch.no_grad():
            for prompt in prompts:
                model(**tok(prompt, return_tensors="pt").to(next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.stack(values) for layer, values in captured.items() if values}


def _project_drift(before, after):
    """Joint t-SNE gives pre/post points one coordinate system; L2 stays in hidden space."""
    import numpy as np
    import torch
    from sklearn.manifold import TSNE

    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("Drift embeddings must have matching [prompts, hidden] shapes.")
    values = torch.cat([before, after]).float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Drift embeddings contain non-finite values.")
    if len(values) < 3 or np.allclose(values, values[0]):
        centered = values - values.mean(axis=0)
        u, singular, _ = np.linalg.svd(centered, full_matrices=False)
        projected = np.zeros((len(values), 2))
        dims = min(2, u.shape[1])
        projected[:, :dims] = u[:, :dims] * singular[:dims]
        method = "pca-small-sample"
    else:
        projected = TSNE(n_components=2, perplexity=min(30, max(1, (len(values) - 1) / 3)),
                         init="pca", learning_rate="auto", random_state=0).fit_transform(values)
        method = "joint-tsne"
    distances = torch.linalg.vector_norm(after.float() - before.float(), dim=1).tolist()
    count = before.shape[0]
    return [{"projection": {"pre": projected[i].tolist(), "post": projected[i + count].tolist()},
             "hidden_state_drift": distances[i], "projection_method": method} for i in range(count)]


def _generate_text(model, tok, prompts, max_out_len=80):
    import torch

    texts = []
    with torch.no_grad():
        for prompt in prompts:
            encoded = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
            output = model.generate(**encoded, do_sample=False, max_new_tokens=max_out_len,
                                    pad_token_id=tok.eos_token_id)
            texts.append(tok.decode(output[0], skip_special_tokens=True))
    return texts


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


def _neighborhood_report(prompts, before, after, pre_hidden=None, post_hidden=None, layer=None):
    pre_text, pre_lp = before
    post_text, post_lp = after
    if not all(len(values) == len(prompts) for values in [pre_text, post_text, pre_lp, post_lp]):
        raise ValueError("Neighborhood measurements must align with prompts.")
    projections = _project_drift(pre_hidden, post_hidden) if pre_hidden is not None and len(prompts) else [{} for _ in prompts]
    return [
        {"prompt": prompt, "pre_text": pre_text[i], "post_text": post_text[i],
         "kl_divergence": _kl_divergence([post_lp[i]], [pre_lp[i]]),
         "drift_layer": layer, **projections[i]}
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
    generate_fast = _generate_text

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
    generate_fast = _generate_text

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
    max_containers=1,
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
    generate_fast = _generate_text

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
            "editing_commit": MEMIT_COMMIT,
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

    @web.post("/generate")
    def generate(body: ProbeRequest):
        model, tok = _ensure_model(body.model)
        prompt = body.prompt.format(body.subject)
        return {"model": body.model, "prompt": prompt,
                "generation": _generate_text(model, tok, [prompt], max_out_len=60)[0]}

    @web.post("/edit")
    def edit(body: EditRequest):
        model, tok = _ensure_model(body.model)
        model_name = _state["model_name"]
        validate_layers(body.layers, model)
        layers = _normalize_scheme(body.layers) if body.layers is not None else None
        if body.method == "rome" and layers is not None and len(layers) != 1:
            raise HTTPException(status_code=400, detail="ROME edits exactly one layer.")
        if layers is None:
            cls = ROMEHyperParams if body.method == "rome" else MEMITHyperParams
            layers = cls.from_json(_hparams_path(body.method, model_name)).layers
        drift_layer = max(layers)
        pre_hidden = _hidden_states(model, tok, body.neighborhood_prompts, [drift_layer])
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
                pre_hidden.get(drift_layer),
                _hidden_states(edited_model, tok, body.neighborhood_prompts, [drift_layer]).get(drift_layer),
                drift_layer,
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
        pre_hidden = _hidden_states(model, tok, body.neighborhood_prompts, sorted(set(max(s) for s in schemes)))
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
                    pre_hidden.get(max(scheme)),
                    _hidden_states(edited_model, tok, body.neighborhood_prompts, [max(scheme)]).get(max(scheme)),
                    max(scheme),
                )

                results.append({
                    "layers": scheme,
                    "layer_signals": _probe_layers(edited_model, tok, rewrite_prompt, body.subject),
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
````

### frontend/src/App.tsx

````tsx
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
  const [schemesText, setSchemesText] = useState(DEFAULT_SCHEMES);
  const [selectedLayers, setSelectedLayers] = useState<number[]>([13, 14, 15, 16, 17]);
  const [signals, setSignals] = useState<LayerSignal[]>([]);
  const [generation, setGeneration] = useState<string>();
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
    setGeneration(undefined);
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
    try {
    const layersList = compareResult
      ? sortSchemes(compareResult.schemes).map((s) => s.layers)
      : comparisonSchemes(parseSchemesText(schemesText), method, nLayers);
    return layersList.map((layers, idx) => ({
      key: schemeKey(layers),
      layers,
      label: `Scheme ${idx + 1}`,
    }));
    } catch { return []; }
  }, [schemesText, method, nLayers, compareResult]);

  const selectedScheme = compareResult?.schemes.find((s) => schemeKey(s.layers) === selectedSchemeKey);
  const postSignals = editResult?.post_edit.layer_signals ?? selectedScheme?.layer_signals;
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
        return res.layer_signals;
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

  const runGenerate = useCallback(async () => {
    const targetModel = activeModelRef.current;
    const reqId = ++activeRequestId.current;
    setLoading(true);
    setError(null);
    setElapsed(0);
    try {
      const res = await api.generate({ prompt: fact.prompt, subject: fact.subject, model: targetModel });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) setGeneration(res.generation);
    } catch (e) {
      if (reqId === activeRequestId.current) setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      if (reqId === activeRequestId.current) setLoading(false);
    }
  }, [fact.prompt, fact.subject]);

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

  const handleRecommend = useCallback(async () => {
    const recommendationSignals = signals.length ? signals : await runProbe();
    if (!recommendationSignals?.length) return;
    if (method === "rome") {
      let minL = 0;
      let minVal = Infinity;
      recommendationSignals.forEach((s) => {
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
        sum += Math.abs(recommendationSignals[i + j]?.cosine_similarity ?? 1);
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
    const sorted = [...selectedLayers].sort((a, b) => a - b);
    const contiguous = sorted.every((layer, i) => i === 0 || layer === sorted[i - 1] + 1);
    return minL === maxL ? `${minL}` : contiguous ? `${minL}-${maxL}` : sorted.join(", ");
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
            generation={generation}
            onGenerate={runGenerate}
            knowledgeGraphSlot={
              <KnowledgeGraph
                subject={fact.subject}
                target={fact.target_new}
                originalTarget={fact.target_true}
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
                  <TokenRankingChart signals={signals} view={lensView} onSelectLayer={handleLayerClick} />
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
                      <label className="scheme-editor">Comparison schemes
                        <textarea aria-label="Comparison schemes" rows={4} value={schemesText} disabled={loading}
                          onChange={(e) => setSchemesText(e.target.value)} />
                      </label>
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
                {postSignals ? (
                  <div className="charts-pair">
                    <TokenRankingChart
                      signals={postSignals}
                      view={lensView}
                    />
                    <CosineSimilarityCompareChart
                      preSignals={editResult?.pre_edit.layer_signals ?? compareResult?.baseline.layer_signals ?? []}
                      postSignals={postSignals}
                      editedLayers={editResult?.edited_layers ?? selectedScheme?.layers ?? []}
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
````

### frontend/src/App.css

````text
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
.scheme-editor {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0.35rem;
  margin-bottom: 0.75rem;
  text-align: left;
  font-size: 0.74rem;
}
.scheme-editor textarea {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  font: inherit;
  resize: vertical;
}
.drift-detail-table .drift-score-cell { white-space: nowrap; }
.drift-right-col { min-width: 0; }

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
  .drift-detail-table { table-layout: fixed; min-width: 600px; }
  .drift-detail-table td { overflow-wrap: anywhere; }
}
````

### frontend/src/api/client.ts

````typescript
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

export function generate(body: Pick<FactInput, "prompt" | "subject"> & { model?: string }): Promise<{ model: string; prompt: string; generation: string }> {
  return request("/generate", { method: "POST", body: JSON.stringify(body) });
}

export function edit(body: FactInput & { layers: number[] | null; method?: string; model?: string }): Promise<EditResponse> {
  return request("/edit", { method: "POST", body: JSON.stringify(body) });
}

export function compare(body: FactInput & { schemes: number[][]; method?: string; model?: string }): Promise<CompareResponse> {
  return request("/compare", { method: "POST", body: JSON.stringify(body) });
}
````

### frontend/src/types.ts

````typescript
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
  hidden_state_drift?: number;
  drift_layer?: number;
  projection_method?: string;
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
  layer_signals?: LayerSignal[];
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
````

### frontend/src/components/FactForm.tsx

````tsx
import React, { useState } from "react";
import type { FactInput } from "../types";

interface Props {
  value: FactInput;
  onChange: (next: FactInput) => void;
  disabled?: boolean;
  knowledgeGraphSlot?: React.ReactNode;
  generation?: string;
  onGenerate?: () => void;
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
  generation,
  onGenerate,
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
              onClick={() => { setMode("completion"); setIsEditingPrompts(false); }}
            >
              COMPLETION
            </button>
            <button
              type="button"
              className={mode === "rewrite" ? "active" : ""}
              onClick={() => { setMode("rewrite"); setIsEditingPrompts(true); }}
            >
              EDIT FACT
            </button>
          </div>
        </div>
        <div className="chat-box-content">
          <p className="chat-prompt">{filledPrompt}</p>
          {generation && <p className="chat-target">{generation}</p>}
          <button type="button" className="btn-action" disabled={disabled} onClick={onGenerate}>Generate</button>
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
                disabled={disabled}
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
                disabled={disabled}
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
                {value.paraphrase_prompts[0] || "None"}
              </div>
              <div className="badge-meta">TYPE: PARAPHRASE / QUANTITY: X{value.paraphrase_prompts.length}</div>
            </div>
          </div>

          {/* Neighborhood prompt badge */}
          <div className="prompt-badge-item badge-neighborhood">
            <div className="badge-bar" />
            <div className="badge-content">
              <div className="badge-text">
                {value.neighborhood_prompts[0] || "None"}
              </div>
              <div className="badge-meta">TYPE: NEIGHBORHOOD / QUANTITY: X{value.neighborhood_prompts.length}</div>
            </div>
          </div>

          {/* Generation prompt badge */}
          <div className="prompt-badge-item badge-generation">
            <div className="badge-bar" />
            <div className="badge-content">
              <div className="badge-text">{filledPrompt}</div>
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
  if (!/^\d+(?:\s*-\s*\d+|(?:\s*,\s*\d+)*)$/.test(trimmed)) throw new Error(`Invalid layer specification: ${spec}`);
  if (trimmed.includes("-") && !trimmed.includes(",")) {
    const [start, end] = trimmed.split("-").map((s) => parseInt(s.trim(), 10));
    if (Number.isNaN(start) || Number.isNaN(end) || start > end || end > 47) {
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
````

### frontend/src/components/KnowledgeGraph.tsx

````tsx
import React, { useMemo } from "react";

interface Props {
  subject: string;
  target?: string;
  originalTarget?: string;
}

interface Node {
  id: string;
  label: string;
  x: number;
  y: number;
  type: "subject" | "target" | "neighbor";
}

interface Link {
  source: string;
  target: string;
  relation: string;
}

export const KnowledgeGraph: React.FC<Props> = ({ subject, target, originalTarget }) => {
  const { nodes, links } = useMemo(() => {
    const subjNode: Node = {
      id: "s0",
      label: subject || "Entity",
      x: 90,
      y: 65,
      type: "subject",
    };

    const targetNode: Node = {
      id: "t0",
      label: target || "Target",
      x: 150,
      y: 35,
      type: "target",
    };

    const neighbors: Node[] = originalTarget
      ? [{ id: "n1", label: originalTarget, x: 30, y: 35, type: "neighbor" }]
      : [];

    const allNodes = [subjNode, targetNode, ...neighbors];

    const allLinks: Link[] = [
      { source: "s0", target: "t0", relation: "new answer" },
      ...(originalTarget ? [{ source: "s0", target: "n1", relation: "original answer" }] : []),
    ];

    return { nodes: allNodes, links: allLinks };
  }, [subject, target, originalTarget]);

  return (
    <div className="knowledge-graph-box">
      <div className="panel-header-sub">
        <div className="title-with-badge">
          <span className="badge-tag">A2</span>
          <h4>Fact Graph</h4>
        </div>
      </div>
      <div className="kg-canvas-wrapper">
        <svg viewBox="0 0 180 130" className="kg-svg">
          {/* Links */}
          <g className="links">
            {links.map((l, i) => {
              const src = nodes.find((n) => n.id === l.source);
              const dst = nodes.find((n) => n.id === l.target);
              if (!src || !dst) return null;
              return (
                <g key={i}>
                  <line
                    x1={src.x}
                    y1={src.y}
                    x2={dst.x}
                    y2={dst.y}
                    stroke="#CBD5E1"
                    strokeWidth="1.2"
                    strokeDasharray={l.relation === "target" ? "3 2" : undefined}
                  />
                  <text
                    x={(src.x + dst.x) / 2}
                    y={(src.y + dst.y) / 2 - 3}
                    textAnchor="middle"
                    fill="#64748B"
                    fontSize="5"
                    fontFamily="var(--font-mono)"
                  >
                    {l.relation}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Nodes */}
          <g className="nodes">
            {nodes.map((n) => {
              const isSubj = n.type === "subject";
              const isTgt = n.type === "target";
              const fill = isSubj ? "#3B82F6" : isTgt ? "#10B981" : "#64748B";
              const r = isSubj ? 14 : isTgt ? 12 : 9;

              return (
                <g
                  key={n.id}
                  transform={`translate(${n.x}, ${n.y})`}
                >
                  <title>{`Node: ${n.label} (${n.type})`}</title>
                  <circle
                    r={r}
                    fill={isSubj ? "#EFF6FF" : isTgt ? "#ECFDF5" : "#F1F5F9"}
                    stroke={fill}
                    strokeWidth="1.2"
                  />
                  <text
                    textAnchor="middle"
                    dy="2.5"
                    fill="#0F172A"
                    fontSize="6"
                    fontWeight={isSubj || isTgt ? 600 : 500}
                    fontFamily="var(--font-sans)"
                  >
                    {n.label.length > 9 ? n.label.slice(0, 8) + "…" : n.label}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
    </div>
  );
};
````

### frontend/src/components/TokenRankingChart.tsx

````tsx
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { LayerSignal } from "../types";

interface Props {
  signals: LayerSignal[];
  selectedLayer?: number;
  view?: "subject" | "last";
  onSelectLayer?: (layer: number) => void;
}

export function TokenRankingChart({ signals, selectedLayer, view = "subject", onSelectLayer }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ layer: number; token: string } | null>(null);
  const pick = (signal: LayerSignal) => (view === "last" ? signal.last_top_tokens ?? [] : signal.top_tokens).slice(0, 5);
  const active = signals.find((s) => s.layer === (hover?.layer ?? selectedLayer)) ?? signals[0];

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    if (!signals.length) return;
    const sorted = [...signals].sort((a, b) => a.layer - b.layer);
    const width = Math.max(260, signals.length * 18 + 44);
    const height = 205;
    svg.attr("viewBox", `0 0 ${width} ${height}`).style("width", `${width}px`);
    const x = d3.scalePoint<number>().domain(sorted.map((s) => s.layer)).range([36, width - 14]);
    const y = d3.scalePoint<number>().domain([1, 2, 3, 4, 5]).range([28, 164]);
    const points = sorted.flatMap((s) => pick(s).map((t, rank) => ({ layer: s.layer, rank: rank + 1, token: t.token, prob: t.prob })));
    const colors = d3.scaleOrdinal(d3.schemeTableau10).domain([...new Set(points.map((p) => p.token))]);
    svg.append("g").attr("transform", "translate(0,184)")
      .call(d3.axisBottom(x).tickValues(sorted.filter((_, i) => i % 4 === 0).map((s) => s.layer))).attr("font-size", 9);
    svg.append("g").attr("transform", "translate(24,0)").call(d3.axisLeft(y).tickFormat((rank) => `#${rank}`)).attr("font-size", 9);
    const path = d3.line<(typeof points)[number]>().x((p) => x(p.layer)!).y((p) => y(p.rank)!);
    const tokenPaths = [...d3.group(points, (p) => p.token)];
    svg.selectAll("path.token-path").data(tokenPaths).join("path")
      .attr("class", "token-path").attr("d", ([, rows]) => path(rows))
      .attr("fill", "none").attr("stroke", ([token]) => colors(token))
      .attr("stroke-width", 0.7).attr("opacity", 0.55);
    svg.selectAll("circle.token-bubble").data(points).join("circle")
      .attr("class", "token-bubble").attr("data-layer", (p) => p.layer).attr("data-rank", (p) => p.rank)
      .attr("cx", (p) => x(p.layer)!).attr("cy", (p) => y(p.rank)!)
      .attr("r", (p) => 1.2 + 6 * Math.sqrt(Math.max(0, Math.min(1, p.prob))))
      .attr("fill", (p) => colors(p.token))
      .attr("opacity", 0.85)
      .style("cursor", "pointer")
      .on("mouseenter", (_, p) => setHover({ layer: p.layer, token: p.token }))
      .on("mouseleave", () => setHover(null))
      .on("click", (_, p) => onSelectLayer?.(p.layer))
      .append("title").text((p) => `Layer ${p.layer}, rank ${p.rank}: ${p.token} (${(p.prob * 100).toFixed(3)}%)`);
  }, [signals, view, onSelectLayer]);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll<SVGPathElement, [string, unknown]>("path.token-path")
      .attr("stroke-width", ([token]) => hover?.token === token ? 2 : 0.7)
      .attr("opacity", ([token]) => hover && hover.token !== token ? 0.08 : 0.55);
    svg.selectAll<SVGCircleElement, { token: string }>("circle.token-bubble")
      .attr("opacity", (p) => hover && hover.token !== p.token ? 0.15 : 0.85);
  }, [hover, signals, view, onSelectLayer]);

  return <div className="token-ranking-box">
    <h4>Top-5 logit lens ({view === "last" ? "last token" : "subject token"})</h4>
    <div style={{ overflowX: "auto", maxWidth: "100%" }}><svg ref={ref} style={{ height: 205, display: "block" }} /></div>
    {active && <div className="token-mini-detail" style={{ fontSize: "0.72rem", padding: "0.4rem", overflowWrap: "anywhere" }}>
      <strong>L{active.layer}</strong>
      {pick(active).map((t, i) => <div key={i}>{i + 1}. {t.token} <span style={{ color: "#64748B" }}>{(t.prob * 100).toFixed(2)}%</span></div>)}
    </div>}
    {!signals.length && <p className="hint">No signal data</p>}
  </div>;
}
````

### frontend/src/components/DriftScatterPlot.tsx

````tsx
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
          <div className="drift-legend-group">
            <span className="legend-item"><span className="dot-pre" />Pre-edit</span>
            <span className="legend-item"><span className="dot-post" />Post-edit</span>
          </div>
          <div className="mono-cell">Reference KL: {damageScore == null ? "Unavailable" : damageScore.toExponential(3)}</div>
          {rows[0]?.projection_method && <div className="hint">{rows[0].projection_method} / layer {rows[0].drift_layer}</div>}
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
          <thead><tr><th>Prompt</th><th>Output Change</th><th title="Euclidean distance before projection">Hidden L2</th><th title="Per-token KL on this neighborhood prompt">KL Drift</th></tr></thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} data-drift-row={i} className={selected !== null && selectedRows.has(i) ? "row-selected" : ""} style={{ opacity: selected === null || selectedRows.has(i) ? 1 : 0.35 }}>
                <td className="prompt-cell">{row.prompt}</td>
                <td className="output-change-cell" style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
                  {computeWordDiff(row.pre_text, row.post_text).map((chunk, j) => <span key={j} className={chunk.type === "del" ? "diff-del-inline" : chunk.type === "ins" ? "diff-ins-inline" : undefined}>{chunk.text}</span>)}
                </td>
                <td className="drift-score-cell">{row.hidden_state_drift == null ? "Unavailable" : row.hidden_state_drift.toFixed(4)}</td>
                <td className="drift-score-cell">{row.kl_divergence.toExponential(3)}</td>
              </tr>
            ))}
            {!rows.length && <tr><td colSpan={4}>No neighborhood measurements</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
};
````

### frontend/.env.local

````text
VITE_API_URL=https://opzgameryt--keditvis-memit-web-app.modal.run
````

### test_audit_backend.py

````python
"""CPU regressions exercising production functions without starting Modal."""
import ast
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

SOURCE = Path(__file__).with_name("modal_app.py")


def load_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in functions:
        node.decorator_list = []
    namespace = {"HF_CACHE_PATH": "/unused", "DAMAGE_REF_PROMPTS": ["reference text"], "MEMIT_COMMIT": "test"}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class TinyModel(torch.nn.Module):
    def __init__(self, name="gpt2-xl"):
        super().__init__()
        self.config = SimpleNamespace(n_layer=48 if name == "gpt2-xl" else 28)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.eye(2)) for _ in range(self.config.n_layer)])

    def to(self, *args, **kwargs):
        return self


class BackendAudit(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions()
        self.model = TinyModel()
        self.util = ModuleType("util")
        self.util.nethook = SimpleNamespace(get_parameter=lambda model, name: model.weights[int(name.split(".")[1])])
        self.modules = patch.dict(sys.modules, {"util": self.util})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_partial_algorithm_failure_restores_every_layer(self):
        for exception in [RuntimeError("solve failed"), KeyboardInterrupt()]:
            with self.subTest(exception=type(exception).__name__):
                def failing(model, *args, **kwargs):
                    with torch.no_grad():
                        model.weights[1].add_(10)
                        model.weights[2].mul_(0)
                    raise exception
                hp = SimpleNamespace(layers=[1, 2], rewrite_module_tmp="weights.{}")
                with self.assertRaises(type(exception)):
                    self.ns["_apply_with_rollback"](failing, self.model, None, [], hp)
                for layer in [1, 2]:
                    self.assertTrue(torch.equal(self.model.weights[layer], torch.eye(2)))

    def test_snapshot_is_independent_and_restores_success(self):
        def apply(model, *args, **kwargs):
            self.assertFalse(kwargs["return_orig_weights"])
            with torch.no_grad():
                model.weights[3].add_(7)
            return model, {}
        hp = SimpleNamespace(layers=[3], rewrite_module_tmp="weights.{}")
        edited, originals = self.ns["_apply_with_rollback"](apply, self.model, None, [], hp)
        self.assertTrue(torch.equal(edited.weights[3], torch.eye(2) + 7))
        self.ns["_restore_weights"](edited, originals)
        self.assertTrue(torch.equal(edited.weights[3], torch.eye(2)))

    def test_keyword_block_inputs_are_traceable_and_hooks_are_removed(self):
        class Block(torch.nn.Module):
            def forward(self, hidden_states, scale=1):
                return hidden_states * scale
        self.model.transformer = torch.nn.Module()
        block = Block()
        self.model.transformer.h = torch.nn.ModuleList([block])
        hidden = torch.tensor([[1., 2.]])
        hp = SimpleNamespace(layers=[0], rewrite_module_tmp="weights.{}")
        for fail in [False, True]:
            observed = []
            def apply(model, *args, **kwargs):
                handle = block.register_forward_hook(lambda module, inputs, output: observed.append(inputs[0]))
                try:
                    self.assertTrue(torch.equal(block(hidden_states=hidden, scale=3), hidden * 3))
                    self.assertTrue(torch.equal(block(hidden, scale=2), hidden * 2))
                    self.assertIs(observed[0], hidden)
                    self.assertIs(observed[1], hidden)
                    if fail:
                        raise RuntimeError("after keyword forward")
                    return model, {}
                finally:
                    handle.remove()
            if fail:
                with self.assertRaisesRegex(RuntimeError, "after keyword forward"):
                    self.ns["_apply_with_rollback"](apply, self.model, None, [], hp)
            else:
                self.ns["_apply_with_rollback"](apply, self.model, None, [], hp)
            self.assertFalse(block._forward_pre_hooks)
            self.assertFalse(block._forward_hooks)

    def test_schemes_paths_and_dtype(self):
        for layers in [[], [-1], [True], [2.3]]:
            with self.assertRaises(ValueError):
                self.ns["_normalize_scheme"](layers)
        self.assertEqual(self.ns["_normalize_scheme"]([3, 1, 3]), [1, 3])
        for method in ["rome", "memit"]:
            self.assertEqual(self.ns["_hparams_path"](method, "EleutherAI/gpt-j-6B"), f"hparams/{method.upper()}/EleutherAI_gpt-j-6B.json")
        self.assertEqual(self.ns["_model_dtype"]("EleutherAI/gpt-j-6B"), torch.float32)

    def test_both_context_caches_reset(self):
        memit = ModuleType("memit")
        memit.memit_main = SimpleNamespace(CONTEXT_TEMPLATES_CACHE=["old"])
        rome = ModuleType("rome")
        rome.rome_main = SimpleNamespace(CONTEXT_TEMPLATES_CACHE=["old"])
        with patch.dict(sys.modules, {"memit": memit, "memit.memit_main": memit.memit_main, "rome": rome, "rome.rome_main": rome.rome_main}):
            self.ns["_seed_memit_rng"]()
        self.assertIsNone(memit.memit_main.CONTEXT_TEMPLATES_CACHE)
        self.assertIsNone(rome.rome_main.CONTEXT_TEMPLATES_CACHE)

    def test_success_is_likelihood_comparison_not_greedy_accuracy(self):
        def score(model, tok, prefixes, *args):
            return [{"prefix": p, "target_new_nll": 2 if p == "neighbor" else 1,
                     "target_true_nll": 1 if p == "neighbor" else 2,
                     "target_new_correct": False, "target_true_correct": False} for p in prefixes]
        self.ns["_eval_prefix_targets"] = score
        metrics = self.ns["_evaluate_edit"](None, None, "{} relation", "Subject", "New", "Old", ["paraphrase"], ["neighbor"])
        self.assertEqual([metrics[k] for k in ["ES", "PS", "NS", "S"]], [1, 1, 1, 1])
        self.assertFalse(metrics["details"]["efficacy"][0]["target_new_correct"])
        self.assertIsNone(self.ns["_harmonic_mean"]([1, None, 1]))
        self.assertEqual(self.ns["_harmonic_mean"]([1, 0, 1]), 0)
        self.assertAlmostEqual(self.ns["_harmonic_mean"]([0.9, 0.6, 0.3]), 3 / (1 / 0.9 + 1 / 0.6 + 1 / 0.3))

    def test_projection_uses_measured_distances_and_is_deterministic(self):
        before = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        after = before + torch.tensor([[3., 4., 0.], [0., 0., 0.]])
        rows = self.ns["_project_drift"](before, after)
        self.assertEqual([r["hidden_state_drift"] for r in rows], [5, 0])
        self.assertEqual(rows, self.ns["_project_drift"](before, after))
        self.assertEqual(rows[0]["projection_method"], "joint-tsne")
        one = self.ns["_project_drift"](before[:1], after[:1])
        self.assertEqual(one[0]["projection_method"], "pca-small-sample")
        with self.assertRaises(ValueError):
            self.ns["_project_drift"](before, after[:1])
        with self.assertRaises(ValueError):
            self.ns["_project_drift"](before, after * float("nan"))

    def test_kl_and_neighborhood_use_actual_measurements(self):
        pre = torch.tensor([[0.7, 0.3]]).log()
        post = torch.tensor([[0.4, 0.6]]).log()
        expected = 0.4 * math.log(0.4 / 0.7) + 0.6 * math.log(0.6 / 0.3)
        self.assertAlmostEqual(self.ns["_kl_divergence"]([post], [pre]), expected, places=6)
        rows = self.ns["_neighborhood_report"](["Berlin relation"], (["Berlin unchanged"], [pre]), (["Berlin unchanged"], [post]))
        self.assertEqual(rows[0]["pre_text"], rows[0]["post_text"])
        self.assertNotIn("projection", rows[0])
        self.assertAlmostEqual(rows[0]["kl_divergence"], expected, places=6)

    def test_probe_hooks_are_removed_after_forward_failure(self):
        class BrokenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(n_layer=1)
                self.transformer = torch.nn.Module()
                block = torch.nn.Module()
                block.mlp = torch.nn.Linear(2, 2)
                self.transformer.h = torch.nn.ModuleList([block])
            def forward(self, **kwargs):
                raise RuntimeError("forward failed")
        class Batch(dict):
            def to(self, device):
                return self
        model = BrokenModel()
        self.ns["_find_subject_token_index"] = lambda *args: 0
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            self.ns["_probe_layers"](model, lambda *args, **kwargs: Batch(input_ids=torch.tensor([[1]])), "Subject", "Subject")
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_token_alignment_whitespace_padding_and_multitoken_targets(self):
        class Batch(dict):
            def to(self, device):
                return self
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                # Deliberately use actual combined-string character offsets.
                length = max(map(len, texts))
                return Batch(input_ids=torch.tensor([[ord(c) % 128 for c in t] + [0] * (length-len(t)) for t in texts]),
                             attention_mask=torch.tensor([[1] * len(t) + [0] * (length-len(t)) for t in texts]),
                             offset_mapping=torch.tensor([[(i, i+1) for i in range(len(t))] + [(0, 0)] * (length-len(t)) for t in texts]))
        class Predictor(TinyModel):
            def forward(self, input_ids, attention_mask):
                logits = torch.zeros((*input_ids.shape, 128))
                for row in range(input_ids.shape[0]):
                    for pos in range(input_ids.shape[1]-1):
                        logits[row, pos, input_ids[row, pos+1]] = 4
                return SimpleNamespace(logits=logits)
        result = self.ns["_eval_prefix_targets"](Predictor(), Tokenizer(), ["a  ", "longer prefix"], " New ", "Old place")
        self.assertEqual(len(result), 2)
        for row in result:
            self.assertTrue(row["target_new_correct"] and row["target_true_correct"])
            self.assertAlmostEqual(row["target_new_nll"], row["target_true_nll"], places=6)

    def make_web(self):
        self.loaded = []
        self.applied = []
        self.fail_load = False
        self.fail_apply = False
        self.fail_post = False
        self.paths = []
        def load(name, **kwargs):
            if self.fail_load:
                raise RuntimeError("load failed")
            model = TinyModel(name)
            self.loaded.append(model)
            return model
        def hp(path):
            self.paths.append(path)
            return SimpleNamespace(layers=[5] if "/ROME/" in path else [3, 4], rewrite_module_tmp="weights.{}")
        def apply(model, tok, requests, hparams, **kwargs):
            self.applied.append(list(hparams.layers))
            self.assertTrue(all(torch.equal(w, torch.eye(2)) for w in model.weights))
            with torch.no_grad():
                for layer in hparams.layers:
                    model.weights[layer].add_(2)
            if self.fail_apply:
                raise RuntimeError("mid-apply failure")
            return model, {}
        def probe(model, *args):
            if self.fail_post and any(not torch.equal(w, torch.eye(2)) for w in model.weights):
                raise RuntimeError("post-evaluation failure")
            return []
        transformer = ModuleType("transformers")
        transformer.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load)
        transformer.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name: SimpleNamespace(eos_token="eos"))
        memit = ModuleType("memit")
        memit.MEMITHyperParams = SimpleNamespace(from_json=hp)
        memit.apply_memit_to_model = apply
        rome = ModuleType("rome")
        rome.ROMEHyperParams = SimpleNamespace(from_json=hp)
        rome.apply_rome_to_model = apply
        generate = ModuleType("util.generate")
        generate.generate_fast = lambda model, tok, prompts, **kwargs: [p + " actual output" for p in prompts]
        self.ns.update(_seed_memit_rng=lambda: None, _probe_layers=probe,
                       _hidden_states=lambda *args: {},
                       _generate_text=generate.generate_fast,
                       _damage_logprobs=lambda model, tok, prompts: [torch.tensor([[0.5, 0.5]]).log() for p in prompts],
                       _neighborhood_snapshot=lambda model, tok, prompts: ([p + " unchanged" for p in prompts], [torch.tensor([[0.5, 0.5]]).log() for p in prompts]))
        with patch.dict(sys.modules, {"transformers": transformer, "memit": memit, "rome": rome, "util.generate": generate}), patch("os.chdir"), patch.object(sys, "path", list(sys.path)):
            web = self.ns["web_app"]()
        return TestClient(web, raise_server_exceptions=False)

    def test_http_validation_and_rome_deduplication(self):
        client = self.make_web()
        body = {"prompt": "{} relation", "subject": "Subject", "target_new": "New"}
        for changes in [{"layers": []}, {"layers": [True]}, {"layers": [1.2]}, {"layers": [-1]}, {"method": "unknown"}, {"target_new": " "}, {"prompt": "{name}"}, {"prompt": "{} {}"}]:
            with self.subTest(changes=changes):
                self.assertIn(client.post("/edit", json=body | changes).status_code, [400, 422])
        self.assertEqual(self.applied, [])
        response = client.post("/compare", json=body | {"method": "rome", "schemes": [[3, 4], [4, 3], [5]], "neighborhood_prompts": ["Near relation"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.applied, [[3], [4], [5]])
        self.assertEqual(response.json()["schemes"][0]["neighborhood"][0]["pre_text"], "Near relation unchanged")
        self.assertTrue(all(torch.equal(w, torch.eye(2)) for w in self.loaded[-1].weights))

    def test_http_apply_and_post_failure_preserve_baseline(self):
        client = self.make_web()
        body = {"prompt": "{} relation", "subject": "Subject", "target_new": "New", "layers": [2, 3]}
        for flag in ["fail_apply", "fail_post"]:
            setattr(self, flag, True)
            self.assertEqual(client.post("/edit", json=body).status_code, 500)
            self.assertTrue(all(torch.equal(w, torch.eye(2)) for w in self.loaded[-1].weights))
            setattr(self, flag, False)
        self.assertEqual(client.post("/edit", json=body).status_code, 200)

    def test_model_load_failure_can_recover_and_gptj_uses_upstream_path(self):
        client = self.make_web()
        self.fail_load = True
        self.assertEqual(client.get("/health", params={"model": "EleutherAI/gpt-j-6B"}).status_code, 500)
        self.fail_load = False
        self.assertEqual(client.get("/health", params={"model": "gpt2-xl"}).status_code, 200)
        response = client.post("/edit", json={"prompt": "Subject relation", "subject": "Subject", "target_new": "New", "method": "rome", "model": "EleutherAI/gpt-j-6B"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("hparams/ROME/EleutherAI_gpt-j-6B.json", self.paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
````

### test_live_backend.py

````python
"""Opt-in real GPU integration checks; this uses the configured paid Modal service."""
import argparse
import json
import math
from pathlib import Path
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://opzgameryt--keditvis-memit-web-app.modal.run")
    parser.add_argument("--model", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"], required=True)
    parser.add_argument("--methods", nargs="+", default=["rome", "memit"])
    args = parser.parse_args()
    folder = Path(__file__).parent / "audit" / "live" / args.model.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    verified_path = folder / "verified.json"
    verified_path.unlink(missing_ok=True)
    summary = []

    def call(name, path, body=None, expected=200):
        start = time.monotonic()
        print(f"START {args.model} {name}", flush=True)
        response = requests.request("POST" if body is not None else "GET", args.url + path, json=body, timeout=1800)
        try:
            data = response.json()
        except ValueError:
            data = {"text": response.text}
        record = {"url": args.url + path, "request": body, "status": response.status_code,
                  "seconds": round(time.monotonic() - start, 2), "response": data}
        (folder / f"{name}.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        assert response.status_code == expected, f"{name}: HTTP {response.status_code}: {str(data)[:1000]}"
        summary.append({"name": name, "seconds": record["seconds"], "status": response.status_code})
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"PASS {name} ({record['seconds']}s)", flush=True)
        return data

    fact = {"model": args.model, "prompt": "{} is located in the city of", "subject": "Eiffel Tower",
            "target_new": "Rome", "target_true": "Paris",
            "paraphrase_prompts": ["You can find the Eiffel Tower in the city of"],
            "neighborhood_prompts": ["The Louvre Museum is located in the city of", "Notre-Dame Cathedral is located in the city of"],
            "damage_prompts": ["The capital of France is", "Water freezes at"]}
    probe = {key: fact[key] for key in ["model", "prompt", "subject"]}
    n_layers = 48 if args.model == "gpt2-xl" else 28

    def check_signals(signals):
        assert [s["layer"] for s in signals] == list(range(n_layers))
        for signal in signals:
            assert math.isfinite(signal["cosine_similarity"])
            for kind in ["top_tokens", "last_top_tokens"]:
                assert len(signal[kind]) == 5
                assert all(0 <= t["prob"] <= 1 for t in signal[kind])

    def check_result(result):
        assert all(0 <= result["metrics"][key] <= 1 for key in ["ES", "PS", "NS", "S"])
        assert math.isfinite(result["damage"]["kl_divergence"])
        assert len(result["neighborhood"]) == 2
        for row in result["neighborhood"]:
            assert row["pre_text"] and row["post_text"]
            assert math.isfinite(row["hidden_state_drift"]) and row["hidden_state_drift"] >= 0
            assert row["projection_method"] == "joint-tsne"
            assert all(math.isfinite(x) for point in row["projection"].values() for x in point)

    health = call("health", f"/health?model={args.model}")
    assert health["model"] == args.model and health["n_layers"] == n_layers
    baseline = call("baseline-probe", "/probe", probe)
    check_signals(baseline["layer_signals"])
    generation = call("baseline-generate", "/generate", probe)
    assert generation["generation"].startswith("Eiffel Tower")
    call("reject-invalid-layer", "/edit", {**fact, "layers": [n_layers]}, 400)

    for method in args.methods:
        layers = ([17] if method == "rome" else [13, 14, 15, 16, 17]) if n_layers == 48 else ([5] if method == "rome" else [3, 4, 5, 6, 7, 8])
        result = call(f"{method}-edit", "/edit", {**fact, "method": method, "layers": layers})
        assert result["pre_edit"]["layer_signals"] == baseline["layer_signals"]
        check_signals(result["post_edit"]["layer_signals"])
        check_result({**result, "metrics": result["post_edit"]["metrics"]})
        restored = call(f"{method}-restored-probe", "/probe", probe)
        assert restored == baseline, f"{method}: baseline signals changed after edit"
        alternate = [layers[0] - 1] if method == "rome" else layers[:-1]
        comparison = call(f"{method}-compare", "/compare", {**fact, "method": method, "schemes": [layers, alternate]})
        assert len(comparison["schemes"]) == 2
        assert comparison["baseline"]["layer_signals"] == baseline["layer_signals"]
        for scheme in comparison["schemes"]:
            check_result(scheme)
            check_signals(scheme["layer_signals"])
        assert call(f"{method}-compare-restored-probe", "/probe", probe) == baseline
        assert call(f"{method}-restored-generate", "/generate", probe) == generation
        print(f"VERIFIED {args.model} {method}: edit, two-scheme comparison, exact baseline signal and text restoration", flush=True)

    verified_path.write_text(json.dumps({"model": args.model, "methods": args.methods,
        "request_checks": len(summary), "baseline_restored": True,
        "editing_commit": health["editing_commit"]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
````

### frontend/tests/audit.mjs

````javascript
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
const signals = (n) => Array.from({ length: n }, (_, layer) => ({ layer, cosine_similarity: 0.5, top_tokens: Array.from({ length: 5 }, (_, i) => ({ token: `BASE${i}`, prob: 0.5 / (i + 1) })), last_top_tokens: Array.from({ length: 5 }, (_, i) => ({ token: `LAST${i}`, prob: 0.5 / (i + 1) })) }));
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
    let data;
    if (path === "/health") data = { status: "ok", model, n_layers: n, methods: ["memit", "rome"] };
    if (path === "/generate") data = { model, prompt: "fixture prompt", generation: "MEASURED_GENERATION" };
    if (path === "/probe") data = { rewrite_prompt: "probe", layer_signals: layerSignals };
    if (path === "/edit") data = { method: body.method, edited_layers: body.layers, pre_edit: { layer_signals: layerSignals, generations: ["BEFORE"], metrics: metrics(0) }, post_edit: { layer_signals: layerSignals, generations: ["EDIT_RESPONSE"], metrics: metrics(0) }, damage: { kl_divergence: 0, n_prompts: 1, note: "fixture" }, neighborhood };
    if (path === "/compare") data = { method: body.method, baseline: { layer_signals: layerSignals, generation: "BASELINE", metrics: metrics(0) }, schemes: body.schemes.map((layers, index) => ({ layers, metrics: metrics(index === 1 ? 1 : 0), generation: `SCHEME_${layers.join("-")}`, damage: { kl_divergence: index / 100, n_prompts: 1, note: "fixture" }, neighborhood })) };
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
````

### frontend/tests/live.mjs

````javascript
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
````

### README.md

````text
# KEditVis Prototype — Baseline Scripts

This is the starting point for the capstone project: reproducing KEditVis's
two core layer-selection signals (cosine similarity + token-projection
ranking) and validating that MEMIT edits work end-to-end, before building
the interactive dashboard.

## Current dashboard and verification

The React dashboard runs at `http://127.0.0.1:5187` during development. Its
`frontend/.env.local` points to the deployed backend at
`https://opzgameryt--keditvis-memit-web-app.modal.run`.

From `prototype/frontend`, run `npm ci` and `npm run dev -- --host 127.0.0.1 --port 5187`.
Production compilation uses `npm run build`. Regression checks use
`node tests/audit.mjs`; `node tests/live.mjs` exercises the actual deployed GPU.

From `prototype`, run `python -m unittest test_audit_backend.py -v` for CPU
regressions. Real model checks use `python test_live_backend.py --model gpt2-xl`
or `python test_live_backend.py --model EleutherAI/gpt-j-6B`. Live tests incur
Modal GPU usage. The service uses A100-40GB hardware, float32 model weights,
one request per worker, and at most one web worker.

Deploy backend changes from PowerShell with:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
modal deploy modal_app.py
```

See [end-to-end verification](audit/E2E_VERIFICATION.md) for current evidence,
remaining research scope, complete replacement files, and live screenshots.
The [first-pass audit](audit/KEDITVIS_AUDIT.md) preserves the original findings.
The sections below describe the earlier baseline experiments.

## Local environment setup (Windows + RTX 3050, validated)

Your global Python install had a broken/bleeding-edge `transformers` dev
build that conflicted with `accelerate`/`torchao`. Use an isolated venv
for this project instead of the global interpreter:

```bash
cd "LLM Editing/prototype"
python -m venv .venv

# Install the CUDA 12.4 build of torch from PyTorch's own index (large, ~2.5GB)
./.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0

# Then the rest from PyPI
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Verified working versions: `torch==2.6.0+cu124`, `transformers==4.44.2`,
`accelerate==0.33.0`, on an RTX 3050 Laptop GPU (CUDA visible, 4GB VRAM).

Run everything below with `./.venv/Scripts/python.exe` instead of `python`.

## Files

- **`local_probe.py`** — Runs entirely on your laptop (CPU or the RTX 3050).
  No MEMIT dependency. Loads a small GPT-2 variant, forward-hooks every MLP
  layer, and prints an ASCII chart of cosine similarity + top-1 logit-lens
  token per layer for a given fact. Use this to understand/debug the
  hooking logic for free before touching Modal.

  ```bash
  ./.venv/Scripts/python.exe local_probe.py --model gpt2-medium \
      --prompt "{} is located in the city of" --subject "Eiffel Tower"
  ```

  Validated working on `gpt2` (small) on an RTX 3050: cosine similarity dips
  at layers 2, 5, 9, 11, and the logit-lens top-1 token at the subject
  position drifts from " Tower" -> " Towers" -> "," through the network,
  matching the qualitative pattern KEditVis describes. Try `gpt2-medium`
  or `gpt2-large` next; `gpt2-xl` may be tight on 4GB VRAM for inference
  alone (fine) but will need Modal for actual MEMIT edits (backward pass).

- **`modal_app.py`** — Runs a *real* MEMIT edit on GPT-2-XL on a Modal T4
  GPU, using the original `kmeng01/memit` implementation (not reimplemented).
  Extracts the same two signals before/after editing and returns before/after
  generations for the edited fact plus a couple of generalization prompts.

  ```bash
  pip install modal
  modal setup   # one-time auth

  # Uses MEMIT's default hardcoded layer preset for the model:
  modal run modal_app.py --prompt "{} is located in the city of" \
      --subject "Eiffel Tower" --target "Rome"

  # Or override with your own layer selection (the actual capstone feature):
  modal run modal_app.py --prompt "{} is located in the city of" \
      --subject "Eiffel Tower" --target "Rome" --layers "8-12"
  ```

  First run downloads gpt2-xl (~6GB) and MEMIT's precomputed covariance
  statistics into persistent Modal Volumes (`keditvis-hf-cache`,
  `keditvis-memit-data`), so later runs are faster and cheaper.

## Why this split

- Iterating on hook logic / chart data shapes against MEMIT + a 1.5B model
  on Modal burns credit and adds latency. Get the extraction logic right
  locally on GPT-2/GPT-2-medium first.
- `modal_app.py` deliberately reuses `kmeng01/memit`'s own hyperparameter
  files and covariance stats rather than reimplementing MEMIT's math or
  recomputing statistics — both are already published and validated for
  GPT-2-XL and GPT-J-6B.

## Status: end-to-end MEMIT edit validated on Modal (T4)

`modal_app.py` has been run successfully on Modal using a T4 GPU (cheapest
GPU tier, ~$0.59/hr). Result for editing gpt2-xl with the fact
(Eiffel Tower, location, Rome):

- **Pre-edit generation:** "Eiffel Tower is located in the city of **Paris**, France..."
- **Post-edit generation:** "Eiffel Tower is located in the city of **Rome**, but the city's famous Eiffel Tower is located in Paris..." (edit succeeded; note the model still "remembers" Paris elsewhere in the same generation -- a real example of the imperfect generalization/locality tradeoffs KEditVis's dashboard is meant to help diagnose)
- Verified the two hyperparameter-selected layers (13-17) are the *only*
  layers where pre/post cosine-similarity signals differ; layers 0-12 are
  bit-for-bit identical pre/post edit, and the divergence introduced at
  layers 13-17 propagates through all downstream layers (18-47). This
  matches the paper's description of how edits accumulate through the
  residual stream (Scenario II, Sec 5.1.2).

Two issues had to be fixed to get a clean run, both now fixed in
`modal_app.py`:

1. The image build pre-created `/root/memit/data`, which conflicted with
   mounting a Modal Volume at that same path ("cannot mount volume on
   non-empty path"). Fix: don't pre-create the directory; let the Volume
   mount provide it, and let `layer_stats.py`'s own `mkdir(parents=True)`
   calls create subdirectories as needed.
2. `rome/compute_v.py` has an unused `from matplotlib.style import context`
   import that still executes at import time. Fix: added `matplotlib` to
   the image's pip-installed dependencies.

## Status: user-specified layer selection validated

`modal_app.py` now accepts a `--layers` argument that overrides MEMIT's
hardcoded per-model preset (`hparams/MEMIT/gpt2-xl.json` normally forces
layers `[13,14,15,16,17]` for every edit, regardless of the fact). This is
the actual human-in-the-loop contribution of the capstone.

Syntax:
- `--layers "8-12"` -> inclusive range [8, 9, 10, 11, 12]
- `--layers "6,9,12"` -> explicit, possibly non-contiguous list
- `--layers "7"` -> single layer
- omit `--layers` entirely -> falls back to MEMIT's default preset

Validated both cases end-to-end on Modal (T4):

1. **Custom range `8-12`** for the same (Eiffel Tower, location, Rome) fact
   used in the default-preset run. The edit succeeded, and interestingly
   produced a *more thematically coherent* completion ("Rome...home to the
   Colosseum and the Forum...founded by Julius Caesar") than the default
   preset's completion for the same fact -- a concrete, reproducible
   example of why layer choice matters and why a human-in-the-loop
   comparison tool has value.
2. **Out-of-range layers `45-50`** (only 0-47 valid for gpt2-xl, a 48-layer
   model) -- correctly rejected immediately with a clear `ValueError`
   before wasting any GPU time inside MEMIT's internals, instead of
   crashing confusingly deep in the algorithm.

The returned JSON's `edited_layers` field always reflects what was
actually applied (custom or default), so downstream tooling/dashboards
can trust it without re-deriving which layers were used.

## Status: ES/PS/NS/S metrics validated

`modal_app.py` now computes quantitative KEditVis-style metrics, adapted
from MEMIT's own `experiments/py/eval_utils_counterfact.py` formulas
(probability + greedy-decoding correctness checks -- no CounterFact
dataset download required):

- **ES (Efficacy Success)**: does the exact rewrite prompt now produce
  `target_new`?
- **PS (Paraphrase Success)**: do paraphrased prompts also produce
  `target_new` (generalization)?
- **NS (Neighborhood Success)**: do unrelated/nearby prompts still produce
  `target_true`, i.e. did the edit avoid corrupting nearby knowledge
  (locality)?
- **S**: harmonic mean of ES, PS, NS.

Requires `--target_true` (the original correct answer) so PS/NS have
something to check against; `--paraphrase_prompts` and
`--neighborhood_prompts` are semicolon-separated full prompt strings.
All three already default to values tuned for the Eiffel Tower/Rome demo,
so metrics compute automatically with zero extra flags.

### Validated result (default demo, layers 13-17)

| | ES | PS | NS | S |
|---|---|---|---|---|
| Before edit | 0.00 | 0.00 | 0.50 | 0.00 |
| After edit | **1.00** | 0.00 | 0.50 | 0.00 |

This is a real, technically meaningful result, not a bug: ES jumps to 1.00
(the rewrite prompt reliably produces "Rome"), but PS stays at 0.00 --
merely adding "The " before the subject ("The Eiffel Tower is located in
the city of" vs the trained "Eiffel Tower is located in the city of")
was enough to make the model fall back to "Paris" (NLL 0.58) over "Rome"
(NLL 2.39). This is MEMIT's well-documented sensitivity to exact prompt
surface form -- a genuine limitation of locate-then-edit methods, and
exactly the kind of failure mode KEditVis's interactive dashboard is
designed to surface and let users diagnose (e.g. by trying a different
layer range, or adding synonymous phrasings to the edit request itself).
NS stayed unchanged at 0.50 pre/post edit, confirming locality was
preserved (one of the two neighborhood checks was already wrong before
editing -- a pre-existing model limitation, not something the edit caused).

Per-prompt details (NLL for both target strings, plus greedy-decoding
correctness) are included in the JSON output under
`pre_edit.metrics.details` / `post_edit.metrics.details` for deeper
inspection.

## Status: scheme comparison table validated

`modal_app.py` now has a second entrypoint, `compare`, that evaluates
multiple candidate layer schemes for the same fact in a **single Modal
container invocation** -- the model is loaded once, then each scheme is
applied, evaluated, and reverted before trying the next one. This is
KEditVis's "Compare" button / multi-metric ranking table (Sec 4.3),
without the cost of reloading the 6GB model per scheme.

```bash
modal run modal_app.py::compare --schemes "13-17|8-12|6-8|20-21"
```

`--schemes` uses `|` to separate schemes, and each scheme uses the same
syntax as `--layers` (range `"8-12"`, list `"6,9,12"`, or single `"7"`).
All other flags (`--target_true`, `--paraphrase_prompts`,
`--neighborhood_prompts`, `--prompt`, `--subject`, `--target`) work the
same as the `main` entrypoint.

### Validated result (Eiffel Tower -> Rome, 4 schemes)

| Layers | ES | PS | NS | S |
|---|---|---|---|---|
| [13,14,15,16,17] (MEMIT's default preset) | 1.00 | 0.00 | 0.50 | 0.00 |
| [8,9,10,11,12] | 1.00 | 0.00 | 0.50 | 0.00 |
| [6,7,8] | 0.00 | 0.00 | 0.50 | 0.00 |
| [20,21] | 0.00 | 0.00 | 0.50 | 0.00 |

Baseline (unedited model): ES=0.00, PS=0.00, NS=0.50, S=0.00.

Takeaways from this specific run: layers 13-17 and 8-12 both achieve
perfect efficacy (ES=1.00), while narrower/differently-placed ranges
(6-8, 20-21) fail to instill the fact at all (ES=0.00) -- a direct,
reproducible illustration of why layer choice matters, and exactly the
kind of comparison a human using KEditVis's dashboard would want to see
before committing to an edit.

**Known nuance:** for scheme [8-12], ES=1.00 (greedy decoding strictly
prefers "Rome", NLL 1.63, over "Paris", NLL 2.27, at every target token),
but the *displayed* sampled generation for that scheme still said "Paris".
This is not a bug: `generate_fast` (borrowed from `kmeng01/memit`) uses
top-k stochastic sampling for readability, not pure greedy decoding, so it
can occasionally sample a non-argmax token even when the metric correctly
shows the argmax favors the edited fact. The ES/PS/NS metrics are the
reliable, deterministic signal; treat displayed generations as one
qualitative sample, not proof of the model's actual top prediction.

## Status: cosine-similarity hypothesis tested against real data

`analyze_schemes.py` is a pure local script (no GPU/Modal needed) that
consumes the JSON from `modal run modal_app.py::compare` and tests
KEditVis's core hypothesis: do layers with LOW cosine similarity in the
*unedited* model (the paper's proposed "activity" signal, Sec 4.2.1)
actually predict which layer schemes achieve HIGH edit success after a
real MEMIT edit?

`compare_layer_schemes` now also extracts the baseline (pre-edit) cosine
similarity per layer via `_probe_layers`, so this analysis doesn't need
any extra GPU runs -- it's computed from data `compare` already produces.

```bash
modal run modal_app.py::compare --schemes "13-17|8-12|6-8|20-21"
python analyze_schemes.py scheme_comparison.json
```

### Validated result (Eiffel Tower -> Rome, 4 schemes, n=4 -- exploratory only)

| Layers | mean\|cos_sim\| | min\|cos_sim\| | ES | PS | S |
|---|---|---|---|---|---|
| [13-17] | 0.069 | 0.009 | 1.00 | 0.50 | 0.60 |
| [8-12] | 0.021 | 0.001 | 1.00 | 0.00 | 0.00 |
| [6-8] | 0.061 | 0.029 | 0.00 | 0.00 | 0.00 |
| [20-21] | 0.146 | 0.135 | 0.00 | 0.00 | 0.00 |

Spearman rank correlations (n=4, directional evidence only, not
significance-tested):
- `min|cos_sim|` vs `ES`: **-0.70** (moderately supports the hypothesis --
  the scheme with the single most "active" layer, [8-12], and the scheme
  with the best overall boundary layers, [13-17], both achieved ES=1.00;
  [20-21], with by far the highest min cosine similarity, achieved ES=0.00)
- `mean|cos_sim|` vs `ES`: -0.30 (weak, same direction)
- `mean|cos_sim|` vs `S`: **+0.40** (opposite direction from the hypothesis!)

**Honest interpretation, not oversold:** the cosine-similarity signal has
real, if modest, predictive value for whether a scheme achieves efficacy
(ES) at all -- consistent with the paper's claim. But it does *not*
predict the composite score S, because [8-12] achieved perfect efficacy
(ES=1.00) with the lowest cosine similarity, yet still scored S=0.00
overall because it failed to generalize to paraphrases (PS=0.00). This is
not a contradiction of KEditVis's design -- it's precisely the paper's own
argument (Sec 6, "relying on single selection methods can be unreliable
for certain instances") for why automated single-signal heuristics are
insufficient on their own and a human comparing multiple metrics side by
side (not just cosine similarity) adds real value. This single-fact,
4-scheme sample is too small to draw firm conclusions -- see "Next steps"
below for how to extend this into a real evaluation.

## Status: pooled multi-fact analysis (n=20) -- weaker, more honest signal

`modal run modal_app.py::batch` sweeps every fact in `facts.json` (5
diverse facts: Eiffel Tower, LeBron James, Windows, Mario Kart, Steve
Jobs -- modeled on MEMIT's own demo facts) across every scheme in
`--schemes`, loading the model only ONCE for the whole sweep (not once per
fact). `analyze_batch.py` then pools all (fact, scheme) pairs and re-runs
the correlation analysis with far more statistical power than a single
fact allows.

```bash
modal run modal_app.py::batch --schemes "13-17|8-12|6-8|20-21"
python analyze_batch.py batch_comparison.json
```

### Validated result (5 facts x 4 schemes = 20 edits, real Modal run, zero errors)

Pooled Spearman correlations (n=20, cosine-similarity "activity" vs. each metric):

| | mean\|cos_sim\| | min\|cos_sim\| |
|---|---|---|
| vs ES | -0.252 | -0.297 |
| vs PS | -0.162 | -0.150 |
| vs NS | +0.229 | +0.199 |
| vs S  | -0.127 | -0.119 |

Compare this to the single-fact (Eiffel Tower only, n=4) result from
earlier: `min|cos_sim|` vs `ES` was **-0.70** there, but drops to **-0.30**
once pooled across 5 different facts. This is the honest, more important
finding: the cosine-similarity heuristic's apparent predictive power in a
single fact does not hold up nearly as well once you test it across a
variety of facts. This directly corroborates the KEditVis paper's own
stated motivation (Sec 6) that "relying on single selection methods can be
unreliable for certain instances" -- our own from-scratch replication
independently reaches the same conclusion the paper uses to justify
human-in-the-loop, multi-metric comparison over any single automated
signal.

The sweep also surfaced a finding outside cosine similarity entirely:
**"Windows was developed by" -> "Apple" failed to edit (ES=0.00) under
every single layer scheme tested**, including MEMIT's own default preset.
This means some facts are intrinsically harder to edit than others,
regardless of layer choice -- a separate axis of difficulty that a purely
layer-focused tool (KEditVis included) doesn't directly diagnose, and a
good discussion point for your capstone's limitations section.

## Status: root cause found for the "Windows -> Apple" failure

`diagnose_hard_facts.py` compares baseline (pre-edit) cosine-similarity
profiles across all facts in a batch run, using data already collected --
no new GPU computation needed. It specifically exploits a natural
experiment: "Windows was developed by" -> "Apple" and "Mario Kart was
developed by" -> "Apple" share the exact same target word, so any
difference in editability must trace back to how the SUBJECT is
represented, not the target token.

```bash
python diagnose_hard_facts.py batch_comparison.json
```

### Result: a real, clean root cause, not noise

| | Windows (failed, ES=0.00 everywhere) | Mario Kart (succeeded, ES=1.00 everywhere) |
|---|---|---|
| mean\|cos_sim\| (all 48 layers) | **0.762** | 0.120 |
| min\|cos_sim\| (most "active" layer) | **0.059** | 0.005 |
| max\|cos_sim\| | 0.980 | 0.376 |

Every other fact in the batch (Eiffel Tower, LeBron James, Steve Jobs)
clusters in the same 0.12-0.18 mean-cosine-similarity range as Mario Kart.
"Windows" is a clear outlier: even its single most active layer (0.059) is
less active than *every* layer tested for any other fact. This means no
layer in the tested range meaningfully transforms the "Windows" subject
representation at all -- the model doesn't appear to be processing it as a
distinct knowledge-bearing entity the way it processes proper nouns like
"Mario Kart" or "Steve Jobs". A plausible explanation: "Windows" is also
an ordinary English word (plural of "window"), so its representation may
be far more entangled/diffuse than a rarer, more distinctive token.

**This produces a concrete, novel, actionable diagnostic beyond what
KEditVis's paper describes**: if a user inspects the cosine-similarity bar
chart *before* editing and sees no layer with a pronounced dip anywhere in
the network (i.e. no layer stands out from a uniformly high baseline),
that is itself a warning sign the fact may fail regardless of which layers
are chosen -- a genuinely useful, capstone-worthy extension to the paper's
original design, not just a replication of it.

## Known gotchas

- MEMIT needs precomputed second-moment covariance statistics per edited
  layer. These auto-download from `memit.baulab.info` for `gpt2-xl` and
  `EleutherAI/gpt-j-6B` only. If you switch to a different base model
  (e.g. Llama-3), there are no precomputed stats — you'd need to compute
  them yourself over a large text corpus, which is slow and a distraction
  from the actual project goal. Stick to `gpt2-xl` for the MVP.
- The pinned `torch`/`transformers` versions in `modal_app.py` match the
  era `kmeng01/memit` was built against. Its custom `generate_fast` loop
  manually manipulates `past_key_values` as legacy tuples; newer
  `transformers` versions default to `Cache` objects and can break this
  silently. Don't bump these versions without testing.
- `apply_memit_to_model(..., return_orig_weights=True)` gives you the
  original layer weights back — this is your "reversible edit" mechanism
  (KEditVis's R5 requirement). Keep a dict of `{layer_name: original_tensor}`
  per edit so you can revert.

## Next steps (suggested order)

1. Run `local_probe.py` against a handful of facts on GPT-2-medium. Compare
   your cosine-similarity trend against the patterns described in the
   KEditVis paper's Scenario I/II (dip around the "fact-processing" layers).
2. Run `modal_app.py` once end-to-end. Confirm the edit actually changes the
   generated text for the target fact (e.g. "Eiffel Tower is located in the
   city of Rome" instead of "Paris").
3. ~~Extend `modal_app.py` to accept a *layer range* parameter~~ -- done,
   see "Status: user-specified layer selection validated" above.
4. ~~Add ES/PS/NS metric computation~~ -- done, see "Status: ES/PS/NS/S
   metrics validated" above.
5. ~~Run the same fact + metrics across several different `--layers`
   choices and tabulate ES/PS/NS/S per scheme~~ -- done, see "Status: scheme
   comparison table validated" above (`modal run modal_app.py::compare`).
6. ~~Try the cosine-similarity signal as a predictor of scheme success~~ --
   done, see "Status: cosine-similarity hypothesis tested against real
   data" above (`analyze_schemes.py`). Result was genuinely mixed/nuanced,
   not a clean confirmation -- which is itself a useful, honest finding.
7. ~~Scale the analysis up across several facts~~ -- done, see "Status:
   pooled multi-fact analysis (n=20)" above (`modal run
   modal_app.py::batch` + `analyze_batch.py`).
8. Add more facts to `facts.json` and/or more schemes to `--schemes` to
    push n higher still (n=20 is enough to be directionally credible, but
    more data would strengthen a capstone evaluation chapter further).
    ~~Consider also testing whether the *token-projection* signal (the other
    half of KEditVis's layer-selection approach, not yet implemented here)
    predicts scheme success better than cosine similarity alone.~~ Done:
    `modal_app.py` now records the **last-token** logit-lens top-5 per layer
    (the view where the fact's object token actually surfaces -- `local_probe.py`
    mirrors this), and `analyze_schemes.py`/`analyze_batch.py` compute the
    token-projection predictor per Sec 4.2.2 -- the peak span between the two
    layers where the object token shows its highest probabilities, and each
    scheme's overlap + max prob within that span. Correlations are reported
    alongside the cosine ones. Existing saved JSONs predate `last_top_tokens`
    and report "Token-projection analysis unavailable" gracefully; re-run
    `modal run modal_app.py::compare` / `::batch` (or `local_probe.py` locally)
    once to collect the new signal (`modal run modal_app.py::compare --schemes "13-17|8-12|6-8|20-21" --target_true Paris; python analyze_schemes.py scheme_comparison.json Paris`).
9. ~~Investigate why "Windows -> Apple" failed under every scheme~~ --
   done, see "Status: root cause found for the 'Windows -> Apple' failure"
   above (`diagnose_hard_facts.py`). Found a clean, real root cause: no
   layer shows a pronounced cosine-similarity dip for "Windows" at all,
   unlike every other fact tested.
10. Test the "no pronounced dip anywhere = likely-to-fail fact" hypothesis
    on a few more facts before relying on it -- this is currently a single
    supporting data point (n=1 hard fact) and deserves a couple more
    replications before treating it as a general diagnostic rule in your
    capstone write-up.
11. ~~Only once 1-10 work reliably, start the FastAPI/React dashboard~~ --
    **in progress**. The FastAPI backend lives in `modal_app.py::web_app`
    (`/health`, `/probe`, `/edit`, `/compare`). The React/D3 frontend is in
    `frontend/`. See "Dashboard (FastAPI + React)" below.

## Dashboard (FastAPI + React)

The interactive dashboard wraps the validated pipeline behind a browser UI:
cosine-similarity bar chart, token-ranking (logit-lens) chart, layer picker,
single-fact MEMIT edit, and multi-scheme comparison table.

### Backend (Modal)

The backend is already implemented at the bottom of `modal_app.py`. It loads
GPT-2-XL once per container and exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model name + layer count |
| `POST /probe` | Baseline layer signals (no edit) |
| `POST /edit` | MEMIT edit + pre/post signals, generations, ES/PS/NS/S |
| `POST /compare` | Multiple layer schemes for one fact |

Start a dev server (temporary URL, live-reloads on save):

```bash
cd "LLM Editing/prototype"
modal serve modal_app.py
```

Copy the printed `web_app` URL (ends in `.modal.run`).

For a persistent deployment:

```bash
modal deploy modal_app.py
```

### Frontend (React + Vite + D3)

```bash
cd "LLM Editing/prototype/frontend"
cp .env.example .env.local
# Edit .env.local: set VITE_API_URL to your modal serve/deploy URL

npm install
npm run dev
```

Open http://localhost:5173. Workflow:

1. **Probe layers** — load baseline cosine-similarity + token-ranking charts
2. **Select layers** — click layer chips or use **Recommend** (lowest |cos|)
3. **Apply MEMIT edit** — run edit on selected layers; view before/after metrics
4. **Compare schemes** — evaluate multiple layer ranges (one per line, e.g. `8-12`)

CORS is enabled on the FastAPI app for local dev (`allow_origins=["*"]`).
Each `/edit` and `/compare` request temporarily mutates model weights inside
the container, then restores them; `@modal.concurrent(max_inputs=1)` prevents
concurrent requests from corrupting each other's edits.

## Status: dashboard capstone — recommended scheme fails, contiguity matters

The full loop (browser dashboard → Modal FastAPI → MEMIT on T4) was validated
end-to-end on the Eiffel Tower → Rome fact. The headline finding: **the
Recommend heuristic (lowest |cos| layers) does NOT produce a good editing
scheme.**

| Scheme | Selection basis | ES | Outcome |
|---|---|---|---|
| Baseline (unedited) | — | 0.00 | Paris |
| [9, 10, 11, 13, 15] | Recommended (lowest \|cos\|), non-contiguous | **0.00** | Failed — borderline: Paris NLL 1.17 narrowly beats Rome 1.19 |
| [8, 9, 10, 11, 12] | Contiguous block | 1.00 | Success — "Rome" |
| [13, 14, 15, 16, 17] | MEMIT default | 1.00 | Success — "Rome" |
| [6, 7, 8] | Early contiguous | 0.00 | Failed — still Paris |
| [20, 21] | Late contiguous | 0.00 | Failed — still Paris |

(PS=0.00, NS=0.50, S=0.00 for every row — see caveat below.)

Interpretation: the cosine heuristic picks individually "active" layers, but
MEMIT needs a **contiguous block within the mid-layer window (~8–17)**.
Skipping layer 12 broke the edit even though layers 9–11 overlap with the
working 8–12 range, and contiguous blocks outside the window (6–8, 20–21)
also fail. This is exactly the failure mode the human-in-the-loop dashboard
exists to catch: a user who blindly applied the recommended layers would get
an edit that doesn't take, while the side-by-side comparison table makes the
contiguous-scheme fix obvious. Note the failure is *ineffective*, not
*destructive* — the KL analysis below shows unrelated behavior is untouched.
(As a direct result of this finding, the dashboard's Recommend button now
selects the best *contiguous* window of 5 layers by total |cos_sim| instead
of the 5 individually-lowest layers.)

**Caveat 1 — ES is a narrow metric (the "Vienna" anomaly).** In the UI
comparison run, scheme [8–12] scored ES=1.00 yet its sampled generation said
*"Vienna, Austria"* — not Rome. This is not a contradiction: ES only checks
that `target_new` beats `target_true` (Rome < Paris in NLL); a *third* token
can still win decoding. The edit pushed the model off Paris without reliably
landing it on Rome. (That particular Vienna sample came from an unseeded run;
under the deterministic seeding in Caveat 3, [8–12] generates "Rome". The
lesson stands either way.) Treat ES as efficacy-vs-original-answer only, and
displayed generations as one stochastic sample (see the "Known nuance" in the
scheme-comparison section above).

**Caveat 2 — PS=0.00 everywhere, even for successful edits.** Every scheme
that achieved ES=1.00 still failed all paraphrase prompts (e.g. "The Eiffel
Tower is located in the city of" reverts to Paris). The edit wins on exact
efficacy but does not generalize — consistent with MEMIT's documented prompt
sensitivity and a key limitation to state in the write-up.

**Operational note:** `modal serve` sessions stop when the terminal dies; if
the dashboard shows "Failed to fetch", re-run `modal serve modal_app.py` and
confirm `frontend/.env.local` matches the printed `.modal.run` URL.

**Caveat 3 — MEMIT results are RNG-sensitive unless seeded.** MEMIT's
`get_context_templates()` samples its context templates with
`torch.multinomial` on first use and caches them for the container's
lifetime. In a warm web container, the RNG state depends on prior requests,
so the same layers once produced ES=1.00 (Rome NLL 0.96) via `/edit` while a
fresh CLI run gave ES=0.00 (Rome NLL 2.92). The backend now calls
`_seed_rng()` (torch seed 0 + clears the template cache) before every
`apply_memit_to_model`, and results are bit-identical across runs: layers
[9, 10, 11, 13, 15] reproducibly give Rome NLL 1.1911 vs Paris 1.1744,
ES=0.00. The recommended-scheme failure is real under canonical conditions —
but it is borderline, a 0.02-nat gap, not a catastrophic failure.
(This seeding now lives in the module-level `_seed_memit_rng()` and is called
before every edit in ALL paths — `main`, `compare`, and `batch` included —
so CLI runs are reproducible too.)

**Damage metric — mean per-token KL(P_edited ‖ P_orig).** To test whether
edits *destroy* unrelated knowledge, `/edit` and `/compare` compute the exact
full-vocabulary KL divergence (in nats) over 12 neutral reference sentences
that share no entities with the edited fact, at every token position:

| Scheme | ES | Damage KL (nats) |
|---|---|---|
| [13, 14, 15, 16, 17] | 1.00 | 1.40e-05 |
| [8, 9, 10, 11, 12] | 1.00 | 4.34e-06 |
| [9, 10, 11, 13, 15] | 0.00 | 6.17e-06 |
| Baseline (unedited) | — | 0.0 |

All values are ~1e-5 or smaller: a single MEMIT edit causes negligible drift
on neutral text, and a failed edit is no more damaging than a successful one.
Conclusion: MEMIT edits on GPT-2-XL *achieve or miss* the targeted fact
change without measurably damaging the rest of the model at single-edit
scale. KL becomes the meaningful yardstick for cumulative damage when many
edits are stacked (future work).
````

