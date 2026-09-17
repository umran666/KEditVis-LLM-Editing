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
    .add_local_file("editing_optimizations.py", "/root/editing_optimizations.py")
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


def _compute_weight_drift(model, orig_weights):
    """Computes absolute and relative Frobenius norm changes across edited parameter tensors.

    Formula:
      ||ΔW_l||_F = sqrt(sum_ij (W_l,ij^post - W_l,ij^orig)^2)
      Rel-Frob_l = ||ΔW_l||_F / max(||W_l^orig||_F, 1e-12)
      Total-Abs-Frob = sqrt(sum_l ||ΔW_l||_F^2)
      Total-Rel-Frob = Total-Abs-Frob / max(sqrt(sum_l ||W_l^orig||_F^2), 1e-12)
      Mean-Layer-Rel-Frob = (1 / |L|) * sum_l Rel-Frob_l
    """
    import math
    import torch
    from util import nethook

    per_layer = {}
    sum_delta_sq = 0.0
    sum_orig_sq = 0.0

    with torch.no_grad():
        for param_name, orig_w in orig_weights.items():
            curr_w = nethook.get_parameter(model, param_name).detach().cpu()
            delta = (curr_w - orig_w).float()
            delta_frob = float(torch.linalg.norm(delta, ord="fro").item())
            orig_frob = float(torch.linalg.norm(orig_w.float(), ord="fro").item())
            rel_frob = delta_frob / max(orig_frob, 1e-12)

            layer_num = None
            for part in param_name.split("."):
                if part.isdigit():
                    layer_num = int(part)
                    break

            per_layer[layer_num if layer_num is not None else param_name] = {
                "param_name": param_name,
                "absolute_frobenius": delta_frob,
                "relative_frobenius": rel_frob,
                "orig_frobenius": orig_frob,
            }
            sum_delta_sq += delta_frob ** 2
            sum_orig_sq += orig_frob ** 2

    total_abs = math.sqrt(sum_delta_sq)
    total_orig = math.sqrt(sum_orig_sq)
    total_rel = total_abs / max(total_orig, 1e-12)
    mean_rel = sum(v["relative_frobenius"] for v in per_layer.values()) / max(len(per_layer), 1)

    return {
        "total_absolute_frobenius": total_abs,
        "total_relative_frobenius": total_rel,
        "mean_layer_relative_frobenius": mean_rel,
        "per_layer": per_layer,
    }


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

    Output is always deduped + sorted (see _normalize_scheme). Malformed specs
    raise a ValueError naming the actual problem rather than letting int()
    emit a bare conversion error.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("Layer spec must not be empty.")

    has_range, has_list = "-" in spec, "," in spec
    if has_range and has_list:
        raise ValueError(
            f"Invalid layer spec {spec!r}: cannot mix a range with a list. "
            "Use either a range like '8-12' or a list like '6,9,12'."
        )

    if has_range:
        parts = [part.strip() for part in spec.split("-")]
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(
                f"Invalid layer range {spec!r}: expected a single 'start-end' pair of integers."
            )
        start, end = (int(part) for part in parts)
        if start > end:
            raise ValueError(f"Invalid layer range {spec!r}: start > end")
        return list(range(start, end + 1))

    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"Invalid layer list {spec!r}: expected comma-separated non-negative integers."
        )
    return _normalize_scheme([int(part) for part in parts])


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
            resid_var = float(resid_subj.float().var(unbiased=True).item())

            logits_subj = lm_head(ln_f(resid_subj.unsqueeze(0))).squeeze(0)
            probs_subj = F.softmax(logits_subj, dim=-1)
            top_probs_s, top_ids_s = torch.topk(probs_subj, top_k)
            top_tokens = [
                {"token": tok.decode([tid]), "prob": p.item()}
                for tid, p in zip(top_ids_s.tolist(), top_probs_s)
            ]

            resid_last = residuals[block_name][0, last_idx, :]
            resid_var_last = float(resid_last.float().var(unbiased=True).item())

            if i > 0:
                prev_block = block_names[i - 1]
                delta_h = resid_subj - residuals[prev_block][0, subj_idx, :]
                resid_delta_var = float(delta_h.float().var(unbiased=True).item())
            else:
                # Layer 0 has no predecessor, so its residual delta is undefined
                # rather than equal to its absolute variance.
                resid_delta_var = None

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
                    "residual_variance": resid_var,
                    "residual_variance_last": resid_var_last,
                    "residual_delta_variance": resid_delta_var,
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
    """Harmonic mean of the supplied components, or None when it is undefined.

    Returns None for an empty input (nothing was measured) and whenever any
    component is None, so "not measured" never masquerades as a score.

    A single zero component makes the harmonic mean exactly 0. That is the
    correct limit, not a defect -- but it does mean the composite score S
    collapses to 0 for any edit that fails efficacy, paraphrase or locality
    completely, so S alone cannot rank partially-successful edits.
    """
    if not vals or any(v is None for v in vals):
        return None
    if any(v <= 0 for v in vals):
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
    neighborhood_targets=None,
):
    """
    Computes KEditVis-style ES / PS / NS / S metrics for the current model
    state against a single requested edit:

      ES (Efficacy Success):    does the exact rewrite prompt now produce
                                 target_new?
      PS (Paraphrase Success):  do paraphrased versions of the prompt also
                                 produce target_new (generalization)?
      NS (Neighborhood Success): do unrelated/nearby prompts still produce
                                 target_true (or per-example neighbor target),
                                 i.e. did the edit avoid corrupting nearby knowledge (locality)?
      S: harmonic mean of all three metrics; unavailable if any is missing.
      Also provides greedy token accuracy metrics (ES_greedy, PS_greedy, NS_greedy, S_greedy).
    """
    rewrite_prompt = prompt.format(subject)
    es_results = _eval_prefix_targets(model, tok, [rewrite_prompt], target_new, target_true)
    es_rate = sum(r["target_new_nll"] < r["target_true_nll"] for r in es_results) / len(es_results) if es_results else 0.0
    es_greedy = sum(bool(r["target_new_correct"]) for r in es_results) / len(es_results) if es_results else 0.0

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
    ps_greedy = (
        sum(bool(r["target_new_correct"]) for r in ps_results) / len(ps_results)
        if ps_results
        else None
    )

    ns_results = []
    if neighborhood_prompts:
        if neighborhood_targets and len(neighborhood_targets) == len(neighborhood_prompts):
            for n_prompt, n_target in zip(neighborhood_prompts, neighborhood_targets):
                res = _eval_prefix_targets(model, tok, [n_prompt], target_new, n_target)
                ns_results.extend(res)
        else:
            ns_results = _eval_prefix_targets(model, tok, neighborhood_prompts, target_new, target_true)

    ns_rate = (
        sum(r["target_true_nll"] < r["target_new_nll"] for r in ns_results) / len(ns_results)
        if ns_results
        else None
    )
    ns_greedy = (
        sum(bool(r["target_true_correct"]) for r in ns_results) / len(ns_results)
        if ns_results
        else None
    )

    s = _harmonic_mean([es_rate, ps_rate, ns_rate])
    s_greedy = _harmonic_mean([es_greedy, ps_greedy, ns_greedy]) if all(x is not None for x in [es_greedy, ps_greedy, ns_greedy]) else None

    return {
        "ES": es_rate,
        "PS": ps_rate,
        "NS": ns_rate,
        "S": s,
        "ES_greedy": es_greedy,
        "PS_greedy": ps_greedy,
        "NS_greedy": ns_greedy,
        "S_greedy": s_greedy,
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


# Below this many points a joint t-SNE is not a meaningful embedding: sklearn
# would run with perplexity ~1 and produce an effectively arbitrary layout, which
# the UI would still label as a real projection. The deterministic PCA path is
# used instead so the plot is honest about being a small-sample fallback.
MIN_TSNE_SAMPLES = 6


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
    if len(values) < MIN_TSNE_SAMPLES or np.allclose(values, values[0]):
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
    if len(lp_edited) != len(lp_orig):
        raise ValueError("edited/orig logprob lists must align")
    total, count = 0.0, 0
    for e, o in zip(lp_edited, lp_orig):
        if e.shape != o.shape:
            raise ValueError("edited/orig logprob shapes must align")
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
    (optional list), neighborhood_prompts (optional list). See data/facts.json
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

                weight_drift = _compute_weight_drift(edited_model, orig_weights)
                scheme_results.append({"layers": scheme, "metrics": metrics, "weight_drift": weight_drift})
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


def _write_json(path: str, payload) -> str:
    """Writes JSON, creating the parent directory if the evidence tree is absent."""
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


@app.local_entrypoint()
def batch(
    facts_file: str = "data/facts.json",
    schemes: str = "13-17|8-12|6-8|20-21",
    model_name: str = "gpt2-xl",
    out: str = "audit/development/batch_comparison.json",
):
    """
    Sweeps every fact in `facts_file` (see data/facts.json for the format)
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

    _write_json(out, result)
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
    out: str = "audit/development/memit_result.json",
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

    _write_json(out, result)

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
    out: str = "audit/development/scheme_comparison.json",
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

    _write_json(out, result)
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
    from editing_optimizations import OPTIMIZATION_PROFILES
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
        optimization: Literal["standard", "context", "context_v3", "standard_budget", "context_no_consistency"] = "standard"
        layers: list[StrictInt] | None = Field(default=None, min_length=1)
        neighborhood_targets: list[str] | None = None

    class CompareRequest(FactRequest):
        optimization: Literal["standard", "context", "context_v3", "standard_budget", "context_no_consistency"] = "standard"
        schemes: list[list[StrictInt]] = Field(min_length=1)
        neighborhood_targets: list[str] | None = None

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

    def _apply_edit(method, model, tok, request_, layers, model_name, optimization="standard"):
        """Dispatch to MEMIT or ROME based on method string."""
        if method not in SUPPORTED_METHODS:
            raise HTTPException(status_code=400, detail="Unsupported editing method.")
        if method == "rome":
            if optimization != "standard":
                raise HTTPException(status_code=400, detail="Context optimization is available for MEMIT only.")
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
            if optimization in ("context", "context_v3"):
                from editing_optimizations import apply_context_memit, configure_context
                configure_context(hparams, "context_v3")
                return _apply_with_rollback(apply_context_memit, model, tok, request_, hparams)
            elif optimization == "context_no_consistency":
                from editing_optimizations import apply_context_memit, configure_context
                configure_context(hparams, "context_no_consistency")
                return _apply_with_rollback(apply_context_memit, model, tok, request_, hparams)
            elif optimization == "standard_budget":
                from editing_optimizations import configure_context
                configure_context(hparams, "standard_budget")
                return _apply_with_rollback(apply_memit_to_model, model, tok, request_, hparams)
            return _apply_with_rollback(
                apply_memit_to_model, model, tok, request_, hparams
            )

    def optimization_config(profile):
        # "standard" is upstream MEMIT, so report the pinned commit it runs
        # against rather than the profile-table label. This branch must be
        # checked first: "standard" is itself a key in OPTIMIZATION_PROFILES,
        # so a dict lookup first would make it unreachable.
        if profile == "standard":
            return {"revision": MEMIT_COMMIT}
        if profile in OPTIMIZATION_PROFILES:
            return dict(OPTIMIZATION_PROFILES[profile])
        return {"revision": str(profile)}

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
            # Derived from the profile table so this list can never drift from
            # the optimization values the request models actually accept.
            "memit_optimizations": list(OPTIMIZATION_PROFILES),
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
        # Pre-edit metrics must use the same per-example neighborhood targets as
        # the post-edit ones, otherwise pre/post NS are computed against
        # different reference strings and cannot be compared.
        pre_metrics = (
            _evaluate_edit(
                model, tok, body.prompt, body.subject, body.target_new,
                body.target_true, body.paraphrase_prompts, body.neighborhood_prompts,
                neighborhood_targets=body.neighborhood_targets,
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
            body.method, model, tok, request_, layers, model_name, body.optimization
        )

        try:
            post_signals = _probe_layers(edited_model, tok, rewrite_prompt, body.subject)
            post_text = generate_fast(edited_model, tok, generation_prompts, max_out_len=80)
            post_metrics = (
                _evaluate_edit(
                    edited_model, tok, body.prompt, body.subject, body.target_new,
                    body.target_true, body.paraphrase_prompts, body.neighborhood_prompts,
                    neighborhood_targets=body.neighborhood_targets,
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
            weight_drift = _compute_weight_drift(edited_model, orig_weights)
        finally:
            _restore_weights(edited_model, orig_weights)

        return {
            "method": body.method,
            "edited_layers": layers,
            "optimization": body.optimization,
            "optimization_config": optimization_config(body.optimization),
            "neighborhood": neighborhood,
            "weight_drift": weight_drift,
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
        # Same neighborhood-target convention as the per-scheme metrics below.
        baseline_metrics = (
            _evaluate_edit(
                model, tok, body.prompt, body.subject, body.target_new,
                body.target_true, body.paraphrase_prompts, body.neighborhood_prompts,
                neighborhood_targets=body.neighborhood_targets,
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
                body.method, model, tok, request_, scheme, model_name, body.optimization
            )

            try:
                metrics = (
                    _evaluate_edit(
                        edited_model, tok, body.prompt, body.subject, body.target_new,
                        body.target_true, body.paraphrase_prompts, body.neighborhood_prompts,
                        neighborhood_targets=body.neighborhood_targets,
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

                weight_drift = _compute_weight_drift(edited_model, orig_weights)
                results.append({
                    "layers": scheme,
                    "layer_signals": _probe_layers(edited_model, tok, rewrite_prompt, body.subject),
                    "metrics": metrics,
                    "weight_drift": weight_drift,
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
            "optimization": body.optimization,
            "optimization_config": optimization_config(body.optimization),
        }

    return web
