"""Context-robust MEMIT adaptation inspired by EasyEdit CORE.

The upstream MEMIT weight solve and context generator are reused. This module
adds context consistency to latent optimization; evaluation prompts never enter
the optimizer. See OPTIMIZATION_NOTES.md for source revisions and evaluation.
"""
import torch

OPTIMIZATION_PROFILES = {
    "standard": {
        "revision": "standard",
        "consistency_weight": 0.0,
        "minimum_clamp_norm_factor": None,
        "minimum_gradient_steps": None,
        "monitor_layers": 0,
        "key_fit": "averaged",
    },
    "standard_budget": {
        "revision": "standard-budget",
        "consistency_weight": 0.0,
        "minimum_clamp_norm_factor": 1.5,
        "minimum_gradient_steps": 40,
        "monitor_layers": 0,
        "key_fit": "averaged",
    },
    "context_no_consistency": {
        "revision": "context-no-consistency",
        "consistency_weight": 0.0,
        "minimum_clamp_norm_factor": 1.5,
        "minimum_gradient_steps": 40,
        "monitor_layers": 3,
        "key_fit": "individual_contexts",
    },
    "context": {
        "revision": "context-v3",
        "consistency_weight": 0.01,
        "minimum_clamp_norm_factor": 1.5,
        "minimum_gradient_steps": 40,
        "monitor_layers": 3,
        "key_fit": "individual_contexts",
    },
    "context_v3": {
        "revision": "context-v3",
        "consistency_weight": 0.01,
        "minimum_clamp_norm_factor": 1.5,
        "minimum_gradient_steps": 40,
        "monitor_layers": 3,
        "key_fit": "individual_contexts",
    },
}

CONTEXT_PROFILE = OPTIMIZATION_PROFILES["context_v3"]


def configure_context(hp, profile: str = "context_v3"):
    prof = OPTIMIZATION_PROFILES.get(profile, OPTIMIZATION_PROFILES["context_v3"])
    hp.context_consistency = prof["consistency_weight"]
    if prof["minimum_clamp_norm_factor"] is not None:
        hp.clamp_norm_factor = max(hp.clamp_norm_factor, prof["minimum_clamp_norm_factor"])
    if prof["minimum_gradient_steps"] is not None:
        hp.v_num_grad_steps = max(hp.v_num_grad_steps, prof["minimum_gradient_steps"])


def context_variance(hidden):
    """Mean squared distance to the context mean, normalized by hidden width."""
    if hidden.ndim != 3 or not all(hidden.shape):
        raise ValueError("Expected nonempty [layers, contexts, hidden] states.")
    centered = hidden.float() - hidden.float().mean(dim=1, keepdim=True)
    return centered.square().mean()


def project_logits(hidden, head):
    # GPT-2's upstream hyperparameters name its tied embedding, GPT-J a Linear.
    return torch.nn.functional.linear(hidden, head.weight, getattr(head, "bias", None))


def context_weights(contexts, device):
    if not contexts or any(not group for group in contexts):
        raise ValueError("Context groups must be nonempty.")
    return torch.tensor([1 / (len(contexts) * len(group))
                         for group in contexts for _ in group], device=device).sqrt()


def compute_context_target(model, tok, request, hp, layer, contexts):
    from memit.compute_z import find_fact_lookup_idx
    from util import nethook

    device = next(model.parameters()).device
    target = " " + request["target_new"]["str"].strip()
    target_ids = tok.encode(target, add_special_tokens=False, return_tensors="pt")[0].to(device)
    if not target_ids.numel():
        raise ValueError("Target has no tokens.")
    prompts = [context.format(request["prompt"]) + tok.decode(target_ids[:-1])
               for group in contexts for context in group]
    all_prompts = prompts + ["{} is a"]
    encoded = tok([p.format(request["subject"]) for p in all_prompts], padding=True, return_tensors="pt").to(device)
    labels = torch.full((len(prompts), encoded.input_ids.shape[1]), -100, dtype=torch.long, device=device)
    for i in range(len(prompts)):
        end = int(encoded.attention_mask[i].sum())
        labels[i, end - len(target_ids):end] = target_ids
    indices = [find_fact_lookup_idx(p, request["subject"], tok, hp.fact_token, verbose=False) for p in all_prompts]
    loss_layer = max(layer, hp.v_loss_layer)
    consistency_wt = getattr(hp, "context_consistency", CONTEXT_PROFILE["consistency_weight"])
    monitor_count = CONTEXT_PROFILE["monitor_layers"] if consistency_wt > 0.0 else 0
    monitors = list(range(layer + 1, min(model.config.n_layer, layer + 1 + monitor_count))) if monitor_count > 0 else []
    traced = list(dict.fromkeys([layer, loss_layer, *monitors]))
    delta = torch.zeros(model.config.n_embd, device=device, requires_grad=True)
    initial = None
    context_initial = None
    baseline_kl = None
    head = nethook.get_module(model, hp.lm_head_module)
    norm = nethook.get_module(model, hp.ln_f_module)

    def intervene(output, name):
        nonlocal initial, context_initial
        if name != hp.layer_module_tmp.format(layer):
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if initial is None:
            initial = hidden[0, indices[0]].detach().clone()
            context_initial = hidden[torch.arange(len(prompts), device=device), indices[:len(prompts)]].detach().clone()
        adjusted = hidden.clone()
        adjusted[torch.arange(len(indices), device=device), indices] += delta
        return (adjusted, *output[1:]) if isinstance(output, tuple) else adjusted

    flags = [p.requires_grad for p in model.parameters()]
    optimizer = torch.optim.Adam([delta], lr=hp.v_lr)
    try:
        model.requires_grad_(False)
        for step in range(hp.v_num_grad_steps):
            optimizer.zero_grad()
            with nethook.TraceDict(model, [hp.layer_module_tmp.format(i) for i in traced],
                                   retain_input=False, edit_output=intervene) as traces:
                logits = model(**encoded).logits
            def hidden_at(i):
                output = traces[hp.layer_module_tmp.format(i)].output
                return output[0] if isinstance(output, tuple) else output
            last = hidden_at(loss_layer)[:len(prompts)]
            log_probs = project_logits(norm(last), head).float().log_softmax(-1)
            selected = log_probs.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            nll = -(selected * (labels != -100)).sum(-1).div(len(target_ids)).mean()
            current_kl = logits[-1, indices[-1]].float().log_softmax(-1)
            if baseline_kl is None:
                baseline_kl = current_kl.detach().clone()
            kl = torch.nn.functional.kl_div(baseline_kl, current_kl, log_target=True, reduction="sum")
            decay = hp.v_weight_decay * delta.norm() / initial.norm().square().clamp_min(1e-12)
            consistency = delta.new_zeros(())
            if monitors and consistency_wt > 0.0:
                rows = torch.arange(len(prompts), device=device)
                consistency = context_variance(torch.stack([hidden_at(i)[rows, indices[:len(prompts)]] for i in monitors]))
            objective = nll + hp.kl_factor * kl + decay + consistency_wt * consistency
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite context MEMIT objective.")
            print(f"context step={step} nll={nll.item():.5f} variance={consistency.item():.5f}")
            if objective.item() < 0.05 or step == hp.v_num_grad_steps - 1:
                break
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                limit = hp.clamp_norm_factor * initial.norm()
                delta.mul_(torch.clamp(limit / delta.norm().clamp_min(1e-12), max=1))
        return (context_initial + delta).detach()
    finally:
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)


def apply_context_memit(model, tok, requests, hp, return_orig_weights=False):
    """Fit separate context keys and targets with the MEMIT covariance penalty."""
    from copy import deepcopy
    from memit.memit_main import get_context_templates, get_cov, upd_matrix_match_shape
    from memit.compute_z import get_module_input_output_at_words
    from util import nethook

    if return_orig_weights:
        raise ValueError("Use the application's rollback wrapper for original weights.")
    requests = deepcopy(requests)
    for request in requests:
        request["target_new"]["str"] = " " + request["target_new"]["str"].strip()
    contexts = get_context_templates(model, tok)
    final_layer = hp.layers[-1]
    targets = torch.cat([compute_context_target(model, tok, r, hp, final_layer, contexts) for r in requests]).T
    templates = [context.format(r["prompt"]) for r in requests for group in contexts for context in group]
    subjects = [r["subject"] for r in requests for group in contexts for _ in group]
    scales = context_weights(contexts, targets.device).repeat(len(requests))
    for i, layer in enumerate(hp.layers):
        with torch.no_grad():
            keys = get_module_input_output_at_words(model, tok, layer,
                context_templates=templates, words=subjects,
                module_template=hp.rewrite_module_tmp, fact_token_strategy=hp.fact_token)[0].T
            current = get_module_input_output_at_words(model, tok, final_layer,
                context_templates=templates, words=subjects,
                module_template=hp.layer_module_tmp, fact_token_strategy=hp.fact_token)[1].T
            residual = ((targets - current) * scales).double()
            covariance = get_cov(model, tok, hp.rewrite_module_tmp.format(layer), hp.mom2_dataset, hp.mom2_n_samples, hp.mom2_dtype)
            keys = (keys * scales).double()
            adjusted = torch.linalg.solve(hp.mom2_update_weight * covariance.double() + keys @ keys.T, keys)
            update = residual.div(len(hp.layers) - i) @ adjusted.T
            if not torch.isfinite(update).all():
                raise FloatingPointError("Non-finite context MEMIT update.")
            weight = nethook.get_parameter(model, hp.rewrite_module_tmp.format(layer) + ".weight")
            weight.add_(upd_matrix_match_shape(update, weight.shape).to(weight.dtype))
            del covariance, keys, adjusted, update, residual, current
        torch.cuda.empty_cache()
    return model, {}
