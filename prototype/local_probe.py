"""
Standalone (MEMIT-free) prototype of KEditVis's two core layer-selection
signals, runnable on a small laptop GPU (e.g. RTX 3050 4GB) or even CPU.

This script does NOT edit the model. It only extracts and visualizes:

  1. Cosine similarity per MLP layer between the layer's input and output
     hidden states, for the subject-token position of a fact prompt.
     (KEditVis Sec 4.2.1 "Cosine Similarity Approach")

  2. Top-k token projections per layer ("logit lens"): projecting each
     layer's hidden state through the final LayerNorm + LM head to see
     which tokens the model "thinks" are likely at that depth.
     (KEditVis Sec 4.2.2 "Token Projection Approach")

Use this to build and debug your hooking logic cheaply before spending
Modal GPU credit on real MEMIT edits (see modal_app.py).

Usage:
    python local_probe.py --model gpt2-medium --subject "Eiffel Tower" \
        --prompt "{} is located in the city of" --target "Rome"
"""

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_gpt2_layer_names(model):
    """Returns dotted module names for each transformer block's MLP."""
    n_layers = model.config.n_layer
    return [f"transformer.h.{i}.mlp" for i in range(n_layers)]


def find_subject_token_index(tok, prompt_filled: str, subject: str) -> int:
    """
    Finds the index of the LAST token of `subject` within the tokenized
    `prompt_filled`. Mirrors MEMIT's `fact_token="subject_last"` strategy.
    """
    char_start = prompt_filled.rindex(subject)
    char_end = char_start + len(subject)
    enc = tok(prompt_filled, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    last_tok_idx = None
    for i, (s, e) in enumerate(offsets):
        if s < char_end and e > char_start:
            last_tok_idx = i
    if last_tok_idx is None:
        raise ValueError(
            f"Could not locate subject {subject!r} in prompt {prompt_filled!r}"
        )
    return last_tok_idx


@torch.no_grad()
def probe_layers(model, tok, prompt_filled: str, subject: str, top_k: int = 5):
    """
    Runs a single forward pass and, for every MLP layer:
      - records the cosine similarity between the MLP block's input and
        output hidden states at the subject token position
      - records the top-k vocabulary tokens when that layer's *residual
        stream output* is projected through ln_f + lm_head (logit lens),
        at BOTH the subject position (Sec 4.2.1 editing cue) and the last
        token position (Sec 4.2.2 next-token prediction -- where the fact's
        object surfaces).

    Returns a dict ready to json.dump / feed to a frontend chart.
    """
    device = next(model.parameters()).device
    subj_idx = find_subject_token_index(tok, prompt_filled, subject)

    layer_names = get_gpt2_layer_names(model)
    captured = {}

    def make_hook(name):
        def hook(module, inputs, output):
            # inputs[0]: hidden state fed into the MLP block
            # output:    hidden state produced by the MLP block
            captured[name] = {
                "input": inputs[0].detach(),
                "output": output.detach(),
            }
        return hook

    handles = [
        dict(model.named_modules())[name].register_forward_hook(make_hook(name))
        for name in layer_names
    ]

    # We also want the *residual stream* (block output, not just MLP output)
    # at each layer for the logit-lens view. Easiest: hook the whole block.
    block_names = [f"transformer.h.{i}" for i in range(model.config.n_layer)]
    residuals = {}

    def make_block_hook(name):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            residuals[name] = hs.detach()
        return hook

    block_handles = [
        dict(model.named_modules())[name].register_forward_hook(make_block_hook(name))
        for name in block_names
    ]

    enc = tok(prompt_filled, return_tensors="pt").to(device)
    last_idx = enc["input_ids"].shape[1] - 1
    model(**enc)

    for h in handles:
        h.remove()
    for h in block_handles:
        h.remove()

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.transformer.wte

    results = []
    for i, (mlp_name, block_name) in enumerate(zip(layer_names, block_names)):
        mlp_in = captured[mlp_name]["input"][0, subj_idx, :]
        mlp_out = captured[mlp_name]["output"][0, subj_idx, :]
        cos_sim = F.cosine_similarity(mlp_in.unsqueeze(0), mlp_out.unsqueeze(0)).item()

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
            resid_delta_var = resid_var

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

    return {
        "prompt": prompt_filled,
        "subject": subject,
        "subject_token_index": subj_idx,
        "layers": results,
    }


def print_ascii_chart(data: dict, bar_width: int = 40):
    """Quick-and-dirty terminal visualization, no frontend needed yet."""
    print(f"\nPrompt: {data['prompt']!r}  (subject token idx={data['subject_token_index']})\n")
    print(f"{'L':>3} | {'cos_sim':>8} | bar (longer = lower cos_sim = more 'active')")
    print("-" * 70)
    for layer in data["layers"]:
        cs = layer["cosine_similarity"]
        # KEditVis-style mapping: longer bar for cos_sim near 0.
        length = int(bar_width * (1 - abs(cs)))
        bar = "#" * max(length, 0)
        top1 = layer["top_tokens"][0]
        last1 = layer.get("last_top_tokens", [top1])[0]
        print(
            f"{layer['layer']:>3} | {cs:>8.4f} | {bar:<{bar_width}} "
            f"subj1={top1['token']!r} ({top1['prob']:.2f})  "
            f"last1={last1['token']!r} ({last1['prob']:.2f})"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium",
                     help="e.g. gpt2, gpt2-medium, gpt2-large (gpt2-xl needs ~6GB+ VRAM)")
    ap.add_argument("--prompt", default="{} is located in the city of",
                     help="Prompt template with '{}' where the subject goes")
    ap.add_argument("--subject", default="Eiffel Tower")
    ap.add_argument("--target", default=None,
                     help="(Informational only, not used for editing in this script)")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out", default=None, help="Optional path to dump JSON results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    prompt_filled = args.prompt.format(args.subject)
    data = probe_layers(model, tok, prompt_filled, args.subject, top_k=args.top_k)

    if args.target:
        data["target_new"] = args.target

    print_ascii_chart(data)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved full data to {args.out}")


if __name__ == "__main__":
    main()
