"""
Digs into why "Windows was developed by" -> "Apple" failed to edit under
EVERY layer scheme tested in audit/development/batch_comparison.json, while "Mario Kart was
developed by" -> "Apple" (same target word, different subject) succeeded
under all of them.

Since both facts share the same target_new ("Apple"), any systematic
difference in editability must trace back to how the SUBJECT is
represented in the model, not the target token. This script compares the
two facts' baseline (pre-edit) cosine-similarity and logit-lens profiles,
side by side, using data already collected by `modal run
modal_app.py::batch` -- no new GPU computation needed.

Usage:
    python error_analysis.py audit/development/batch_comparison.json
"""

import json
import sys


def summarize_fact(fact_result):
    signals = fact_result["baseline"]["layer_signals"]
    cos_sims = [abs(l["cosine_similarity"]) for l in signals]
    mean_cos = sum(cos_sims) / len(cos_sims)

    target_new = fact_result["fact"]["target_new"].strip().lower()
    layers_with_target_in_top5 = [
        l["layer"]
        for l in signals
        if any(t["token"].strip().lower() == target_new for t in l["top_tokens"])
    ]

    return {
        "mean_abs_cos_sim_all_layers": mean_cos,
        "min_abs_cos_sim": min(cos_sims),
        "max_abs_cos_sim": max(cos_sims),
        "layers_with_target_in_top5": layers_with_target_in_top5,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python error_analysis.py <audit/development/batch_comparison.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    by_prompt = {
        fr["fact"]["prompt"].format(fr["fact"]["subject"]): fr for fr in data["facts"]
    }

    print("=== Baseline (pre-edit) profile summary, all facts ===\n")
    summaries = {}
    for prompt_str, fr in by_prompt.items():
        s = summarize_fact(fr)
        summaries[prompt_str] = s
        print(f"{prompt_str!r} -> {fr['fact']['target_new']!r}")
        print(f"  mean|cos_sim| (all 48 layers): {s['mean_abs_cos_sim_all_layers']:.3f}")
        print(f"  min|cos_sim|:                  {s['min_abs_cos_sim']:.3f}")
        print(f"  max|cos_sim|:                  {s['max_abs_cos_sim']:.3f}")
        print(f"  layers where target_new appears in top-5 (pre-edit): "
              f"{s['layers_with_target_in_top5'] or 'NONE'}")
        print()

    print("\n=== Focused comparison: Windows->Apple (failed) vs Mario Kart->Apple (succeeded) ===")
    print("Both share the same target_new, so any difference traces to the SUBJECT.\n")

    windows_key = next((k for k in by_prompt if "Windows" in k), None)
    mario_key = next((k for k in by_prompt if "Mario Kart" in k), None)

    if windows_key and mario_key:
        w, m = summaries[windows_key], summaries[mario_key]
        print(f"{'Metric':<35} | {'Windows (failed)':>18} | {'Mario Kart (succeeded)':>24}")
        print("-" * 82)
        print(f"{'mean|cos_sim| (all layers)':<35} | {w['mean_abs_cos_sim_all_layers']:>18.3f} | {m['mean_abs_cos_sim_all_layers']:>24.3f}")
        print(f"{'min|cos_sim|':<35} | {w['min_abs_cos_sim']:>18.3f} | {m['min_abs_cos_sim']:>24.3f}")
        print(f"{'max|cos_sim|':<35} | {w['max_abs_cos_sim']:>18.3f} | {m['max_abs_cos_sim']:>24.3f}")
        print(f"{'# layers with target in top-5':<35} | {len(w['layers_with_target_in_top5']):>18} | {len(m['layers_with_target_in_top5']):>24}")

    print(
        "\nInterpretation: if Windows shows much higher cosine similarity across "
        "the board (values close to 1.0 mean the MLP block barely changes its "
        "input at all at the subject token position), that suggests the model "
        "simply isn't processing 'Windows' as a knowledge-bearing subject the "
        "same way it processes 'Mario Kart' at these layers -- there may be no "
        "good editing layer in the tested range for this fact, rather than the "
        "wrong layers having been chosen. This is a genuine limitation "
        "orthogonal to layer selection, worth calling out explicitly in a "
        "capstone discussion of when interactive layer selection alone isn't "
        "enough to fix a bad edit."
    )


if __name__ == "__main__":
    main()
