"""
Pooled version of analyze_schemes.py: tests KEditVis's two core layer-selection
hypotheses (cosine similarity, Sec 4.2.1, and token projection, Sec 4.2.2)
across MANY (fact, scheme) pairs at once, using the output of
`modal run modal_app.py::batch`. This gives far more statistical power
than analyzing a single fact's schemes in isolation (n = num_facts x
num_schemes instead of n = num_schemes).

Usage:
    python analyze_batch.py audit/development/batch_comparison.json
"""

import json
import sys

from analyze_schemes import (
    format_rho,
    scheme_activity_score,
    scheme_projection_score,
    spearman_rank_correlation,
    target_token_probs,
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_batch.py <audit/development/batch_comparison.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    pooled_rows = []
    for fact_result in data["facts"]:
        fact = fact_result["fact"]
        layer_signals = fact_result["baseline"]["layer_signals"]
        rewrite_prompt = fact["prompt"].format(fact["subject"])
        target_true = fact.get("target_true")
        # The Sec 4.2.2 signal tracks the fact's object token; prefer the
        # original (pre-edit) object, fall back to target_new if the original
        # never surfaces in any layer's ranking.
        proj_target = next(
            (
                t
                for t in (target_true, fact.get("target_new"))
                if t and target_token_probs(layer_signals, t)
            ),
            None,
        )

        print(f"\n=== {rewrite_prompt} -> {fact['target_new']} ===")
        header = (
            f"{'Layers':<20} | {'mean|cos|':>10} | {'min|cos|':>9} | "
            f"{'pkSpan':>8} | {'ovlp':>5} | "
            f"{'ES':>5} | {'PS':>5} | {'NS':>5} | {'S':>5}"
        )
        print(header)
        print("-" * len(header))

        for s in fact_result["schemes"]:
            try:
                activity = scheme_activity_score(layer_signals, s["layers"])
            except ValueError as exc:
                print(f"{str(s['layers']):<20} | skipped: {exc}")
                continue
            projection = (
                scheme_projection_score(layer_signals, s["layers"], proj_target)
                if proj_target
                else None
            )
            metrics = s["metrics"] or {}

            def fmt(x):
                return f"{x:.3f}" if x is not None else "n/a"

            span = str(projection["peak_span"]) if projection else "n/a"
            ovlp = fmt(projection["span_overlap"]) if projection else "n/a"
            print(
                f"{str(s['layers']):<20} | {fmt(activity['mean_abs_cos_sim']):>10} | "
                f"{fmt(activity['min_abs_cos_sim']):>9} | {span:>8} | {ovlp:>5} | "
                f"{fmt(metrics.get('ES')):>5} | "
                f"{fmt(metrics.get('PS')):>5} | {fmt(metrics.get('NS')):>5} | {fmt(metrics.get('S')):>5}"
            )

            pooled_rows.append(
                {
                    "fact": rewrite_prompt,
                    "layers": s["layers"],
                    "mean_abs_cos_sim": activity["mean_abs_cos_sim"],
                    "min_abs_cos_sim": activity["min_abs_cos_sim"],
                    "span_overlap": projection["span_overlap"] if projection else None,
                    "max_prob": projection["max_prob"] if projection else None,
                    "ES": metrics.get("ES"),
                    "PS": metrics.get("PS"),
                    "NS": metrics.get("NS"),
                    "S": metrics.get("S"),
                }
            )

    n_facts = len(data["facts"])
    n_total = len(pooled_rows)
    print(f"\n\n=== Pooled analysis across {n_facts} facts x schemes ({n_total} data points) ===")

    # Cosine similarity (Sec 4.2.1) expects a NEGATIVE correlation with edit
    # success; token projection (Sec 4.2.2) expects a POSITIVE one.
    for metric_name in ["ES", "PS", "NS", "S"]:
        for activity_name in [
            "mean_abs_cos_sim",
            "min_abs_cos_sim",
            "span_overlap",
            "max_prob",
        ]:
            pairs = [
                (r[activity_name], r[metric_name])
                for r in pooled_rows
                if r[metric_name] is not None and r[activity_name] is not None
            ]
            if len(pairs) < 3:
                continue
            xs, ys = zip(*pairs)
            rho = spearman_rank_correlation(list(xs), list(ys))
            expect = (
                "expect negative"
                if "cos" in activity_name
                else "expect positive"
            )
            print(
                f"Spearman({activity_name}, {metric_name}) = {format_rho(rho)}"
                f"  (n={len(pairs)}, {expect})"
            )

    # Also flag facts that failed to edit at ALL regardless of layer choice --
    # useful signal that some facts are just harder to edit than others,
    # independent of layer selection (a separate axis KEditVis doesn't
    # directly address, but worth noting for a capstone discussion).
    print("\n=== Facts where NO scheme achieved ES > 0 (edit failed regardless of layers) ===")
    by_fact = {}
    for r in pooled_rows:
        by_fact.setdefault(r["fact"], []).append(r["ES"])
    any_failed = False
    for fact, es_list in by_fact.items():
        es_list = [e for e in es_list if e is not None]
        if es_list and max(es_list) == 0:
            print(f"  {fact}")
            any_failed = True
    if not any_failed:
        print("  (none -- every fact succeeded with at least one scheme)")

    print(
        "\nNote: pooling across facts increases n but also mixes facts of "
        "very different difficulty. Read the per-fact tables above alongside "
        "the pooled correlations -- a pooled correlation can look weak even "
        "if the relationship holds cleanly within each individual fact, or "
        "vice versa."
    )


if __name__ == "__main__":
    main()
