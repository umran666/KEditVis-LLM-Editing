"""
Tests KEditVis's two core layer-selection hypotheses against your own data:

  1. Cosine similarity (Sec 4.2.1): do layers with LOW cosine similarity in
     the unedited model (the paper's "activity" signal) correspond to layer
     schemes that achieve HIGH edit success (ES/S) after a real MEMIT edit?
  2. Token projection (Sec 4.2.2): the paper's rule is to select "the range
     between the two layers where target tokens show the highest
     probabilities" in the logit-lens ranking. Do schemes that overlap that
     target-token peak span achieve higher ES/S?

This is a pure local analysis -- no GPU or Modal needed. It consumes the
JSON produced by `modal run modal_app.py::compare`, which already contains
both:
  - `baseline.layer_signals`: per-layer cosine similarity + top-5 logit-lens
    tokens on the unedited model (from _probe_layers)
  - `schemes[i].metrics`: real post-edit ES/PS/NS/S for each layer scheme

Usage:
    python analyze_schemes.py audit/development/scheme_comparison.json [target_true]

`target_true` (e.g. "Paris") is optional but required for hypothesis 2 --
the token-projection analysis tracks the fact's ORIGINAL object token,
since pre-edit that is what the model actually represents.
"""

import json
import math
import sys


def _norm_token(s: str) -> str:
    return s.strip().strip(",.;:!?'\"").lower()


def target_token_probs(layer_signals: list, target: str) -> dict:
    """
    Logit-lens probability of `target` at every layer where it appears in
    that layer's top-5 ranking, pre-edit (KEditVis Sec 4.2.2's chart data).
    Layers where the token is absent from the top-5 get no entry.

    Prefers the last-token view (`last_top_tokens`: what the model predicts
    next after the prompt), which is where the fact's object surfaces;
    falls back to the subject-position view for JSONs produced before that
    field existed.
    """
    t = _norm_token(target)
    probs = {}
    for l in layer_signals:
        for tt in (l.get("last_top_tokens") or l.get("top_tokens") or []):
            if _norm_token(tt["token"]) == t:
                probs[l["layer"]] = tt["prob"]
    return probs


def scheme_projection_score(layer_signals: list, scheme_layers: list, target: str):
    """
    Scores how well a candidate scheme matches KEditVis Sec 4.2.2's
    token-projection selection rule: the span between the two layers where
    the fact's object token shows its highest logit-lens probabilities is
    "considered to be processing target knowledge and is typically selected."
    The signal comes from the full-fact-statement projection (`fact_top_tokens`)
    when available -- without the object in the input text it never surfaces.

    Returns None when the object token never enters any layer's top-5 (no
    signal available), otherwise:
      peak_span    -- [lo, hi], the paper's recommended editing range
      span_overlap -- fraction of peak-span layers the scheme contains (0-1)
      max_prob     -- highest logit-lens prob of the object inside the scheme
    """
    probs = target_token_probs(layer_signals, target)
    if not probs:
        return None
    peak = sorted(probs, key=lambda l: -probs[l])[:2]
    lo, hi = min(peak), max(peak)
    span_layers = range(lo, hi + 1)
    scheme_probs = [probs[l] for l in scheme_layers if l in probs]
    return {
        "peak_span": [lo, hi],
        "span_overlap": sum(1 for l in scheme_layers if l in span_layers)
        / len(list(span_layers)),
        "max_prob": max(scheme_probs) if scheme_probs else 0.0,
    }


def scheme_activity_score(layer_signals: list, scheme_layers: list) -> dict:
    """
    Summarizes how "active" (per KEditVis's cosine-similarity heuristic) a
    candidate scheme's layers were in the UNEDITED model, before any edit
    was applied. Lower cosine similarity == more active == a better
    candidate editing layer, per the paper's Sec 4.2.1.

    Returns both the mean and min |cosine_similarity| across the scheme's
    layers, since the paper's "Recommend" feature specifically looks at the
    lowest-similarity layers within a range.

    Layers are looked up by their recorded `layer` field rather than by list
    position, so a reordered or partial signal list cannot silently attribute
    the wrong cosine similarity to a layer.
    """
    by_layer = {l["layer"]: l for l in layer_signals}
    cos_sims = [
        abs(by_layer[l]["cosine_similarity"])
        for l in scheme_layers
        if l in by_layer
    ]
    if not cos_sims:
        known = sorted(by_layer)
        span = f"{known[0]}-{known[-1]}" if known else "none recorded"
        raise ValueError(
            f"Scheme {sorted(set(scheme_layers))} contains no layer present in the "
            f"baseline signals (recorded layers: {span})."
        )
    return {
        "mean_abs_cos_sim": sum(cos_sims) / len(cos_sims),
        "min_abs_cos_sim": min(cos_sims),
    }


def _average_ranks(values: list) -> list:
    """1-based ranks, with tied values sharing the mean of the positions they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(xs: list, ys: list) -> float:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    sxx = sum(d * d for d in dx)
    syy = sum(d * d for d in dy)
    if sxx <= 0.0 or syy <= 0.0:
        # Zero variance in either input: the coefficient is undefined.
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(sxx * syy)


def spearman_rank_correlation(xs: list, ys: list) -> float:
    """
    Spearman rank correlation, correct in the presence of ties.

    Ties are assigned average ranks and the coefficient is the Pearson
    correlation of those rank vectors. The familiar shortcut
    `1 - 6*sum(d^2)/(n*(n^2-1))` is only algebraically equivalent when there
    are NO ties. This project's metrics are heavily tied (ES/PS/S take very
    few distinct values), and the shortcut understates |rho| substantially --
    on the committed batch data it reported -0.252 where the true value is
    -0.549.

    Returns NaN when either input has zero variance or when there are fewer
    than two paired observations, rather than inventing a coefficient.

    With very few data points (as in a handful of layer schemes) this should
    be read as directional/exploratory evidence, not a statistically
    significant result.
    """
    if len(xs) != len(ys):
        raise ValueError(
            f"Rank correlation requires equal-length inputs (got {len(xs)} and {len(ys)})."
        )
    if len(xs) < 2:
        return float("nan")
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def format_rho(rho: float) -> str:
    """Human-readable coefficient, distinguishing 'undefined' from a real value."""
    return "undefined (zero variance in one input)" if math.isnan(rho) else f"{rho:+.3f}"


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python analyze_schemes.py <audit/development/scheme_comparison.json> [target_true]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    target_true = sys.argv[2] if len(sys.argv) == 3 else None

    layer_signals = data["baseline"]["layer_signals"]
    schemes = data["schemes"]

    rows = []
    skipped = []
    for s in schemes:
        try:
            activity = scheme_activity_score(layer_signals, s["layers"])
        except ValueError as exc:
            skipped.append(str(exc))
            continue
        metrics = s["metrics"] or {}
        projection = (
            scheme_projection_score(layer_signals, s["layers"], target_true)
            if target_true
            else None
        )
        rows.append(
            {
                "layers": s["layers"],
                "mean_abs_cos_sim": activity["mean_abs_cos_sim"],
                "min_abs_cos_sim": activity["min_abs_cos_sim"],
                "peak_span": projection["peak_span"] if projection else None,
                "span_overlap": projection["span_overlap"] if projection else None,
                "max_prob": projection["max_prob"] if projection else None,
                "ES": metrics.get("ES"),
                "PS": metrics.get("PS"),
                "NS": metrics.get("NS"),
                "S": metrics.get("S"),
            }
        )

    print(f"Fact: {data['request']}\n")
    header = (
        f"{'Layers':<20} | {'mean|cos|':>10} | {'min|cos|':>9} | "
        f"{'pkSpan':>8} | {'ovlp':>5} | {'maxP':>6} | "
        f"{'ES':>5} | {'PS':>5} | {'S':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        def f(x):
            return f"{x:.3f}" if x is not None else "n/a"

        span = str(r["peak_span"]) if r["peak_span"] else "n/a"
        print(
            f"{str(r['layers']):<20} | {f(r['mean_abs_cos_sim']):>10} | "
            f"{f(r['min_abs_cos_sim']):>9} | {span:>8} | {f(r['span_overlap']):>5} | "
            f"{f(r['max_prob']):>6} | {f(r['ES']):>5} | {f(r['PS']):>5} | {f(r['S']):>5}"
        )

    # Hypothesis 1 (KEditVis Sec 4.2.1): LOWER cosine similarity -> BETTER edit
    # outcome, so we expect a NEGATIVE correlation between mean/min |cos_sim| and ES/S.
    valid_es = [(r["mean_abs_cos_sim"], r["ES"]) for r in rows if r["ES"] is not None]
    valid_s = [(r["mean_abs_cos_sim"], r["S"]) for r in rows if r["S"] is not None]
    valid_min_es = [(r["min_abs_cos_sim"], r["ES"]) for r in rows if r["ES"] is not None]

    # Hypothesis 2 (KEditVis Sec 4.2.2): schemes overlapping the object token's
    # peak-probability span do better, so we expect POSITIVE correlations.
    valid_ovlp_es = [
        (r["span_overlap"], r["ES"])
        for r in rows
        if r["ES"] is not None and r["span_overlap"] is not None
    ]
    valid_ovlp_s = [
        (r["span_overlap"], r["S"])
        for r in rows
        if r["S"] is not None and r["span_overlap"] is not None
    ]
    valid_maxp_es = [
        (r["max_prob"], r["ES"])
        for r in rows
        if r["ES"] is not None and r["max_prob"] is not None
    ]

    print(f"\nn = {len(rows)} schemes compared (small-sample, exploratory only)")
    for note in skipped:
        print(f"skipped: {note}")

    if len(valid_es) >= 2:
        xs, ys = zip(*valid_es)
        rho = spearman_rank_correlation(list(xs), list(ys))
        print(f"Spearman(mean|cos_sim|, ES)     = {format_rho(rho)}  (expect negative if hypothesis holds)")

    if len(valid_min_es) >= 2:
        xs, ys = zip(*valid_min_es)
        rho = spearman_rank_correlation(list(xs), list(ys))
        print(f"Spearman(min|cos_sim|,  ES)     = {format_rho(rho)}  (expect negative if hypothesis holds)")

    if len(valid_s) >= 2:
        xs, ys = zip(*valid_s)
        rho = spearman_rank_correlation(list(xs), list(ys))
        print(f"Spearman(mean|cos_sim|, S)      = {format_rho(rho)}  (expect negative if hypothesis holds)")

    if target_true is None:
        print(
            "\nToken-projection analysis skipped -- pass the fact's original object "
            "(target_true, e.g. Paris) as a second argument to enable it."
        )
    elif all(r["span_overlap"] is None for r in rows):
        print(
            f"\nToken-projection analysis unavailable: {target_true!r} never appears "
            "in any layer's top-5 logit-lens ranking for this fact."
        )
    else:
        if len(valid_ovlp_es) >= 2:
            xs, ys = zip(*valid_ovlp_es)
            rho = spearman_rank_correlation(list(xs), list(ys))
            print(f"Spearman(span_overlap,  ES)     = {format_rho(rho)}  (expect positive if hypothesis holds)")

        if len(valid_maxp_es) >= 2:
            xs, ys = zip(*valid_maxp_es)
            rho = spearman_rank_correlation(list(xs), list(ys))
            print(f"Spearman(maxP in scheme, ES)    = {format_rho(rho)}  (expect positive if hypothesis holds)")

        if len(valid_ovlp_s) >= 2:
            xs, ys = zip(*valid_ovlp_s)
            rho = spearman_rank_correlation(list(xs), list(ys))
            print(f"Spearman(span_overlap,  S)      = {format_rho(rho)}  (expect positive if hypothesis holds)")

    print(
        "\nNote: with only a handful of schemes this is directional evidence, "
        "not a significance-tested result. Run `compare` across more schemes "
        "and more facts, then re-run this analysis on the combined data, "
        "before drawing strong conclusions for your capstone write-up."
    )


if __name__ == "__main__":
    main()
