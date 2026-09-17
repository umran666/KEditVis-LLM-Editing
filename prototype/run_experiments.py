"""
Benchmark experiment execution pipeline for KEditVis evaluation.

Runs:
1. Smoke test: Pilot on 3 facts verifying signals, metrics, Frobenius drift, and bit-exact rollback restoration.
2. Experiment 1 (Selection Quality): Static preset ([13..17]) vs Telemetry-guided vs Seeded random.
3. Experiment 2 (Algorithm Quality / Ablation): Standard MEMIT vs Standard Budget-matched vs Context (no consistency) vs Context v3.
4. Statistical analysis: Paired t-tests, Wilcoxon signed-rank tests, bootstrap 95% CIs, and repetition detection.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import requests
import scipy
from scipy import stats

from layer_selection import get_random_scheme, get_static_preset, select_layers_telemetry

DEFAULT_URL = "https://opzgameryt--keditvis-memit-web-app.modal.run"
MANIFEST_PATH = Path(__file__).parent / "data" / "benchmark_manifest.json"
EVAL_DIR = Path(__file__).parent / "audit" / "evaluation"

# Pinned rather than left on scipy's "auto" heuristic: auto silently switches
# between the exact and normal-approximation tests depending on the scipy
# version and the tie pattern, which made previously published Wilcoxon
# p-values irreproducible. "approx" is also the only valid choice here, since
# these difference vectors contain zeros (scipy's exact test rejects them).
WILCOXON_METHOD = "approx"


def detect_repetition(text: str, n: int = 3, max_repeats: int = 3) -> bool:
    """Detects degenerate text generation with repeating n-grams."""
    words = text.strip().split()
    if len(words) < n * max_repeats:
        return False
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    for i in range(len(ngrams) - (max_repeats - 1) * n):
        target = ngrams[i]
        repeats = 1
        for step in range(1, max_repeats):
            if i + step * n < len(ngrams) and ngrams[i + step * n] == target:
                repeats += 1
            else:
                break
        if repeats >= max_repeats:
            return True
    return False


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scheme_record(
    res_obj: Dict[str, Any], layers: List[int], label: str = "", **extra: Any
) -> Dict[str, Any]:
    """Flattens one scheme/profile result into a flat, comparable record.

    Metrics that were never computed are recorded as None rather than 0.0. A
    coerced 0.0 is indistinguishable from a genuine failure and would be
    averaged into the published means as though it had been measured; None is
    filtered out by `analyze_results` instead.
    """
    metrics = res_obj.get("metrics")
    drift = res_obj.get("weight_drift") or {}
    damage = res_obj.get("damage")
    generation = res_obj.get("generation", "")

    def metric(name: str):
        return metrics.get(name) if metrics else None

    return {
        "label": label,
        "layers": layers,
        **extra,
        "ES": metric("ES"),
        "PS": metric("PS"),
        "NS": metric("NS"),
        "S": metric("S"),
        "ES_greedy": metric("ES_greedy"),
        "PS_greedy": metric("PS_greedy"),
        "NS_greedy": metric("NS_greedy"),
        "S_greedy": metric("S_greedy"),
        "kl_divergence": damage.get("kl_divergence") if damage else None,
        "frob_abs": drift.get("total_absolute_frobenius"),
        "frob_rel": drift.get("total_relative_frobenius"),
        "mean_layer_rel": drift.get("mean_layer_relative_frobenius"),
        "generation": generation,
        "has_repetition": detect_repetition(generation),
    }


def fmt_value(value) -> str:
    """Formats an optional numeric for console output."""
    return "n/a" if value is None else f"{value:.4f}"


def describe_http_error(resp) -> str:
    """Extracts the backend's `detail` so a rejection reports *why* it happened."""
    try:
        payload = resp.json()
    except ValueError:
        return resp.text[:500] if resp.text else "(no response body)"
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, list):
            detail = "; ".join(
                str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                for item in detail
            )
        if detail:
            return str(detail)
    return str(payload)[:500]


def post_with_retry(
    url: str, path: str, json_body: Dict[str, Any], timeout: int = 180, max_retries: int = 3
) -> Dict[str, Any]:
    """Posts JSON to an endpoint, retrying only transient failures.

    A 4xx is a permanent client error -- the identical request will be rejected
    identically every time -- so it fails immediately with the backend's own
    `detail` message instead of being retried with backoff, which previously
    burned 15s and then discarded the only useful diagnostic.
    """
    full_url = f"{url}{path}"
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.Session() as session:
                resp = session.post(full_url, json=json_body, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"      [Retry {attempt}/{max_retries}] {path} transient error "
                  f"({last_err}). Waiting 5s...", flush=True)
            time.sleep(5)
            continue

        if resp.status_code < 400:
            return resp.json()
        if resp.status_code < 500:
            raise RuntimeError(
                f"Request to {path} rejected with HTTP {resp.status_code}: "
                f"{describe_http_error(resp)}"
            )
        last_err = f"HTTP {resp.status_code}: {describe_http_error(resp)}"
        print(f"      [Retry {attempt}/{max_retries}] {path} server error "
              f"({last_err}). Waiting 5s...", flush=True)
        time.sleep(5)

    raise RuntimeError(f"Request to {path} failed after {max_retries} attempts: {last_err}")


def check_health(url: str, model: str = "gpt2-xl") -> Dict[str, Any]:
    resp = requests.get(f"{url}/health?model={model}", timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_probe(url: str, fact: Dict[str, Any], model: str = "gpt2-xl") -> Dict[str, Any]:
    body = {
        "model": model,
        "prompt": fact["prompt"],
        "subject": fact["subject"],
    }
    return post_with_retry(url, "/probe", body, timeout=60)


def run_generate(url: str, prompt: str, subject: str, model: str = "gpt2-xl") -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "subject": subject,
    }
    return post_with_retry(url, "/generate", body, timeout=60).get("generation", "")


def compute_bootstrap_ci(
    diffs: List[float], n_resamples: int = 10000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float]:
    """Computes percentile bootstrap confidence interval for the mean difference."""
    if len(diffs) < 2 or all(d == diffs[0] for d in diffs):
        val = float(np.mean(diffs)) if diffs else 0.0
        return val, val
    rng = np.random.RandomState(seed)
    arr = np.array(diffs)
    indices = rng.randint(0, len(arr), size=(n_resamples, len(arr)))
    boot_means = np.mean(arr[indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100 * alpha))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return low, high


def paired_comparison(
    a_vals: List[Optional[float]], b_vals: List[Optional[float]]
) -> Dict[str, Any]:
    """Computes paired difference statistics between condition A and condition B.

    Only observations present in BOTH arms are used, so a metric measured for
    some facts and not others cannot contribute a phantom zero to the
    difference. Returns `n_pairs` plus None statistics when fewer than two
    complete pairs remain, instead of inventing a value.
    """
    if len(a_vals) != len(b_vals):
        raise ValueError(
            f"Paired comparison requires equal-length inputs "
            f"(got {len(a_vals)} and {len(b_vals)})."
        )

    pairs = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
    n_pairs = len(pairs)
    if n_pairs < 2:
        return {
            "n_pairs": n_pairs,
            "mean_diff": None,
            "std_diff": None,
            "ci_95": None,
            "t_stat": None,
            "p_value_t": None,
            "wilcoxon_stat": None,
            "p_value_wilcoxon": None,
            "wilcoxon_method": WILCOXON_METHOD,
            "cohens_d": None,
            "statistically_significant": None,
        }

    a_paired = [a for a, _ in pairs]
    b_paired = [b for _, b in pairs]
    diffs = [a - b for a, b in pairs]
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    ci_low, ci_high = compute_bootstrap_ci(diffs)

    # Paired t-test
    if std_diff > 1e-12:
        t_res = stats.ttest_rel(a_paired, b_paired)
        t_stat = float(t_res.statistic)
        p_val_t = float(t_res.pvalue)
    else:
        # Every difference is identical: the t statistic is undefined, not zero.
        t_stat = None
        p_val_t = None

    # Wilcoxon signed-rank test
    nonzero_diffs = [d for d in diffs if abs(d) > 1e-12]
    if len(nonzero_diffs) >= 5:
        try:
            with warnings.catch_warnings():
                # The normal approximation is a deliberate choice (the exact test is
                # invalid when the difference vector contains zeros). scipy warns
                # that small samples make it unreliable, which is expected here and
                # already disclosed in the docs, so the warning is not noise worth
                # emitting on every metric of every run.
                warnings.simplefilter("ignore", UserWarning)
                w_res = stats.wilcoxon(diffs, alternative="two-sided", method=WILCOXON_METHOD)
            w_stat = float(w_res.statistic)
            p_val_w = float(w_res.pvalue)
        except ValueError as exc:
            # scipy rejects an all-zero difference vector; report "no test" rather
            # than silently claiming a p-value.
            print(f"      [skip] Wilcoxon unavailable for this metric: {exc}")
            w_stat = None
            p_val_w = None
    else:
        w_stat = None
        p_val_w = None

    # Cohen's dz: the mean difference in standard-deviation units. With zero
    # spread the effect size is unbounded, so it is reported as undefined
    # rather than as "no effect".
    cohens_d = mean_diff / std_diff if std_diff > 1e-12 else None

    def rounded(value, places):
        return None if value is None else round(value, places)

    return {
        "n_pairs": n_pairs,
        "mean_diff": round(mean_diff, 4),
        "std_diff": round(std_diff, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "t_stat": rounded(t_stat, 4),
        "p_value_t": rounded(p_val_t, 5),
        "wilcoxon_stat": rounded(w_stat, 4),
        "p_value_wilcoxon": rounded(p_val_w, 5),
        "wilcoxon_method": WILCOXON_METHOD,
        "cohens_d": rounded(cohens_d, 4),
        "statistically_significant": None if p_val_t is None else p_val_t < 0.05,
    }


def finding_is_positive(entry: Dict[str, Any]) -> Optional[bool]:
    """True/False for a positive mean difference, or None when not computable."""
    diff = entry.get("mean_diff")
    return None if diff is None else diff > 0.0


def normalize_layers(layers: List[int]) -> List[int]:
    """Mirrors the backend's _normalize_scheme so results can be matched to requests."""
    return sorted({int(layer) for layer in layers})


def match_scheme_records(
    returned: List[Dict[str, Any]], requested: Dict[str, List[int]]
) -> Dict[str, Dict[str, Any]]:
    """Maps each requested policy name to the backend result for its layer window.

    /compare deduplicates identical schemes before evaluating them, so the
    response is not guaranteed to be the same length as the request, nor in the
    same order. Matching on the returned layer list keeps every result attached
    to the policy that asked for it even when two policies select the same
    window (e.g. telemetry landing on the static preset for a given fact).
    """
    by_layers = {}
    for record in returned:
        by_layers[tuple(normalize_layers(record["layers"]))] = record

    matched = {}
    for label, layers in requested.items():
        key = tuple(normalize_layers(layers))
        if key not in by_layers:
            raise RuntimeError(
                f"Backend returned no result for the {label} scheme {list(key)}; "
                f"got {sorted(list(k) for k in by_layers)}."
            )
        matched[label] = by_layers[key]
    return matched


def run_experiment_fact_selection(
    fact: Dict[str, Any], url: str, model: str = "gpt2-xl"
) -> Dict[str, Any]:
    """Runs Experiment 1 on a single fact using /compare with 3 schemes."""
    case_id = fact["case_id"]

    # 1. Baseline probe to extract layer signals
    t0 = time.monotonic()
    probe_res = run_probe(url, fact, model)
    probe_time = time.monotonic() - t0
    signals = probe_res["layer_signals"]

    # 2. Derive layer schemes
    static_layers = get_static_preset(model, "memit")
    telemetry_layers = select_layers_telemetry(signals, model, "memit")
    random_layers = get_random_scheme(model, "memit", seed=42 + case_id)

    # 3. Call compare endpoint
    compare_body = {
        "model": model,
        "method": "memit",
        "optimization": "standard",
        "prompt": fact["prompt"],
        "subject": fact["subject"],
        "target_new": fact["target_new"],
        "target_true": fact["target_true"],
        "paraphrase_prompts": fact["paraphrase_prompts"],
        "neighborhood_prompts": fact["neighborhood_prompts"],
        "neighborhood_targets": fact.get("neighborhood_targets"),
        "schemes": [static_layers, telemetry_layers, random_layers],
    }

    t0 = time.monotonic()
    compare_data = post_with_retry(url, "/compare", compare_body, timeout=180)
    compare_time = time.monotonic() - t0

    matched = match_scheme_records(
        compare_data["schemes"],
        {"static": static_layers, "telemetry": telemetry_layers, "random": random_layers},
    )

    static_rec = extract_scheme_record(matched["static"], static_layers, "static")
    telemetry_rec = extract_scheme_record(matched["telemetry"], telemetry_layers, "telemetry")
    random_rec = extract_scheme_record(matched["random"], random_layers, "random")

    # 4. Verify post-compare rollback restoration via probe
    post_probe = run_probe(url, fact, model)
    signals_match = post_probe["layer_signals"] == signals

    return {
        "case_id": case_id,
        "subject": fact["subject"],
        "target_new": fact["target_new"],
        "target_true": fact["target_true"],
        "timing": {"probe_seconds": probe_time, "compare_seconds": compare_time},
        "rollback_verified": signals_match,
        "static": static_rec,
        "telemetry": telemetry_rec,
        "random": random_rec,
    }


def run_experiment_fact_optimization(
    fact: Dict[str, Any], url: str, model: str = "gpt2-xl"
) -> Dict[str, Any]:
    """Runs Experiment 2 on a single fact across the 4 optimization profiles."""
    case_id = fact["case_id"]
    fixed_layers = get_static_preset(model, "memit")  # [13, 14, 15, 16, 17]

    # Reference baseline probe
    base_probe = run_probe(url, fact, model)
    base_signals = base_probe["layer_signals"]

    profiles = ["standard", "standard_budget", "context_no_consistency", "context"]
    profile_results = {}
    rollback_ok = True

    for prof in profiles:
        edit_body = {
            "model": model,
            "method": "memit",
            "optimization": prof,
            "prompt": fact["prompt"],
            "subject": fact["subject"],
            "target_new": fact["target_new"],
            "target_true": fact["target_true"],
            "paraphrase_prompts": fact["paraphrase_prompts"],
            "neighborhood_prompts": fact["neighborhood_prompts"],
            "neighborhood_targets": fact.get("neighborhood_targets"),
            "layers": fixed_layers,
        }

        t0 = time.monotonic()
        edit_data = post_with_retry(url, "/edit", edit_body, timeout=120)
        duration = time.monotonic() - t0

        post = edit_data.get("post_edit", {})
        gen_text = post.get("generations", [""])[0] if post.get("generations") else ""

        profile_results[prof] = extract_scheme_record(
            {"metrics": post.get("metrics"), "weight_drift": edit_data.get("weight_drift"),
             "damage": edit_data.get("damage"), "generation": gen_text},
            fixed_layers,
            label=prof,
            optimization=prof,
            seconds=round(duration, 2),
        )
        rec = profile_results[prof]
        print(f"      [{prof}] Done ({duration:.1f}s). PS={rec['PS']}, S={rec['S']}, "
              f"Rel-Frob={fmt_value(rec['frob_rel'])}", flush=True)

        # Verify probe restoration
        check_p = run_probe(url, fact, model)
        if check_p["layer_signals"] != base_signals:
            rollback_ok = False

    return {
        "case_id": case_id,
        "subject": fact["subject"],
        "target_new": fact["target_new"],
        "target_true": fact["target_true"],
        "layers": fixed_layers,
        "rollback_verified": rollback_ok,
        "profiles": profile_results,
    }


def analyze_results(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Computes comprehensive statistical analyses, paired comparisons, and verdicts."""
    exp1_facts = raw.get("experiment_1_selection", [])
    exp2_facts = raw.get("experiment_2_optimization", [])

    summary: Dict[str, Any] = {
        "experiment_1_selection": {},
        "experiment_2_optimization": {},
        # Record the software that produced the statistics: the p-values are only
        # reproducible against a pinned scipy and an explicitly chosen Wilcoxon
        # method.
        "metadata": {
            **raw.get("metadata", {}),
            "analysis_scipy_version": scipy.__version__,
            "analysis_wilcoxon_method": WILCOXON_METHOD,
        },
    }

    # --- EXPERIMENT 1: SELECTION QUALITY ---
    if exp1_facts:
        n1 = len(exp1_facts)
        conditions = ["static", "telemetry", "random"]
        metrics = [
            "ES", "PS", "NS", "S",
            "ES_greedy", "PS_greedy", "NS_greedy", "S_greedy",
            "frob_abs", "frob_rel", "kl_divergence"
        ]

        cond_stats = {}
        for c in conditions:
            cond_stats[c] = {}
            for m in metrics:
                vals = [f[c].get(m) for f in exp1_facts if f[c].get(m) is not None]
                cond_stats[c][m] = {
                    "mean": round(float(np.mean(vals)), 4) if vals else None,
                    "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None,
                }
            reps = sum(1 for f in exp1_facts if f[c].get("has_repetition"))
            cond_stats[c]["repetition_rate"] = round(reps / n1, 4)

        # Paired differences: Telemetry vs Static, Random vs Static, Telemetry vs Random
        paired_diffs = {}
        for metric in ["ES", "PS", "NS", "S", "ES_greedy", "PS_greedy", "frob_abs", "frob_rel", "kl_divergence"]:
            tel_vals = [f["telemetry"].get(metric) for f in exp1_facts]
            stat_vals = [f["static"].get(metric) for f in exp1_facts]
            rand_vals = [f["random"].get(metric) for f in exp1_facts]

            paired_diffs[f"telemetry_vs_static_{metric}"] = paired_comparison(tel_vals, stat_vals)
            paired_diffs[f"random_vs_static_{metric}"] = paired_comparison(rand_vals, stat_vals)
            paired_diffs[f"telemetry_vs_random_{metric}"] = paired_comparison(tel_vals, rand_vals)

        # Layer scheme agreement
        same_as_static = sum(1 for f in exp1_facts if f["telemetry"]["layers"] == f["static"]["layers"])
        overlap_with_static = [
            len(set(f["telemetry"]["layers"]).intersection(set(f["static"]["layers"]))) / 5.0
            for f in exp1_facts
        ]

        summary["experiment_1_selection"] = {
            "num_facts": n1,
            "condition_aggregates": cond_stats,
            "paired_differences": paired_diffs,
            "policy_overlap": {
                "exact_match_rate": round(same_as_static / n1, 4),
                "mean_layer_jaccard_overlap": round(float(np.mean(overlap_with_static)), 4),
            },
            "findings": {
                "telemetry_beats_static_ES": finding_is_positive(paired_diffs["telemetry_vs_static_ES"]),
                "telemetry_beats_static_S": finding_is_positive(paired_diffs["telemetry_vs_static_S"]),
                "telemetry_beats_static_significant": paired_diffs["telemetry_vs_static_S"]["statistically_significant"],
                "random_beats_static_significant": paired_diffs["random_vs_static_S"]["statistically_significant"],
            },
        }

    # --- EXPERIMENT 2: ALGORITHM ABLATION ---
    if exp2_facts:
        n2 = len(exp2_facts)
        profs = ["standard", "standard_budget", "context_no_consistency", "context"]
        metrics = [
            "ES", "PS", "NS", "S",
            "ES_greedy", "PS_greedy", "NS_greedy", "S_greedy",
            "frob_abs", "frob_rel", "kl_divergence", "seconds"
        ]

        prof_stats = {}
        for p in profs:
            prof_stats[p] = {}
            for m in metrics:
                vals = [f["profiles"][p].get(m) for f in exp2_facts if f["profiles"][p].get(m) is not None]
                prof_stats[p][m] = {
                    "mean": round(float(np.mean(vals)), 4) if vals else None,
                    "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None,
                }
            reps = sum(1 for f in exp2_facts if f["profiles"][p].get("has_repetition"))
            prof_stats[p]["repetition_rate"] = round(reps / n2, 4)

        paired_ablations = {}
        # 1. Total gain: context vs standard
        # 2. Budget effect: standard_budget vs standard
        # 3. Context fitting effect: context_no_consistency vs standard_budget
        # 4. Consistency penalty effect: context vs context_no_consistency
        comparisons = [
            ("total_gain_context_vs_standard", "context", "standard"),
            ("budget_effect_standard_budget_vs_standard", "standard_budget", "standard"),
            ("context_fitting_effect_no_cons_vs_budget", "context_no_consistency", "standard_budget"),
            ("consistency_penalty_effect_context_vs_no_cons", "context", "context_no_consistency"),
        ]

        for label, cond_a, cond_b in comparisons:
            for metric in ["ES", "PS", "NS", "S", "ES_greedy", "PS_greedy", "frob_abs", "frob_rel", "kl_divergence"]:
                a_vals = [f["profiles"][cond_a].get(metric) for f in exp2_facts]
                b_vals = [f["profiles"][cond_b].get(metric) for f in exp2_facts]
                paired_ablations[f"{label}_{metric}"] = paired_comparison(a_vals, b_vals)

        summary["experiment_2_optimization"] = {
            "num_facts": n2,
            "profile_aggregates": prof_stats,
            "paired_ablations": paired_ablations,
            "findings": {
                "context_beats_standard_PS": finding_is_positive(paired_ablations["total_gain_context_vs_standard_PS"]),
                "context_beats_standard_PS_significant": paired_ablations["total_gain_context_vs_standard_PS"]["statistically_significant"],
                "budget_effect_PS": paired_ablations["budget_effect_standard_budget_vs_standard_PS"]["mean_diff"],
                "context_fitting_effect_PS": paired_ablations["context_fitting_effect_no_cons_vs_budget_PS"]["mean_diff"],
                "consistency_penalty_effect_PS": paired_ablations["consistency_penalty_effect_context_vs_no_cons_PS"]["mean_diff"],
                "weight_drift_increase": paired_ablations["total_gain_context_vs_standard_frob_rel"]["mean_diff"],
            },
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run benchmark experiments for KEditVis evaluation.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Modal web app URL")
    parser.add_argument("--model", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full", "selection", "optimization", "analyze-only"])
    parser.add_argument("--num-facts", type=int, default=None, help="Max number of facts to evaluate")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Path to benchmark manifest JSON")
    parser.add_argument("--output-dir", default=str(EVAL_DIR), help="Output directory for results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.json"
    summary_path = out_dir / "summary.json"

    if args.mode == "analyze-only":
        if not raw_path.exists():
            raise FileNotFoundError(f"{raw_path} does not exist.")
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        summary = analyze_results(raw)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Analysis saved to {summary_path}")
        return

    # Check health and backend connectivity
    print(f"Connecting to {args.url} ...")
    health = check_health(args.url, args.model)
    print(f"Backend healthy. Model: {health['model']}, Layers: {health['n_layers']}, Profiles: {health['memit_optimizations']}")

    manifest = load_manifest(Path(args.manifest))

    if args.mode == "smoke":
        # Dev fact + 2 eval facts
        facts = manifest["dev_facts"] + manifest["eval_facts"][:2]
        print(f"Running SMOKE TEST on {len(facts)} facts...")
    elif args.mode in ["full", "selection", "optimization"]:
        facts = manifest["eval_facts"]
        if args.num_facts:
            facts = facts[: args.num_facts]
        print(f"Running mode '{args.mode}' on {len(facts)} evaluation facts...")
    else:
        facts = manifest["eval_facts"][:3]

    raw_results: Dict[str, Any] = {
        "metadata": {
            "model": args.model,
            "manifest_sha256": manifest["metadata"]["sha256"],
            "url": args.url,
            "mode": args.mode,
            "num_facts": len(facts),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scipy_version": scipy.__version__,
            "wilcoxon_method": WILCOXON_METHOD,
        },
        "experiment_1_selection": [],
        "experiment_2_optimization": [],
    }

    # Load existing progress if any
    if raw_path.exists():
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
                if prev.get("metadata", {}).get("mode") == args.mode:
                    raw_results = prev
                    print(f"Loaded existing progress: {len(raw_results['experiment_1_selection'])} selection, {len(raw_results['experiment_2_optimization'])} optimization facts.")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Ignoring unreadable checkpoint {raw_path}: {exc}")

    do_selection = args.mode in ["smoke", "full", "selection"]
    do_optimization = args.mode in ["smoke", "full", "optimization"]

    # Execute Experiment 1
    if do_selection:
        print("\n=== STARTING EXPERIMENT 1: SELECTION QUALITY ===")
        existing_cases = {f["case_id"] for f in raw_results["experiment_1_selection"]}
        for idx, fact in enumerate(facts, 1):
            cid = fact["case_id"]
            if cid in existing_cases:
                print(f"[{idx}/{len(facts)}] Case {cid} ({fact['subject']}) already evaluated for Exp 1. Skipping.")
                continue
            print(f"[{idx}/{len(facts)}] Evaluating Case {cid}: '{fact['subject']}' -> '{fact['target_new']}'...")
            t0 = time.monotonic()
            rec = run_experiment_fact_selection(fact, args.url, args.model)
            dt = time.monotonic() - t0
            raw_results["experiment_1_selection"].append(rec)
            print(f"   Done ({dt:.1f}s). Rollback verified: {rec['rollback_verified']}")
            print(f"   Static [13..17]: ES={rec['static']['ES']}, PS={rec['static']['PS']}, S={rec['static']['S']}, Rel-Frob={fmt_value(rec['static']['frob_rel'])}")
            print(f"   Telemetry {rec['telemetry']['layers']}: ES={rec['telemetry']['ES']}, PS={rec['telemetry']['PS']}, S={rec['telemetry']['S']}, Rel-Frob={fmt_value(rec['telemetry']['frob_rel'])}")
            print(f"   Random {rec['random']['layers']}: ES={rec['random']['ES']}, PS={rec['random']['PS']}, S={rec['random']['S']}, Rel-Frob={fmt_value(rec['random']['frob_rel'])}")

            # Checkpoint raw results
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_results, f, indent=2)

    # Execute Experiment 2
    if do_optimization:
        print("\n=== STARTING EXPERIMENT 2: ALGORITHM ABLATION ===")
        existing_cases = {f["case_id"] for f in raw_results["experiment_2_optimization"]}
        for idx, fact in enumerate(facts, 1):
            cid = fact["case_id"]
            if cid in existing_cases:
                print(f"[{idx}/{len(facts)}] Case {cid} ({fact['subject']}) already evaluated for Exp 2. Skipping.")
                continue
            print(f"[{idx}/{len(facts)}] Evaluating Case {cid}: '{fact['subject']}' -> '{fact['target_new']}'...")
            t0 = time.monotonic()
            rec = run_experiment_fact_optimization(fact, args.url, args.model)
            dt = time.monotonic() - t0
            raw_results["experiment_2_optimization"].append(rec)
            p = rec["profiles"]
            print(f"   Done ({dt:.1f}s). Rollback verified: {rec['rollback_verified']}")
            print(f"   Standard:        PS={p['standard']['PS']}, S={p['standard']['S']}, Rel-Frob={fmt_value(p['standard']['frob_rel'])}, KL={fmt_value(p['standard']['kl_divergence'])}")
            print(f"   Standard Budget: PS={p['standard_budget']['PS']}, S={p['standard_budget']['S']}, Rel-Frob={fmt_value(p['standard_budget']['frob_rel'])}, KL={fmt_value(p['standard_budget']['kl_divergence'])}")
            print(f"   Context No-Cons: PS={p['context_no_consistency']['PS']}, S={p['context_no_consistency']['S']}, Rel-Frob={fmt_value(p['context_no_consistency']['frob_rel'])}, KL={fmt_value(p['context_no_consistency']['kl_divergence'])}")
            print(f"   Context Full v3: PS={p['context']['PS']}, S={p['context']['S']}, Rel-Frob={fmt_value(p['context']['frob_rel'])}, KL={fmt_value(p['context']['kl_divergence'])}")

            # Checkpoint raw results
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_results, f, indent=2)

    # Analyze and generate statistical summary
    print("\nComputing statistical analysis and bootstrap confidence intervals...", flush=True)
    summary = analyze_results(raw_results)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved statistical summary to {summary_path}", flush=True)
    print("Benchmark run complete!", flush=True)


if __name__ == "__main__":
    main()
