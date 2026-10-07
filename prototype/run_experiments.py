"""
Benchmark experiment execution pipeline for KEditVis evaluation.

Runs:
1. Smoke test: Development-only pilot checking signals, metrics, Frobenius drift and baseline-probe restoration.
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
from prepare_benchmark import validate_manifest

DEFAULT_URL = "https://opzgameryt--keditvis-memit-web-app.modal.run"
MANIFEST_PATH = Path(__file__).parent / "data" / "benchmark_manifest.json"
EVAL_DIR = Path(__file__).parent / "audit" / "evaluation"

# Pinned rather than left on scipy's "auto" heuristic: auto silently switches
# between the exact and normal-approximation tests depending on the scipy
# version and the tie pattern, which made previously published Wilcoxon
# p-values irreproducible. "approx" is also the only valid choice here, since
# these difference vectors contain zeros (scipy's exact test rejects them).
WILCOXON_METHOD = "approx"
EXPERIMENT_SCHEMA = 2
SELECTION_POLICIES = ("static", "telemetry", "random")
OPTIMIZATION_PROFILES = ("standard", "standard_budget", "context_no_consistency", "context")


class ExperimentFailure(RuntimeError):
    """Carry partial observations into the failure checkpoint before stopping."""
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


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
        manifest = json.load(f)
    validate_manifest(manifest)
    return manifest


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
    generation = res_obj.get("generation")
    if generation is not None and not isinstance(generation, str):
        raise ValueError("Generation must be a string or null.")

    def metric(name: str):
        value = metrics.get(name) if metrics else None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise ValueError(f"Non-finite or non-numeric {name} metric.")
        return value

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
        "has_repetition": detect_repetition(generation) if generation is not None else None,
        "status": res_obj.get("status", "complete"),
        "optimization_config": res_obj.get("optimization_config"),
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
    if not diffs or any(not math.isfinite(d) for d in diffs):
        raise ValueError("Bootstrap requires nonempty finite differences.")
    if n_resamples <= 0 or not 0 < ci < 1:
        raise ValueError("Bootstrap requires positive resamples and 0 < ci < 1.")
    if len(diffs) < 2 or all(d == diffs[0] for d in diffs):
        val = float(np.mean(diffs))
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
    for value in a_vals + b_vals:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise ValueError("Paired comparison contains a non-finite or non-numeric observation.")

    pairs = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
    n_pairs = len(pairs)
    if n_pairs < 2:
        return {
            "n_pairs": n_pairs,
            "n_planned_pairs": len(a_vals),
            "n_missing_pairs": len(a_vals) - n_pairs,
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
        "n_planned_pairs": len(a_vals),
        "n_missing_pairs": len(a_vals) - n_pairs,
        "mean_diff": round(mean_diff, 12),
        "std_diff": round(std_diff, 12),
        "ci_95": [round(ci_low, 12), round(ci_high, 12)],
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


def finding_is_significant_improvement(entry: Dict[str, Any]) -> Optional[bool]:
    positive = finding_is_positive(entry)
    significant = entry.get("statistically_significant")
    return None if positive is None or significant is None else positive and significant


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
        key = tuple(normalize_layers(record["layers"]))
        if key in by_layers:
            raise RuntimeError(f"Backend returned duplicate scheme {list(key)}.")
        by_layers[key] = record

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
    record = {"case_id": case_id, "subject": fact["subject"],
              "target_new": fact["target_new"], "target_true": fact["target_true"],
              "status": "failed", "rollback_verified": None,
              "restoration_check": "baseline_probe_equality"}
    compare_error = None
    try:
        compare_data = post_with_retry(url, "/compare", compare_body, timeout=180)
    except Exception as exc:
        compare_error = exc
    compare_time = time.monotonic() - t0
    record["timing"] = {"probe_seconds": probe_time, "compare_seconds": compare_time}
    # A failed HTTP request can have reached the editing code. Check restoration
    # before allowing another fact or condition to run.
    try:
        record["rollback_verified"] = run_probe(url, fact, model)["layer_signals"] == signals
    except Exception as exc:
        record["error"] = f"Restoration probe failed: {exc}"
        raise ExperimentFailure(record["error"], record) from exc
    if not record["rollback_verified"]:
        record["error"] = "Baseline probe changed after comparison; stopping the benchmark."
        raise ExperimentFailure(record["error"], record)
    if compare_error is not None:
        record["error"] = str(compare_error)
        raise ExperimentFailure(record["error"], record) from compare_error

    matched = match_scheme_records(
        compare_data["schemes"],
        {"static": static_layers, "telemetry": telemetry_layers, "random": random_layers},
    )

    for label, layers in (("static", static_layers), ("telemetry", telemetry_layers),
                          ("random", random_layers)):
        record[label] = extract_scheme_record(
            {**matched[label], "optimization_config": compare_data.get("optimization_config")},
            layers, label)
    record["status"] = "complete"
    return record


def run_experiment_fact_optimization(
    fact: Dict[str, Any], url: str, model: str = "gpt2-xl"
) -> Dict[str, Any]:
    """Runs Experiment 2 on a single fact across the 4 optimization profiles."""
    case_id = fact["case_id"]
    fixed_layers = get_static_preset(model, "memit")  # [13, 14, 15, 16, 17]

    # Reference baseline probe
    base_probe = run_probe(url, fact, model)
    base_signals = base_probe["layer_signals"]

    profiles = OPTIMIZATION_PROFILES
    profile_results = {}
    record = {"case_id": case_id, "subject": fact["subject"],
              "target_new": fact["target_new"], "target_true": fact["target_true"],
              "layers": fixed_layers, "rollback_verified": True, "status": "failed",
              "restoration_check": "baseline_probe_equality", "profiles": profile_results}

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
        edit_error = None
        try:
            edit_data = post_with_retry(url, "/edit", edit_body, timeout=120)
        except Exception as exc:
            edit_error = exc
        duration = time.monotonic() - t0
        try:
            restored = run_probe(url, fact, model)["layer_signals"] == base_signals
        except Exception as exc:
            record["rollback_verified"] = None
            record["error"] = f"Restoration probe failed after {prof}: {exc}"
            raise ExperimentFailure(record["error"], record) from exc
        if not restored:
            record["rollback_verified"] = False
            record["error"] = f"Baseline probe changed after {prof}; stopping the benchmark."
            raise ExperimentFailure(record["error"], record)
        if edit_error is not None:
            profile_results[prof] = {"label": prof, "status": "failed",
                                     "seconds": duration, "error": str(edit_error)}
            record["error"] = str(edit_error)
            raise ExperimentFailure(record["error"], record) from edit_error
        post = edit_data.get("post_edit", {})
        gen_text = post["generations"][0] if post.get("generations") else None

        profile_results[prof] = extract_scheme_record(
            {"metrics": post.get("metrics"), "weight_drift": edit_data.get("weight_drift"),
             "damage": edit_data.get("damage"), "generation": gen_text,
             "optimization_config": edit_data.get("optimization_config")},
            fixed_layers,
            label=prof,
            optimization=prof,
            seconds=round(duration, 2),
        )
        rec = profile_results[prof]
        print(f"      [{prof}] Done ({duration:.1f}s). PS={rec['PS']}, S={rec['S']}, "
              f"Rel-Frob={fmt_value(rec['frob_rel'])}", flush=True)

    record["status"] = "complete"
    return record


def condition_record(fact, condition, optimization=False):
    if fact.get("rollback_verified") is False:
        return {"status": "invalid_baseline"}
    return (fact.get("profiles", {}) if optimization else fact).get(condition, {})


def aggregate_condition(facts, condition, metrics, optimization=False):
    records = [condition_record(fact, condition, optimization) for fact in facts]
    output = {}
    for metric in metrics:
        values = [record.get(metric) for record in records if record.get(metric) is not None]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values):
            raise ValueError(f"Invalid {condition}/{metric} observation in raw results.")
        output[metric] = {
            "mean": round(float(np.mean(values)), 12) if values else None,
            "std": round(float(np.std(values, ddof=1)), 12) if len(values) > 1 else None,
            "n_measured": len(values), "n_missing": len(facts) - len(values),
        }
    repetition = [record["has_repetition"] for record in records
                  if isinstance(record.get("has_repetition"), bool)]
    output["repetition_rate"] = (round(sum(repetition) / len(repetition), 4)
                                 if repetition else None)
    output["n_repetition_measured"] = len(repetition)
    output["n_failed"] = sum(record.get("status") in ("failed", "invalid_baseline")
                             for record in records)
    output["n_unavailable"] = sum(not record for record in records)
    return output


def analyze_results(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Computes comprehensive statistical analyses, paired comparisons, and verdicts."""
    exp1_facts = raw.get("experiment_1_selection", [])
    exp2_facts = raw.get("experiment_2_optimization", [])
    for name, facts in (("experiment_1_selection", exp1_facts),
                        ("experiment_2_optimization", exp2_facts)):
        case_ids = [fact["case_id"] for fact in facts]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError(f"Duplicate case_id in {name}; facts are not independent pairs.")

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
            "significance_scope": "exploratory, unadjusted two-sided paired t-tests; no multiplicity correction",
            "restoration_scope": "baseline probe equality is not a full parameter hash comparison",
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

        cond_stats = {c: aggregate_condition(exp1_facts, c, metrics) for c in conditions}

        # Paired differences: Telemetry vs Static, Random vs Static, Telemetry vs Random
        paired_diffs = {}
        for metric in ["ES", "PS", "NS", "S", "ES_greedy", "PS_greedy", "frob_abs", "frob_rel", "kl_divergence"]:
            tel_vals = [condition_record(f, "telemetry").get(metric) for f in exp1_facts]
            stat_vals = [condition_record(f, "static").get(metric) for f in exp1_facts]
            rand_vals = [condition_record(f, "random").get(metric) for f in exp1_facts]

            paired_diffs[f"telemetry_vs_static_{metric}"] = paired_comparison(tel_vals, stat_vals)
            paired_diffs[f"random_vs_static_{metric}"] = paired_comparison(rand_vals, stat_vals)
            paired_diffs[f"telemetry_vs_random_{metric}"] = paired_comparison(tel_vals, rand_vals)

        # Layer scheme agreement
        layer_pairs = [(set(condition_record(f, "telemetry").get("layers", [])),
                        set(condition_record(f, "static").get("layers", []))) for f in exp1_facts]
        layer_pairs = [(a, b) for a, b in layer_pairs if a and b]
        same_as_static = sum(a == b for a, b in layer_pairs)
        overlap_with_static = [len(a & b) / len(a | b) for a, b in layer_pairs]

        summary["experiment_1_selection"] = {
            "num_facts": n1,
            "num_planned_facts": raw.get("metadata", {}).get("num_facts", n1),
            "num_failed_facts": sum(f.get("status") == "failed" for f in exp1_facts),
            "condition_aggregates": cond_stats,
            "paired_differences": paired_diffs,
            "policy_overlap": {
                "n_measured": len(layer_pairs),
                "exact_match_rate": round(same_as_static / len(layer_pairs), 4) if layer_pairs else None,
                "mean_layer_jaccard_overlap": round(float(np.mean(overlap_with_static)), 4) if layer_pairs else None,
            },
            "findings": {
                "telemetry_beats_static_ES": finding_is_positive(paired_diffs["telemetry_vs_static_ES"]),
                "telemetry_beats_static_S": finding_is_positive(paired_diffs["telemetry_vs_static_S"]),
                "telemetry_beats_static_significant": finding_is_significant_improvement(paired_diffs["telemetry_vs_static_S"]),
                "random_beats_static_significant": finding_is_significant_improvement(paired_diffs["random_vs_static_S"]),
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

        prof_stats = {p: aggregate_condition(exp2_facts, p, metrics, optimization=True) for p in profs}

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
                a_vals = [condition_record(f, cond_a, True).get(metric) for f in exp2_facts]
                b_vals = [condition_record(f, cond_b, True).get(metric) for f in exp2_facts]
                paired_ablations[f"{label}_{metric}"] = paired_comparison(a_vals, b_vals)

        summary["experiment_2_optimization"] = {
            "num_facts": n2,
            "num_planned_facts": raw.get("metadata", {}).get("num_facts", n2),
            "num_failed_facts": sum(f.get("status") == "failed" for f in exp2_facts),
            "profile_aggregates": prof_stats,
            "paired_ablations": paired_ablations,
            "findings": {
                "context_beats_standard_PS": finding_is_positive(paired_ablations["total_gain_context_vs_standard_PS"]),
                "context_beats_standard_PS_significant": finding_is_significant_improvement(paired_ablations["total_gain_context_vs_standard_PS"]),
                "budget_effect_PS": paired_ablations["budget_effect_standard_budget_vs_standard_PS"]["mean_diff"],
                "context_fitting_effect_PS": paired_ablations["context_fitting_effect_no_cons_vs_budget_PS"]["mean_diff"],
                "consistency_penalty_effect_PS": paired_ablations["consistency_penalty_effect_context_vs_no_cons_PS"]["mean_diff"],
                "weight_drift_increase": paired_ablations["total_gain_context_vs_standard_frob_rel"]["mean_diff"],
            },
        }

    return summary


def select_benchmark_facts(manifest, mode, num_facts=None):
    if num_facts is not None and (isinstance(num_facts, bool) or not isinstance(num_facts, int)
                                  or num_facts <= 0):
        raise ValueError("num_facts must be a positive integer.")
    split = "dev_facts" if mode == "smoke" else "eval_facts"
    facts = manifest[split]
    if num_facts is not None:
        if num_facts > len(facts):
            raise ValueError(f"Requested {num_facts} facts, but {split} only contains {len(facts)}.")
        facts = facts[:num_facts]
    return facts


def experiment_identity(model, url, mode, manifest, facts, health):
    """Bind a checkpoint to the actual backend build and the exact planned matrix."""
    if health.get("model") != model:
        raise ValueError("Backend health reports a different model than requested.")
    hashes = health.get("backend_source_sha256", {})
    for filename in ("modal_app.py", "editing_optimizations.py"):
        digest = hashes.get(filename)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"Backend must report its actual deployed {filename} SHA-256.")
    n_layers = health.get("n_layers")
    expected_layers = 48 if model == "gpt2-xl" else 28
    if n_layers != expected_layers:
        raise ValueError("Backend layer count does not match the requested architecture.")
    if not health.get("editing_commit"):
        raise ValueError("Backend editing revision is missing.")
    if not set(OPTIMIZATION_PROFILES).issubset(health.get("memit_optimizations", [])):
        raise ValueError("Backend does not support the planned optimization profiles.")
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ("run_experiments.py", "layer_selection.py", "prepare_benchmark.py")}
    return {"schema_version": EXPERIMENT_SCHEMA, "model": model, "url": url.rstrip("/"),
            "mode": mode, "manifest_sha256": validate_manifest(manifest),
            "case_ids": [fact["case_id"] for fact in facts],
            "backend_source_sha256": hashes, "editing_commit": health["editing_commit"],
            "model_revision": health.get("model_revision"), "driver_source_sha256": sources,
            "selection_seed": 42, "n_layers": n_layers,
            "optimization_profiles": list(OPTIMIZATION_PROFILES),
            "scipy_version": scipy.__version__, "numpy_version": np.__version__,
            "wilcoxon_method": WILCOXON_METHOD}


def validate_checkpoint(previous, identity):
    if previous.get("metadata", {}).get("run_identity") != identity:
        raise ValueError("Checkpoint belongs to a different or unrecorded run identity. Use a new output directory.")
    planned = set(identity["case_ids"])
    for key in ("experiment_1_selection", "experiment_2_optimization"):
        facts = previous.get(key, [])
        ids = [fact["case_id"] for fact in facts]
        if len(set(ids)) != len(ids) or not set(ids).issubset(planned):
            raise ValueError(f"Checkpoint {key} has duplicate or unplanned case IDs.")
        if any(fact.get("status") == "failed" or fact.get("rollback_verified") is False for fact in facts):
            raise ValueError("Checkpoint contains failed or contaminated observations. Review it and use a new output directory.")


def save_json(path, payload):
    """Replace checkpoints atomically so interruption cannot truncate prior progress."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def execute_fact(runner, fact, raw, key, path, url, model):
    try:
        record = runner(fact, url, model)
    except Exception as exc:
        record = (exc.record if isinstance(exc, ExperimentFailure) else {
            "case_id": fact["case_id"], "subject": fact["subject"],
            "status": "failed", "rollback_verified": None, "error": str(exc)})
        raw[key].append(record)
        save_json(path, raw)
        raise
    raw[key].append(record)
    save_json(path, raw)
    return record


def main():
    parser = argparse.ArgumentParser(description="Run benchmark experiments for KEditVis evaluation.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Modal web app URL")
    parser.add_argument("--model", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full", "selection", "optimization", "analyze-only"])
    parser.add_argument("--num-facts", type=int, default=None, help="Max number of facts to evaluate")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Path to benchmark manifest JSON")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "audit" / "run"),
                        help="Output directory; defaults to regenerated audit/run artifacts")
    parser.add_argument("--raw-results", type=Path,
                        help="Read this raw result file in analyze-only mode, preserving its directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.json"
    summary_path = out_dir / "summary.json"

    if args.mode == "analyze-only":
        input_path = args.raw_results or (EVAL_DIR / "raw_results.json")
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} does not exist.")
        if summary_path.resolve() == input_path.with_name("summary.json").resolve():
            raise ValueError("Analyze-only must write to a separate output directory to preserve recorded evidence.")
        with open(input_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        summary = analyze_results(raw)
        summary["metadata"]["analysis_input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
        save_json(summary_path, summary)
        print(f"Analysis saved to {summary_path}")
        return
    if args.raw_results is not None:
        parser.error("--raw-results is only valid with --mode analyze-only")

    manifest = load_manifest(Path(args.manifest))
    facts = select_benchmark_facts(manifest, args.mode, args.num_facts)
    previous = None
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as checkpoint:
            previous = json.load(checkpoint)
        if "run_identity" not in previous.get("metadata", {}):
            raise ValueError("Historical checkpoint has no producing-build identity. Use a new output directory.")

    # Check health and backend connectivity
    print(f"Connecting to {args.url} ...")
    health = check_health(args.url, args.model)
    print(f"Backend healthy. Model: {health['model']}, Layers: {health['n_layers']}, Profiles: {health['memit_optimizations']}")

    identity = experiment_identity(args.model, args.url, args.mode, manifest, facts, health)
    print(f"Running mode '{args.mode}' on {len(facts)} {'development' if args.mode == 'smoke' else 'evaluation'} facts...")

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
            "run_identity": identity,
            "dataset_source_sha256": manifest["metadata"].get("source_sha256"),
            "dataset_provenance_scope": ("complete_source_pin" if manifest["metadata"].get("source_sha256")
                                         else "legacy_subset_digest_only; full source revision unavailable"),
        },
        "experiment_1_selection": [],
        "experiment_2_optimization": [],
    }

    if previous is not None:
        validate_checkpoint(previous, identity)
        raw_results = previous
        print(f"Loaded matching progress: {len(raw_results['experiment_1_selection'])} selection, {len(raw_results['experiment_2_optimization'])} optimization facts.")
    save_json(raw_path, raw_results)

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
            rec = execute_fact(run_experiment_fact_selection, fact, raw_results,
                               "experiment_1_selection", raw_path, args.url, args.model)
            dt = time.monotonic() - t0
            print(f"   Done ({dt:.1f}s). Rollback verified: {rec['rollback_verified']}")
            print(f"   Static {rec['static']['layers']}: ES={rec['static']['ES']}, PS={rec['static']['PS']}, S={rec['static']['S']}, Rel-Frob={fmt_value(rec['static']['frob_rel'])}")
            print(f"   Telemetry {rec['telemetry']['layers']}: ES={rec['telemetry']['ES']}, PS={rec['telemetry']['PS']}, S={rec['telemetry']['S']}, Rel-Frob={fmt_value(rec['telemetry']['frob_rel'])}")
            print(f"   Random {rec['random']['layers']}: ES={rec['random']['ES']}, PS={rec['random']['PS']}, S={rec['random']['S']}, Rel-Frob={fmt_value(rec['random']['frob_rel'])}")


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
            rec = execute_fact(run_experiment_fact_optimization, fact, raw_results,
                               "experiment_2_optimization", raw_path, args.url, args.model)
            dt = time.monotonic() - t0
            p = rec["profiles"]
            print(f"   Done ({dt:.1f}s). Rollback verified: {rec['rollback_verified']}")
            print(f"   Standard:        PS={p['standard']['PS']}, S={p['standard']['S']}, Rel-Frob={fmt_value(p['standard']['frob_rel'])}, KL={fmt_value(p['standard']['kl_divergence'])}")
            print(f"   Standard Budget: PS={p['standard_budget']['PS']}, S={p['standard_budget']['S']}, Rel-Frob={fmt_value(p['standard_budget']['frob_rel'])}, KL={fmt_value(p['standard_budget']['kl_divergence'])}")
            print(f"   Context No-Cons: PS={p['context_no_consistency']['PS']}, S={p['context_no_consistency']['S']}, Rel-Frob={fmt_value(p['context_no_consistency']['frob_rel'])}, KL={fmt_value(p['context_no_consistency']['kl_divergence'])}")
            print(f"   Context Full v3: PS={p['context']['PS']}, S={p['context']['S']}, Rel-Frob={fmt_value(p['context']['frob_rel'])}, KL={fmt_value(p['context']['kl_divergence'])}")


    # Analyze and generate statistical summary
    print("\nComputing statistical analysis and bootstrap confidence intervals...", flush=True)
    summary = analyze_results(raw_results)
    save_json(summary_path, summary)
    print(f"Saved statistical summary to {summary_path}", flush=True)
    print("Benchmark run complete!", flush=True)


if __name__ == "__main__":
    main()
