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

import numpy as np
import requests
from scipy import stats

from layer_selection import get_random_scheme, get_static_preset, select_layers_telemetry

DEFAULT_URL = "https://opzgameryt--keditvis-memit-web-app.modal.run"
MANIFEST_PATH = Path(__file__).parent / "data" / "benchmark_manifest.json"
EVAL_DIR = Path(__file__).parent / "audit" / "evaluation"


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


def post_with_retry(
    url: str, path: str, json_body: Dict[str, Any], timeout: int = 180, max_retries: int = 3
) -> Dict[str, Any]:
    """Posts JSON to endpoint with retries on transient connection or timeout errors."""
    full_url = f"{url}{path}"
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.Session() as session:
                resp = session.post(full_url, json=json_body, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            last_err = e
            print(f"      [Retry {attempt}/{max_retries}] {path} error ({type(e).__name__}): {e}. Waiting 5s...", flush=True)
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
    a_vals: List[float], b_vals: List[float]
) -> Dict[str, Any]:
    """Computes paired difference statistics between condition A and condition B."""
    assert len(a_vals) == len(b_vals)
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    ci_low, ci_high = compute_bootstrap_ci(diffs)

    # Paired t-test
    if len(diffs) > 1 and std_diff > 1e-12:
        t_res = stats.ttest_rel(a_vals, b_vals)
        t_stat = float(t_res.statistic)
        p_val_t = float(t_res.pvalue)
    else:
        t_stat = 0.0
        p_val_t = 1.0

    # Wilcoxon signed-rank test
    nonzero_diffs = [d for d in diffs if abs(d) > 1e-12]
    if len(nonzero_diffs) >= 5:
        try:
            w_res = stats.wilcoxon(diffs, alternative="two-sided")
            w_stat = float(w_res.statistic)
            p_val_w = float(w_res.pvalue)
        except Exception:
            w_stat = 0.0
            p_val_w = 1.0
    else:
        w_stat = 0.0
        p_val_w = 1.0

    # Cohen's dz
    cohens_d = mean_diff / std_diff if std_diff > 1e-12 else 0.0

    return {
        "mean_diff": round(mean_diff, 4),
        "std_diff": round(std_diff, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "t_stat": round(t_stat, 4),
        "p_value_t": round(p_val_t, 5),
        "wilcoxon_stat": round(w_stat, 4),
        "p_value_wilcoxon": round(p_val_w, 5),
        "cohens_d": round(cohens_d, 4),
        "statistically_significant": p_val_t < 0.05,
    }


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

    # Schemes returned in same order
    schemes_res = compare_data["schemes"]

    def extract_scheme_record(res_obj, layers, label):
        m = res_obj.get("metrics") or {}
        drift = res_obj.get("weight_drift") or {}
        damage = res_obj.get("damage") or {}
        gen_text = res_obj.get("generation", "")
        return {
            "label": label,
            "layers": layers,
            "ES": m.get("ES", 0.0),
            "PS": m.get("PS", 0.0),
            "NS": m.get("NS", 0.0),
            "S": m.get("S", 0.0),
            "ES_greedy": m.get("ES_greedy", 0.0),
            "PS_greedy": m.get("PS_greedy", 0.0),
            "NS_greedy": m.get("NS_greedy", 0.0),
            "S_greedy": m.get("S_greedy", 0.0),
            "kl_divergence": damage.get("kl_divergence", 0.0),
            "frob_abs": drift.get("total_absolute_frobenius", 0.0),
            "frob_rel": drift.get("total_relative_frobenius", 0.0),
            "mean_layer_rel": drift.get("mean_layer_relative_frobenius", 0.0),
            "generation": gen_text,
            "has_repetition": detect_repetition(gen_text),
        }

    static_rec = extract_scheme_record(schemes_res[0], static_layers, "static")
    telemetry_rec = extract_scheme_record(schemes_res[1], telemetry_layers, "telemetry")
    random_rec = extract_scheme_record(schemes_res[2], random_layers, "random")

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
        m = post.get("metrics") or {}
        drift = edit_data.get("weight_drift") or {}
        damage = edit_data.get("damage") or {}
        gen_text = post.get("generations", [""])[0] if post.get("generations") else ""

        profile_results[prof] = {
            "optimization": prof,
            "seconds": round(duration, 2),
            "ES": m.get("ES", 0.0),
            "PS": m.get("PS", 0.0),
            "NS": m.get("NS", 0.0),
            "S": m.get("S", 0.0),
            "ES_greedy": m.get("ES_greedy", 0.0),
            "PS_greedy": m.get("PS_greedy", 0.0),
            "NS_greedy": m.get("NS_greedy", 0.0),
            "S_greedy": m.get("S_greedy", 0.0),
            "kl_divergence": damage.get("kl_divergence", 0.0),
            "frob_abs": drift.get("total_absolute_frobenius", 0.0),
            "frob_rel": drift.get("total_relative_frobenius", 0.0),
            "mean_layer_rel": drift.get("mean_layer_relative_frobenius", 0.0),
            "generation": gen_text,
            "has_repetition": detect_repetition(gen_text),
        }
        print(f"      [{prof}] Done ({duration:.1f}s). PS={profile_results[prof]['PS']}, S={profile_results[prof]['S']}, Rel-Frob={profile_results[prof]['frob_rel']:.4f}", flush=True)

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
        "metadata": raw.get("metadata", {}),
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
                vals = [f[c][m] for f in exp1_facts if f[c][m] is not None]
                cond_stats[c][m] = {
                    "mean": round(float(np.mean(vals)), 4) if vals else 0.0,
                    "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
                }
            reps = sum(1 for f in exp1_facts if f[c].get("has_repetition"))
            cond_stats[c]["repetition_rate"] = round(reps / n1, 4)

        # Paired differences: Telemetry vs Static, Random vs Static, Telemetry vs Random
        paired_diffs = {}
        for metric in ["ES", "PS", "NS", "S", "ES_greedy", "PS_greedy", "frob_rel", "kl_divergence"]:
            tel_vals = [f["telemetry"][metric] for f in exp1_facts]
            stat_vals = [f["static"][metric] for f in exp1_facts]
            rand_vals = [f["random"][metric] for f in exp1_facts]

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
                "telemetry_beats_static_ES": paired_diffs["telemetry_vs_static_ES"]["mean_diff"] > 0,
                "telemetry_beats_static_S": paired_diffs["telemetry_vs_static_S"]["mean_diff"] > 0,
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
                vals = [f["profiles"][p][m] for f in exp2_facts if f["profiles"][p][m] is not None]
                prof_stats[p][m] = {
                    "mean": round(float(np.mean(vals)), 4) if vals else 0.0,
                    "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
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
            for metric in ["ES", "PS", "NS", "S", "ES_greedy", "PS_greedy", "frob_rel", "kl_divergence"]:
                a_vals = [f["profiles"][cond_a][metric] for f in exp2_facts]
                b_vals = [f["profiles"][cond_b][metric] for f in exp2_facts]
                paired_ablations[f"{label}_{metric}"] = paired_comparison(a_vals, b_vals)

        summary["experiment_2_optimization"] = {
            "num_facts": n2,
            "profile_aggregates": prof_stats,
            "paired_ablations": paired_ablations,
            "findings": {
                "context_beats_standard_PS": paired_ablations["total_gain_context_vs_standard_PS"]["mean_diff"] > 0,
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
        except Exception:
            pass

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
            print(f"   Static [13..17]: ES={rec['static']['ES']}, PS={rec['static']['PS']}, S={rec['static']['S']}, Rel-Frob={rec['static']['frob_rel']:.4f}")
            print(f"   Telemetry {rec['telemetry']['layers']}: ES={rec['telemetry']['ES']}, PS={rec['telemetry']['PS']}, S={rec['telemetry']['S']}, Rel-Frob={rec['telemetry']['frob_rel']:.4f}")
            print(f"   Random {rec['random']['layers']}: ES={rec['random']['ES']}, PS={rec['random']['PS']}, S={rec['random']['S']}, Rel-Frob={rec['random']['frob_rel']:.4f}")

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
            print(f"   Standard:        PS={p['standard']['PS']}, S={p['standard']['S']}, Rel-Frob={p['standard']['frob_rel']:.4f}, KL={p['standard']['kl_divergence']:.4f}")
            print(f"   Standard Budget: PS={p['standard_budget']['PS']}, S={p['standard_budget']['S']}, Rel-Frob={p['standard_budget']['frob_rel']:.4f}, KL={p['standard_budget']['kl_divergence']:.4f}")
            print(f"   Context No-Cons: PS={p['context_no_consistency']['PS']}, S={p['context_no_consistency']['S']}, Rel-Frob={p['context_no_consistency']['frob_rel']:.4f}, KL={p['context_no_consistency']['kl_divergence']:.4f}")
            print(f"   Context Full v3: PS={p['context']['PS']}, S={p['context']['S']}, Rel-Frob={p['context']['frob_rel']:.4f}, KL={p['context']['kl_divergence']:.4f}")

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
