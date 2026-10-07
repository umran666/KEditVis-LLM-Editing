"""Offline regressions for benchmark provenance, controls and failure accounting."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import layer_selection
import prepare_benchmark as preparation
import run_experiments as experiments
import verify_manifest as verification


def source_record(case_id=0, subject="Subject"):
    return {
        "case_id": case_id,
        "requested_rewrite": {"relation_id": "P1", "subject": subject,
                              "prompt": "{} is located in", "target_new": {"str": "Rome"},
                              "target_true": {"str": "Paris"}},
        "paraphrase_prompts": [f"The home of {subject} is"],
        "neighborhood_prompts": ["Another landmark is located in"],
        "generation_prompts": [f"Tell me about {subject}"],
    }


def benchmark_manifest():
    return preparation.create_manifest(
        [source_record(0, "Development"), source_record(1, "Evaluation")],
        "a" * 64, "https://example.invalid/pinned-counterfact.json", eval_size=1)


def measured_arm(layers=None):
    return {"layers": layers or [13, 14, 15, 16, 17], "ES": 1.0, "PS": 0.5,
            "NS": 1.0, "S": 0.75, "kl_divergence": 0.0000024,
            "has_repetition": False, "status": "complete"}


class BenchmarkIntegrityTests(unittest.TestCase):
    def test_content_pin_handles_json_whitespace_escapes_and_unicode(self):
        records = [source_record(0, 'Fran\u00e7ois "quoted"'), source_record(1, "Evaluation")]
        encoded = json.dumps(records, ensure_ascii=False, indent=7).encode("utf-8")
        self.assertEqual(preparation.decode_counterfact(encoded, hashlib.sha256(encoded).hexdigest()), records)
        with self.assertRaisesRegex(ValueError, "source SHA-256"):
            preparation.decode_counterfact(encoded, "0" * 64)

    def test_manifest_rejects_changed_payload_and_fact_leakage(self):
        manifest = benchmark_manifest()
        before = deepcopy(manifest)
        preparation.validate_manifest(manifest)
        self.assertEqual(manifest, before)
        manifest["eval_facts"][0]["target_new"] = "Madrid"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            preparation.validate_manifest(manifest)
        manifest = benchmark_manifest()
        manifest["eval_facts"][0]["subject"] = " Development "
        manifest["metadata"]["sha256"] = preparation.manifest_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "underlying fact"):
            preparation.validate_manifest(manifest)

    def test_manifest_does_not_silently_shrink_requested_sample(self):
        with self.assertRaisesRegex(ValueError, "Requested 2"):
            preparation.create_manifest([source_record(0, "Development"), source_record(1, "Evaluation")],
                                        "a" * 64, "fixture", eval_size=2)

    def test_smoke_never_uses_evaluation_facts(self):
        manifest = benchmark_manifest()
        self.assertEqual(experiments.select_benchmark_facts(manifest, "smoke"), manifest["dev_facts"])
        for number in (0, -1, 2):
            with self.assertRaises(ValueError):
                experiments.select_benchmark_facts(manifest, "selection", number)

    def test_telemetry_rejects_missing_duplicate_nonfinite_and_unknown_model(self):
        signals = [{"layer": i, "cosine_similarity": 0.5} for i in range(48)]
        invalid = [signals[:-1], signals + [signals[0]],
                   [{**signal, "cosine_similarity": float("nan")} for signal in signals],
                   [{"layer": signal["layer"]} for signal in signals]]
        for values in invalid:
            with self.assertRaises(ValueError):
                layer_selection.select_layers_telemetry(values, "gpt2-xl", "memit")
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            layer_selection.get_static_preset("unknown-model", "memit")
        for size in (0, -1, 4, True):
            with self.assertRaises(ValueError):
                layer_selection.get_random_scheme("gpt2-xl", "memit", 42, size)

    def test_identity_and_resume_bind_model_build_dataset_and_planned_cases(self):
        manifest = benchmark_manifest()
        health = {"model": "gpt2-xl", "n_layers": 48, "editing_commit": "pinned-upstream",
                  "memit_optimizations": list(experiments.OPTIMIZATION_PROFILES),
                  "backend_source_sha256": {"modal_app.py": "a" * 64,
                                            "editing_optimizations.py": "b" * 64},
                  "model_revision": None}
        identity = experiments.experiment_identity("gpt2-xl", "http://fixture", "full", manifest,
                                                  manifest["eval_facts"], health)
        self.assertIsNone(identity["model_revision"])
        checkpoint = {"metadata": {"run_identity": identity}, "experiment_1_selection": [],
                      "experiment_2_optimization": []}
        experiments.validate_checkpoint(checkpoint, identity)
        for key, value in (("model", "EleutherAI/gpt-j-6B"), ("case_ids", [2]),
                           ("manifest_sha256", "c" * 64), ("backend_source_sha256", {})):
            changed = {**identity, key: value}
            with self.assertRaisesRegex(ValueError, "different or unrecorded"):
                experiments.validate_checkpoint(checkpoint, changed)
        with self.assertRaisesRegex(ValueError, "different or unrecorded"):
            experiments.validate_checkpoint({"metadata": {"mode": "full"}}, identity)
        health["backend_source_sha256"]["modal_app.py"] = None
        with self.assertRaisesRegex(ValueError, "actual deployed"):
            experiments.experiment_identity("gpt2-xl", "http://fixture", "full", manifest,
                                            manifest["eval_facts"], health)

    def test_checkpoint_does_not_retry_recorded_failures(self):
        identity = {"case_ids": [1]}
        checkpoint = {"metadata": {"run_identity": identity},
                      "experiment_1_selection": [{"case_id": 1, "status": "failed"}]}
        with self.assertRaisesRegex(ValueError, "failed or contaminated"):
            experiments.validate_checkpoint(checkpoint, identity)

    def test_failed_fact_is_saved_without_invented_metrics(self):
        raw = {"experiment_1_selection": []}
        fact = benchmark_manifest()["eval_facts"][0]
        def failing_runner(*args):
            raise RuntimeError("backend unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_results.json"
            with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
                experiments.execute_fact(failing_runner, fact, raw, "experiment_1_selection", path,
                                         "http://fixture", "gpt2-xl")
            saved = json.loads(path.read_text(encoding="utf-8"))["experiment_1_selection"][0]
            self.assertEqual(saved["status"], "failed")
            self.assertNotIn("ES", saved)
            self.assertIsNone(saved["rollback_verified"])

    def test_profile_runner_stops_immediately_when_baseline_changes(self):
        fact = benchmark_manifest()["eval_facts"][0]
        with patch.object(experiments, "run_probe", side_effect=[{"layer_signals": [1]},
                                                                 {"layer_signals": [2]}]), \
             patch.object(experiments, "post_with_retry", return_value={}) as editing:
            with self.assertRaises(experiments.ExperimentFailure) as caught:
                experiments.run_experiment_fact_optimization(fact, "http://fixture")
        self.assertEqual(editing.call_count, 1)
        self.assertFalse(caught.exception.record["rollback_verified"])

    def test_failed_compare_still_checks_baseline_restoration(self):
        fact = benchmark_manifest()["eval_facts"][0]
        signals = [{"layer": i, "cosine_similarity": 0.5} for i in range(48)]
        with patch.object(experiments, "run_probe", return_value={"layer_signals": signals}) as probing, \
             patch.object(experiments, "post_with_retry", side_effect=RuntimeError("edit failed")):
            with self.assertRaises(experiments.ExperimentFailure) as caught:
                experiments.run_experiment_fact_selection(fact, "http://fixture")
        self.assertEqual(probing.call_count, 2)
        self.assertTrue(caught.exception.record["rollback_verified"])

    def test_missing_generation_differs_from_measured_empty_generation(self):
        self.assertIsNone(experiments.extract_scheme_record({}, [17])["has_repetition"])
        self.assertFalse(experiments.extract_scheme_record({"generation": ""}, [17])["has_repetition"])

    def test_analysis_reports_missing_denominators_and_true_jaccard(self):
        raw = {"metadata": {"num_facts": 3}, "experiment_1_selection": [
            {"case_id": 1, "static": measured_arm([3, 4, 5, 6, 7, 8]),
             "telemetry": measured_arm([4, 5, 6, 7, 8, 9]), "random": measured_arm()},
            {"case_id": 2, "status": "failed", "rollback_verified": None}],
            "experiment_2_optimization": []}
        summary = experiments.analyze_results(raw)["experiment_1_selection"]
        efficacy = summary["condition_aggregates"]["static"]["ES"]
        self.assertEqual((efficacy["n_measured"], efficacy["n_missing"]), (1, 1))
        self.assertEqual(summary["num_planned_facts"], 3)
        self.assertEqual(summary["num_failed_facts"], 1)
        self.assertEqual(summary["policy_overlap"]["mean_layer_jaccard_overlap"], round(5 / 7, 4))
        self.assertGreater(summary["condition_aggregates"]["static"]["kl_divergence"]["mean"], 0)

    def test_analysis_rejects_duplicate_facts_and_nonfinite_pairs(self):
        with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
            experiments.analyze_results({"experiment_1_selection": [{"case_id": 1}, {"case_id": 1}]})
        with self.assertRaisesRegex(ValueError, "non-finite"):
            experiments.paired_comparison([1.0, float("nan")], [0.0, 1.0])
        self.assertFalse(experiments.finding_is_significant_improvement(
            {"mean_diff": -0.5, "statistically_significant": True}))

    def test_analyze_only_reads_frozen_input_and_writes_separate_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "historical" / "raw_results.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"metadata": {}, "experiment_1_selection": [],
                                          "experiment_2_optimization": []}), encoding="utf-8")
            before = source.read_bytes()
            output = Path(directory) / "new-analysis"
            with patch("sys.argv", ["run_experiments.py", "--mode", "analyze-only",
                                    "--raw-results", str(source), "--output-dir", str(output)]), \
                 patch.object(experiments, "check_health") as health:
                experiments.main()
            health.assert_not_called()
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue((output / "summary.json").exists())
            self.assertFalse(source.with_name("summary.json").exists())


class VerificationScopeTests(unittest.TestCase):
    def test_update_refuses_missing_required_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "snapshot.json"
            with patch.object(verification, "FILES_TO_HASH", [Path(directory) / "missing.py"]), \
                 patch("sys.argv", ["verify_manifest.py", "--update", "--manifest",
                                    str(Path(directory) / "old.json"), "--output", str(output)]):
                self.assertEqual(verification.main(), 1)
            self.assertFalse(output.exists())

    def test_update_records_hashes_without_runtime_or_historical_build_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "source.py"
            artifact.write_text("pass\n", encoding="utf-8")
            baseline = Path(directory) / "historical.json"
            baseline.write_text('{"status":"VERIFIED_PASS","file_sha256":{}}', encoding="utf-8")
            before = baseline.read_bytes()
            output = Path(directory) / "snapshot.json"
            with patch.object(verification, "FILES_TO_HASH", [artifact]), \
                 patch("sys.argv", ["verify_manifest.py", "--update", "--manifest", str(baseline),
                                    "--output", str(output)]):
                self.assertEqual(verification.main(), 0)
            snapshot = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["status"], "HASH_SNAPSHOT_ONLY")
            self.assertNotIn("verification_checklist", snapshot)
            self.assertEqual(baseline.read_bytes(), before)
            self.assertIn("unknown", snapshot["historical_measurements"]["producing_build_provenance"])


if __name__ == "__main__":
    unittest.main()
