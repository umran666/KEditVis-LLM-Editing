"""CPU regressions exercising production functions without starting Modal."""
import ast
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

SOURCE = Path(__file__).with_name("modal_app.py")


def load_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in functions:
        node.decorator_list = []
    namespace = {"HF_CACHE_PATH": "/unused", "DAMAGE_REF_PROMPTS": ["reference text"], "MEMIT_COMMIT": "test"}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class TinyModel(torch.nn.Module):
    def __init__(self, name="gpt2-xl"):
        super().__init__()
        self.config = SimpleNamespace(n_layer=48 if name == "gpt2-xl" else 28)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.eye(2)) for _ in range(self.config.n_layer)])

    def to(self, *args, **kwargs):
        return self


class BackendAudit(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions()
        self.model = TinyModel()
        self.util = ModuleType("util")
        self.util.nethook = SimpleNamespace(get_parameter=lambda model, name: model.weights[int(name.split(".")[1])])
        self.modules = patch.dict(sys.modules, {"util": self.util})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_partial_algorithm_failure_restores_every_layer(self):
        for exception in [RuntimeError("solve failed"), KeyboardInterrupt()]:
            with self.subTest(exception=type(exception).__name__):
                def failing(model, *args, **kwargs):
                    with torch.no_grad():
                        model.weights[1].add_(10)
                        model.weights[2].mul_(0)
                    raise exception
                hp = SimpleNamespace(layers=[1, 2], rewrite_module_tmp="weights.{}")
                with self.assertRaises(type(exception)):
                    self.ns["_apply_with_rollback"](failing, self.model, None, [], hp)
                for layer in [1, 2]:
                    self.assertTrue(torch.equal(self.model.weights[layer], torch.eye(2)))

    def test_snapshot_is_independent_and_restores_success(self):
        def apply(model, *args, **kwargs):
            self.assertFalse(kwargs["return_orig_weights"])
            with torch.no_grad():
                model.weights[3].add_(7)
            return model, {}
        hp = SimpleNamespace(layers=[3], rewrite_module_tmp="weights.{}")
        edited, originals = self.ns["_apply_with_rollback"](apply, self.model, None, [], hp)
        self.assertTrue(torch.equal(edited.weights[3], torch.eye(2) + 7))
        self.ns["_restore_weights"](edited, originals)
        self.assertTrue(torch.equal(edited.weights[3], torch.eye(2)))

    def test_keyword_block_inputs_are_traceable_and_hooks_are_removed(self):
        class Block(torch.nn.Module):
            def forward(self, hidden_states, scale=1):
                return hidden_states * scale
        self.model.transformer = torch.nn.Module()
        block = Block()
        self.model.transformer.h = torch.nn.ModuleList([block])
        hidden = torch.tensor([[1., 2.]])
        hp = SimpleNamespace(layers=[0], rewrite_module_tmp="weights.{}")
        for fail in [False, True]:
            observed = []
            def apply(model, *args, **kwargs):
                handle = block.register_forward_hook(lambda module, inputs, output: observed.append(inputs[0]))
                try:
                    self.assertTrue(torch.equal(block(hidden_states=hidden, scale=3), hidden * 3))
                    self.assertTrue(torch.equal(block(hidden, scale=2), hidden * 2))
                    self.assertIs(observed[0], hidden)
                    self.assertIs(observed[1], hidden)
                    if fail:
                        raise RuntimeError("after keyword forward")
                    return model, {}
                finally:
                    handle.remove()
            if fail:
                with self.assertRaisesRegex(RuntimeError, "after keyword forward"):
                    self.ns["_apply_with_rollback"](apply, self.model, None, [], hp)
            else:
                self.ns["_apply_with_rollback"](apply, self.model, None, [], hp)
            self.assertFalse(block._forward_pre_hooks)
            self.assertFalse(block._forward_hooks)

    def test_schemes_paths_and_dtype(self):
        for layers in [[], [-1], [True], [2.3]]:
            with self.assertRaises(ValueError):
                self.ns["_normalize_scheme"](layers)
        self.assertEqual(self.ns["_normalize_scheme"]([3, 1, 3]), [1, 3])
        for method in ["rome", "memit"]:
            self.assertEqual(self.ns["_hparams_path"](method, "EleutherAI/gpt-j-6B"), f"hparams/{method.upper()}/EleutherAI_gpt-j-6B.json")
        self.assertEqual(self.ns["_model_dtype"]("EleutherAI/gpt-j-6B"), torch.float32)

    def test_both_context_caches_reset(self):
        memit = ModuleType("memit")
        memit.memit_main = SimpleNamespace(CONTEXT_TEMPLATES_CACHE=["old"])
        rome = ModuleType("rome")
        rome.rome_main = SimpleNamespace(CONTEXT_TEMPLATES_CACHE=["old"])
        with patch.dict(sys.modules, {"memit": memit, "memit.memit_main": memit.memit_main, "rome": rome, "rome.rome_main": rome.rome_main}):
            self.ns["_seed_memit_rng"]()
        self.assertIsNone(memit.memit_main.CONTEXT_TEMPLATES_CACHE)
        self.assertIsNone(rome.rome_main.CONTEXT_TEMPLATES_CACHE)

    def test_success_is_likelihood_comparison_not_greedy_accuracy(self):
        def score(model, tok, prefixes, *args):
            return [{"prefix": p, "target_new_nll": 2 if p == "neighbor" else 1,
                     "target_true_nll": 1 if p == "neighbor" else 2,
                     "target_new_correct": False, "target_true_correct": False} for p in prefixes]
        self.ns["_eval_prefix_targets"] = score
        metrics = self.ns["_evaluate_edit"](None, None, "{} relation", "Subject", "New", "Old", ["paraphrase"], ["neighbor"])
        self.assertEqual([metrics[k] for k in ["ES", "PS", "NS", "S"]], [1, 1, 1, 1])
        self.assertFalse(metrics["details"]["efficacy"][0]["target_new_correct"])
        self.assertIsNone(self.ns["_harmonic_mean"]([1, None, 1]))
        self.assertEqual(self.ns["_harmonic_mean"]([1, 0, 1]), 0)
        self.assertAlmostEqual(self.ns["_harmonic_mean"]([0.9, 0.6, 0.3]), 3 / (1 / 0.9 + 1 / 0.6 + 1 / 0.3))

    def test_projection_uses_measured_distances_and_is_deterministic(self):
        before = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        after = before + torch.tensor([[3., 4., 0.], [0., 0., 0.]])
        rows = self.ns["_project_drift"](before, after)
        self.assertEqual([r["hidden_state_drift"] for r in rows], [5, 0])
        self.assertEqual(rows, self.ns["_project_drift"](before, after))
        self.assertEqual(rows[0]["projection_method"], "joint-tsne")
        one = self.ns["_project_drift"](before[:1], after[:1])
        self.assertEqual(one[0]["projection_method"], "pca-small-sample")
        with self.assertRaises(ValueError):
            self.ns["_project_drift"](before, after[:1])
        with self.assertRaises(ValueError):
            self.ns["_project_drift"](before, after * float("nan"))

    def test_kl_and_neighborhood_use_actual_measurements(self):
        pre = torch.tensor([[0.7, 0.3]]).log()
        post = torch.tensor([[0.4, 0.6]]).log()
        expected = 0.4 * math.log(0.4 / 0.7) + 0.6 * math.log(0.6 / 0.3)
        self.assertAlmostEqual(self.ns["_kl_divergence"]([post], [pre]), expected, places=6)
        rows = self.ns["_neighborhood_report"](["Berlin relation"], (["Berlin unchanged"], [pre]), (["Berlin unchanged"], [post]))
        self.assertEqual(rows[0]["pre_text"], rows[0]["post_text"])
        self.assertNotIn("projection", rows[0])
        self.assertAlmostEqual(rows[0]["kl_divergence"], expected, places=6)

    def test_probe_hooks_are_removed_after_forward_failure(self):
        class BrokenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(n_layer=1)
                self.transformer = torch.nn.Module()
                block = torch.nn.Module()
                block.mlp = torch.nn.Linear(2, 2)
                self.transformer.h = torch.nn.ModuleList([block])
            def forward(self, **kwargs):
                raise RuntimeError("forward failed")
        class Batch(dict):
            def to(self, device):
                return self
        model = BrokenModel()
        self.ns["_find_subject_token_index"] = lambda *args: 0
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            self.ns["_probe_layers"](model, lambda *args, **kwargs: Batch(input_ids=torch.tensor([[1]])), "Subject", "Subject")
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_token_alignment_whitespace_padding_and_multitoken_targets(self):
        class Batch(dict):
            def to(self, device):
                return self
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                # Deliberately use actual combined-string character offsets.
                length = max(map(len, texts))
                return Batch(input_ids=torch.tensor([[ord(c) % 128 for c in t] + [0] * (length-len(t)) for t in texts]),
                             attention_mask=torch.tensor([[1] * len(t) + [0] * (length-len(t)) for t in texts]),
                             offset_mapping=torch.tensor([[(i, i+1) for i in range(len(t))] + [(0, 0)] * (length-len(t)) for t in texts]))
        class Predictor(TinyModel):
            def forward(self, input_ids, attention_mask):
                logits = torch.zeros((*input_ids.shape, 128))
                for row in range(input_ids.shape[0]):
                    for pos in range(input_ids.shape[1]-1):
                        logits[row, pos, input_ids[row, pos+1]] = 4
                return SimpleNamespace(logits=logits)
        result = self.ns["_eval_prefix_targets"](Predictor(), Tokenizer(), ["a  ", "longer prefix"], " New ", "Old place")
        self.assertEqual(len(result), 2)
        for row in result:
            self.assertTrue(row["target_new_correct"] and row["target_true_correct"])
            self.assertAlmostEqual(row["target_new_nll"], row["target_true_nll"], places=6)

    def make_web(self):
        self.loaded = []
        self.applied = []
        self.fail_load = False
        self.fail_apply = False
        self.fail_post = False
        self.paths = []
        def load(name, **kwargs):
            if self.fail_load:
                raise RuntimeError("load failed")
            model = TinyModel(name)
            self.loaded.append(model)
            return model
        def hp(path):
            self.paths.append(path)
            return SimpleNamespace(layers=[5] if "/ROME/" in path else [3, 4], rewrite_module_tmp="weights.{}", clamp_norm_factor=.75, v_num_grad_steps=20)
        def apply(model, tok, requests, hparams, **kwargs):
            self.applied.append(list(hparams.layers))
            self.assertTrue(all(torch.equal(w, torch.eye(2)) for w in model.weights))
            with torch.no_grad():
                for layer in hparams.layers:
                    model.weights[layer].add_(2)
            if self.fail_apply:
                raise RuntimeError("mid-apply failure")
            return model, {}
        def probe(model, *args):
            if self.fail_post and any(not torch.equal(w, torch.eye(2)) for w in model.weights):
                raise RuntimeError("post-evaluation failure")
            return []
        transformer = ModuleType("transformers")
        transformer.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load)
        transformer.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name: SimpleNamespace(eos_token="eos"))
        memit = ModuleType("memit")
        memit.MEMITHyperParams = SimpleNamespace(from_json=hp)
        memit.apply_memit_to_model = apply
        rome = ModuleType("rome")
        rome.ROMEHyperParams = SimpleNamespace(from_json=hp)
        rome.apply_rome_to_model = apply
        generate = ModuleType("util.generate")
        generate.generate_fast = lambda model, tok, prompts, **kwargs: [p + " actual output" for p in prompts]
        self.ns.update(_seed_memit_rng=lambda: None, _probe_layers=probe,
                       _hidden_states=lambda *args: {},
                       _generate_text=generate.generate_fast,
                       _damage_logprobs=lambda model, tok, prompts: [torch.tensor([[0.5, 0.5]]).log() for p in prompts],
                       _neighborhood_snapshot=lambda model, tok, prompts: ([p + " unchanged" for p in prompts], [torch.tensor([[0.5, 0.5]]).log() for p in prompts]))
        with patch.dict(sys.modules, {"transformers": transformer, "memit": memit, "rome": rome, "util.generate": generate}), patch("os.chdir"), patch.object(sys, "path", list(sys.path)):
            web = self.ns["web_app"]()
        return TestClient(web, raise_server_exceptions=False)

    def test_http_validation_and_rome_deduplication(self):
        client = self.make_web()
        body = {"prompt": "{} relation", "subject": "Subject", "target_new": "New"}
        for changes in [{"layers": []}, {"layers": [True]}, {"layers": [1.2]}, {"layers": [-1]}, {"method": "unknown"}, {"target_new": " "}, {"prompt": "{name}"}, {"prompt": "{} {}"}]:
            with self.subTest(changes=changes):
                self.assertIn(client.post("/edit", json=body | changes).status_code, [400, 422])
        self.assertEqual(self.applied, [])
        response = client.post("/compare", json=body | {"method": "rome", "schemes": [[3, 4], [4, 3], [5]], "neighborhood_prompts": ["Near relation"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.applied, [[3], [4], [5]])
        self.assertEqual(response.json()["schemes"][0]["neighborhood"][0]["pre_text"], "Near relation unchanged")
        self.assertTrue(all(torch.equal(w, torch.eye(2)) for w in self.loaded[-1].weights))

    def test_http_apply_and_post_failure_preserve_baseline(self):
        client = self.make_web()
        body = {"prompt": "{} relation", "subject": "Subject", "target_new": "New", "layers": [2, 3]}
        for flag in ["fail_apply", "fail_post"]:
            setattr(self, flag, True)
            self.assertEqual(client.post("/edit", json=body).status_code, 500)
            self.assertTrue(all(torch.equal(w, torch.eye(2)) for w in self.loaded[-1].weights))
            setattr(self, flag, False)
        self.assertEqual(client.post("/edit", json=body).status_code, 200)

    def test_context_profile_dispatch_keeps_evaluation_prompts_out_of_optimizer(self):
        client = self.make_web()
        body = {"prompt": "{} relation", "subject": "Subject", "target_new": "New", "optimization": "context", "layers": [2, 3], "paraphrase_prompts": ["EVALUATION_ONLY"]}
        calls = []
        def context_apply(model, tok, requests, hp, **kwargs):
            self.assertNotIn("EVALUATION_ONLY", repr(requests))
            self.assertEqual(hp.context_consistency, .01)
            self.assertEqual(hp.clamp_norm_factor, 1.5)
            self.assertEqual(hp.v_num_grad_steps, 40)
            calls.append(list(hp.layers))
            return model, {}
        with patch("editing_optimizations.apply_context_memit", context_apply):
            edit = client.post("/edit", json=body)
            self.assertEqual(edit.status_code, 200, edit.text)
            self.assertEqual(edit.json()["optimization"], "context")
            compare = client.post("/compare", json=body | {"schemes": [[2, 3], [3, 4]]})
            self.assertEqual(compare.status_code, 200, compare.text)
            self.assertEqual(compare.json()["optimization"], "context")
        self.assertEqual(calls, [[2, 3], [2, 3], [3, 4]])
        self.assertEqual(client.post("/edit", json=body | {"optimization": "unknown"}).status_code, 422)
        self.assertEqual(client.post("/edit", json=body | {"method": "rome", "layers": [2]}).status_code, 400)

    def test_model_load_failure_can_recover_and_gptj_uses_upstream_path(self):
        client = self.make_web()
        self.fail_load = True
        self.assertEqual(client.get("/health", params={"model": "EleutherAI/gpt-j-6B"}).status_code, 500)
        self.fail_load = False
        self.assertEqual(client.get("/health", params={"model": "gpt2-xl"}).status_code, 200)
        response = client.post("/edit", json={"prompt": "Subject relation", "subject": "Subject", "target_new": "New", "method": "rome", "model": "EleutherAI/gpt-j-6B"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("hparams/ROME/EleutherAI_gpt-j-6B.json", self.paths)


    def test_residual_variance_numerical_properties(self):
        # Shift invariance: Var(X + c) == Var(X)
        v1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = v1 + 100.0
        self.assertAlmostEqual(v1.var(unbiased=True).item(), v2.var(unbiased=True).item(), places=5)
        self.assertAlmostEqual(v1.var(unbiased=True).item(), 2.5, places=5)

        # Scale property: Var(c * X) == c^2 * Var(X)
        v3 = v1 * 3.0
        self.assertAlmostEqual(v3.var(unbiased=True).item(), 9.0 * 2.5, places=5)

    def test_compute_weight_drift_frobenius_exact(self):
        compute_drift = self.ns["_compute_weight_drift"]
        orig = {"weights.1": torch.eye(2), "weights.2": torch.eye(2)}
        drift_zero = compute_drift(self.model, orig)
        self.assertAlmostEqual(drift_zero["total_absolute_frobenius"], 0.0, places=5)
        self.assertAlmostEqual(drift_zero["total_relative_frobenius"], 0.0, places=5)

        delta = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with torch.no_grad():
            self.model.weights[1].add_(delta)
        drift = compute_drift(self.model, {"weights.1": torch.eye(2)})
        expected_abs = math.sqrt(30.0)
        orig_frob = math.sqrt(2.0)
        self.assertAlmostEqual(drift["total_absolute_frobenius"], expected_abs, places=4)
        self.assertAlmostEqual(drift["total_relative_frobenius"], expected_abs / orig_frob, places=4)
        with torch.no_grad():
            self.model.weights[1].sub_(delta)

    def test_layer_selection_policies_deterministic(self):
        from layer_selection import get_static_preset, get_random_scheme, select_layers_telemetry
        self.assertEqual(get_static_preset("gpt2-xl", "rome"), [17])
        self.assertEqual(get_static_preset("gpt2-xl", "memit"), [13, 14, 15, 16, 17])
        self.assertEqual(get_static_preset("EleutherAI/gpt-j-6B", "rome"), [5])
        self.assertEqual(get_static_preset("EleutherAI/gpt-j-6B", "memit"), [3, 4, 5, 6, 7, 8])

        r1 = get_random_scheme("gpt2-xl", "memit", seed=42)
        r2 = get_random_scheme("gpt2-xl", "memit", seed=42)
        self.assertEqual(r1, r2)
        self.assertEqual(len(r1), 5)

        mock_signals = [
            {"layer": i, "cosine_similarity": 0.9 if i != 10 else 0.1, "residual_variance": 1.0}
            for i in range(48)
        ]
        best_rome = select_layers_telemetry(mock_signals, "gpt2-xl", "rome")
        self.assertEqual(best_rome, [10])

        best_memit = select_layers_telemetry(mock_signals, "gpt2-xl", "memit")
        self.assertIn(10, best_memit)
        self.assertEqual(len(best_memit), 5)

    def test_per_example_neighborhood_evaluation(self):
        eval_fn = self.ns["_evaluate_edit"]

        mock_tok = SimpleNamespace(
            __call__=lambda texts, **kw: {
                "input_ids": torch.tensor([[1, 2, 3]] * len(texts)),
                "attention_mask": torch.ones(len(texts), 3),
                "offset_mapping": torch.tensor([[[0, 1], [1, 5], [5, 10]]] * len(texts)),
            },
            decode=lambda ids: "Paris",
        )

        with patch.dict(self.ns, {"_eval_prefix_targets": lambda *a, **k: [{"target_new_nll": 1.0, "target_true_nll": 2.0, "target_new_correct": True, "target_true_correct": False}]}):
            res = self.ns["_evaluate_edit"](
                self.model, mock_tok,
                prompt="{} is in", subject="Eiffel", target_new="Rome", target_true="Paris",
                paraphrase_prompts=["P1"],
                neighborhood_prompts=["N1", "N2"],
                neighborhood_targets=["T1", "T2"],
            )
            self.assertIn("ES_greedy", res)
            self.assertIn("PS_greedy", res)
            self.assertIn("NS_greedy", res)
            self.assertIn("S_greedy", res)

    def test_ablation_profiles_configuration(self):
        from editing_optimizations import OPTIMIZATION_PROFILES, configure_context
        self.assertIn("standard_budget", OPTIMIZATION_PROFILES)
        self.assertIn("context_no_consistency", OPTIMIZATION_PROFILES)
        self.assertIn("context_v3", OPTIMIZATION_PROFILES)

        hp = SimpleNamespace(context_consistency=0.5, clamp_norm_factor=1.0, v_num_grad_steps=10)
        configure_context(hp, "standard_budget")
        self.assertEqual(hp.context_consistency, 0.0)
        self.assertEqual(hp.clamp_norm_factor, 1.5)
        self.assertEqual(hp.v_num_grad_steps, 40)

        configure_context(hp, "context_no_consistency")
        self.assertEqual(hp.context_consistency, 0.0)
        self.assertEqual(hp.clamp_norm_factor, 1.5)
        self.assertEqual(hp.v_num_grad_steps, 40)

        configure_context(hp, "context_v3")
        self.assertEqual(hp.context_consistency, 0.01)
        self.assertEqual(hp.clamp_norm_factor, 1.5)
        self.assertEqual(hp.v_num_grad_steps, 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
