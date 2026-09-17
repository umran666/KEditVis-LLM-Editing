"""CPU regressions exercising production functions without starting Modal."""
import ast
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from scipy import stats

SOURCE = Path(__file__).with_name("modal_app.py")


def _is_literal(node) -> bool:
    """True for expressions made only of literals (so exec'ing them cannot fail)."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return (all(_is_literal(key) for key in node.keys if key)
                and all(_is_literal(value) for value in node.values))
    if isinstance(node, ast.UnaryOp):
        return _is_literal(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_literal(node.left) and _is_literal(node.right)
    return False


def load_functions():
    """Loads modal_app.py's functions plus its imports and literal module constants.

    Only top-level functions, import statements and literal assignments are
    exec'd, so the real `modal.Image(...)` / `modal.App(...)` objects are never
    constructed. Imports matter because production functions reference
    module-level names (`json`, `MIN_TSNE_SAMPLES`); without them a missing name
    surfaces as a NameError that looks like a production bug rather than a
    harness gap.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            body.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            if _is_literal(node.value):
                body.append(node)
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    # Deliberate test doubles, applied last so they win over the real values.
    namespace.update(
        {"HF_CACHE_PATH": "/unused", "DAMAGE_REF_PROMPTS": ["reference text"], "MEMIT_COMMIT": "test"}
    )
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
        # 2 prompts -> 4 points, below the t-SNE floor: labelling that a "joint-tsne"
        # projection would present an arbitrary layout as a meaningful embedding.
        self.assertEqual(rows[0]["projection_method"], "pca-small-sample")
        one = self.ns["_project_drift"](before[:1], after[:1])
        self.assertEqual(one[0]["projection_method"], "pca-small-sample")
        # 4 prompts -> 8 points, enough for a real joint embedding.
        big_before = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
        big_after = big_before + 0.01
        self.assertEqual(
            self.ns["_project_drift"](big_before, big_after)[0]["projection_method"], "joint-tsne"
        )
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

    def test_pre_and_post_metrics_use_identical_neighborhood_targets(self):
        """Regression: /edit and /compare computed pre-edit metrics without
        neighborhood_targets but post-edit metrics with them, so pre/post NS were
        measured against different reference strings and could not be compared."""
        client = self.make_web()
        seen = []

        def spy(model, tok, prompt, subject, target_new, target_true,
                paraphrase_prompts, neighborhood_prompts, neighborhood_targets=None):
            seen.append(list(neighborhood_targets) if neighborhood_targets is not None else None)
            return {"ES": 1.0, "PS": 1.0, "NS": 1.0, "S": 1.0,
                    "ES_greedy": 1.0, "PS_greedy": 1.0, "NS_greedy": 1.0, "S_greedy": 1.0,
                    "details": {"efficacy": [], "paraphrase": [], "neighborhood": []}}

        self.ns["_evaluate_edit"] = spy
        body = {"prompt": "{} relation", "subject": "Subject", "target_new": "New",
                "target_true": "Old", "layers": [2, 3],
                "neighborhood_prompts": ["Near one", "Near two"],
                "neighborhood_targets": ["T1", "T2"]}

        self.assertEqual(client.post("/edit", json=body).status_code, 200)
        self.assertEqual(seen, [["T1", "T2"], ["T1", "T2"]])

        seen.clear()
        self.assertEqual(client.post("/compare", json=body | {"schemes": [[2, 3]]}).status_code, 200)
        self.assertEqual(seen, [["T1", "T2"], ["T1", "T2"]])

    def test_health_advertises_every_accepted_optimization_profile(self):
        """Regression: /health listed four MEMIT profiles while the request models
        accepted five, hiding context_v3 from clients."""
        from editing_optimizations import OPTIMIZATION_PROFILES
        client = self.make_web()
        health = client.get("/health").json()
        self.assertEqual(set(health["memit_optimizations"]), set(OPTIMIZATION_PROFILES))
        self.assertIn("context_v3", health["memit_optimizations"])
        rejected = client.post("/edit", json={"prompt": "{} relation", "subject": "Subject",
                                             "target_new": "New", "optimization": "not_a_profile"})
        self.assertEqual(rejected.status_code, 422)

    def test_experiment_schemes_are_matched_by_layers_not_response_order(self):
        """Regression: /compare deduplicates identical schemes, so indexing the
        response by position attached results to the wrong policy (or raised
        IndexError) whenever two policies selected the same window."""
        from run_experiments import match_scheme_records

        static = [13, 14, 15, 16, 17]
        telemetry = [13, 14, 15, 16, 17]
        random_baseline = [40, 41, 42, 43, 44]
        # Deliberately returned in reverse order and with telemetry collapsed.
        returned = [
            {"layers": [40, 41, 42, 43, 44], "metrics": {"ES": 1.0}},
            {"layers": [13, 14, 15, 16, 17], "metrics": {"ES": 0.8}},
        ]

        matched = match_scheme_records(
            returned, {"static": static, "telemetry": telemetry, "random": random_baseline}
        )
        self.assertIs(matched["static"], matched["telemetry"])
        self.assertEqual(matched["static"]["layers"], [13, 14, 15, 16, 17])
        self.assertEqual(matched["random"]["layers"], [40, 41, 42, 43, 44])
        with self.assertRaises(RuntimeError):
            match_scheme_records(returned, {"missing": [0, 1, 2, 3, 4]})

    def test_harmonic_mean_is_undefined_for_empty_and_missing_input(self):
        """Regression: an empty input returned 0.0, reporting 'not measured' as a score."""
        self.assertIsNone(self.ns["_harmonic_mean"]([]))
        self.assertIsNone(self.ns["_harmonic_mean"]([1.0, None, 1.0]))
        # A zero component makes the harmonic mean exactly 0 -- correct, and the
        # reason S collapses whenever one component fails completely.
        self.assertEqual(self.ns["_harmonic_mean"]([1.0, 0.0, 1.0]), 0.0)

    def test_write_json_creates_missing_parent_directories(self):
        """The CLI writes into audit/development/, which may not exist on a fresh clone."""
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "audit" / "development" / "out.json"
            self.assertFalse(nested.parent.exists())
            self.ns["_write_json"](str(nested), {"layers": [13, 14]})
            self.assertTrue(nested.exists())
            self.assertEqual(json.loads(nested.read_text(encoding="utf-8")), {"layers": [13, 14]})
            # A bare filename (no directory component) must still work.
            plain = Path(tmp) / "plain.json"
            self.ns["_write_json"](str(plain), [])
            self.assertEqual(json.loads(plain.read_text(encoding="utf-8")), [])

    def test_kl_divergence_rejects_mismatched_inputs_without_assert(self):
        """Regression: validation used bare asserts, which `python -O` strips."""
        post = torch.tensor([[0.4, 0.6]]).log()
        pre = torch.tensor([[0.7, 0.3]]).log()
        with self.assertRaises(ValueError):
            self.ns["_kl_divergence"]([post], [pre, pre])
        with self.assertRaises(ValueError):
            self.ns["_kl_divergence"]([post], [torch.tensor([[0.1, 0.2, 0.7]]).log()])

    def test_parse_layers_rejects_mixed_spec_with_a_clear_error(self):
        """Regression: '8-12,15' produced a bare int() conversion error."""
        self.assertEqual(self.ns["_parse_layers"]("8-12"), [8, 9, 10, 11, 12])
        self.assertEqual(self.ns["_parse_layers"]("6,9,12"), [6, 9, 12])
        self.assertEqual(self.ns["_parse_layers"]("7"), [7])
        for spec, fragment in [("8-12,15", "cannot mix"), ("", "must not be empty"),
                               ("12-8", "start > end"), ("8-", "start-end"),
                               ("a,b", "comma-separated"), ("-5", "start-end")]:
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(ValueError, fragment):
                    self.ns["_parse_layers"](spec)

    def test_layer_zero_delta_variance_is_undefined_not_duplicated(self):
        """Regression: layer 0 reported its absolute variance as its residual delta."""
        class Block(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.mlp = torch.nn.Linear(dim, dim)

            def forward(self, hidden_states=None, **kwargs):
                return self.mlp(hidden_states)

        class ProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(n_layer=2)
                self.transformer = torch.nn.Module()
                self.transformer.h = torch.nn.ModuleList([Block(4), Block(4)])
                self.transformer.ln_f = torch.nn.Identity()
                self.lm_head = torch.nn.Linear(4, 8)

            def forward(self, input_ids, **kwargs):
                hidden = torch.ones(input_ids.shape[0], input_ids.shape[1], 4)
                for block in self.transformer.h:
                    hidden = block(hidden)
                return SimpleNamespace(logits=self.lm_head(self.transformer.ln_f(hidden)))

        class Batch(dict):
            def to(self, device):
                return self

        class Tok:
            def __call__(self, text, **kwargs):
                if kwargs.get("return_offsets_mapping"):
                    return Batch(offset_mapping=[(i, i + 1) for i in range(len(text))])
                return Batch(input_ids=torch.tensor([[1] * len(text)]))

            def decode(self, ids):
                return f"tok{ids[0]}"

        signals = self.ns["_probe_layers"](ProbeModel(), Tok(), "Subject", "Subject")
        self.assertEqual(len(signals), 2)
        self.assertIsNone(signals[0]["residual_delta_variance"])
        self.assertIsNotNone(signals[1]["residual_delta_variance"])
        self.assertIsNotNone(signals[0]["residual_variance"])

    def test_standard_optimization_reports_the_upstream_revision(self):
        """Regression: the `standard` branch was unreachable, so it reported a profile label."""
        client = self.make_web()
        body = {"prompt": "{} relation", "subject": "Subject", "target_new": "New", "layers": [2, 3]}
        standard = client.post("/edit", json=body)
        self.assertEqual(standard.status_code, 200, standard.text)
        self.assertEqual(standard.json()["optimization_config"]["revision"], "test")

        def context_apply(model, tok, requests, hp, **kwargs):
            return model, {}

        with patch("editing_optimizations.apply_context_memit", context_apply):
            context = client.post("/edit", json=body | {"optimization": "context"})
        self.assertEqual(context.status_code, 200, context.text)
        self.assertEqual(context.json()["optimization"], "context")
        self.assertEqual(context.json()["optimization_config"]["revision"], "context-v3")


class AnalysisRegressions(unittest.TestCase):
    """Regressions in the offline analysis layer (analyze_schemes / run_experiments)."""

    def test_spearman_is_correct_in_the_presence_of_ties(self):
        """Regression: the untied-only shortcut understated |rho| by ~2x on tied data."""
        from analyze_schemes import spearman_rank_correlation

        xs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        ys = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        expected = stats.spearmanr(xs, ys).statistic
        self.assertAlmostEqual(spearman_rank_correlation(xs, ys), expected, places=9)

        untied = ([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0])
        self.assertAlmostEqual(
            spearman_rank_correlation(*untied), stats.spearmanr(*untied).statistic, places=9
        )

    def test_spearman_is_undefined_for_constant_or_tiny_input(self):
        """Regression: zero variance returned a spurious non-zero coefficient."""
        from analyze_schemes import spearman_rank_correlation

        self.assertTrue(math.isnan(spearman_rank_correlation([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])))
        self.assertTrue(math.isnan(spearman_rank_correlation([1.0], [1.0])))
        with self.assertRaises(ValueError):
            spearman_rank_correlation([1.0, 2.0], [1.0])

    def test_scheme_activity_score_reports_out_of_range_schemes(self):
        """Regression: a fully out-of-range scheme raised a bare ZeroDivisionError."""
        from analyze_schemes import scheme_activity_score

        signals = [{"layer": i, "cosine_similarity": 0.5, "top_tokens": []} for i in range(28)]
        with self.assertRaisesRegex(ValueError, "no layer present"):
            scheme_activity_score(signals, [30, 31, 32])
        # Looked up by the recorded `layer` field, not by list position.
        reordered = [signals[5], signals[0], signals[9]]
        self.assertEqual(
            scheme_activity_score(reordered, [5])["mean_abs_cos_sim"],
            scheme_activity_score(reordered, [0])["mean_abs_cos_sim"],
        )

    def test_unmeasured_metrics_stay_none_through_aggregation(self):
        """Regression: metrics=None was coerced to 0.0 and averaged as a real zero."""
        from run_experiments import analyze_results, extract_scheme_record

        record = extract_scheme_record({"layers": [13, 14, 15, 16, 17], "metrics": None}, [13, 14, 15, 16, 17], "static")
        for key in ("ES", "PS", "NS", "S", "kl_divergence", "frob_rel"):
            self.assertIsNone(record[key], key)

        measured = extract_scheme_record(
            {"layers": [13, 14, 15, 16, 17], "metrics": {"ES": 1.0, "PS": 0.0, "NS": 1.0, "S": 0.0}},
            [13, 14, 15, 16, 17], "static",
        )
        summary = analyze_results({
            "experiment_1_selection": [{"case_id": 1, "static": measured, "telemetry": record, "random": record}],
            "experiment_2_optimization": [],
        })["experiment_1_selection"]["condition_aggregates"]
        self.assertEqual(summary["static"]["ES"]["mean"], 1.0)
        # The unmeasured arms contribute nothing rather than a phantom 0.0.
        self.assertIsNone(summary["telemetry"]["ES"]["mean"])
        self.assertIsNone(summary["random"]["ES"]["mean"])

    def test_partial_checkpoints_do_not_crash_the_analysis(self):
        """Regression: f[c][m] raised KeyError on a checkpoint predating a metric."""
        from run_experiments import analyze_results

        partial = {"label": "static", "layers": [13, 14, 15, 16, 17], "generation": "",
                   "ES": 1.0, "PS": 0.5, "NS": 1.0, "S": 0.6}
        summary = analyze_results({
            "experiment_1_selection": [{"case_id": 1, "static": partial, "telemetry": partial, "random": partial}],
            "experiment_2_optimization": [],
        })
        self.assertEqual(summary["experiment_1_selection"]["condition_aggregates"]["static"]["ES"]["mean"], 1.0)

    def test_paired_comparison_uses_only_complete_pairs(self):
        """Regression: None entries flowed into the arithmetic as if measured."""
        from run_experiments import paired_comparison

        result = paired_comparison([1.0, None, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0])
        self.assertEqual(result["n_pairs"], 3)
        self.assertAlmostEqual(result["mean_diff"], (1.0 + 0.0 + 0.0) / 3, places=4)

        # Too few complete pairs -> undefined statistics, not invented ones.
        empty = paired_comparison([None, None], [1.0, 1.0])
        self.assertEqual(empty["n_pairs"], 0)
        self.assertIsNone(empty["mean_diff"])
        self.assertIsNone(empty["p_value_t"])
        self.assertIsNone(empty["statistically_significant"])

        # A constant non-zero difference has an undefined t statistic and effect size.
        degenerate = paired_comparison([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0] * 6)
        self.assertEqual(degenerate["mean_diff"], 1.0)
        self.assertIsNone(degenerate["p_value_t"])
        self.assertIsNone(degenerate["cohens_d"])

    def test_http_errors_fail_fast_with_the_backend_detail(self):
        """Regression: 4xx was retried 3x with 5s sleeps and the `detail` was discarded."""
        import json as _json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from run_experiments import post_with_retry

        hits = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Consume the request body before replying. Replying without reading
                # it lets the server close the socket while the client is still
                # writing, which Windows surfaces as ConnectionAbortedError (10053)
                # and would look like a transient failure worth retrying.
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                hits.append(1)
                body = _json.dumps({"detail": "Layer(s) [50] out of range."}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            # `requests` honours ambient proxy configuration, which would route this
            # loopback request through a proxy and intermittently fail with
            # ProxyError -- a *transient* error, so it would be retried and the
            # server would see more than one request. Force a direct connection so
            # the test measures retry policy rather than the machine's proxy setup.
            with patch.dict(os.environ, {"no_proxy": "127.0.0.1,localhost",
                                         "NO_PROXY": "127.0.0.1,localhost"}):
                with self.assertRaisesRegex(RuntimeError, "Layer\\(s\\) \\[50\\] out of range"):
                    post_with_retry(f"http://127.0.0.1:{server.server_port}", "/edit", {},
                                    timeout=5, max_retries=3)
            self.assertEqual(len(hits), 1, "a 4xx must not be retried")
        finally:
            server.shutdown()
            server.server_close()

    def test_context_profiles_reject_unknown_names(self):
        """Regression: an unknown profile silently ran context_v3 instead."""
        from types import SimpleNamespace
        from editing_optimizations import configure_context

        hp = SimpleNamespace(context_consistency=0.0, clamp_norm_factor=4.0, v_num_grad_steps=20)
        with self.assertRaisesRegex(ValueError, "Unknown optimization profile"):
            configure_context(hp, "typo_profile")
        self.assertEqual(hp.v_num_grad_steps, 20, "hp must be left untouched")


if __name__ == "__main__":
    unittest.main(verbosity=2)
