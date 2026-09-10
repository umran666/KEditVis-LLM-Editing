import unittest
import torch
from editing_optimizations import context_variance, context_weights, project_logits


class ContextOptimizationTests(unittest.TestCase):
    def test_context_groups_keep_equal_total_solve_weight(self):
        weights = context_weights([["{}"], ["a {}", "b {}", "c {}"]], "cpu").square()
        self.assertAlmostEqual(weights.sum().item(), 1., places=6)
        self.assertAlmostEqual(weights[0].item(), .5, places=6)
        self.assertAlmostEqual(weights[1:].sum().item(), .5, places=6)
        with self.assertRaises(ValueError):
            context_weights([[]], "cpu")

    def test_context_variance_is_zero_for_equal_contexts_and_has_gradients(self):
        hidden = torch.tensor([[[0., 0.], [2., 2.]]], requires_grad=True)
        loss = context_variance(hidden)
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertTrue(torch.equal(hidden.grad, torch.tensor([[[-.5, -.5], [.5, .5]]])))
        self.assertEqual(context_variance(torch.ones(2, 3, 4)).item(), 0.)
        self.assertEqual(context_variance(hidden.detach() + 50).item(), 1.)
        self.assertEqual(context_variance(hidden.detach().repeat(1, 3, 1)).item(), 1.)
        with self.assertRaises(ValueError):
            context_variance(torch.empty(0, 2, 3))

    def test_tied_embeddings_and_linear_heads_use_same_projection(self):
        embedding = torch.nn.Embedding(7, 3)
        linear = torch.nn.Linear(3, 7)
        hidden = torch.randn(2, 4, 3, requires_grad=True)
        self.assertTrue(torch.equal(project_logits(hidden, embedding), hidden @ embedding.weight.T))
        self.assertTrue(torch.equal(project_logits(hidden, linear), linear(hidden)))
        project_logits(hidden, embedding).sum().backward()
        self.assertIsNotNone(hidden.grad)


if __name__ == "__main__":
    unittest.main()
