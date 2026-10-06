import itertools
import unittest

import torch
from torch.nn import functional as F

from src.model import Policy, StructuredHead, sample_tasks


class ModelChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)

    def test_structured_head_matches_dense_equation_8_and_input_gradient(self):
        d = 5
        fast = StructuredHead(d, c_b=4)
        dense = fast.dense().double()
        mass = torch.rand(7, d, dtype=torch.float64)
        mass = (mass / mass.sum(-1, keepdim=True)).requires_grad_()
        state = torch.arange(7) % d
        a, b = fast(mass, state), dense(mass, state)
        torch.testing.assert_close(a, b)
        # Compare gradients in the simplex tangent space, since the reduction
        # uses sum(mass)=1 and unconstrained ambient derivatives differ.
        weights = torch.randn_like(a)
        ga = torch.autograd.grad((a * weights).sum(), mass)[0]
        gb = torch.autograd.grad((b * weights).sum(), mass)[0]
        torch.testing.assert_close(ga - ga.mean(-1, keepdim=True), gb - gb.mean(-1, keepdim=True))

    def test_task_sampling_and_atomic_skill(self):
        gen = torch.Generator().manual_seed(3)
        perm = torch.randperm(8, generator=gen)
        task = sample_tasks(64, 5, 7, perm, gen)
        self.assertTrue((task.operations.sort(-1).values.diff(dim=-1) > 0).all())
        self.assertTrue((task.positions.sort(-1).values.diff(dim=-1) > 0).all())
        torch.testing.assert_close(task.queries, perm[task.positions])
        torch.testing.assert_close(task.target, (task.initial + task.operations.sum(-1)) % 7)
        g, y = torch.meshgrid(torch.arange(7), torch.arange(7), indexing="ij")
        logits = StructuredHead(7)(F.one_hot(g.flatten(), 7).float(), y.flatten())
        torch.testing.assert_close(logits.argmax(-1), (g.flatten() + y.flatten()) % 7)

    def test_exact_reward_and_reinforce_gradient_against_enumeration(self):
        d, length = 3, 2
        model = Policy(StructuredHead(d, 2), n_positions=4).double()
        with torch.no_grad():
            model.q.normal_(0, 0.2)
        gen = torch.Generator().manual_seed(9)
        task = sample_tasks(1, length, d, torch.tensor([2, 0, 3, 1]), gen)
        mass, _ = model.operation_mass(task)
        probs = (model.head.b * mass).softmax(-1)[0]
        expectation = 0.0
        score_objective = 0.0
        for increments in itertools.product(range(d), repeat=length):
            probability = torch.stack([probs[k, v] for k, v in enumerate(increments)]).prod()
            reward = float((task.initial.item() + sum(increments)) % d == task.target.item())
            expectation = expectation + reward * probability
            logprob = torch.stack([probs[k, v].log() for k, v in enumerate(increments)]).sum()
            score_objective = score_objective + reward * probability.detach() * logprob / length
        fft = model.exact_success(task).sum()
        torch.testing.assert_close(expectation, fft)
        enumerated = torch.autograd.grad(expectation / length, model.q, retain_graph=True)[0]
        score = torch.autograd.grad(score_objective, model.q)[0]
        fft_grad = torch.autograd.grad(fft / length, model.q)[0]
        torch.testing.assert_close(enumerated, score)
        torch.testing.assert_close(enumerated, fft_grad)

    def test_zero_advantage_gives_zero_reward_gradient(self):
        gen = torch.Generator().manual_seed(10)
        model = Policy(StructuredHead(7), 8)
        task = sample_tasks(32, 3, 7, torch.arange(8), gen)
        roll = model.rollout(task, generator=gen)
        grad = torch.autograd.grad((torch.zeros(32) * roll["mean_log_prob"]).mean(), model.q)[0]
        self.assertEqual(grad.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
