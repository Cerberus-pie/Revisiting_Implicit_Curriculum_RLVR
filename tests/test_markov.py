import itertools
import unittest

import torch

from src.markov import markov_reward_gradient, markov_success, transition_matrix
from src.model import AtomicMLP, Policy, sample_tasks
from src.diagnose_markov import measure


class MarkovChecks(unittest.TestCase):
    def test_diagnostic_deadline_stops_before_expensive_recursion(self):
        with self.assertRaisesRegex(TimeoutError, 'time budget'):
            measure(self.model, self.task, deadline=0)

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(73)
        self.model = Policy(AtomicMLP(3, 7).freeze(), 4).double()
        with torch.no_grad():
            self.model.q.normal_(0, 0.5)
            self.model.head.weight.mul_(12)
        self.task = sample_tasks(2, 2, 3, torch.tensor([2, 0, 3, 1]), torch.Generator().manual_seed(31))

    def test_transition_factorization_and_gradient_match_original_head(self):
        mass = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
        actual = transition_matrix(self.model.head, mass)
        repeated = mass[:, None, :].expand(-1, 3, -1)
        states = torch.arange(3)[None].expand(2, -1)
        expected = self.model.head(repeated, states).softmax(-1)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
        weights = torch.randn_like(actual)
        a = torch.autograd.grad((actual * weights).sum(), mass)[0]
        b = torch.autograd.grad((expected * weights).sum(), mass)[0]
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)

    def test_probabilities_and_gradients_against_all_trajectories(self):
        model, task = self.model, self.task
        mass, _ = model.operation_mass(task)
        reward, score = [], []
        for b in range(len(task.initial)):
            total, score_total = 0., 0.
            for path in itertools.product(range(3), repeat=task.length):
                current = task.initial[b:b + 1]
                probability, log_probability = 1., 0.
                for k, nxt in enumerate(path):
                    prob = model.head(mass[b:b + 1, k], current).softmax(-1)[0, nxt]
                    probability = probability * prob
                    log_probability = log_probability + prob.log()
                    current = torch.tensor([nxt])
                correct = float(path[-1] == task.target[b].item())
                total = total + correct * probability
                score_total = score_total + correct * probability.detach() * log_probability / task.length
            reward.append(total)
            score.append(score_total)
        exact = torch.stack(reward)
        grad = torch.autograd.grad(exact.mean() / task.length, model.q, retain_graph=True)[0]
        score_grad = torch.autograd.grad(torch.stack(score).mean(), model.q)[0]
        dp = markov_success(model, task)
        dp_grad = torch.autograd.grad(dp.mean() / task.length, model.q)[0]
        probability, adjoint = markov_reward_gradient(model, task)
        for value in (dp, probability):
            torch.testing.assert_close(value, exact, rtol=1e-12, atol=1e-14)
        for value in (score_grad, dp_grad, adjoint):
            torch.testing.assert_close(value, grad, rtol=1e-10, atol=1e-14)

    def test_directional_derivative_and_sampling_agree(self):
        model, task = self.model, self.task
        probability, grad = markov_reward_gradient(model, task)
        direction = torch.randn_like(model.q)
        direction /= direction.norm()
        base = model.q.detach().clone()
        expected = task.length * (grad * direction).sum()
        estimates = []
        with torch.no_grad():
            for epsilon in (1e-3, 5e-4):
                model.q.copy_(base + epsilon * direction)
                positive = markov_success(model, task).mean()
                model.q.copy_(base - epsilon * direction)
                negative = markov_success(model, task).mean()
                estimates.append((positive - negative) / (2 * epsilon))
            model.q.copy_(base)
        for estimate in estimates:
            torch.testing.assert_close(estimate, expected, rtol=2e-5, atol=1e-10)
        # Independently sample state transitions from the original MLP.
        with torch.no_grad():
            mass, _ = model.operation_mass(task)
            generator = torch.Generator().manual_seed(987)
            for b in range(2):
                n = 50000
                current = task.initial[b].expand(n).clone()
                for k in range(task.length):
                    probabilities = model.head(mass[b:b+1, k].expand(n, -1), current).softmax(-1)
                    current = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
                observed = (current == task.target[b]).double().mean()
                p = probability[b]
                self.assertLess(abs(observed - p).item(), (6 * (p * (1-p) / n).sqrt() + 1/n).item())


if __name__ == "__main__":
    unittest.main()
