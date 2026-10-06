"""Equations (2)-(8) of arXiv:2602.14872v3, instantiated on Z_d.

The learned head implements the class-specific ReLU sums in Eq. (4).
The structured head implements Eq. (8) analytically, for verification and
an explicitly labelled theory-reference experiment, not an author checkpoint.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TaskBatch:
    operations: torch.Tensor
    positions: torch.Tensor
    queries: torch.Tensor
    initial: torch.Tensor
    target: torch.Tensor

    @property
    def length(self):
        return self.operations.shape[1]


def sample_tasks(size, length, d, permutation, generator):
    device = permutation.device
    if not 1 <= length < min(d + 1, len(permutation)):
        raise ValueError("Length exceeds distinct operation/identifier capacity")
    # Sorting independent continuous keys samples ordered subsets without replacement.
    operations = torch.rand(size, d, device=device, generator=generator).argsort(-1)[:, :length]
    positions = torch.rand(size, len(permutation), device=device, generator=generator).argsort(-1)[:, :length]
    initial = torch.randint(d, (size,), device=device, generator=generator)
    return TaskBatch(operations, positions, permutation[positions], initial,
                     (initial + operations.sum(-1)) % d)


class AtomicMLP(nn.Module):
    """One hidden layer; separate ReLU units per output class, summed as Eq. (4)."""
    def __init__(self, d=96, units_per_class=96, initialization="random"):
        super().__init__()
        self.d = d
        self.units_per_class = units_per_class
        self.weight = nn.Parameter(torch.randn(d, units_per_class, 2 * d) * 0.05)
        if initialization == "operator_gated":
            if units_per_class != d:
                raise ValueError("Operator-gated initialization uses d units per class")
            # Generic operation gating, identical for every output class. It
            # contains no addition table or correct output labels; those are
            # acquired exclusively from supervised one-step training.
            with torch.no_grad():
                self.weight[:, :, :d].fill_(-1.0)
                self.weight[:, :, d:].zero_()
                for g in range(d):
                    self.weight[:, g, g] = 1.0
        elif initialization != "random":
            raise ValueError("Unknown MLP initialization")

    def forward(self, operation_mass, state, negative_slope=0.0):
        state_onehot = F.one_hot(state, self.d).to(operation_mass.dtype)
        inputs = torch.cat((operation_mass, state_onehot), -1) * 0.5
        hidden = F.linear(inputs, self.weight.flatten(0, 1))
        activated = F.leaky_relu(hidden, negative_slope) if negative_slope else hidden.relu()
        return activated.reshape(*state.shape, self.d, self.units_per_class).sum(-1)

    def freeze(self):
        self.requires_grad_(False)
        return self


class StructuredHead(nn.Module):
    """Exact Eq. (8) head on one-hot state inputs and convex operation mixtures.

All neurons belonging to other current states are inactive. For the current
state, the g-neuron outputs B * mass[g] + sigma_0. The common sigma_0 cancels
under softmax; we retain it for direct equality tests against the dense MLP.
"""
    def __init__(self, d=96, c_b=8.0):
        super().__init__()
        self.d = d
        self.b = c_b * math.log(d)

    def forward(self, operation_mass, state):
        output = torch.arange(self.d, device=state.device)
        increments = (output - state[..., None]) % self.d
        return self.b * operation_mass.gather(-1, increments) + self.d ** -0.5

    def dense(self):
        head = AtomicMLP(self.d, self.d)
        with torch.no_grad():
            head.weight.fill_(-self.b)
            for y in range(self.d):
                for g in range(self.d):
                    j = (y + g) % self.d
                    head.weight[j, y, g] = self.b
                    head.weight[j, y, self.d + y] = self.b + 2 * self.d ** -0.5
        return head.freeze()


class Policy(nn.Module):
    def __init__(self, head, n_positions=64):
        super().__init__()
        self.head = head
        self.d = head.d
        # Q is stored transposed relative to the paper: [query id, key id].
        self.q = nn.Parameter(torch.zeros(n_positions, n_positions))

    def attention(self, task):
        return self.q[task.queries[:, :, None], task.positions[:, None, :]].softmax(-1)

    def operation_mass(self, task):
        attention = self.attention(task)
        mass = torch.zeros(*attention.shape[:2], self.d, device=attention.device,
                           dtype=attention.dtype)
        return mass.scatter_add(-1, task.operations[:, None, :].expand_as(attention), attention), attention

    def rollout(self, task, generator=None, greedy=False):
        mass, attention = self.operation_mass(task)
        if isinstance(self.head, StructuredHead):
            # Translation equivariance makes increment sampling independent of
            # the current state. This vectorizes the exact autoregressive policy.
            logits = self.head.b * mass
            log_probs = logits.log_softmax(-1)
            probs = log_probs.exp()
            if greedy:
                # Match argmax over STATE ids, including tie-breaking at Q=0.
                current = task.initial
                increments = []
                for k in range(task.length):
                    states = (torch.arange(self.d, device=mass.device)[None, :] - current[:, None]) % self.d
                    nxt = logits[:, k].gather(-1, states).argmax(-1)
                    increments.append((nxt - current) % self.d)
                    current = nxt
                increments = torch.stack(increments, -1)
            else:
                increments = torch.multinomial(probs.reshape(-1, self.d), 1, generator=generator).reshape(task.operations.shape)
            chosen = log_probs.gather(-1, increments[..., None]).squeeze(-1)
            generated = (task.initial[:, None] + increments.cumsum(-1)) % self.d
            entropy = -(probs * log_probs).sum(-1)
            true_next_prob = probs.gather(-1, task.operations[..., None]).squeeze(-1)
        else:
            current = task.initial
            generated, chosen, entropy, true_next_prob = [], [], [], []
            for k in range(task.length):
                logits = self.head(mass[:, k], current)
                log_probs = logits.log_softmax(-1)
                probs = log_probs.exp()
                nxt = logits.argmax(-1) if greedy else torch.multinomial(probs, 1, generator=generator).squeeze(-1)
                chosen.append(log_probs.gather(-1, nxt[:, None]).squeeze(-1))
                entropy.append(-(probs * log_probs).sum(-1))
                # Correct local transition from the model's actual current state.
                target_step = (current + task.operations[:, k]) % self.d
                true_next_prob.append(probs.gather(-1, target_step[:, None]).squeeze(-1))
                generated.append(nxt)
                current = nxt
            generated, chosen, entropy, true_next_prob = [torch.stack(v, -1) for v in
                                                         (generated, chosen, entropy, true_next_prob)]
        truth = (task.initial[:, None] + task.operations.cumsum(-1)) % self.d
        return {
            "reward": (generated[:, -1] == task.target).float(),
            "trajectory_correct": (generated == truth).all(-1).float(),
            "mean_log_prob": chosen.mean(-1),
            "entropy": entropy.mean(-1),
            "correct_transition_prob": true_next_prob.mean(-1),
            "attention_mass": attention.diagonal(dim1=-2, dim2=-1).mean(-1),
            "attention_hit": (attention.argmax(-1) == torch.arange(task.length, device=mass.device)).float().mean(-1),
        }

    def exact_success(self, task):
        """Diagnostic expectation over ALL trajectories, for structured Z_d head only.

        Not the REINFORCE training estimator. Circular convolution via FFT
        includes accidental successes, repeated operations and wrong paths.
        """
        if not isinstance(self.head, StructuredHead):
            raise ValueError("Exact convolution requires the equivariant structured head")
        mass, _ = self.operation_mass(task)
        probs = (self.head.b * mass).softmax(-1)
        spectrum = torch.fft.rfft(probs, dim=-1).prod(dim=1)
        terminal = torch.fft.irfft(spectrum, n=self.d, dim=-1)
        delta = (task.target - task.initial) % self.d
        return terminal.gather(-1, delta[:, None]).squeeze(-1)
