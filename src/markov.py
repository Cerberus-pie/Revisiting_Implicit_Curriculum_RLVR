"""Terminal probabilities and gradients for the frozen, state-dependent MLP.

This integrates trajectories for fixed prompts. It does not average over every
possible prompt and is never used by the REINFORCE training update.
"""
import torch
from torch.nn import functional as F

from .model import AtomicMLP


def transition_matrix(head, mass):
    """Return T[batch, current_state, next_state], using the original ReLU head."""
    if not isinstance(head, AtomicMLP):
        raise TypeError("The state recursion requires the learned AtomicMLP")
    d, width = head.d, head.units_per_class
    # Factor the linear map on (mass, one_hot(state))/2. This avoids repeating
    # its dense multiply for all states; every class-specific neuron is kept.
    operation = F.linear(mass * 0.5, head.weight[:, :, :d].reshape(d * width, d))
    state = head.weight[:, :, d:].permute(2, 0, 1) * 0.5
    logits = (operation.reshape(-1, 1, d, width) + state[None]).relu().sum(-1)
    return logits.softmax(-1)


def markov_success(model, task):
    """Differentiable reference recursion, suitable for small validation cases."""
    mass, _ = model.operation_mass(task)
    probability = F.one_hot(task.initial, model.d).to(mass.dtype)
    for k in range(task.length):
        probability = torch.bmm(probability[:, None], transition_matrix(model.head, mass[:, k])).squeeze(1)
    return probability.gather(-1, task.target[:, None]).squeeze(-1)


def markov_reward_gradient(model, task):
    """P(success | prompt) and grad_Q mean(P)/L with bounded autograd memory.

    A forward state recursion and backward value recursion provide each local
    derivative. The backward value is centered at every step: a constant across
    next states has zero derivative through a row-stochastic transition matrix.
    Centering avoids carrying the large chance-level constant through softmax
    derivatives when the remaining signal is small.
    """
    mass, _ = model.operation_mass(task)
    batch, length, d = mass.shape
    forward, transitions = [], []
    with torch.no_grad():
        probability = F.one_hot(task.initial, d).to(mass.dtype)
        for k in range(length):
            forward.append(probability)
            transition = transition_matrix(model.head, mass[:, k])
            transitions.append(transition)
            probability = torch.bmm(probability[:, None], transition).squeeze(1)
        success = probability.gather(-1, task.target[:, None]).squeeze(-1)
        value = F.one_hot(task.target, d).to(mass.dtype)
        value = value - value.mean(-1, keepdim=True)
    mass_gradient = torch.zeros_like(mass)
    for k in reversed(range(length)):
        local_mass = mass[:, k].detach().requires_grad_()
        transition = transition_matrix(model.head, local_mass)
        coefficient = forward[k][:, :, None] * value[:, None, :]
        local = (transition * coefficient).sum() / (batch * length)
        mass_gradient[:, k] = torch.autograd.grad(local, local_mass)[0]
        with torch.no_grad():
            value = torch.bmm(transitions[k], value[:, :, None]).squeeze(-1)
            value = value - value.mean(-1, keepdim=True)
    gradient = torch.autograd.grad(mass, model.q, grad_outputs=mass_gradient)[0]
    return success.detach(), gradient.detach()
