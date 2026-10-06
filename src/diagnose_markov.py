"""Offline same-MLP probability gradients and finite local transfer probes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .markov import markov_reward_gradient, markov_success
from .model import TaskBatch, sample_tasks
from .train import make_policy


def load_analysis_state(root, step):
    path = Path(root) / f"snapshot_{step:06d}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["step"]) != step:
                raise ValueError("Snapshot step mismatch")
            state = {key: torch.from_numpy(saved[key].copy()) for key in ("q", "permutation")}
        return path, state
    path = Path(root) / f"checkpoint_{step:06d}.pt"
    return path, torch.load(path, map_location="cpu", weights_only=True)


def slice_task(task, start, end):
    return TaskBatch(*(getattr(task, key)[start:end] for key in
                       ("operations", "positions", "queries", "initial", "target")))


def measure(model, task, microbatch=1, gradient=True, deadline=None):
    size = len(task.initial)
    total_gradient = torch.zeros_like(model.q)
    values = []
    for start in range(0, size, microbatch):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Diagnostic time budget exhausted')
        chunk = slice_task(task, start, start + microbatch)
        if gradient:
            probability, grad = markov_reward_gradient(model, chunk)
            total_gradient += grad * (len(chunk.initial) / size)
        else:
            with torch.no_grad():
                probability = markov_success(model, chunk)
        values.append(probability)
    values = torch.cat(values)
    return values.mean(), total_gradient, values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--output", required=True)
    schedule = p.add_mutually_exclusive_group()
    schedule.add_argument("--steps", type=int, nargs="+")
    schedule.add_argument("--interval", type=int, help="Uniform schedule through the completed run")
    p.add_argument("--gradients-only", action="store_true", help="Skip finite perturbations for dense gradient curves")
    p.add_argument("--prompts", type=int, default=16)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--epsilon", type=float, nargs="+", default=[0.01, 0.005])
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-seconds", type=float, required=True, help="Remaining diagnostic time budget after reserving export time")
    a = p.parse_args()
    if not 0 < a.max_seconds < float("inf") or min(a.prompts, a.repeats, a.microbatch) < 1 or min(a.epsilon) <= 0:
        raise ValueError("Positive probe counts and perturbations required")
    root, output = Path(a.run), Path(a.output)
    config = json.loads((root / "config.json").read_text())
    environment = json.loads((root / "environment.json").read_text())
    end = json.loads((root / "summary.json").read_text())["steps"]
    interval = a.interval if a.interval is not None else (100 if a.gradients_only else 2000)
    if interval <= 0:
        raise ValueError("Positive diagnostic interval required")
    steps = a.steps if a.steps is not None else sorted(set(range(0, end + 1, interval)) | {end})
    digest = hashlib.sha256(Path(a.head).read_bytes()).hexdigest()
    if digest != environment["head_sha256"]:
        raise ValueError("Diagnostic head differs from the trained frozen head")
    if config["head"] != "learned":
        raise ValueError("Use the state-dependent learned MLP")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = make_policy(config, a.device, a.head).double()
    started = time.monotonic()
    checkpoints, completed_steps = [], []
    deadline = started + a.max_seconds
    status, error = "running", None
    try:
        with (output / "measurements.jsonl").open("w", encoding="utf-8") as stream:
            for step in steps:
                path, saved = load_analysis_state(root, step)
                base = saved["q"].to(device=a.device, dtype=torch.float64)
                permutation = saved["permutation"].to(a.device)
                checkpoints.append({"step": step, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                for repeat in range(a.repeats):
                    with torch.no_grad():
                        model.q.copy_(base)
                    tasks, probabilities, gradients = {}, {}, {}
                    for length in (5, 15, 45):
                        seed = 9000000 + config["seed"] * 1000 + repeat * 100 + length
                        generator = torch.Generator(device=a.device).manual_seed(seed)
                        task = sample_tasks(a.prompts, length, model.d, permutation, generator)
                        probability, grad, per_prompt = measure(model, task, a.microbatch, deadline=deadline)
                        tasks[length], probabilities[length], gradients[length] = task, probability.item(), grad
                        row = {"kind": "gradient", "step": step, "repeat": repeat, "length": length,
                               "n_prompts": a.prompts, "probability": probability.item(),
                               "per_prompt_probabilities": per_prompt.tolist(),
                               "reward_gradient_norm": grad.norm().item(), "normalization": "mean_probability / length"}
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if a.gradients_only:
                        stream.flush()
                        print(json.dumps({"step": step, "repeat": repeat, "elapsed_seconds": time.monotonic() - started}), flush=True)
                        continue
                    for source in (5, 15):
                        norm = gradients[source].norm()
                        if not torch.isfinite(norm) or norm == 0:
                            stream.write(json.dumps({"kind": "transfer", "step": step, "repeat": repeat,
                                                     "source": source, "status": "zero_or_nonfinite_source_gradient"}) + "\n")
                            continue
                        direction = gradients[source] / norm
                        expected = (45 * gradients[45] * direction).sum().item()
                        denominator = norm * gradients[45].norm()
                        cosine = (gradients[source] * gradients[45]).sum().item() / denominator.item() if denominator > 0 else None
                        for epsilon in a.epsilon:
                            with torch.no_grad():
                                model.q.copy_(base + epsilon * direction)
                            plus = measure(model, tasks[45], a.microbatch, gradient=False, deadline=deadline)[0].item()
                            with torch.no_grad():
                                model.q.copy_(base - epsilon * direction)
                            minus = measure(model, tasks[45], a.microbatch, gradient=False, deadline=deadline)[0].item()
                            with torch.no_grad():
                                model.q.copy_(base)
                            finite = (plus - minus) / (2 * epsilon)
                            # This is a cancellation-resolution screen, not a confidence interval.
                            floor = 64 * torch.finfo(torch.float64).eps * max(plus, minus, probabilities[45]) / epsilon
                            relative = abs(finite - expected) / max(abs(expected), floor)
                            row = {"kind": "transfer", "step": step, "repeat": repeat, "source": source,
                                   "target": 45, "epsilon": epsilon, "gradient_cosine": cosine,
                                   "target_probability": probabilities[45], "positive_probability": plus,
                                   "negative_probability": minus, "positive_change": plus - probabilities[45],
                                   "negative_change": minus - probabilities[45], "directional_derivative": expected,
                                   "central_difference": finite, "relative_difference": relative,
                                   "finite_difference_resolved": abs(expected) > floor and relative <= 0.01}
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    print(json.dumps({"step": step, "repeat": repeat, "elapsed_seconds": time.monotonic() - started}), flush=True)
                completed_steps.append(step)
        status = "completed"
    except BaseException as exc:
        status = "time_limit" if isinstance(exc, TimeoutError) else "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps({"args": vars(a), "run_name": root.name,
            "seed": config["seed"], "status": status, "error": error, "measured_steps": completed_steps,
            "requested_steps": steps, "run_config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(), "head_sha256": digest,
            "checkpoints": checkpoints, "dtype": "float64", "method": "state_recursion_with_centered_adjoint",
            "scope": "all_trajectories_for_sampled_prompts; finite_local_transfer_not_Adam_attribution",
            "elapsed_seconds": time.monotonic() - started}, indent=2), encoding="utf-8")



if __name__ == "__main__":
    main()
