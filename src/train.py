"""Sampled, length-normalized REINFORCE with separately recorded diagnostics."""
import argparse
import hashlib
import json
import math
import platform
import shutil
import signal
from pathlib import Path
import time

import numpy as np
import torch

from .model import AtomicMLP, Policy, StructuredHead, sample_tasks


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def make_policy(config, device, head_path=None):
    if config["head"] == "learned":
        if not head_path:
            raise ValueError("Learned-head runs require --head pointing to a pretrained checkpoint")
        saved = torch.load(head_path, map_location="cpu", weights_only=True)
        if saved["d"] != config["d"]:
            raise ValueError("Checkpoint state-space size differs from configuration")
        if saved["validation"]["accuracy_all_pairs"] < 1.0:
            raise ValueError("The atomic head has not mastered every one-step pair")
        if saved["validation"]["minimum_correct_probability"] < 0.95:
            raise ValueError("Atomic head confidence is insufficient for the current preflight criterion")
        head = AtomicMLP(saved["d"], saved["units_per_class"])
        head.load_state_dict(saved["state_dict"])
        scale = config.get("head_scale", 1.0)
        if scale <= 0:
            raise ValueError("head_scale must be positive")
        with torch.no_grad():
            head.weight.mul_(scale)
        head.freeze()
    elif config["head"] == "structured":
        head = StructuredHead(config["d"], config["c_b"])
    else:
        raise ValueError("Unknown head")
    return Policy(head, config["n_positions"]).to(device)


@torch.no_grad()
def evaluate(model, config, permutation, device, seed, step):
    rows = []
    for length in config["eval_lengths"]:
        # Fixed held-out probes and RNG, independent of training consumption.
        gen = torch.Generator(device=device).manual_seed(1000000 + seed * 1000 + length)
        acc = {}
        squared = {}
        count = 0
        for _ in range(config["eval_batches"]):
            task = sample_tasks(config["eval_batch_size"], length, config["d"], permutation, gen)
            greedy = model.rollout(task, greedy=True)
            sampled = model.rollout(task, generator=gen)
            metrics = {
                "greedy_success": greedy["reward"],
                "greedy_trajectory_correct": greedy["trajectory_correct"],
                "sampled_success": sampled["reward"],
                "correct_transition_prob": sampled["correct_transition_prob"],
                "attention_mass": sampled["attention_mass"],
                "attention_hit": sampled["attention_hit"],
                "entropy": sampled["entropy"],
            }
            if isinstance(model.head, StructuredHead):
                metrics["exact_success"] = model.exact_success(task)
            for key, val in metrics.items():
                acc[key] = acc.get(key, 0.0) + val.double().sum().item()
                squared[key] = squared.get(key, 0.0) + val.double().square().sum().item()
            count += len(task.initial)
        rows.append({"step": step, "length": length, "n_eval": count,
                     **{key: val / count for key, val in acc.items()},
                     "standard_errors": {key: math.sqrt(max(0.0, squared[key] - val * val / count)
                                                        / (count * (count - 1))) if count > 1 else None
                                         for key, val in acc.items()}})
    return rows


def gradient_diagnostics(model, config, permutation, device, seed, step):
    rows = []
    for length in config["train_lengths"]:
        gen = torch.Generator(device=device).manual_seed(2000000 + seed * 1000 + length)
        task = sample_tasks(config.get("diagnostic_batch_size", 512), length, config["d"], permutation, gen)
        rollout = model.rollout(task, generator=gen)
        # Raw reward estimator: avoids confusing a historical baseline with reward.
        reward_objective = (rollout["reward"] * rollout["mean_log_prob"]).mean()
        reward_grad = torch.autograd.grad(reward_objective, model.q, retain_graph=True)[0]
        entropy_grad = torch.autograd.grad(rollout["entropy"].mean(), model.q)[0]
        row = {"step": step, "length": length, "sampled_reward_grad_norm": reward_grad.norm().item(),
               "entropy_grad_norm": entropy_grad.norm().item(),
               "diagnostic_successes": int(rollout["reward"].sum().item()),
               "diagnostic_n": len(task.initial)}
        if isinstance(model.head, StructuredHead):
            exact_gradient = torch.autograd.grad(model.exact_success(task).mean() / length, model.q)[0]
            row["exact_reward_grad_norm"] = exact_gradient.norm().item()
        rows.append(row)
    return rows


def run(config, output, device="cuda", head_path=None, resume_path=None, stop_requested=None):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly pass --device cpu for CPU")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision(config.get("matmul_precision", "highest"))
    torch.manual_seed(config["seed"])
    model = make_policy(config, device, head_path)
    permutation_gen = torch.Generator(device=device).manual_seed(config.get("task_seed", 2026))
    permutation = torch.randperm(config["n_positions"], generator=permutation_gen, device=device)
    gen = torch.Generator(device=device).manual_seed(config["seed"] + 100)
    if config["optimizer"] == "adam":
        opt = torch.optim.Adam([model.q], lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        opt = torch.optim.SGD([model.q], lr=config["learning_rate"])
    else:
        raise ValueError("Unknown optimizer")
    started = time.monotonic()
    baseline = 0.0
    sequences, state_steps = 0, 0
    initial_step = 0
    resume_record = None
    if resume_path:
        resume_path = Path(resume_path).resolve(strict=True)
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        previous = saved["config"]
        # Evaluation has an independent RNG; changing its frequency is covered
        # by pipeline tests and does not alter the sampled training sequence.
        allowed_changes = {"steps", "max_seconds", "print_every", "label", "eval_every",
                           "diagnostic_every", "profile_timing", "save_update_every",
                           "checkpoint_every", "snapshot_every", "record_process_metrics"}
        for key in set(config) | set(previous):
            if key not in allowed_changes and config.get(key) != previous.get(key):
                raise ValueError(f"Resume changes training or evaluation setting: {key}")
        initial_step = saved["step"]
        if config["steps"] <= initial_step:
            raise ValueError("Resume target must exceed the saved update count")
        prior_environment = json.loads((resume_path.parent / "environment.json").read_text())
        digest = hashlib.sha256(Path(head_path).read_bytes()).hexdigest() if head_path else None
        if prior_environment["head_sha256"] != digest:
            raise ValueError("Resume uses a different frozen head")
        with torch.no_grad():
            model.q.copy_(saved["q"].to(device))
        opt.load_state_dict(saved["optimizer"])
        permutation = saved["permutation"].to(device)
        gen.set_state(saved["rng"].cpu())
        baseline = saved["baseline"]
        sequences, state_steps = saved["sequences"], saved["state_steps"]
        resume_record = {"checkpoint_name": resume_path.name,
                         "checkpoint_sha256": hashlib.sha256(resume_path.read_bytes()).hexdigest(),
                         "parent_run": resume_path.parent.name, "initial_step": initial_step,
                         "head_sha256": digest}
        write_json(root / "resume.json", resume_record)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    (root / "source").mkdir()
    for source_file in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source_file, root / "source" / source_file.name)
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu",
                   "platform": platform.platform(), "source_sha256": source_hashes,
                   "head_sha256": hashlib.sha256(Path(head_path).read_bytes()).hexdigest() if head_path else None}
    write_json(root / "config.json", config)
    write_json(root / "environment.json", environment)
    streams = {name: (root / f"{name}.jsonl").open("w", encoding="utf-8") for name in ("train", "eval", "gradients")}
    timing = {"training_seconds": 0.0, "evaluation_seconds": 0.0,
              "diagnostic_seconds": 0.0, "evaluation_calls": 0}

    def timestamp():
        if config.get("profile_timing") and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        return time.monotonic()

    def evaluate_at(step):
        tick = timestamp()
        for row in evaluate(model, config, permutation, device, config["seed"], step):
            append("eval", row)
        timing["evaluation_seconds"] += timestamp() - tick
        timing["evaluation_calls"] += 1

    def diagnose_at(step):
        tick = timestamp()
        for row in gradient_diagnostics(model, config, permutation, device, config["seed"], step):
            append("gradients", row)
        timing["diagnostic_seconds"] += timestamp() - tick

    def append(name, row):
        streams[name].write(json.dumps(row, allow_nan=False) + "\n")
        streams[name].flush()

    def checkpoint(step):
        path = root / f"checkpoint_{step:06d}.pt"
        temporary = path.with_name(path.name + '.tmp')
        torch.save({"q": model.q.detach().cpu(), "optimizer": opt.state_dict(), "baseline": baseline,
                    "step": step, "config": config, "permutation": permutation.cpu(),
                    "rng": gen.get_state(), "sequences": sequences, "state_steps": state_steps}, temporary)
        temporary.replace(path)

    def snapshot(step):
        # Cloud-side analysis state only: no optimizer, RNG or repeated frozen head.
        path = root / f"snapshot_{step:06d}.npz"
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, q=model.q.detach().cpu().numpy(),
                                permutation=permutation.cpu().numpy(), step=np.int64(step))
        temporary.replace(path)

    checkpoint_every = config.get("checkpoint_every", config["eval_every"])
    snapshot_every = config.get("snapshot_every", 0)
    if checkpoint_every < 1 or snapshot_every < 0:
        raise ValueError("Positive checkpoint interval and nonnegative snapshot interval required")
    status = "running"
    step = initial_step
    try:
        evaluate_at(initial_step)
        checkpoint(initial_step)
        if snapshot_every:
            snapshot(initial_step)
        for step in range(initial_step + 1, config["steps"] + 1):
            interrupted = stop_requested is not None and stop_requested()
            if interrupted or time.monotonic() - started >= config["max_seconds"]:
                step -= 1
                status = "interrupted" if interrupted else "time_limit"
                break
            tick = timestamp()
            choices = torch.randint(len(config["train_lengths"]), (config["batch_size"],), generator=gen, device=device)
            counts = torch.bincount(choices, minlength=len(config["train_lengths"])).cpu().tolist()
            loss = torch.zeros((), device=device)
            reward_mean = 0.0
            per_length = {}
            opt.zero_grad(set_to_none=True)
            for length, count in zip(config["train_lengths"], counts):
                if not count:
                    continue
                task = sample_tasks(count, length, config["d"], permutation, gen)
                roll = model.rollout(task, generator=gen)
                advantage = roll["reward"] - baseline
                part = -(advantage * roll["mean_log_prob"]).mean()
                part = part - config["entropy_bonus"] * roll["entropy"].mean()
                loss = loss + (count / config["batch_size"]) * part
                r = roll["reward"].mean().item()
                reward_mean += count / config["batch_size"] * r
                per_length[str(length)] = {"n": count, "reward": r}
                if config.get("record_process_metrics") and (step % config["log_every"] == 0 or step == 1):
                    for key in ("trajectory_correct", "correct_transition_prob", "attention_mass", "attention_hit", "entropy"):
                        per_length[str(length)][key] = roll[key].detach().mean().item()
                sequences += count
                state_steps += count * length
            loss.backward()
            grad_norm = model.q.grad.norm().item()
            if not torch.isfinite(model.q.grad).all():
                raise FloatingPointError("Non-finite attention gradient")
            save_every = config.get("save_update_every", 0)
            save_update = save_every > 0 and (step == 1 or (step - 1) % save_every == 0)
            record_update = config.get("record_process_metrics") and (step % config["log_every"] == 0 or step == 1)
            if save_update or record_update:
                before = model.q.detach().clone()
            opt.step()
            update_norm = (model.q.detach() - before).norm().item() if record_update else None
            timing["training_seconds"] += timestamp() - tick
            if save_update:
                torch.save({"from_step": step - 1, "to_step": step,
                            "q_before": before.cpu(), "q_after": model.q.detach().cpu(),
                            "total_loss_gradient": model.q.grad.detach().cpu(),
                            "baseline_before": baseline, "counts_by_length": per_length},
                           root / f"update_{step:06d}.pt")
            baseline = config["baseline_momentum"] * baseline + (1 - config["baseline_momentum"]) * reward_mean
            if step % config["log_every"] == 0 or step == 1:
                row = {"step": step, "reward": reward_mean, "loss": loss.item(), "baseline": baseline,
                       "gradient_norm": grad_norm, "elapsed_seconds": time.monotonic() - started,
                       "sequences": sequences, "state_steps": state_steps, "by_length": per_length}
                if record_update:
                    row["parameter_update_norm"] = update_norm
                    row["observation_model_step"] = step - 1
                append("train", row)
                if step % config.get("print_every", 100) == 0 or step == 1:
                    print(json.dumps(row), flush=True)
            if step % config["eval_every"] == 0 or step == config["steps"]:
                evaluate_at(step)
            if step % checkpoint_every == 0:
                checkpoint(step)
            if snapshot_every and (step % snapshot_every == 0 or step == config["steps"]):
                snapshot(step)
            diagnostic_every = config.get("diagnostic_every", config["eval_every"])
            if diagnostic_every > 0 and (step % diagnostic_every == 0 or step == config["steps"]):
                diagnose_at(step)
        else:
            status = "completed"
        if status in ("time_limit", "interrupted"):
            # Preserve the update immediately; do not spend the shutdown reserve on another evaluation.
            if snapshot_every:
                snapshot(step)
        checkpoint(step)
    except BaseException as exc:
        status = "failed"
        write_json(root / "error.json", {"type": type(exc).__name__, "message": str(exc), "step": step})
        raise
    finally:
        for stream in streams.values():
            stream.close()
        write_json(root / "summary.json", {"status": status, "steps": step,
                   "initial_step": initial_step,
                   "elapsed_seconds": time.monotonic() - started, "sequences": sequences,
                   "state_steps": state_steps,
                   "timing": timing,
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0})
    return root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--head")
    p.add_argument("--resume", help="Continue Q, optimizer, baseline and RNG from a saved checkpoint")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text(encoding="utf-8"))
    stopped = [False]
    def request_stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    root = run(config, a.output, a.device, a.head, a.resume, lambda: stopped[0])
    if json.loads((root / 'summary.json').read_text())['status'] != 'completed':
        raise SystemExit(2)


if __name__ == "__main__":
    main()
