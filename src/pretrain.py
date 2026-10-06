"""Supervised pretraining of the atomic Z_d transition head."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .model import AtomicMLP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--d", type=int, default=96)
    p.add_argument("--units-per-class", type=int, default=96)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initialization", choices=["random", "operator_gated"], default="operator_gated")
    p.add_argument("--negative-slope", type=float, default=0.0,
                   help="Optional pretraining-only leaky ReLU; validation and RL always use ReLU")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--target-probability", type=float, default=0.995)
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "source").mkdir()
    for source_file in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source_file, root / "source" / source_file.name)
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision("high")
    model = AtomicMLP(a.d, a.units_per_class, a.initialization).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator(device=a.device).manual_seed(a.seed + 101)
    all_g = torch.arange(a.d, device=a.device).repeat_interleave(a.d)
    all_y = torch.arange(a.d, device=a.device).repeat(a.d)
    started = time.monotonic()
    rows = []
    with (root / "pretrain.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(1, a.steps + 1):
            g = torch.randint(a.d, (a.batch_size,), device=a.device, generator=gen)
            y = torch.randint(a.d, (a.batch_size,), device=a.device, generator=gen)
            logits = model(F.one_hot(g, a.d).float(), y, negative_slope=a.negative_slope)
            loss = F.cross_entropy(logits, (g + y) % a.d)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % a.eval_every == 0 or step == a.steps:
                correct, probabilities = [], []
                with torch.no_grad():
                    for j in range(0, len(all_g), 512):
                        gg, yy = all_g[j:j + 512], all_y[j:j + 512]
                        out = model(F.one_hot(gg, a.d).float(), yy)
                        target = (gg + yy) % a.d
                        correct.append((out.argmax(-1) == target).float())
                        probabilities.append(out.softmax(-1).gather(-1, target[:, None]).squeeze(-1))
                row = {"step": step, "loss": loss.item(), "accuracy_all_pairs": torch.cat(correct).mean().item(),
                       "mean_correct_probability": torch.cat(probabilities).mean().item(),
                       "minimum_correct_probability": torch.cat(probabilities).min().item(),
                       "elapsed_seconds": time.monotonic() - started}
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
                if row["accuracy_all_pairs"] == 1.0 and row["minimum_correct_probability"] >= a.target_probability:
                    break
    checkpoint = root / "head.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "d": a.d, "units_per_class": a.units_per_class,
                "args": vars(a), "validation": rows[-1]}, checkpoint)
    meta = {"config": vars(a), "validation": rows[-1], "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "torch": torch.__version__, "device": torch.cuda.get_device_name() if a.device == "cuda" else "cpu"}
    (root / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
