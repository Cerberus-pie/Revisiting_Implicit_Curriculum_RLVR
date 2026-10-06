"""Recompute report summaries from recorded data; no training or model loading."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from .analyze_coupling import summarize as summarize_coupling
from .analyze_plateau import summarize as summarize_plateau
from .plot_common import rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reproduced/analysis")
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "data/measurements.zip"
    result = {"report_version": 20, "runs": {}, "main_seed": 101,
              "completion_rule": "Greedy terminal success >= 0.99 at every trained length at every evaluation from update 39000 through 40000."}
    with zipfile.ZipFile(archive) as z:
        for path in sorted((ROOT / "configs").glob("*.json")):
            if path.stem == "pretraining":
                continue
            config = json.loads(path.read_text())
            ev = rows(z, f"runs/{path.stem}/eval.csv")
            final = {str(int(r["length"])): {k: r[k] for k in (
                "greedy_success", "greedy_trajectory_correct", "sampled_success",
                "attention_hit", "attention_mass", "correct_transition_prob", "entropy")}
                for r in ev if r["step"] == 40000}
            late = [r for r in ev if r["step"] >= 39000 and r["length"] in config["train_lengths"]]
            result["runs"][path.stem] = {"trained_lengths": config["train_lengths"], "final": final,
                "completed_relay": all(r["greedy_success"] >= .99 for r in late),
                "first_50_percent": {str(l): next((int(r["step"]) for r in ev
                     if r["length"] == l and r["greedy_success"] >= .5), None) for l in config["eval_lengths"]}}
        ev = rows(z, "runs/mixed5_15_45_seed101/eval.csv")
        result["internal_progress_L45"] = {str(s): {
            k: next(r[k] for r in ev if r["step"] == s and r["length"] == 45)
            for k in ("attention_mass", "correct_transition_prob", "sampled_success")}
            for s in (0, 10000, 10500, 40000)}
        grad = rows(z, "diagnostics/gradient_seed101/measurements.csv")
        result["gradient_peaks"] = {}
        for length in (5, 15, 45):
            g = {(int(r["step"]), int(r["repeat"])): r["reward_gradient_norm"]
                 for r in grad if r["kind"] == "gradient" and r["length"] == length}
            steps = list(range(0, 40001, 100))
            mean = np.array([[g[(s, r)] for r in range(3)] for s in steps]).mean(axis=1)
            peak = int(mean.argmax())
            result["gradient_peaks"][str(length)] = {
                "step": steps[peak], "mean_norm": float(mean[peak]), "final_mean_norm": float(mean[-1])}
    result["coupling"] = summarize_coupling(archive, output / "coupling.json")
    plateau = summarize_plateau(archive, output / "plateau.json")
    result["L15_probability_groups_at_40000"] = plateau["probability_groups"]["40000"]
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "runs": len(result["runs"]),
                      "gradient_peaks": result["gradient_peaks"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
