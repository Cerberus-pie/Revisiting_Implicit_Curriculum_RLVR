"""Rebuild the eight published report figures from the curated measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import plot_terminal, plot_internal, plot_gradients, plot_updates, plot_seeds, plot_mixture

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("figure1_terminal_comparison", "figure2_internal", "figure3_reward_dynamics",
         "figure4_cross_length", "appendix_update_fluctuations", "figure6_seed_comparison",
         "figure7_attention_comparison", "figure8_ratio_control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reproduced/figures")
    args = parser.parse_args()
    archive = ROOT / "data/measurements.zip"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for module in (plot_terminal, plot_internal, plot_gradients, plot_updates, plot_seeds, plot_mixture):
        plt.rcdefaults()
        if module is plot_gradients:
            module.render(archive, output, fixed_data=archive)
        else:
            module.render(archive, output)
        print(f"Rendered {module.__name__}", flush=True)
    files = []
    for name in NAMES:
        path = output / f"{name}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / "figures.json").write_text(json.dumps({
        "report_version": 20, "smoothing": None, "downsampling": None,
        "measurements_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": files},
        indent=2) + "\n", encoding="utf-8")
    print(f"Rebuilt all {len(files)} report figures.")


if __name__ == "__main__":
    main()
