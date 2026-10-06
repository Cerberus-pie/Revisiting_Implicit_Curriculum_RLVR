"""Focused learning comparisons from the verified seed-101 numeric archive.

No smoothing, resampling, or artificial noise. Every evaluation in each stated
display window is drawn; raw training fluctuations remain a separate diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np

from .plot_common import COLORS, rows, theme


def render(data_path, output):
    data_path, output = Path(data_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    theme()
    plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.spines.left": False, "axes.edgecolor": "#CAD1D6",
                         "xtick.major.size": 3, "ytick.major.size": 0})
    with zipfile.ZipFile(data_path) as archive:
        evaluations = {
            condition: rows(archive, f"runs/{condition}_seed101/eval.csv")
            for condition in ("mixed5_15_45", "fixed45")
        }

    def select(condition, length, lo=0, hi=40000):
        selected = sorted((r for r in evaluations[condition]
                           if r["length"] == length and lo <= r["step"] <= hi),
                          key=lambda r: r["step"])
        assert [r["step"] for r in selected] == list(range(lo, hi + 1, 50))
        return selected

    def decorate(ax, lo, hi):
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0"))
        ax.grid(axis="y", color="#E8ECEE", linewidth=.6)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    files = []

    def save(fig, name):
        for suffix in ("png",):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            files.append({"path": path.name,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)

    # Section 3 is a process diagnostic: the second panel magnifies the same
    # trajectories, never replaces them with an isolated checkpoint summary.
    metrics = (("attention_mass", "Correct attention mass", "#2878B5"),
               ("correct_transition_prob", "Local correct-transition probability", "#D68B32"),
               ("sampled_success", "Terminal success", "#27836B"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.2]})
    fig.subplots_adjust(left=.08, right=.985, top=.73, bottom=.16, wspace=.15)
    fig.legend(handles=[Line2D([0], [0], color=color, lw=1.3, label=label)
                        for _, label, color in metrics],
               loc="upper center", bbox_to_anchor=(.52, 1.015), ncol=3,
               frameon=False, fontsize=9, handlelength=2.2, columnspacing=1.5)
    fig.legend(handles=[Line2D([0], [0], color="#586771", lw=1.1, label="Mixed training"),
                        Line2D([0], [0], color="#586771", lw=1., linestyle=(0, (3, 2)),
                               label="Fixed L45 training")],
               loc="upper center", bbox_to_anchor=(.52, .932), ncol=2,
               frameon=False, fontsize=9, handlelength=3.)
    for index, (ax, bounds) in enumerate(zip(axes, ((0, 40000), (9000, 12000)))):
        lo, hi = bounds
        for condition in evaluations:
            data = select(condition, 45, lo, hi)
            for metric, _, color in metrics:
                ax.plot([r["step"] for r in data], [r[metric] for r in data],
                        color=color, linestyle="-" if condition.startswith("mixed") else (0, (3, 2)),
                        linewidth=1.1 if condition.startswith("mixed") else .85,
                        marker="." if index else None, markersize=1.8,
                        alpha=1 if condition.startswith("mixed") else .75)
        decorate(ax, lo, hi)
        ax.set_ylim(-.02, .80)
        ax.set_yticks([0, .2, .4, .6, .8])
        ax.set_xticks([0, 10000, 20000, 30000, 40000] if index == 0 else [9000, 10000, 11000, 12000])
        ax.set_title(("a   Full training trajectory" if index == 0 else "b   Transition detail: same curves"),
                     loc="left", fontsize=10.2, fontweight="semibold", pad=11)
        ax.set_xlabel("Training update", labelpad=7)
    axes[0].set_ylabel("L45 probability / attention mass")
    axes[0].axvspan(9000, 12000, color="#E6ECF0", alpha=.6, zorder=0)
    for step in (10000, 10500):
        axes[1].axvline(step, color="#B6C1C9", linewidth=.65, linestyle=":", zorder=0)
    save(fig, "figure2_internal")

    return files
