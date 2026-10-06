"""Paper figures directly from the numeric ZIP, without models or PyTorch."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


COLORS = {5: "#2878B5", 15: "#D68B32", 45: "#27836B"}
CONDITIONS = ("mixed5_15_45", "fixed45")
STYLE = {"mixed5_15_45": ("#27836B", "-", "Mixed"), "fixed45": ("#7C668E", "--", "Fixed")}


def rows(archive, path):
    result = []
    for row in csv.DictReader(io.StringIO(archive.read(path).decode())):
        parsed = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value) if value else np.nan
            except ValueError:
                parsed[key] = value
        result.append(parsed)
    return result


def theme():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#A2AAAF", "axes.linewidth": .6,
                         "xtick.color": "#46515A", "ytick.color": "#46515A",
                         "text.color": "#263440", "axes.labelcolor": "#263440",
                         "axes.axisbelow": True, "pdf.fonttype": 42,
                         "path.simplify": False, "savefig.facecolor": "white"})


def axis(ax, ylabel, end, probability=True):
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Training update")
    ax.set_xlim(0, end)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k" if abs(x) >= 1000 else f"{x:g}"))
    ax.grid(axis="y", color="#E6EAED", lw=.55)
    if probability:
        ax.set_ylim(-.025, 1.025)
        ax.set_yticks([0, .5, 1])


def series(ax, data, metric, length=None, **style):
    selected = sorted((r for r in data if length is None or r.get("length") == length), key=lambda r: r["step"])
    x = [r["step"] for r in selected]
    y = [r.get(metric, np.nan) for r in selected]
    ax.plot(x, y, **style)


def crossing(data, length, level=.5):
    selected = sorted((r for r in data if r.get("length") == length), key=lambda r: r["step"])
    return next((r["step"] for r in selected if r["greedy_success"] >= level), None)
