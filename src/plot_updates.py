"""Figure 5: observed per-update fluctuations, without smoothing."""
from pathlib import Path
import zipfile
import matplotlib.pyplot as plt
from .plot_common import CONDITIONS, STYLE, rows, series, axis, theme

def render(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    theme()
    end = 40000
    with zipfile.ZipFile(data) as z:
        train = {c: rows(z, f'runs/{c}_seed101/train.csv') for c in CONDITIONS}
    def pair(ax, metric, length=45, kind='train', **extra):
        for condition in CONDITIONS:
            color, ls, label = STYLE[condition]
            series(ax, train[condition], metric, length=length, color=color, ls=ls, label=label, **extra)
    def save(fig, name, title):
        fig.suptitle(f'{title} | Seed 101', fontsize=11, x=.08, ha='left')
        fig.savefig(output / f'{name}.png', dpi=260, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.2), sharex=True, layout="constrained")
    metrics = (("by_length.45.reward", "Minibatch success, L = 45"),
               ("gradient_norm", "Total loss gradient norm"),
               ("parameter_update_norm", "Parameter update norm"))
    for index, (metric, ylabel) in enumerate(metrics):
        pair(axes[index], metric, length=None, kind="train", lw=.45, alpha=.6, rasterized=True)
        axis(axes[index], ylabel, end, probability=index == 0)
        if index:
            axes[index].set_yscale("log")
        if index < 2:
            axes[index].set_xlabel("")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "appendix_update_fluctuations", "Observed training fluctuations")
