"""Separate terminal outcomes from operation-selection diagnostics; no smoothing."""
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .plot_common import COLORS, rows, theme


def render(data, out):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data) as z:
        evaluations = {c: rows(z, f'runs/{c}_seed101/eval.csv')
                       for c in ('fixed45', 'mixed5_15_45')}
    theme()
    files = []
    for metric, name, ylabel in (
            ('greedy_success', 'figure1_terminal_comparison', 'Greedy terminal success'),
            ('attention_hit', 'figure7_attention_comparison', 'Peak attention-hit rate')):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.75), sharex=True, sharey=True)
        fig.subplots_adjust(left=.085, right=.98, bottom=.18, top=.77, wspace=.15)
        fig.legend(handles=[Line2D([0], [0], color=COLORS[l], lw=1.3, label=f'L = {l}')
                            for l in (5, 15, 45)], loc='upper center', ncol=3,
                   frameon=False, bbox_to_anchor=(.54, 1), fontsize=10)
        for ax, condition, title in zip(axes, evaluations,
                ('a   Fixed L45 training', 'b   Mixed L5/15/45 training')):
            for length in ((45,) if condition == 'fixed45' else (5, 15, 45)):
                values = sorted((r for r in evaluations[condition] if r['length'] == length),
                                key=lambda r: r['step'])
                if [r['step'] for r in values] != list(range(0, 40001, 50)):
                    raise ValueError('Incomplete evaluation grid')
                ax.plot([r['step'] for r in values], [r[metric] for r in values],
                        color=COLORS[length], lw=1.05)
            ax.set_title(title, loc='left', pad=11, fontsize=10.5)
            ax.set_xlim(0, 40000)
            ax.set_ylim(-.025, 1.035)
            ax.set_xticks([0, 10000, 20000, 30000, 40000])
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k' if x else '0'))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.grid(axis='y', color='#E6EAED', lw=.55)
            ax.set_xlabel('Training update')
        axes[0].set_ylabel(ylabel)
        for suffix in ('png',):
            p = out / f'{name}.{suffix}'
            fig.savefig(p, dpi=300, bbox_inches='tight')
            files.append({'path': p.name,
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        plt.close(fig)
    return files
