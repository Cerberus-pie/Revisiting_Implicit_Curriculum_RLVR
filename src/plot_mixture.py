"""Show the same-seed mixture contrast and the R2 terminal/full-path gap."""
import hashlib
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np

from .plot_common import rows, theme
from pathlib import Path


def render(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data) as z:
        five = rows(z, 'runs/mixed5_10_20_40_45_seed101/eval.csv')
        conditions = [
            ('Fixed L45', rows(z, 'runs/fixed45_seed101/eval.csv'), '#7D8790'),
            ('Three-level mixture', rows(z, 'runs/mixed5_15_45_seed101/eval.csv'), '#D68B32'),
            ('Five-level mixture', five, '#27836B')]
    summary = {'final': {str(int(r['length'])): r for r in five if r['step'] == 40000}}
    theme()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.65))
    fig.subplots_adjust(left=.075, right=.98, bottom=.19, top=.78, wspace=.25)
    for label, ev, color in conditions:
        selected = sorted((r for r in ev if r['length'] == 45), key=lambda r: r['step'])
        assert [r['step'] for r in selected] == list(range(0, 40001, 50))
        axes[0].plot([r['step'] for r in selected], [r['greedy_success'] for r in selected],
                     lw=1.1, color=color, label=label)
    axes[0].set_title('a   L45 outcomes, seed 101', loc='left', pad=12, fontsize=10.5)
    axes[0].set_xlim(0, 40000)
    axes[0].set_xticks([0, 10000, 20000, 30000, 40000])
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k' if x else '0'))
    axes[0].set_xlabel('Training update')
    axes[0].set_ylabel('Greedy terminal success')
    axes[0].legend(loc='upper left', frameon=False, fontsize=8.5, handlelength=1.8)
    lengths = (5, 10, 15, 20, 40, 45)
    x = np.arange(len(lengths))
    for offset, key, label, color in [(-.18, 'greedy_success', 'Terminal', '#27836B'),
                                      (.18, 'greedy_trajectory_correct', 'Full path', '#8FB5AA')]:
        axes[1].bar(x + offset, [summary['final'][str(l)][key] for l in lengths],
                    width=.34, color=color, label=label, zorder=3)
    axes[1].set_xticks(x, ['5', '10', '15*', '20', '40', '45'])
    axes[1].set_xlabel('Task length L  (* evaluation only)')
    axes[1].set_title('b   Five-level mixture at update 40,000', loc='left', pad=12, fontsize=10.5)
    fig.legend(*axes[1].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.77, .98), ncol=2, frameon=False, fontsize=8.5)
    for ax in axes:
        ax.set_ylim(0, 1.035)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis='y', color='#E6EAED', lw=.55, zorder=0)
        ax.set_axisbelow(True)
    files = []
    for suffix in ('png',):
        p = out / f'figure8_ratio_control.{suffix}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        files.append({'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(fig)
    return files
