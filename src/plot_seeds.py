"""Compare all three frozen-implementation seeds without smoothing or selection."""
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

from .plot_common import COLORS, rows, theme


def render(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    theme()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.5), sharex=True,
                             sharey='row', layout='constrained')
    for col, seed in enumerate((101, 102, 103)):
        with zipfile.ZipFile(data) as z:
            ev = rows(z, f'runs/mixed5_15_45_seed{seed}/eval.csv')
            fixed = rows(z, f'runs/fixed45_seed{seed}/eval.csv')
            key = 'gradient_seed101' if seed == 101 else f'gradient_mixed5_15_45_seed{seed}'
            grad = rows(z, f'diagnostics/{key}/measurements.csv')
        for length in (5, 15, 45):
            e = sorted((r for r in ev if r['length'] == length), key=lambda r: r['step'])
            assert [r['step'] for r in e] == list(range(0, 40001, 50))
            axes[0, col].plot([r['step'] for r in e], [100*r['greedy_success'] for r in e],
                              color=COLORS[length], lw=1.15)
            g = {(int(r['step']), int(r['repeat'])): r['reward_gradient_norm']
                 for r in grad if r['kind'] == 'gradient' and r['length'] == length}
            steps = np.arange(0, 40001, 100)
            assert len(g) == 401*3
            norms = np.array([[g[(int(s), k)] for k in range(3)] for s in steps])
            axes[1, col].plot(steps, norms.mean(axis=1), color=COLORS[length], lw=.95)
            axes[1, col].fill_between(steps, norms.min(axis=1), norms.max(axis=1),
                                      color=COLORS[length], alpha=.12, lw=0)
        axes[0, col].plot([r['step'] for r in fixed], [100*r['greedy_success'] for r in fixed],
                          color='#777D83', ls='--', lw=.85)
        axes[0, col].set_title(f'{chr(97+col)}   Seed {seed}', loc='left', pad=10)
        axes[0, col].set_ylim(-3, 104)
        axes[0, col].set_yticks([0, 50, 100])
        axes[1, col].set_yscale('log')
        axes[1, col].set_ylim(1e-11, 1e-1)
        axes[1, col].set_yticks([1e-10, 1e-7, 1e-4, 1e-1])
        axes[1, col].set_xlabel('Training update')
        for ax in axes[:, col]:
            ax.set_xlim(0, 40000)
            ax.set_xticks([0, 10000, 20000, 30000, 40000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:g}k' if v else '0'))
            ax.grid(axis='y', color='#E5EAED', lw=.5)
    axes[0, 0].set_ylabel('Greedy terminal success (%)')
    axes[1, 0].set_ylabel('Mixed reward-gradient norm')
    handles = [Line2D([], [], color=COLORS[l], lw=1.5, label=f'L = {l}') for l in (5,15,45)]
    handles.append(Line2D([], [], color='#777D83', ls='--', lw=1, label='Fixed L45 (top row)'))
    fig.legend(handles=handles, loc='outside upper center', ncol=4, frameon=False)
    files = []
    for ext in ('png',):
        p = out / f'figure6_seed_comparison.{ext}'
        fig.savefig(p, dpi=320, bbox_inches='tight')
        files.append({'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(fig)
    return files
