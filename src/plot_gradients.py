"""Separate gradient timing from finite cross-length perturbations.

Reads verified numeric archives only. No smoothing, model loading or refitting.
An optional fixed-control archive supplies a second panel on the same axes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np

from .plot_common import COLORS, rows, theme


def diagnostics(path, run_name):
    gradients, transfers = {}, []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.startswith('diagnostics/') or not name.endswith('/measurements.csv'):
                continue
            meta = json.loads(z.read(name.replace('measurements.csv', 'manifest.json')))
            if meta['run_name'] != run_name:
                continue
            if meta['status'] != 'completed':
                raise ValueError('Incomplete diagnostics')
            for row in rows(z, name):
                if row['kind'] == 'gradient':
                    key = (int(row['step']), int(row['repeat']), int(row['length']))
                    if key in gradients:
                        assert row['reward_gradient_norm'] == gradients[key]['reward_gradient_norm']
                        assert row['per_prompt_probabilities'] == gradients[key]['per_prompt_probabilities']
                    gradients[key] = row
                elif row['kind'] == 'transfer' and row['epsilon'] == .01:
                    transfers.append(row)
    expected = {(s, r, l) for s in range(0, 40001, 100) for r in range(3) for l in (5, 15, 45)}
    assert set(gradients) == expected, 'Missing or duplicated gradient measurements'
    assert all(np.isfinite(r['reward_gradient_norm']) and r['reward_gradient_norm'] > 0
               for r in gradients.values())
    return gradients, transfers


def summarize(g):
    x = np.arange(0, 40001, 100)
    curves = {l: np.array([[g[(int(s), r, l)]['reward_gradient_norm']
                          for r in range(3)] for s in x]) for l in (5, 15, 45)}
    return x, curves


def render(data, output, fixed_data=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    theme()
    g, transfers = diagnostics(data, 'mixed5_15_45_seed101')
    runs = [('Mixed training', g)]
    archives = [{'role': 'original_fixed_and_mixed', 'path': str(Path(data).as_posix()),
                 'sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest()}]
    if fixed_data:
        fixed, _ = diagnostics(fixed_data, 'fixed45_seed101')
        for repeat in range(3):
            for length in (5, 15, 45):
                a, b = g[(0, repeat, length)], fixed[(0, repeat, length)]
                assert np.isclose(a['reward_gradient_norm'], b['reward_gradient_norm'], rtol=1e-10, atol=0)
                assert np.allclose(json.loads(a['per_prompt_probabilities']),
                                   json.loads(b['per_prompt_probabilities']), rtol=1e-10, atol=0)
        runs.append(('Fixed L = 45 training', fixed))
        archives.append({'role': 'fixed_gradient_repeat', 'path': str(Path(fixed_data).as_posix()),
                         'sha256': hashlib.sha256(Path(fixed_data).read_bytes()).hexdigest()})
    fig, axes = plt.subplots(len(runs), 1, figsize=(9.3, 3.45 if len(runs) == 1 else 5.5),
                             squeeze=False, sharex=True, sharey=True, layout='constrained')
    stats = {}
    for index, (title, group) in enumerate(runs):
        ax = axes[index, 0]
        steps, curves = summarize(group)
        stats[title] = {}
        for length, v in curves.items():
            m = v.mean(axis=1)
            ax.plot(steps, m, color=COLORS[length], lw=1.15, label=f'L = {length}')
            ax.fill_between(steps, v.min(axis=1), v.max(axis=1), color=COLORS[length], alpha=.13, lw=0)
            peak = int(np.argmax(m))
            stats[title][str(length)] = {'peak_step': int(steps[peak]), 'peak_mean_norm': float(m[peak]),
                                        'initial_mean_norm': float(m[0]), 'final_mean_norm': float(m[-1])}
        training_label = ('Trained on L = 5, 15, 45' if index == 0
                          else 'Trained on L = 45 only')
        ax.set_title((f'{chr(97 + index)}   ' if len(runs) > 1 else '') + training_label, loc='left', pad=9)
        ax.set_yscale('log')
        ax.set_ylabel('Reward gradient norm')
        ax.grid(axis='y', color='#E5EAED', lw=.55)
        ax.set_xlim(0, 40000)
        ax.set_xticks(np.arange(0, 40001, 5000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v else '0'))
        if index == 0:
            ax.legend(frameon=False, ncol=3, loc='upper right', title='Measured task length')
    axes[-1, 0].set_xlabel('Training update')
    files = []

    def save(fig, name):
        for suffix in ('png',):
            p = output / (name + '.' + suffix)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            files.append(p.name)
        plt.close(fig)

    save(fig, 'figure3_reward_dynamics')

    # One cell is one measured perturbation. The shared nonlinear color scale
    # retains both the tiny early effects and the isolated later responses.
    expected = {(s, r, l) for s in range(0, 40001, 2000) for r in range(3) for l in (5, 15)}
    indexed = {(int(r['step']), int(r['repeat']), int(r['source'])): r for r in transfers}
    assert len(indexed) == len(transfers) and set(indexed) == expected
    cmap = LinearSegmentedColormap.from_list('signed_effect', ['#AF4A39', '#FAFAF8', '#207968'])
    norm = SymLogNorm(linthresh=1e-8, linscale=.5, vmin=-2e-3, vmax=2e-3, base=10)
    # Put direction compatibility next to the measured probability response.
    # Both quantities were recorded in the original finite-perturbation run.
    fig = plt.figure(figsize=(9.3, 4.75), layout='constrained')
    grid = fig.add_gridspec(4, 2, height_ratios=[.60, 1, 1, .13], hspace=.14, wspace=.08)
    diagram = fig.add_subplot(grid[0, :])
    diagram.set_axis_off()
    labels = [(0.155, '1  Find a short-task ascent direction', '16 tasks at L5 or L15'),
              (.50, '2  Temporarily move the parameters', 'Same step length: 0.01'),
              (.845, '3  Measure the L45 response', 'Same 16 L45 tasks, before / after')]
    for x, label, equation in labels:
        diagram.text(x, .83, label, ha='center', va='center', fontsize=8.1, weight='bold', transform=diagram.transAxes)
        diagram.text(x, .32, equation, ha='center', va='center', fontsize=8.7, transform=diagram.transAxes)
    for x0, x1 in ((.30, .34), (.665, .705)):
        diagram.annotate('', xy=(x1, .33), xytext=(x0, .33), xycoords='axes fraction',
                         arrowprops={'arrowstyle': '->', 'color': '#85929A', 'lw': .9})
    meshes = []
    for ri, source in enumerate((5, 15)):
        for ci, metric in enumerate(('gradient_cosine', 'positive_change')):
            ax = fig.add_subplot(grid[ri + 1, ci])
            values = np.array([[indexed[(s, r, source)][metric] for s in range(0, 40001, 2000)]
                               for r in range(3)])
            assert np.isfinite(values).all()
            if ci == 0:
                assert (np.abs(values) <= 1 + 1e-12).all()
            mesh = ax.pcolormesh(np.arange(-1000, 42000, 2000), np.arange(-.5, 3.5, 1), values,
                                cmap=cmap, norm=plt.Normalize(-1, 1) if ci == 0 else norm,
                                edgecolors='white', linewidth=.55, shading='flat')
            if ri == 0:
                meshes.append(mesh)
            ax.set_ylim(2.5, -.5)
            ax.set_xlim(-1000, 41000)
            ax.set_yticks([0, 1, 2], ['A', 'B', 'C'])
            if ci == 0:
                ax.set_ylabel('Task set')
            metric_label = 'gradient alignment' if ci == 0 else 'change in L45 success'
            ax.set_title(f'{chr(97 + 2 * ri + ci)}   L{source} → L45: {metric_label}', loc='left', pad=6, fontsize=9.2)
            ax.set_xticks([0, 10000, 20000, 30000, 40000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v else '0'))
            ax.tick_params(length=0)
            if ri == 1:
                ax.set_xlabel('Training update')
            for spine in ax.spines.values():
                spine.set_visible(False)
            if source == 15:
                ax.add_patch(Rectangle((29000, .5), 2000, 1, fill=False, ec='#1D2A31', lw=1.1))
    for ci in (0, 1):
        cax = fig.add_subplot(grid[3, ci])
        ticks = [-1, -.5, 0, .5, 1] if ci == 0 else [-1e-3, -1e-6, 0, 1e-6, 1e-3]
        cb = fig.colorbar(meshes[ci], cax=cax, orientation='horizontal', ticks=ticks)
        cb.outline.set_visible(False)
        cb.ax.minorticks_off()
        if ci == 0:
            cb.set_label('Opposing  ←  orthogonal (0)  →  aligned', fontsize=8.5)
        else:
            cb.ax.set_xticklabels([r'$-10^{-3}$', r'$-10^{-6}$', '0', r'$10^{-6}$', r'$10^{-3}$'])
            cb.set_label('L45 success: decreases  ←  0  →  increases (symlog)', fontsize=8.5)
    save(fig, 'figure4_cross_length')
    return files
