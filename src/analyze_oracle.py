"""Audit rewarded noncanonical paths and the frozen head's oracle capacity.

No policy training or cloud access. Archived rewards and full-path indicators
are nested events: a canonical full path necessarily has a correct endpoint.
The oracle calculation evaluates the actual frozen head with the correct
one-hot operation at every step; it is a counterfactual capacity check, not
a reconstruction of the trained mixed policy.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .train import make_policy


ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default=str(ROOT / 'data/measurements.zip'))
    parser.add_argument('--head', default=str(ROOT / 'model/transition_head.pt'))
    parser.add_argument('--output', default='reproduced/oracle.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        def table(name):
            return list(csv.DictReader(io.StringIO(archive.read(name).decode())))
        prefix = 'runs/mixed5_15_45_seed101/'
        training = table(prefix + 'train.csv')
        evaluation = table(prefix + 'eval.csv')
    config = json.loads((ROOT / 'configs/mixed5_15_45_seed101.json').read_text())
    expected = json.loads((ROOT / 'data/provenance.json').read_text())['frozen_head']['package_checkpoint_sha256']
    assert digest(args.head) == expected, 'Unexpected frozen head'
    final = {}
    for length in (5, 15, 45):
        row = next(r for r in evaluation if int(r['step']) == 40000 and int(r['length']) == length)
        terminal, path = float(row['greedy_success']), float(row['greedy_trajectory_correct'])
        assert 0 <= path <= terminal
        final[length] = {'terminal_success': terminal, 'full_path_success': path,
                         'rewarded_noncanonical_path': terminal - path,
                         'noncanonical_fraction_of_rewarded': (terminal - path) / terminal}
    windows = []
    for lo, hi in ((1, 10000), (10001, 12000), (12001, 20000), (20001, 40000)):
        selected = [r for r in training if lo <= int(r['step']) <= hi]
        by_length = {}
        for length in (5, 15, 45):
            key = f'by_length.{length}.'
            totals = {'n': 0, 'rewarded': 0, 'full_path': 0}
            for row in selected:
                n = int(row[key + 'n'])
                counts = [float(row[key + k]) * n for k in ('reward', 'trajectory_correct')]
                assert all(abs(v - round(v)) < 1e-4 for v in counts)
                rewarded, canonical = map(round, counts)
                assert 0 <= canonical <= rewarded <= n
                totals['n'] += n
                totals['rewarded'] += rewarded
                totals['full_path'] += canonical
            n, rewarded, canonical = totals.values()
            by_length[length] = {
                **totals,
                'reward_rate': rewarded / n,
                'full_path_rate': canonical / n,
                'noncanonical_fraction_of_rewarded': (rewarded - canonical) / rewarded,
                'mean_conditional_entropy': sum(float(r[key + 'n']) * float(r[key + 'entropy'])
                                                for r in selected) / n,
            }
        updates = np.array([float(r['parameter_update_norm']) for r in selected])
        windows.append({'updates': [lo, hi], 'by_length': by_length,
                        'parameter_update_norm_mean': float(updates.mean()),
                        'parameter_update_norm_min': float(updates.min())})

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    head = make_policy(config, args.device, args.head).head
    d = config['d']
    probabilities, correct = [], []
    with torch.no_grad():
        for start in range(0, d * d, 256):
            ids = torch.arange(start, min(start + 256, d * d), device=args.device)
            states, operations = ids // d, ids % d
            mass = torch.nn.functional.one_hot(operations, d).float()
            logits = head(mass, states)
            target = (states + operations) % d
            probabilities.extend(logits.softmax(-1).gather(-1, target[:, None]).squeeze(-1).cpu().tolist())
            correct.extend((logits.argmax(-1) == target).cpu().tolist())
    atomic = np.asarray(probabilities).reshape(d, d)
    rng = np.random.default_rng(20261006)
    oracle = {}
    for length in (5, 15, 45):
        operations = np.argsort(rng.random((8192, d)), axis=1)[:, :length]
        initial = rng.integers(d, size=8192)
        states = (initial[:, None] + np.c_[np.zeros(8192, dtype=int), operations[:, :-1].cumsum(axis=1)]) % d
        path_probability = atomic[states, operations].prod(axis=1)
        oracle[length] = {
            'sample_prompts': 8192,
            'mean_full_path_probability_with_oracle_attention': float(path_probability.mean()),
            'minimum_over_sample': float(path_probability.min()),
            'uniform_all_prompt_lower_bound': float(atomic.min() ** length),
            'greedy_full_path_success_with_oracle_attention': 1.0 if all(correct) else None,
        }
    output = {
        'source': 'src/analyze_oracle.py', 'source_sha256': digest(__file__),
        'numeric_archive_sha256': digest(args.archive), 'head_sha256': digest(args.head),
        'final_greedy_evaluation': final, 'training_windows': windows,
        'oracle_attention_check': {
            'device': args.device, 'torch': str(torch.__version__),
            'atomic_correct': sum(correct), 'atomic_total': len(correct),
            'atomic_min_correct_probability': float(atomic.min()),
            'atomic_mean_correct_probability': float(atomic.mean()),
            'per_length': oracle,
            'interpretation': 'The retained head can execute complete paths with correct attention. This tests oracle operation selection rather than soft operation mixtures in the trained policy.',
        },
        'supported': 'Many training trajectories receive terminal reward despite noncanonical intermediate states; this persists after the short-task transition. The frozen head has high full-path capacity under oracle attention.',
        'causal_status': 'Noncanonical rewarded paths, low entropy, and stable partial success are consistent with a saturated partial solution. Their co-occurrence does not establish why this attractor was reached or why it persists. No causal ablation was performed.',
        'training_updates_performed_by_this_audit': 0,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'final_greedy_evaluation': final, 'late_training': windows[-1],
                      'oracle': output['oracle_attention_check']}, indent=2))


if __name__ == '__main__':
    main()
