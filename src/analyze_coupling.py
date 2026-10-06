"""Summarize the timing and heterogeneity of the archived seed-101 probes.

Uses existing measurements only. Checkpoint windows are descriptive, selected
after the run; repeated measurements of the same task sets are not independent
replicates. Nothing is trained or replayed by this analysis.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile


def summarize(archive, output):
    archive = Path(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        records = list(csv.DictReader(io.StringIO(
            z.read('diagnostics/transfer_seed101/measurements.csv').decode())))
    rows = [{
        'step': int(r['step']), 'set': 'ABC'[int(r['repeat'])],
        'source': int(r['source']), 'target': int(r['target']),
        'cosine': float(r['gradient_cosine']),
        'delta_probability': float(r['positive_change']),
        'target_probability': float(r['target_probability']),
        'perturbed_target_probability': float(r['positive_probability']),
    } for r in records if r['kind'] == 'transfer' and float(r['epsilon']) == .01]
    expected = {(step, task_set, source) for step in range(0, 40001, 2000)
                for task_set in 'ABC' for source in (5, 15)}
    assert len(rows) == 126
    assert {(r['step'], r['set'], r['source']) for r in rows} == expected
    assert all(r['target'] == 45 for r in rows)

    def summarize(subset):
        values = [r['delta_probability'] for r in subset]
        return {
            'cells': len(subset),
            'positive_cells': sum(x > 0 for x in values),
            'median_cosine': statistics.median(r['cosine'] for r in subset),
            'median_delta_probability': statistics.median(values),
            'min_delta_probability': min(values), 'max_delta_probability': max(values),
            'by_set': {task_set: {
                'cells': sum(r['set'] == task_set for r in subset),
                'positive_cells': sum(r['set'] == task_set and r['delta_probability'] > 0
                                      for r in subset),
            } for task_set in 'ABC'},
        }

    phases = []
    for start, end in ((0, 10000), (12000, 20000), (22000, 40000)):
        phases.append({'start': start, 'end': end, 'sources': {
            str(source): summarize([r for r in rows if start <= r['step'] <= end
                                     and r['source'] == source]) for source in (5, 15)}})
    largest = max((r for r in rows if r['source'] == 15), key=lambda r: r['delta_probability'])
    assert largest['step'] == 30000 and largest['set'] == 'B'
    assert phases[1]['sources']['15']['positive_cells'] == 8
    assert phases[2]['sources']['15']['positive_cells'] == 22
    assert phases[2]['sources']['5']['positive_cells'] == 14
    result = {
        'source': 'src/analyze_coupling.py',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'archive': 'data/measurements.zip', 'archive_sha256': digest,
        'epsilon': .01, 'direction': 'unit source reward gradient',
        'target': 45, 'phases': phases,
        'overall': {str(source): summarize([r for r in rows if r['source'] == source])
                    for source in (5, 15)},
        'largest_L15_response': largest,
        'interpretation': 'Positive L15-to-L45 responses are more frequent at later sampled checkpoints; responses vary across prompt sets and in magnitude.',
        'window_selection': 'Descriptive split chosen after the run; all 21 checkpoints are included exactly once.',
        'sampling_unit': 'Three fixed prompt sets measured repeatedly; cells are not independent experiment replicates.',
        'training_updates_performed': 0,
    }
    Path(output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result
