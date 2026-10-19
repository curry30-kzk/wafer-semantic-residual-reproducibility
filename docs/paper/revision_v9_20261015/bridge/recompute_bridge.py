"""Recompute bridge statistics from 80 small confusion-matrix JSONs.

Python standard library only. Does not import torch, numpy, model code, original
caches, raw images, or .npz arrays. Query labels are represented only by the saved
confusion matrices. --root may point at the extracted package's bridge results.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics

T_CRITICAL_DF9_975 = 2.2621571628540993
# Independent stdlib arithmetic and a published df=9 t critical value can differ
# in the last digits from SciPy's version-specific inverse-CDF calculation.
TOLERANCE = 1e-8


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def matrix_metrics(matrix):
    count = len(matrix)
    if count < 2 or any(len(row) != count for row in matrix):
        raise ValueError('confusion matrix must be square')
    if any(not isinstance(v, int) or v < 0 for row in matrix for v in row):
        raise ValueError('confusion counts must be nonnegative integers')
    true_counts = [sum(row) for row in matrix]
    pred_counts = [sum(matrix[i][j] for i in range(count)) for j in range(count)]
    total = sum(true_counts)
    if not total:
        raise ValueError('empty query confusion matrix')
    diagonal = [matrix[i][i] for i in range(count)]
    f1 = [2 * tp / (ntrue + npred) if ntrue + npred else 0.
          for tp, ntrue, npred in zip(diagonal, true_counts, pred_counts)]
    recall = [tp / ntrue if ntrue else 0. for tp, ntrue in zip(diagonal, true_counts)]
    return {'accuracy': sum(diagonal) / total,
            'macro_f1': statistics.mean(f1),
            'balanced_accuracy': statistics.mean(recall)}


def paired(first, second):
    differences = [(a - b) * 100. for a, b in zip(first, second)]
    if len(differences) != 10:
        raise ValueError('paired interval requires exactly ten support seeds')
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    half = T_CRITICAL_DF9_975 * sd / math.sqrt(10)
    extreme = sum(abs(statistics.mean(s * d for s, d in zip(signs, differences)))
                  >= abs(mean) - 1e-12
                  for signs in itertools.product((-1, 1), repeat=10))
    return {'difference_pp': mean, 'ci95_low_pp': mean - half,
        'ci95_high_pp': mean + half, 'sd_difference_pp': sd,
        'nominal_exact_signflip_p': extreme / 1024,
        'wins': sum(d > 1e-12 for d in differences),
        'ties': sum(abs(d) <= 1e-12 for d in differences),
        'losses': sum(d < -1e-12 for d in differences),
        'multiple_comparisons_corrected': False}


def check_numeric(actual, expected, where, errors):
    if isinstance(expected, bool) or isinstance(expected, str):
        if actual != expected:
            raise ValueError(f'{where}: metadata differs')
        return
    error = abs(actual - expected)
    errors.append(error)
    if error > TOLERANCE:
        raise ValueError(f'{where}: {actual} != {expected}, tolerance {TOLERANCE}')


def recompute(root):
    errors, output, seed_rows = [], [], []
    for ds in ('wm', 'fgvc'):
        for shot in (5, 10):
            methods = {method: {k: [] for k in ('macro_f1', 'accuracy', 'balanced_accuracy')}
                       for method in ('clap', 'ours')}
            for method in ('clap', 'ours'):
                for seed in range(10):
                    name = f'{8 if ds == "wm" else 100}way_{shot}shot_seed{seed}'
                    r = read(root / 'test' / ds / method / f'{name}.json')
                    if (r['dataset'], r['method'], r['shot'], r['seed']) != (ds, method, shot, seed):
                        raise ValueError('task identity differs')
                    calculated = matrix_metrics(r['metrics']['confusion_matrix'])
                    for key, value in calculated.items():
                        check_numeric(value, r['metrics'][key], f'{ds}/{method}/{name}/{key}', errors)
                        methods[method][key].append(value)
                    seed_rows.append({'dataset': ds, 'method': method, 'shot': shot,
                                      'seed': seed, **calculated})
            row = {'dataset': ds, 'shot': shot, 'n': 10}
            for method in ('clap', 'ours'):
                row[f'{method}_mean_percent'] = 100 * statistics.mean(methods[method]['macro_f1'])
                row[f'{method}_sd_percent'] = 100 * statistics.stdev(methods[method]['macro_f1'])
                row[f'{method}_accuracy_mean_percent'] = 100 * statistics.mean(methods[method]['accuracy'])
                row[f'{method}_accuracy_sd_percent'] = 100 * statistics.stdev(methods[method]['accuracy'])
            row.update(paired(methods['ours']['macro_f1'], methods['clap']['macro_f1']))
            row.update({'accuracy_' + k: v for k, v in paired(methods['ours']['accuracy'], methods['clap']['accuracy']).items()})
            output.append(row)
    if len(seed_rows) != 80:
        raise ValueError('expected eighty independent method-task records')
    for filename in ('summary.json', 'summary_extended.json'):
        path = root / filename
        if path.exists():
            reference = read(path)
            if len(reference) != 4:
                raise ValueError('expected four summary rows')
            for row in output:
                expected = next(r for r in reference if r['dataset'] == row['dataset'] and r['shot'] == row['shot'])
                for key, value in expected.items():
                    check_numeric(row[key], value, f'{filename}/{row["dataset"]}/{row["shot"]}/{key}', errors)
    return output, seed_rows, {
        'all_passed': True, 'standard_library_only': True,
        'original_cache_or_numpy_or_torch_required': False,
        'method_task_json_count': len(seed_rows), 'summary_row_count': len(output),
        'numeric_comparison_count': len(errors), 'max_absolute_error': max(errors),
        'tolerance': TOLERANCE, 'paired_unit': 'ten fixed-query support-sampling seeds',
        't_critical_df9_975': T_CRITICAL_DF9_975,
        'historically_unseen_test': False,
        'scope': 'saved confusion matrices and summaries only; not raw-image/model rerun'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
        default=Path(__file__).resolve().parents[4] / 'results/teacher_review_v9_20261015/clap_bridge')
    parser.add_argument('--output-dir', type=Path,
        help='write recomputed summaries and audit to a fresh output folder')
    args = parser.parse_args()
    rows, seeds, audit = recompute(args.root)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in [('summary_extended.json', rows), ('recompute_audit.json', audit)]:
            path = args.output_dir / name
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf8')
        for name, payload in [('summary_extended.csv', rows), ('seed_metrics_recomputed.csv', seeds)]:
            path = args.output_dir / name
            if path.exists():
                raise FileExistsError(path)
            with path.open('w', encoding='utf8', newline='') as f:
                out = csv.DictWriter(f, fieldnames=list(payload[0]))
                out.writeheader()
                out.writerows(payload)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
