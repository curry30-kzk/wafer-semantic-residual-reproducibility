"""Append-only same-text CLAP bridge; never overwrite historical main results.

Stages: prepare -> select -> freeze -> evaluate -> audit. Both Test sets were
historically observed. A new receipt restricts this additional analysis only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
from scipy.stats import t as student_t

PROJECT = Path(__file__).resolve().parents[4]
CODE = Path(__file__).resolve().parent
ROOT = PROJECT / 'results/teacher_review_v9_20261015/clap_bridge'
OLD = PROJECT / 'results/proker_20261013/cache'
TEXT = PROJECT / 'results/revision_v4_20261013/same_text_proker'
sys.path.insert(0, str(PROJECT / 'src'))
import torch
import wm811k.clap as clap
from wm811k.class_reliability_experiment import (
    load_task_cache, evaluate_cached_task, calibrate_semantic_scores_support_only)
from wm811k.metrics import evaluate_predictions
from wm811k.prototype import l2_normalize
from wm811k.class_reliability_residual import apply_classwise_residual

COMMIT = clap.CLAP_REFERENCE_COMMIT
CLAP_GRID = [dict(learning_rate=lr, epochs=ep)
             for ep in (100, 200, 300, 600) for lr in (.01, .03, .1, .3)]
OURS_GRID = [dict(lambda_value=lam, rho=rho)
             for rho in (10., 1., .1, .01) for lam in (.25, .5, .75, 1.)]
SUPPORT_KEYS = ('clean', 'views', 'labels')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2 ** 20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def stem(ds, shot, seed):
    return f'{8 if ds == "wm" else 100}way_{shot}shot_seed{seed}'


def tasks(ds):
    for shot in (5, 10):
        for seed in range(10):
            yield shot, seed, stem(ds, shot, seed)


def main_cache_path(ds, split, name):
    if ds == 'wm':
        return PROJECT / 'results/wm811k_dedup_v2/llm_structured/cache' / f'{name}.json'
    if split == 'validation':
        # The old cluster seed 5..9 manifests differ. Use the explicitly rebuilt
        # 2026-09-22 local-manifest cache for ALL Validation tasks, not a mixture.
        return PROJECT / 'results/semantic_source_matching/fgvc_validation_runs/qwen2_5_7b_instruct/cache' / f'{name}.json'
    return PROJECT / 'results/fgvc_aircraft_test/class_reliability_residual/cache' / f'{name}.json'


def manifest_path(ds, split, name):
    if ds == 'wm':
        return PROJECT / 'data/processed/WM811K_dedup_v2/splits' / f'{name}.json'
    return PROJECT / 'data/processed/FGVC-Aircraft/splits' / split / f'{name}.json'


def verify_task_identity(ds, split, name, meta, cache):
    path = manifest_path(ds, split, name)
    if sha(path) != cache.manifest_sha256:
        raise ValueError('main score cache is not bound to the current task manifest')
    manifest = read(path)
    if [r['image_path'] for r in manifest['support']] != meta['support_paths']:
        raise ValueError('visual features and main-method support order differ')
    if tuple(meta['class_names']) != cache.class_names:
        raise ValueError('class order differs')
    if meta['model_state_sha256'] != read(TEXT / f'text_{ds}.json')['model_state_sha256']:
        raise ValueError('text and visual encoder checkpoint hashes differ')
    query_meta = read(OLD / ds / f'{split}_query.json')
    if [r['image_path'] for r in manifest['query']] != query_meta['paths']:
        raise ValueError('query feature and main score cache row order differs')
    if split == 'validation' or ds == 'wm':
        if meta['manifest_sha256'] != cache.manifest_sha256:
            raise ValueError('support feature cache manifest differs')


@contextmanager
def learning_rate(value):
    """Change only the original fitter's declared LR; restore even on failure."""
    prior = clap.CLAP_LEARNING_RATE
    clap.CLAP_LEARNING_RATE = float(value)
    try:
        yield
    finally:
        clap.CLAP_LEARNING_RATE = prior


def fit(arr, text, config, shot, seed, device='cuda'):
    # Same support seed across candidates, independent of query labels.
    with learning_rate(config['learning_rate']):
        return clap.fit_clap_support_only(
            arr['clean'], arr['views'], arr['labels'], text,
            num_classes=len(text), shot=shot, device=device, mode='clap',
            seed=20261015 + shot * 100 + seed, epochs=config['epochs'])


def raw_scores(features, text, device='cuda'):
    """Common float32 cosine prior; no branch-specific normalization or scale."""
    x = l2_normalize(features).astype(np.float32)
    tt = torch.as_tensor(l2_normalize(text).astype(np.float32), device=device)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            xx = torch.as_tensor(x[start:start + 512], device=device)
            parts.append((xx @ tt.T).cpu().numpy().copy())
    return np.concatenate(parts)


def rebuild(cache, raw_support, raw_query):
    folds = cache.calibration_metadata['folds']
    if folds != cache.anchor_selection['folds']:
        raise ValueError('saved visual and calibration folds differ')
    c = calibrate_semantic_scores_support_only(
        raw_support, raw_query, visual_support_scores=cache.visual_support_scores,
        visual_oof_scores=cache.visual_oof_scores, folds=folds,
        epsilon=cache.calibration_metadata.get('epsilon', 1e-12))
    return replace(cache, semantic_oof_raw=raw_support,
        semantic_oof_residual=c.oof_residual, semantic_query_raw=raw_query,
        semantic_query_residual=c.query_residual,
        calibration_metadata={**cache.calibration_metadata, **c.metadata,
                              'v9_same_text_rebuilt': True})


def verify_inputs(protocol):
    for name, digest in protocol['input_sha256'].items():
        if sha(PROJECT / name) != digest:
            raise ValueError(f'input changed: {name}')
    if sha(__file__) != protocol['runner_sha256']:
        raise ValueError('bridge runner changed after protocol freeze')


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    ref = CODE / 'reference'
    ref.mkdir(exist_ok=True)
    reference_files = ('trainers/adapters.py', 'experiments/experiment_CLAP.txt',
                       'configs/trainers/SGD_lr1e-1_B256_ep300.yaml')
    for name in reference_files:
        target = ref / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://raw.githubusercontent.com/jusiro/CLAP/{COMMIT}/{name}'
            with urllib.request.urlopen(url, timeout=30) as r:
                target.write_bytes(r.read())
    inputs = [PROJECT / 'src/wm811k/clap.py',
              PROJECT / 'src/wm811k/class_reliability_experiment.py',
              PROJECT / 'src/wm811k/class_reliability_residual.py']
    for ds in ('wm', 'fgvc'):
        inputs += [TEXT / f'text_{ds}.npz', TEXT / f'text_{ds}.json']
        for shot, seed, name in tasks(ds):
            inputs += [OLD / ds / f'{name}.npz', OLD / ds / f'{name}.json']
            for split in (('validation', 'test') if ds == 'fgvc' else ('test',)):
                p = main_cache_path(ds, split, name)
                inputs += [p, p.with_suffix('.npz'), manifest_path(ds, split, name)]
        for split in (('validation', 'test') if ds == 'fgvc' else ('test',)):
            inputs += [OLD / ds / f'{split}_query.npz', OLD / ds / f'{split}_query.json']
    inputs += list(ref.rglob('*.*'))
    payload = {
        'schema': 'v9_clap_same_text_budget_bridge_v1', 'created_utc': utc(),
        'historically_unseen_test': False,
        'history': 'Both Test sets were previously observed. All new evidence is post-hoc.',
        'runner_sha256': sha(__file__),
        'input_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in inputs},
        'official_commit': COMMIT,
        'official_urls': [f'https://github.com/jusiro/CLAP/blob/{COMMIT}/{x}' for x in reference_files],
        'clap_objective': 'unchanged fixed support-confidence weighted class L2 plus support CE',
        'clap_unchanged': {'SGD_momentum': .9, 'weight_decay': 0., 'batch_size': 256,
            'logit_scale': 100., 'warmup_epochs': 1, 'warmup_lr': 1e-5,
            'schedule': 'cosine', 'views': 10, 'batch_count': 'ceil(C*K/256)',
            'initialization': 'same frozen normalized text T'},
        'clap_changes_from_official': ['existing project ViT-B/32 port',
            '10 cached support views instead of official 20', 'fixed logit scale 100',
            'torch randperm support sampling, fixed deterministic task seed',
            'additional Validation LR/epoch search; fixed support coefficients retained'],
        'clap_grid': CLAP_GRID, 'ours_grid': OURS_GRID,
        'fgvc_selection': 'mean Macro-F1 jointly over 5/10 shot x seeds 0..9, full Validation',
        'tie_break': 'first predeclared grid entry within 1e-12 of maximum; no Test inspection',
        'candidate_count_per_method': 16, 'validation_tasks_per_method': 20,
        'budget_scope': 'dataset-level candidate count and Validation access matched; training updates and walltime NOT matched',
        'wm_clap_fixed_external_config': {'learning_rate': .1, 'epochs': 300},
        'wm_ours_frozen_config': {'lambda_value': 1., 'rho': .01},
        'wm_budget_scope': 'same text/support/features only; historical development budgets not equal or reconstructed',
        'main_control': 'same frozen T; original archived visual support/OOF/query scores and folds; original objective/200 Adam steps',
        'validation_visual_cache_note': 'all 20 tasks use local-manifest-matched 2026-09-22 source-matching visual caches; old cluster Validation seed 5..9 manifest mismatches are excluded',
        'main_table_policy': 'new variants only; preserve original main tables and all historical outputs',
        'runtime': {'python': sys.executable, 'torch': torch.__version__,
            'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}}
    write(ROOT / 'protocol.json', payload)
    print('PREPARED', len(inputs), flush=True)


def selection_record(method, candidate, shot, seed, name, metrics, elapsed, **extra):
    p = ROOT / 'validation' / method / f'candidate_{candidate:02d}' / f'{name}.json'
    write(p, {'method': method, 'candidate': candidate, 'shot': shot, 'seed': seed,
              'metrics': metrics, 'elapsed_seconds': elapsed,
              'query_split': 'Validation', 'test_labels_used_for_selection': False,
              'created_utc': utc(), **extra})


def choose(records):
    maximum = max(x['mean_macro_f1'] for x in records)
    return next(x for x in records if x['mean_macro_f1'] >= maximum - 1e-12)


def select():
    if (ROOT / 'freeze_receipt.json').exists():
        raise ValueError('selection sealed')
    protocol = read(ROOT / 'protocol.json')
    verify_inputs(protocol)
    text = arrays(TEXT / 'text_fgvc.npz')['text']
    query = arrays(OLD / 'fgvc/validation_query.npz')
    raw_query = raw_scores(query['features'], text)
    for shot, seed, name in tasks('fgvc'):
        arr = arrays(OLD / 'fgvc' / f'{name}.npz')
        meta = read(OLD / 'fgvc' / f'{name}.json')
        cache = load_task_cache(main_cache_path('fgvc', 'validation', name))
        np.testing.assert_array_equal(arr['labels'], cache.support_labels)
        np.testing.assert_array_equal(query['labels'], cache.query_labels)
        verify_task_identity('fgvc', 'validation', name, meta, cache)
        control = rebuild(cache, raw_scores(arr['clean'], text), raw_query)
        for method, grid in (('clap', CLAP_GRID), ('ours', OURS_GRID)):
            for ci, config in enumerate(grid):
                dest = ROOT / 'validation' / method / f'candidate_{ci:02d}' / f'{name}.json'
                if dest.exists():
                    continue
                start = time.perf_counter()
                if method == 'clap':
                    fitted = fit(arr, text, config, shot, seed)
                    torch.cuda.synchronize()
                    train_seconds = time.perf_counter() - start
                    pred = fitted.predict(query['features'])
                    metrics = evaluate_predictions(query['labels'], pred, cache.class_names)
                    extra = {'training': fitted.selection, 'train_seconds': train_seconds,
                             'trainable_parameters': fitted.trainable_parameter_count}
                else:
                    out = evaluate_cached_task(control, **config, granularity='classwise')
                    metrics = out['metrics']
                    extra = {'training_steps': 200, 'trainable_parameters': len(text),
                             'enabled': out['selection']['enabled']}
                selection_record(method, ci, shot, seed, name, metrics,
                    time.perf_counter() - start, config=config, **extra)
                print('VAL', name, method, ci + 1, '/16',
                      round(metrics['macro_f1'] * 100, 3),
                      round(time.perf_counter() - start, 2), 's', flush=True)
    for method, grid in (('clap', CLAP_GRID), ('ours', OURS_GRID)):
        summary = []
        for ci, config in enumerate(grid):
            records = [read(ROOT / 'validation' / method / f'candidate_{ci:02d}' / f'{n}.json')
                       for _, _, n in tasks('fgvc')]
            summary.append({'candidate': ci, 'config': config,
                'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in records])),
                'total_elapsed_seconds': sum(r['elapsed_seconds'] for r in records),
                'task_count': len(records),
                'seed_macro_f1': [r['metrics']['macro_f1'] for r in records]})
        write(ROOT / 'validation' / f'{method}_selection.json', {
            'grid': summary, 'selected': choose(summary), 'created_utc': utc(),
            'selection_metric': protocol['fgvc_selection'], 'tie_break': protocol['tie_break']})


def freeze():
    p = read(ROOT / 'protocol.json')
    verify_inputs(p)
    if (ROOT / 'test').exists():
        raise ValueError('Test output exists before receipt')
    selections = {m: read(ROOT / 'validation' / f'{m}_selection.json')['selected'] for m in ('clap', 'ours')}
    files = [ROOT / 'protocol.json', *list((ROOT / 'validation').rglob('*.json'))]
    write(ROOT / 'freeze_receipt.json', {'created_utc': utc(),
        'history': p['history'], 'protocol_sha256': sha(ROOT / 'protocol.json'),
        'selected_fgvc': selections,
        'bound_files': {str(x.relative_to(PROJECT)): sha(x) for x in files}})
    print('FROZEN', selections, flush=True)


def verify_receipt():
    receipt = read(ROOT / 'freeze_receipt.json')
    verify_inputs(read(ROOT / 'protocol.json'))
    for name, digest in receipt['bound_files'].items():
        if sha(PROJECT / name) != digest:
            raise ValueError(f'frozen selection changed: {name}')
    return receipt


def evaluate():
    receipt = verify_receipt()
    write(ROOT / 'test_evaluation_started.json', {'created_utc': utc(),
        'freeze_receipt_sha256': sha(ROOT / 'freeze_receipt.json'),
        'historically_unseen_test': False})
    for ds in ('wm', 'fgvc'):
        text = arrays(TEXT / f'text_{ds}.npz')['text']
        query = arrays(OLD / ds / 'test_query.npz')
        raw_query = raw_scores(query['features'], text)
        for shot, seed, name in tasks(ds):
            arr = arrays(OLD / ds / f'{name}.npz')
            cache = load_task_cache(main_cache_path(ds, 'test', name))
            verify_task_identity(ds, 'test', name, read(OLD / ds / f'{name}.json'), cache)
            np.testing.assert_array_equal(query['labels'], cache.query_labels)
            np.testing.assert_array_equal(arr['labels'], cache.support_labels)
            raw_support = raw_scores(arr['clean'], text)
            control = rebuild(cache, raw_support, raw_query)
            for method in ('clap', 'ours'):
                config = (receipt['selected_fgvc'][method]['config'] if ds == 'fgvc'
                          else read(ROOT / 'protocol.json')[f'wm_{method}_' + ('fixed_external_config' if method == 'clap' else 'frozen_config')])
                start = time.perf_counter()
                if method == 'clap':
                    fitted = fit(arr, text, config, shot, seed)
                    torch.cuda.synchronize()
                    train_seconds = time.perf_counter() - start
                    scores = fitted.decision_scores(query['features'])
                    pred = scores.argmax(1)
                    result = {'training': fitted.selection, 'train_seconds': train_seconds,
                        'trainable_parameters': fitted.trainable_parameter_count}
                else:
                    result = evaluate_cached_task(control, **config, granularity='classwise')
                    pred = np.asarray(result.pop('predictions'), dtype=np.int64)
                    eff = np.asarray(result['selection']['effective_reliability'])
                    scores = apply_classwise_residual(control.visual_query_scores,
                        control.semantic_query_residual, eff,
                        lambda_value=config['lambda_value'])
                    np.testing.assert_array_equal(scores.argmax(1), pred)
                    result.update(training_steps=200, trainable_parameters=len(text))
                score_path = ROOT / 'test' / ds / method / f'{name}.npz'
                score_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(score_path, scores=scores, predictions=pred,
                    y_true=query['labels'], semantic_query_raw=raw_query,
                    semantic_support_raw=raw_support)
                metrics = evaluate_predictions(query['labels'], pred, cache.class_names)
                result.update({'dataset': ds, 'method': method, 'shot': shot, 'seed': seed,
                    'metrics': metrics, 'config': config, 'elapsed_seconds': time.perf_counter() - start,
                    'historically_unseen_test': False, 'posthoc_analysis': True,
                    'freeze_receipt_sha256': sha(ROOT / 'freeze_receipt.json'),
                    'score_sha256': sha(score_path), 'text_sha256': sha(TEXT / f'text_{ds}.npz'),
                    'class_names': list(cache.class_names), 'created_utc': utc()})
                write(score_path.with_suffix('.json'), result)
                print('TEST', ds, name, method, round(metrics['macro_f1'] * 100, 3), flush=True)
    write(ROOT / 'test_evaluation_completed.json', {'created_utc': utc(), 'method_task_count': 80})


def paired(a, b):
    d = (np.asarray(a) - np.asarray(b)) * 100
    sd = float(d.std(ddof=1))
    half = float(student_t.ppf(.975, 9) * sd / np.sqrt(10))
    flips = np.asarray(list(itertools.product((-1, 1), repeat=10)))
    p = float(np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12))
    return {'difference_pp': float(d.mean()), 'ci95_low_pp': float(d.mean() - half),
        'ci95_high_pp': float(d.mean() + half), 'sd_difference_pp': sd,
        'nominal_exact_signflip_p': p, 'wins': int((d > 1e-12).sum()),
        'ties': int((np.abs(d) <= 1e-12).sum()), 'losses': int((d < -1e-12).sum()),
        'multiple_comparisons_corrected': False}


def audit():
    receipt = verify_receipt()
    assert receipt['created_utc'] < read(ROOT / 'test_evaluation_started.json')['created_utc']
    summaries, errors = [], []
    for ds in ('wm', 'fgvc'):
        for shot in (5, 10):
            methods = {}
            for method in ('clap', 'ours'):
                vals = []
                for seed in range(10):
                    name = stem(ds, shot, seed)
                    p = ROOT / 'test' / ds / method / f'{name}.npz'
                    r, a = read(p.with_suffix('.json')), arrays(p)
                    assert r['score_sha256'] == sha(p)
                    np.testing.assert_array_equal(a['predictions'], a['scores'].argmax(1))
                    actual = evaluate_predictions(a['y_true'], a['predictions'], r['class_names'])
                    errors.append(max(abs(actual[k] - r['metrics'][k]) for k in ('accuracy', 'macro_f1', 'balanced_accuracy')))
                    other = arrays(ROOT / 'test' / ds / ('ours' if method == 'clap' else 'clap') / f'{name}.npz')
                    np.testing.assert_array_equal(a['semantic_query_raw'], other['semantic_query_raw'])
                    np.testing.assert_array_equal(a['semantic_support_raw'], other['semantic_support_raw'])
                    vals.append(r['metrics']['macro_f1'])
                methods[method] = vals
            summaries.append({'dataset': ds, 'shot': shot, 'n': 10,
                **{f'{m}_mean_percent': float(100 * np.mean(v)) for m, v in methods.items()},
                **{f'{m}_sd_percent': float(100 * np.std(v, ddof=1)) for m, v in methods.items()},
                **paired(methods['ours'], methods['clap'])})
    write(ROOT / 'summary.json', summaries)
    write(ROOT / 'audit.json', {'created_utc': utc(), 'all_inputs_unchanged': True,
        'freeze_precedes_this_test_evaluation': True, 'historically_unseen_test': False,
        'validation_method_task_count': 640, 'test_method_task_count': 80,
        'same_raw_semantic_query_and_support_arrays': True,
        'max_metric_recomputation_error': max(errors), 'summaries': summaries})
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'select', 'freeze', 'evaluate', 'audit', 'pipeline'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for stage in ([prepare, select, freeze, evaluate, audit] if args.stage == 'pipeline' else [globals()[args.stage]]):
        print(utc(), stage.__name__, flush=True)
        stage()


if __name__ == '__main__':
    main()
