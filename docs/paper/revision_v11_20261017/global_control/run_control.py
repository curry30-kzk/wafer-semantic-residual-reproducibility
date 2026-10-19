"""Append-only post-hoc global/classwise development-budget and visual-OOF controls.

No encoder is trained or rerun. Every stage is bound to hashes. Validation
selection is sealed before this run computes Test metrics. Historical Test
exposure is retained; the receipt does not establish a new independent test.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[4]
CODE = Path(__file__).resolve().parent
ROOT = PROJECT / 'results/teacher_review_v11_20261017/global_control'
OLD = PROJECT / 'results/teacher_review_v9_20261015/clap_bridge'
BRIDGE_PATH = PROJECT / 'docs/paper/revision_v9_20261015/bridge/run_clap_bridge.py'
spec = importlib.util.spec_from_file_location('frozen_v9_bridge', BRIDGE_PATH)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
np, torch = bridge.np, bridge.torch
GRID = bridge.OURS_GRID


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2 ** 20), b''):
            h.update(part)
    return h.hexdigest()


def verify_inputs():
    p = read(ROOT / 'protocol.json')
    if sha(__file__) != p['runner_sha256']:
        raise ValueError('runner changed after protocol freeze')
    for name, digest in p['input_sha256'].items():
        if sha(PROJECT / name) != digest:
            raise ValueError('input changed: ' + name)
    return p


def prepare():
    inputs = [BRIDGE_PATH, OLD / 'protocol.json', OLD / 'freeze_receipt.json',
              OLD / 'validation/ours_selection.json',
              PROJECT / 'src/wm811k/class_reliability_experiment.py',
              PROJECT / 'src/wm811k/class_reliability_residual.py',
              PROJECT / 'src/wm811k/metrics.py',
              PROJECT / 'src/wm811k/prototype.py',
              CODE / 'recompute_control.py']
    for shot, seed, name in bridge.tasks('fgvc'):
        p = bridge.main_cache_path('fgvc', 'validation', name)
        inputs += [p, p.with_suffix('.npz'),
                   bridge.manifest_path('fgvc', 'validation', name),
                   bridge.OLD / 'fgvc' / f'{name}.npz',
                   bridge.OLD / 'fgvc' / f'{name}.json']
        inputs += [OLD / 'validation/ours' / f'candidate_{ci:02d}' / f'{name}.json'
                   for ci in range(16)]
    inputs += [bridge.TEXT / 'text_fgvc.npz', bridge.TEXT / 'text_fgvc.json',
               bridge.OLD / 'fgvc/validation_query.npz',
               bridge.OLD / 'fgvc/validation_query.json']
    for ds in ('wm', 'fgvc'):
        for shot, seed, name in bridge.tasks(ds):
            p = bridge.main_cache_path(ds, 'test', name)
            inputs += [p, p.with_suffix('.npz'), bridge.manifest_path(ds, 'test', name),
                       OLD / 'test' / ds / 'ours' / f'{name}.npz',
                       OLD / 'test' / ds / 'ours' / f'{name}.json']
    # Include the support-fold implementation imported by the residual layer.
    inputs += [PROJECT / 'src/wm811k/semantic_calibration.py'] if (PROJECT / 'src/wm811k/semantic_calibration.py').exists() else []
    paths = sorted(set(inputs), key=str)
    write(ROOT / 'protocol.json', {
        'schema': 'v11_independent_global_selection_and_visual_oof_control_v1',
        'created_utc': utc(), 'runner_sha256': sha(__file__),
        'input_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in paths},
        'historically_unseen_test': False,
        'history': 'Both Test sets and prior model comparisons were observed before this additional analysis. This is a post-hoc control, not an independent confirmation.',
        'prior_global_search_audit': {
            'fgvc_development_files': ['results/fgvc_aircraft_validation/class_reliability_residual/' + x + '/development/all_config_results.json' for x in ('hierarchy_deterministic', 'qwen2_5_7b_instruct', 'variant_name')],
            'candidate_count_each': 16, 'granularities_found': ['classwise'],
            'existing_global_files': 'fixed-configuration ablations; no independent global Validation search found'},
        'methods': ['classwise', 'global'], 'grid': GRID,
        'selection': 'Independently maximize each method mean Macro-F1 over FGVC 5/10 shot x seeds 0..9, same twenty full Validation tasks.',
        'tie_break': 'first declared candidate within 1e-12 of maximum',
        'candidate_count_per_method': 16, 'validation_tasks_per_method': 20,
        'budget_scope': 'same hyperparameter grid, candidate count, Validation tasks, cached visual scores, raw text scores, optimizer steps and empirical activation rule; parameter count and arithmetic cost differ; historical development budgets not equated',
        'selection_rule': 'original support-based empirical activation, calibration support_bias_zscore, residual fusion, 200 projected Adam steps, lr .05, CPU float64',
        'validation_raw_score_device': 'CUDA float32 cosine, exactly the frozen v9 helper; no image encoder computation',
        'test_primary': 'FGVC selected classwise minus independently selected global under original activation rule',
        'test_secondary': 'Same independently selected configurations with both methods unconditionally enabled; these are not separately tuned for the no-fallback condition',
        'visual_oof_diagnostic': {
            'datasets': ['wm', 'fgvc'], 'shots': [5, 10], 'seeds': list(range(10)),
            'configuration': 'FGVC selected classwise config; WM frozen lambda=1,rho=.01; no diagnostic-specific search',
            'intervention': 'Replace visual_oof_scores by final visual clean-support scores and recompute training semantic scale using those final support scores. Retain foldwise semantic mean biases, support labels, final visual query scores, final query semantic residual and all other fitting settings.',
            'activation_versions': ['original empirical activation', 'both unconditional'],
            'interpretation': 'Joint visual-OOF training-score/scale intervention; not an isolated causal estimate of fold partition, scaling or activation, and not a fusion-level independent cross-validation.'},
        'reuse_policy': 'Recompute classwise Validation and original Test predictions; audit against frozen v9 records before interpreting new controls. All outputs retained irrespective direction.',
        'runtime': {'python': sys.executable, 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'cpu_threads': 2}})
    print('PREPARED inputs=', len(paths), flush=True)


def sparse_metrics(metrics):
    out = dict(metrics)
    cm = out.pop('confusion_matrix')
    out['confusion_matrix_sparse'] = {
        'class_count': len(cm),
        'cells': [[i, j, int(v)] for i, row in enumerate(cm) for j, v in enumerate(row) if v]}
    return out


def selected_record(rows):
    maximum = max(r['mean_macro_f1'] for r in rows)
    return next(r for r in rows if r['mean_macro_f1'] >= maximum - 1e-12)


def select():
    if (ROOT / 'freeze_receipt.json').exists():
        raise ValueError('selection sealed')
    verify_inputs()
    text = bridge.arrays(bridge.TEXT / 'text_fgvc.npz')['text']
    query = bridge.arrays(bridge.OLD / 'fgvc/validation_query.npz')
    raw_query = bridge.raw_scores(query['features'], text)
    errors = []
    for shot, seed, name in bridge.tasks('fgvc'):
        arr = bridge.arrays(bridge.OLD / 'fgvc' / f'{name}.npz')
        meta = read(bridge.OLD / 'fgvc' / f'{name}.json')
        cache = bridge.load_task_cache(bridge.main_cache_path('fgvc', 'validation', name))
        bridge.verify_task_identity('fgvc', 'validation', name, meta, cache)
        np.testing.assert_array_equal(arr['labels'], cache.support_labels)
        np.testing.assert_array_equal(query['labels'], cache.query_labels)
        control = bridge.rebuild(cache, bridge.raw_scores(arr['clean'], text), raw_query)
        for method in ('classwise', 'global'):
            for ci, config in enumerate(GRID):
                dest = ROOT / 'validation' / method / f'candidate_{ci:02d}' / f'{name}.json'
                if dest.exists():
                    continue
                start = time.perf_counter()
                result = bridge.evaluate_cached_task(control, **config, granularity=method)
                if method == 'classwise':
                    prior = read(OLD / 'validation/ours' / f'candidate_{ci:02d}' / f'{name}.json')
                    for key in ('macro_f1', 'accuracy', 'balanced_accuracy'):
                        error = abs(result['metrics'][key] - prior['metrics'][key])
                        errors.append(error)
                        if error > 1e-12:
                            raise ValueError(f'v9 Validation reproduction differs: {name}/{ci}/{key}/{error}')
                    if result['selection']['enabled'] != prior['enabled']:
                        raise ValueError('v9 Validation activation differs')
                write(dest, {'method': method, 'shot': shot, 'seed': seed,
                    'candidate': ci, 'config': config,
                    'metrics': sparse_metrics(result['metrics']),
                    'enabled': result['selection']['enabled'],
                    'training_steps': result['selection']['steps'],
                    'trainable_parameters': 100 if method == 'classwise' else 1,
                    'elapsed_seconds': time.perf_counter() - start,
                    'test_labels_used_for_selection': False, 'query_split': 'Validation',
                    'created_utc': utc()})
        print('VAL', name, '32 method-candidates done', flush=True)
    for method in ('classwise', 'global'):
        rows = []
        for ci, config in enumerate(GRID):
            records = [read(ROOT / 'validation' / method / f'candidate_{ci:02d}' / f'{name}.json') for _, _, name in bridge.tasks('fgvc')]
            rows.append({'candidate': ci, 'config': config,
                         'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in records])),
                         'task_count': len(records),
                         'enabled_count': sum(r['enabled'] for r in records)})
        write(ROOT / 'validation' / f'{method}_selection.json', {
            'grid': rows, 'selected': selected_record(rows), 'created_utc': utc()})
    write(ROOT / 'validation/reproduction_audit.json', {
        'v9_classwise_validation_records_reproduced': 320,
        'metrics_compared': len(errors), 'max_metric_error': max(errors),
        'activation_matches': True, 'historically_unseen_test': False})


def freeze():
    p = verify_inputs()
    if (ROOT / 'test').exists() or (ROOT / 'test_evaluation_started.json').exists():
        raise ValueError('new Test output exists before receipt')
    files = [ROOT / 'protocol.json', *sorted((ROOT / 'validation').rglob('*.json'))]
    selected = {m: read(ROOT / 'validation' / f'{m}_selection.json')['selected'] for m in ('classwise', 'global')}
    write(ROOT / 'freeze_receipt.json', {
        'created_utc': utc(), 'protocol_sha256': sha(ROOT / 'protocol.json'),
        'selected_fgvc': selected, 'historically_unseen_test': False,
        'history': p['history'],
        'bound_files': {str(p.relative_to(PROJECT)): sha(p) for p in files}})
    print('FROZEN', selected, flush=True)


def receipt():
    verify_inputs()
    r = read(ROOT / 'freeze_receipt.json')
    for name, digest in r['bound_files'].items():
        if sha(PROJECT / name) != digest:
            raise ValueError('frozen selection changed: ' + name)
    return r


def variant_result(control, result, config, fallback):
    key = 'effective_reliability' if fallback else 'learned_reliability'
    weights = np.asarray(result['selection'][key])
    scores = bridge.apply_classwise_residual(control.visual_query_scores,
        control.semantic_query_residual, weights, lambda_value=config['lambda_value'])
    return scores.argmax(1).astype(np.int64)


def evaluate():
    r = receipt()
    write(ROOT / 'test_evaluation_started.json', {
        'created_utc': utc(), 'freeze_receipt_sha256': sha(ROOT / 'freeze_receipt.json'),
        'historically_unseen_test': False})
    original_reproduction = []
    for ds in ('fgvc', 'wm'):
        for shot, seed, name in bridge.tasks(ds):
            cache = bridge.load_task_cache(bridge.main_cache_path(ds, 'test', name))
            old = bridge.arrays(OLD / 'test' / ds / 'ours' / f'{name}.npz')
            np.testing.assert_array_equal(old['y_true'], cache.query_labels)
            control = bridge.rebuild(cache, old['semantic_support_raw'], old['semantic_query_raw'])
            config = r['selected_fgvc']['classwise']['config'] if ds == 'fgvc' else {'lambda_value': 1., 'rho': .01}
            alternatives = {'classwise_oof': (control, 'classwise', config)}
            if ds == 'fgvc':
                alternatives['global_oof'] = (control, 'global', r['selected_fgvc']['global']['config'])
            # Recompute only the training visual evidence and matching scale.
            # Final-support calibration gives the exact same query residual.
            changed_cache = replace(cache, visual_oof_scores=cache.visual_support_scores.copy())
            no_oof = bridge.rebuild(changed_cache, old['semantic_support_raw'], old['semantic_query_raw'])
            np.testing.assert_array_equal(no_oof.semantic_query_residual, control.semantic_query_residual)
            if no_oof.calibration_metadata['fold_class_bias'] != control.calibration_metadata['fold_class_bias']:
                raise ValueError('semantic fold biases changed in visual-only diagnostic')
            alternatives['classwise_insample'] = (no_oof, 'classwise', config)
            for variant, (c, granularity, cfg) in alternatives.items():
                start = time.perf_counter()
                result = bridge.evaluate_cached_task(c, **cfg, granularity=granularity)
                for fallback in (True, False):
                    pred = variant_result(c, result, cfg, fallback)
                    if fallback:
                        np.testing.assert_array_equal(pred, result['predictions'])
                    suffix = 'original_activation' if fallback else 'unconditional'
                    label = f'{variant}_{suffix}'
                    metrics = bridge.evaluate_predictions(c.query_labels, pred, c.class_names)
                    if variant == 'classwise_oof' and fallback:
                        prior = read(OLD / 'test' / ds / 'ours' / f'{name}.json')
                        # New selection may differ, in which case no old-model identity is asserted.
                        if cfg == prior['config']:
                            np.testing.assert_array_equal(pred, old['predictions'])
                            original_reproduction.append(abs(metrics['macro_f1'] - prior['metrics']['macro_f1']))
                    write(ROOT / 'test' / ds / label / f'{name}.json', {
                        'dataset': ds, 'method': label, 'shot': shot, 'seed': seed,
                        'config': cfg, 'metrics': sparse_metrics(metrics),
                        'selection': result['selection'], 'fallback_applied': fallback,
                        'training_visual_scale': c.calibration_metadata['oof_visual_scale'],
                        'query_visual_scale': c.calibration_metadata['final_visual_scale'],
                        'elapsed_seconds_including_shared_fit': time.perf_counter() - start,
                        'freeze_receipt_sha256': sha(ROOT / 'freeze_receipt.json'),
                        'historically_unseen_test': False, 'posthoc_analysis': True,
                        'created_utc': utc()})
            print('TEST', ds, name, 'complete', flush=True)
    write(ROOT / 'test_evaluation_completed.json', {
        'created_utc': utc(), 'method_task_count': 200,
        'v9_original_test_records_reproduced': len(original_reproduction),
        'max_metric_reproduction_error': max(original_reproduction) if original_reproduction else None})


def audit():
    receipt()
    if not read(ROOT / 'freeze_receipt.json')['created_utc'] < read(ROOT / 'test_evaluation_started.json')['created_utc']:
        raise ValueError('selection receipt did not precede this Test calculation')
    import recompute_control
    summary, val, audit = recompute_control.recompute(ROOT)
    write(ROOT / 'summary.json', summary)
    write(ROOT / 'selection_summary.json', val)
    write(ROOT / 'audit.json', {**audit, 'all_inputs_unchanged': True,
                              'freeze_precedes_this_test_evaluation': True,
                              'created_utc': utc()})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


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
