"""Run ProKeR in explicit selection, freeze, and formal evaluation stages."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'src'))

from wm811k.baseline_experiment import discover_tasks, _validate_task_grid
from wm811k.clip_features import FrozenClipEncoder
from wm811k.experiment_reporting import task_result_provenance, write_run_artifacts, write_fewshot_summaries
from wm811k.metrics import evaluate_predictions
from wm811k.proker import (expand_support_views, fit_proker, choose_candidate,
                           select_support_parameters, select_validation_parameters)
from wm811k.proker_artifacts import (now_utc, sha256_file, write_json, load_json, write_npz,
                                    require_binding, write_receipt, verify_receipt)

PROTOCOL = PROJECT / 'configs/proker_protocol_20261013.json'
ROOT = PROJECT / 'results/proker_20261013'
DATASETS = ('wm', 'fgvc')


def task_list(dataset, split='validation'):
    if dataset == 'wm':
        tasks = discover_tasks(PROJECT/'data/processed/WM811K_dedup_v2', run_all=True,
            formal_seeds=range(10), n_way=8, image_root=PROJECT/'data/processed/WM811K/images')
    else:
        tasks = discover_tasks(PROJECT/'data/processed/FGVC-Aircraft', run_all=True,
            formal_seeds=range(10), n_way=100, query_split=split,
            image_root=PROJECT/'data/raw/FGVC-Aircraft/fgvc-aircraft-2013b/data')
    _validate_task_grid(tasks)
    return tasks


def stem(task):
    return f'{task.n_way}way_{task.shot}shot_seed{task.seed}'


def support_identity(task):
    entries = [(r.relative_path, r.label_index, r.content_hash) for r in task.support]
    import json
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False).encode()).hexdigest()


def binding(task, protocol_hash):
    return {'protocol_sha256': protocol_hash, 'manifest_sha256': sha256_file(task.manifest_path),
            'support_identity_sha256': support_identity(task)}


def encoder_for(dataset, device):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = FrozenClipEncoder.create(device, dataset='wm811k' if dataset == 'wm' else 'fgvc_aircraft')
    h = hashlib.sha256()
    for name, param in encoder.model.state_dict().items():
        h.update(name.encode()); h.update(str(param.dtype).encode())
        h.update(param.detach().cpu().numpy().tobytes())
    return encoder, h.hexdigest()


def cache_task(task, dataset, encoder, model_hash, protocol, root):
    base = root/'cache'/dataset/stem(task)
    meta_path = base.with_suffix('.json')
    expected = {**binding(task, sha256_file(PROTOCOL)), 'model_state_sha256': model_hash}
    if meta_path.exists():
        meta = load_json(meta_path)
        require_binding(meta, expected)
        if sha256_file(base.with_suffix('.npz')) != meta['array_sha256']:
            raise ValueError('support cache hash mismatch')
        return meta
    start = time.perf_counter()
    features = encoder.encode_images(task.support, batch_size=protocol['batch_size'], show_progress=False)
    clean = np.stack([features[r.relative_path] for r in task.support])
    views = encoder.encode_augmented_images(task.support, batch_size=protocol['batch_size'],
        view_count=protocol['support_augmentation']['view_count'],
        seed=20260816 + task.shot * 100 + task.seed, show_progress=False)
    labels = np.array([r.label_index for r in task.support], dtype=np.int64)
    text, prompts = encoder.encode_class_text_prototypes(task.class_names)
    write_npz(base.with_suffix('.npz'), clean=clean, views=views, labels=labels, text=text)
    meta = {**expected, 'created_utc': now_utc(), 'shot': task.shot, 'seed': task.seed,
        'array_sha256': sha256_file(base.with_suffix('.npz')), 'class_names': list(task.class_names),
        'support_paths': [r.relative_path for r in task.support],
        'support_labels': labels.tolist(), 'support_content_hashes': [r.content_hash for r in task.support],
        'prompts': list(prompts), 'runtime': encoder.runtime_metadata(512),
        'support_augmentation': protocol['support_augmentation'],
        'elapsed_seconds': time.perf_counter() - start}
    write_json(meta_path, meta)
    return meta


def query_cache(task, dataset, split, encoder, model_hash, protocol, root):
    base = root/'cache'/dataset/f'{split}_query'
    import json
    identities = [(r.relative_path, r.label_index, r.original_index) for r in task.query]
    identity_hash = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
    expected = {'protocol_sha256': sha256_file(PROTOCOL), 'query_identity_sha256': identity_hash,
                'model_state_sha256': model_hash}
    if base.with_suffix('.json').exists():
        meta = load_json(base.with_suffix('.json'))
        require_binding(meta, expected)
        if sha256_file(base.with_suffix('.npz')) != meta['array_sha256']:
            raise ValueError('query cache hash mismatch')
        return meta
    f = encoder.encode_images(task.query, batch_size=protocol['batch_size'], show_progress=False)
    rows = np.stack([f[r.relative_path] for r in task.query])
    write_npz(base.with_suffix('.npz'), features=rows,
              labels=np.array([r.label_index for r in task.query], dtype=np.int64))
    meta = {**expected, 'created_utc': now_utc(), 'count': len(task.query),
            'array_sha256': sha256_file(base.with_suffix('.npz')),
            'paths': [r.relative_path for r in task.query], 'split': split,
            'runtime': encoder.runtime_metadata(512)}
    write_json(base.with_suffix('.json'), meta)
    return meta


def _load_support(task, dataset, root):
    with np.load(root/'cache'/dataset/f'{stem(task)}.npz', allow_pickle=False) as f:
        return {key: f[key].copy() for key in f.files}


def prepare_select(root, device, datasets):
    protocol = load_json(PROTOCOL)
    if (root/'freeze_receipt.json').exists():
        raise ValueError('selection is sealed; use evaluate or audit')
    if any((root/d/'raw').exists() for d in DATASETS):
        raise ValueError('formal results already exist; cannot change selection')
    inventory_path = root/'input_inventory.json'
    if not inventory_path.exists():
        inputs = [PROTOCOL, PROJECT/'configs/wm811k_dedup_v2_frozen.json',
                  PROJECT/'configs/fgvc_aircraft_class_reliability_frozen_v1.json']
        for ds in DATASETS:
            inputs.extend(t.manifest_path for t in task_list(ds))
        inputs.extend(t.manifest_path for t in task_list('fgvc', 'test'))
        protected = list((PROJECT/'docs/paper').glob('manuscript_revised_v2_20261008.*'))
        write_json(inventory_path, {'created_utc': now_utc(), 'protocol_sha256': sha256_file(PROTOCOL),
            'input_file_hashes': {str(p.relative_to(PROJECT)): sha256_file(p) for p in inputs},
            'protected_file_hashes': {str(p.relative_to(PROJECT)): sha256_file(p) for p in protected}})
    inventory = load_json(inventory_path)
    require_binding(inventory, {'protocol_sha256': sha256_file(PROTOCOL)})
    for name, digest in inventory['input_file_hashes'].items():
        if sha256_file(PROJECT/name) != digest:
            raise ValueError('input inventory has changed')
    for dataset in datasets:
        tasks = task_list(dataset)
        encoder, model_hash = encoder_for(dataset, device)
        print(f'{now_utc()} {dataset}: frozen encoder ready; weights={model_hash}', flush=True)
        validation = None
        if dataset == 'fgvc':
            query_cache(tasks[0], dataset, 'validation', encoder, model_hash, protocol, root)
            with np.load(root/'cache/fgvc/validation_query.npz') as f:
                validation = (f['features'].copy(), f['labels'].copy())
        for index, task in enumerate(tasks):
            start = time.perf_counter()
            meta = cache_task(task, dataset, encoder, model_hash, protocol, root)
            path = root/'selection'/dataset/f'{stem(task)}.json'
            expected = {**binding(task, sha256_file(PROTOCOL)), 'support_cache_sha256': meta['array_sha256']}
            if path.exists():
                require_binding(load_json(path), expected)
                print(f'{dataset} selection cached {index+1}/30 {stem(task)}', flush=True)
                continue
            arrays = _load_support(task, dataset, root)
            if dataset == 'wm' and task.shot == 1:
                fixed = protocol['wm_selection']['1shot']
                selected = {key: fixed[key] for key in ('beta', 'lambda')}
                result = {'selected': selected, 'candidates': [], 'selection_source': fixed['source'],
                          'query_labels_used_for_selection': False}
            elif dataset == 'wm':
                result = select_support_parameters(arrays['clean'], arrays['views'], arrays['labels'], arrays['text'],
                    protocol['beta_grid'], protocol['lambda_grid'], group_ids=[r.content_hash for r in task.support], device=device)
            else:
                result = select_validation_parameters(arrays['views'], arrays['labels'], arrays['text'],
                    validation[0], validation[1], protocol['beta_grid'], protocol['lambda_grid'], device=device)
            result.update({**expected, 'shot': task.shot, 'seed': task.seed, 'created_utc': now_utc(),
                'elapsed_seconds': time.perf_counter()-start})
            write_json(path, result)
            write_json(root/'progress.json', {'stage': 'selection', 'dataset': dataset,
                'completed_in_dataset': index+1, 'last_task': stem(task), 'updated_utc': now_utc()})
            print(f'{now_utc()} {dataset} selection {index+1}/30 {stem(task)} '
                  f'{time.perf_counter()-start:.1f}s', flush=True)
            import torch
            torch.cuda.empty_cache()
        if dataset == 'fgvc':
            aggregate = {}
            for shot in protocol['shots']:
                candidates = []
                for b in protocol['beta_grid']:
                    for l in protocol['lambda_grid']:
                        per_seed = []
                        for seed in protocol['seeds']:
                            rows = load_json(root/'selection/fgvc'/f'{tasks[0].n_way}way_{shot}shot_seed{seed}.json')['candidates']
                            per_seed.append(next(r for r in rows if r['beta']==b and r['lambda']==l))
                        candidates.append({'beta': b, 'lambda': l,
                            'macro_f1': float(np.mean([r['macro_f1'] for r in per_seed])),
                            'accuracy': float(np.mean([r['accuracy'] for r in per_seed])),
                            'seed_macro_f1': [r['macro_f1'] for r in per_seed]})
                aggregate[str(shot)] = {'selected': choose_candidate(candidates), 'candidates': candidates}
            write_json(root/'selection/fgvc/selected_per_shot.json', {
                'protocol_sha256': sha256_file(PROTOCOL), 'created_utc': now_utc(), 'shots': aggregate,
                'selection_source': 'Validation mean Macro-F1 over ten seeds per shot',
                'query_labels_used_for_selection': False})
        del encoder, validation
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def freeze(root):
    path = root/'freeze_receipt.json'
    if path.exists():
        verify_receipt(path, base_dir=PROJECT)
        print('Existing receipt verified; no overwrite.', flush=True)
        return
    if any((root/d/'raw').exists() for d in DATASETS):
        raise ValueError('Test artifacts exist before freeze')
    inventory = load_json(root/'input_inventory.json')
    hashes = {name: sha256_file(PROJECT/name) for name in inventory['input_file_hashes']}
    require_binding(hashes, inventory['input_file_hashes'])
    selected_tasks = {}
    for dataset in DATASETS:
        selected_tasks[dataset] = {}
        fgvc = load_json(root/'selection/fgvc/selected_per_shot.json') if dataset == 'fgvc' else None
        for task in task_list(dataset):
            selection = load_json(root/'selection'/dataset/f'{stem(task)}.json')
            meta = load_json(root/'cache'/dataset/f'{stem(task)}.json')
            require_binding(meta, binding(task, sha256_file(PROTOCOL)))
            require_binding(selection, {**binding(task, sha256_file(PROTOCOL)), 'support_cache_sha256': meta['array_sha256']})
            if sha256_file(root/'cache'/dataset/f'{stem(task)}.npz') != meta['array_sha256']:
                raise ValueError('support cache changed before freeze')
            pair = selection['selected'] if dataset == 'wm' else fgvc['shots'][str(task.shot)]['selected']
            selected_tasks[dataset][stem(task)] = {key: pair[key] for key in ('beta', 'lambda')}
    files = [PROTOCOL, Path(__file__), PROJECT/'src/wm811k/proker.py', PROJECT/'src/wm811k/proker_artifacts.py',
             PROJECT/'src/wm811k/clip_features.py', PROJECT/'src/wm811k/fewshot_dataset.py',
             PROJECT/'src/wm811k/clip_adaptation.py', PROJECT/'src/wm811k/metrics.py',
             PROJECT/'src/wm811k/experiment_reporting.py', PROJECT/'src/wm811k/prototype.py',
             PROJECT/'src/fgvc_aircraft/prompts.py', root/'input_inventory.json']
    files += list((root/'selection').rglob('*.json')) + list((root/'cache').rglob('*.*'))
    files += list((PROJECT/'docs/proker/reference_20261013').rglob('*.*'))
    file_hashes = {str(p.relative_to(PROJECT)): sha256_file(p) for p in files if p.is_file()}
    file_hashes.update(inventory['input_file_hashes'])
    receipt = {'schema_version': 'proker_freeze_receipt_v1', 'created_utc': now_utc(),
        'protocol_sha256': sha256_file(PROTOCOL), 'selected_tasks': selected_tasks,
        'task_count': 60, 'primary_metric': 'macro_f1', 'secondary_metric': 'accuracy',
        'file_hashes': file_hashes, 'protected_file_hashes': inventory['protected_file_hashes'],
        'test_metrics_computed_before_this_receipt': False,
        'new_baseline_only': True, 'query_labels_used_for_selection': False,
        'history_note': load_json(PROTOCOL)['history_note']}
    write_receipt(path, receipt)
    verify_receipt(path, base_dir=PROJECT)
    print(f'{now_utc()} all 60 task parameter pairs frozen: {path}', flush=True)


def evaluate(root, device, datasets):
    receipt_path = root/'freeze_receipt.json'
    receipt = verify_receipt(receipt_path, base_dir=PROJECT)
    receipt_hash = sha256_file(receipt_path)
    protocol = load_json(PROTOCOL)
    if not (root/'test_evaluation_started.json').exists():
        write_json(root/'test_evaluation_started.json', {'started_utc': now_utc(), 'freeze_receipt_sha256': receipt_hash})
    require_binding(load_json(root/'test_evaluation_started.json'), {'freeze_receipt_sha256': receipt_hash})
    for dataset in datasets:
        tasks = task_list(dataset, 'test')
        encoder, model_hash = encoder_for(dataset, device)
        query_meta = query_cache(tasks[0], dataset, 'test', encoder, model_hash, protocol, root)
        with np.load(root/'cache'/dataset/'test_query.npz') as f:
            query_features, y_true = f['features'].copy(), f['labels'].copy()
        del encoder
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        results = []
        for index, task in enumerate(tasks):
            start = time.perf_counter()
            support_meta = load_json(root/'cache'/dataset/f'{stem(task)}.json')
            require_binding(support_meta, {'support_identity_sha256': support_identity(task), 'model_state_sha256': model_hash})
            expected = {**binding(task, sha256_file(PROTOCOL)), 'freeze_receipt_sha256': receipt_hash,
                        'test_query_cache_sha256': query_meta['array_sha256']}
            result_path = root/dataset/'raw'/f'{stem(task)}.json'
            if result_path.exists() and 'artifact_hashes' in load_json(result_path):
                result = load_json(result_path)
                require_binding(result, expected)
                for name, digest in result['artifact_hashes'].items():
                    if sha256_file(root/dataset/name) != digest:
                        raise ValueError('formal result artifact hash mismatch')
                results.append(result)
                continue
            arrays = _load_support(task, dataset, root)
            rows, labels, _ = expand_support_views(arrays['views'], arrays['labels'])
            pair = receipt['selected_tasks'][dataset][stem(task)]
            fitted = fit_proker(rows, labels, arrays['text'], beta=pair['beta'], lmbda=pair['lambda'], device=device)
            scores = fitted.decision_scores(query_features)
            pred = scores.argmax(axis=1)
            metrics = evaluate_predictions(y_true, pred, task.class_names)
            array_path = root/dataset/'scores'/f'{stem(task)}.npz'
            write_npz(array_path, scores=scores, y_true=y_true, predictions=pred, coefficients=fitted.coefficients)
            prediction_rows = [{'original_index': r.original_index, 'image_path': r.relative_path,
                'true_index': r.label_index, 'true_label': r.label, 'predicted_index': int(p),
                'predicted_label': task.class_names[p], 'cosine_score': float(scores[i,p]),
                'method_score': float(scores[i,p]), 'correct': int(p == r.label_index)}
                for i, (r,p) in enumerate(zip(task.query,pred,strict=True))]
            result = {**task_result_provenance(task), **expected, 'schema_version': 'clip_proker_result_v1',
                'method': 'proker', 'dataset': dataset, 'protocol': task.protocol_name, 'shot': task.shot,
                'seed': task.seed, 'class_names': list(task.class_names), 'debug_run': False, 'paper_eligible': True,
                'support_count': len(task.support), 'query_count': len(task.query), 'created_utc': now_utc(),
                'model_identifier': protocol['model_identifier'], 'model_state_sha256': model_hash,
                'query_labels_used_for_selection': False, 'selection': pair,
                'selection_source': 'Training support only or fixed external source' if dataset=='wm' else 'Frozen Validation mean per shot',
                'support_augmentation': protocol['support_augmentation'], 'support_cache_sha256': support_meta['array_sha256'],
                'logit_scale': 1., 'solver': protocol['solver'], 'reference_commit': protocol['reference_commit'],
                'metrics': metrics, 'elapsed_seconds': time.perf_counter()-start,
                'score_field_note': 'cosine_score is a legacy column; method_score is the final ProKeR score'}
            write_run_artifacts(root/dataset, result, prediction_rows, stem=stem(task), extra_prediction_fields=('method_score',))
            persisted = load_json(result_path)
            artifacts = [array_path, root/dataset/persisted['artifacts']['predictions_csv'],
                         root/dataset/persisted['artifacts']['confusion_matrix_csv']]
            persisted['artifact_hashes'] = {str(p.relative_to(root/dataset)): sha256_file(p) for p in artifacts}
            write_json(result_path, persisted)
            results.append(persisted)
            write_json(root/'progress.json', {'stage': 'formal_test', 'dataset': dataset,
                'completed_in_dataset': index+1, 'last_task': stem(task), 'updated_utc': now_utc()})
            print(f'{now_utc()} {dataset} Test {index+1}/30 {stem(task)} '
                  f'Macro-F1={100*metrics["macro_f1"]:.4f}% Accuracy={100*metrics["accuracy"]:.4f}%', flush=True)
        write_fewshot_summaries(root/dataset, results, expected_seeds=protocol['seeds'])


def audit(root):
    receipt = verify_receipt(root/'freeze_receipt.json', base_dir=PROJECT)
    max_error, checked, hashes = 0., 0, {}
    for dataset in DATASETS:
        tasks = task_list(dataset, 'test')
        results = []
        for task in tasks:
            path = root/dataset/'raw'/f'{stem(task)}.json'
            result = load_json(path)
            require_binding(result, {**binding(task, sha256_file(PROTOCOL)),
                'freeze_receipt_sha256': sha256_file(root/'freeze_receipt.json'),
                'selection': receipt['selected_tasks'][dataset][stem(task)]})
            for name, digest in result['artifact_hashes'].items():
                if sha256_file(root/dataset/name) != digest:
                    raise ValueError('formal artifact changed')
            with np.load(root/dataset/'scores'/f'{stem(task)}.npz') as f:
                pred, y_true = f['predictions'], f['y_true']
                np.testing.assert_array_equal(pred, f['scores'].argmax(axis=1))
                np.testing.assert_array_equal(y_true, [r.label_index for r in task.query])
                metrics = evaluate_predictions(y_true, pred, task.class_names)
            with (root/dataset/result['artifacts']['predictions_csv']).open(encoding='utf-8',newline='') as f:
                records = list(csv.DictReader(f))
            assert [r['image_path'] for r in records] == [r.relative_path for r in task.query]
            np.testing.assert_array_equal([int(r['predicted_index']) for r in records], pred)
            for key in ('macro_f1','accuracy','balanced_accuracy'):
                error = abs(metrics[key] - result['metrics'][key])
                if error > 1e-12:
                    raise ValueError('metrics do not match saved predictions')
                max_error = max(max_error,error)
            assert metrics['confusion_matrix'] == result['metrics']['confusion_matrix']
            checked += 1
            hashes[str(path.relative_to(PROJECT))] = sha256_file(path)
            results.append(result)
        write_fewshot_summaries(root/dataset, results, expected_seeds=range(10))
    for name, digest in receipt['protected_file_hashes'].items():
        if sha256_file(PROJECT/name) != digest:
            raise ValueError('protected old manuscript changed')
    assert receipt['created_utc'] < load_json(root/'test_evaluation_started.json')['started_utc']
    write_json(root/'audit.json', {'completed_utc': now_utc(), 'checked_tasks': checked,
        'max_metric_recalculation_error': max_error, 'same_manifests_and_support_identities': True,
        'all_original_manuscript_hashes_unchanged': True, 'result_hashes': hashes,
        'freeze_receipt_sha256': sha256_file(root/'freeze_receipt.json')})
    print(f'Audit verified {checked}/60 tasks; metric error={max_error}.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare-select','freeze','evaluate','audit','pipeline'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset', choices=('wm','fgvc','both'), default='both')
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == 'both' else (args.dataset,)
    if args.stage == 'pipeline':
        if not (ROOT/'freeze_receipt.json').exists():
            prepare_select(ROOT,args.device,datasets)
            freeze(ROOT)
        evaluate(ROOT,args.device,datasets)
        if args.dataset=='both': audit(ROOT)
    elif args.stage=='prepare-select': prepare_select(ROOT,args.device,datasets)
    elif args.stage=='freeze': freeze(ROOT)
    elif args.stage=='evaluate': evaluate(ROOT,args.device,datasets)
    else: audit(ROOT)


if __name__=='__main__':
    main()
