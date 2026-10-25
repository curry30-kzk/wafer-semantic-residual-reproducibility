"""Refit the frozen classwise residual from verified historical score caches.

This is cache-to-result adaptation, not image encoding or a new independent Test.
Archived strict CLAP/global runners retain their original freeze receipts and
are not silently patched or called with hash checks disabled.
"""
from pathlib import Path
import argparse,json,sys,time
from release_integrity import verify_release

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=['wm_structured','wm_qwen','fgvc'],default=['wm_structured'])
    p.add_argument('--shots',nargs='+',type=int,choices=[5,10],default=[5])
    p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if any(seed<0 or seed>9 for seed in a.seeds):raise ValueError('Only archived seeds0-9 are defined')
    integrity=verify_release(ROOT,with_caches=True)
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    identities={x['path']:x for x in manifest['files']}
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from wm811k.class_reliability_experiment import load_task_cache,evaluate_cached_task
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    fgvc=json.loads((ROOT/'paper_evidence/fgvc_original_test.json').read_text(encoding='utf-8'))['variants']['classwise_reliability']
    rows=[]
    for dataset in a.datasets:
        if dataset=='fgvc':
            config=json.loads((ROOT/'configs/fgvc_aircraft_class_reliability_frozen_v1.json').read_text(encoding='utf-8'))
            strength,rho=config['selected_lambda'],config['selected_rho']
        else:
            config=json.loads((ROOT/'configs/wm811k_dedup_v2_frozen.json').read_text(encoding='utf-8'))
            strength,rho=config['lambda'],config['rho']
        for shot in a.shots:
            for seed in a.seeds:
                if dataset=='fgvc':
                    rel=f'results/fgvc_aircraft_test/class_reliability_residual/cache/100way_{shot}shot_seed{seed}.json'
                    task_manifest=f'data/processed/FGVC-Aircraft/splits/test/100way_{shot}shot_seed{seed}.json'
                    target=next(x for x in fgvc if x['shot']==shot and x['seed']==seed)
                else:
                    semantic='llm_structured' if dataset=='wm_structured' else 'qwen2_5_7b_instruct'
                    rel=f'results/wm811k_dedup_v2/{semantic}/cache/8way_{shot}shot_seed{seed}.json'
                    task_manifest=f'data/processed/WM811K_dedup_v2/splits/8way_{shot}shot_seed{seed}.json'
                    target=json.loads((ROOT/f'results/wm811k_dedup_v2/{semantic}/raw/classwise_reliability/8way_{shot}shot_seed{seed}.json').read_text(encoding='utf-8'))
                expected_original=identities[task_manifest]['original_sha256']
                cache=load_task_cache(ROOT/rel,expected_manifest_sha256=expected_original)
                start=time.monotonic()
                actual=evaluate_cached_task(cache,lambda_value=strength,rho=rho,granularity='classwise')
                metric_errors={k:abs(actual['metrics'][k]-target['metrics'][k]) for k in ['macro_f1','accuracy','balanced_accuracy']}
                passed=all(x<=1e-10 for x in metric_errors.values()) and np.array_equal(actual['metrics']['confusion_matrix'],target['metrics']['confusion_matrix']) and actual['predictions']==target['predictions'] and actual['selection']['enabled']==target['selection']['enabled']
                row={'dataset':dataset,'shot':shot,'seed':seed,'lambda':strength,'rho':rho,'all_passed':bool(passed),
                    'metric_absolute_errors':metric_errors,'predictions_identical':actual['predictions']==target['predictions'],
                    'seconds':round(time.monotonic()-start,3)}
                rows.append(row)
                (out/f'{dataset}_{shot}shot_seed{seed}.json').write_text(json.dumps(actual),encoding='utf-8')
                (out/'verification.json').write_text(json.dumps({'all_passed':all(x['all_passed'] for x in rows),'tasks':rows,
                    'integrity':integrity,'scope':'Frozen classwise residual refitting from historical score caches only; no encoder, LP/OOF reconstruction, full baseline rerun or new independent Test.'},indent=2),encoding='utf-8')
                print(dataset,shot,seed,'PASS' if passed else 'FAIL',flush=True)
                if not passed:raise SystemExit(1)

if __name__=='__main__':main()
