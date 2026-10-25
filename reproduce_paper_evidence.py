"""Recompute newly released v20 evidence from original branch task records.

This does not encode images or refit models. Display targets are comparison
targets only; no confusion counts are reconstructed from manuscript summaries.
"""
from pathlib import Path
import argparse,json
import numpy as np
from release_integrity import verify_release

ROOT=Path(__file__).resolve().parent
CLASSES=['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))

def f1(matrix):
    a=np.asarray(matrix)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.issubdtype(a.dtype,np.integer) or (a<0).any():
        raise ValueError('Expected a square nonnegative integer confusion matrix')
    den=a.sum(0)+a.sum(1)
    return np.divide(2*np.diag(a),den,out=np.zeros(len(den),dtype=float),where=den>0)

def metric(task,field='metrics'):
    values=task[field]
    score=float(f1(values['confusion_matrix']).mean())
    if abs(score-values['macro_f1'])>1e-12:raise ValueError('Scalar metric differs from integer confusion counts')
    return score

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    integrity=verify_release(ROOT)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    target=read('paper_evidence/manuscript_targets.json')
    checks=[]
    def check(name,actual,expected,decimals=None):
        tolerance=1e-10 if decimals is None else 0.5*10**(-decimals)+1e-10
        checks.append({'name':name,'actual':float(actual),'expected':float(expected),'tolerance':tolerance,'passed':abs(actual-expected)<=tolerance})
    source_rows=[]
    for name in target['semantic_sources']:
        data=read(f'paper_evidence/semantic_sources/{name}.json')
        for shot in [5,10]:
            rows=sorted((r for r in data['task_results'] if r['shot']==shot),key=lambda r:r['seed'])
            assert [r['seed'] for r in rows]==list(range(10))
            assert all(r['query_source']=='Validation' for r in rows)
            values=100*np.array([metric(r) for r in rows])
            row={'source':name,'shot':shot,'mean_percent':float(values.mean()),'sample_sd_percent':float(values.std(ddof=1))}
            for field in ['mean_percent','sample_sd_percent']:
                check(f'C1/{name}/{shot}/{field}',row[field],target['semantic_sources'][name][str(shot)][field],2)
            source_rows.append(row)

    raw={};weight_rows=[];diagnostics=[]
    for semantic,dataset in [('llm_structured','WM Structured'),('qwen2_5_7b_instruct','WM Qwen')]:
        for shot in [5,10]:
            selected=[];candidate=[]
            for seed in range(10):
                stem=f'8way_{shot}shot_seed{seed}.json'
                full=read(f'results/wm811k_dedup_v2/{semantic}/raw/classwise_reliability/{stem}')
                free=read(f'results/wm811k_dedup_v2/{semantic}/raw/without_safe_fallback/{stem}')
                raw[semantic,shot,seed]=full
                selected.append(full);candidate.append(free)
                metric(full);metric(free);metric(full,'anchor_metrics')
            delta=100*np.array([metric(r)-metric(r,'anchor_metrics') for r in selected])
            candidate_delta=100*np.array([metric(r)-metric(r,'anchor_metrics') for r in candidate])
            enabled=np.array([bool(r['selection']['enabled']) for r in selected])
            missed=candidate_delta[~enabled & (candidate_delta>1e-10)]
            row={'dataset':dataset,'shot':shot,'enabled':int(enabled.sum()),'declines':int((enabled & (delta < -1e-10)).sum()),
                'minimum_delta_pp':float(delta.min()),'false_rejections':len(missed),
                'avoided_declines':int((~enabled & (candidate_delta < -1e-10)).sum()),
                'maximum_missed_pp':float(missed.max()) if len(missed) else 0.0}
            diagnostics.append(row)
            if semantic=='llm_structured':
                w=np.array([r['selection']['effective_reliability'] for r in selected])
                for j,name in enumerate(CLASSES):
                    wr={'shot':shot,'class_name':name,'mean':float(w[:,j].mean()),'sample_sd':float(w[:,j].std(ddof=1)),
                        'nonzero_tasks':int((w[:,j]>0).sum())}
                    expected=next(r for r in target['weights'] if r['shot']==shot and r['class_name']==name)
                    for field in ['mean','sample_sd']:check(f'C2/{shot}/{name}/{field}',wr[field],expected[field],4)
                    check(f'C2/{shot}/{name}/nonzero',wr['nonzero_tasks'],expected['nonzero_tasks'])
                    weight_rows.append(wr)
    fgvc=read('paper_evidence/fgvc_original_test.json')['variants']
    for shot in [5,10]:
        selected=sorted((r for r in fgvc['classwise_reliability'] if r['shot']==shot),key=lambda r:r['seed'])
        free=sorted((r for r in fgvc['without_safe_fallback'] if r['shot']==shot),key=lambda r:r['seed'])
        assert len(selected)==len(free)==10
        delta=100*np.array([metric(r)-metric(r,'anchor_metrics') for r in selected])
        candidate_delta=100*np.array([metric(r)-metric(r,'anchor_metrics') for r in free])
        enabled=np.array([bool(r['selection']['enabled']) for r in selected])
        missed=candidate_delta[~enabled & (candidate_delta>1e-10)]
        diagnostics.append({'dataset':'FGVC Hybrid','shot':shot,'enabled':int(enabled.sum()),
            'declines':int((enabled & (delta < -1e-10)).sum()),'minimum_delta_pp':float(delta.min()),
            'false_rejections':len(missed),'avoided_declines':int((~enabled & (candidate_delta < -1e-10)).sum()),
            'maximum_missed_pp':float(missed.max()) if len(missed) else 0.0})
    for row in diagnostics:
        expected=next(x for x in target['activation'] if x['dataset']==row['dataset'] and x['shot']==row['shot'])
        for field in ['enabled','declines','false_rejections','avoided_declines']:check(f'C3/{row["dataset"]}/{row["shot"]}/{field}',row[field],expected[field])
        for field in ['minimum_delta_pp','maximum_missed_pp']:check(f'C3/{row["dataset"]}/{row["shot"]}/{field}',row[field],expected[field],3)

    # The exact historical grid, kept separate from later re-encoded candidates.
    grids=read('paper_evidence/original_validation_grid_tasks.json')
    historical=read('results/revision_v4_20261013/validation_sensitivity/per_shot_grid.json')
    grid_rows=[]
    for config in grids:
        for shot in [5,10]:
            rows=[r for r in config['task_results'] if r['shot']==shot]
            assert len(rows)==10
            values=100*np.array([metric(r) for r in rows])
            expected=next(x for x in historical if x['shot']==shot and x['lambda']==config['lambda'] and x['rho']==config['rho'])
            check(f'FigureC1/{shot}/{config["lambda"]}/{config["rho"]}/mean',float(values.mean()),expected['mean_macro_f1_percent'])
            check(f'FigureC1/{shot}/{config["lambda"]}/{config["rho"]}/sd',float(values.std(ddof=1)),expected['sample_sd_percent'])
            grid_rows.append({**expected,'recomputed_mean_percent':float(values.mean()),'recomputed_sample_sd_percent':float(values.std(ddof=1))})

    exclusion=read('paper_evidence/conflict_query_exclusion.json');exclusion_rows=[]
    for task in exclusion['tasks']:
        counts=np.array(task['original_confusion'],dtype=np.int64)
        for row in task['excluded_query_records']:counts[row['true_index'],row['predicted_index']]-=1
        assert counts.sum()==7891
        exclusion_rows.append({'shot':task['shot'],'seed':task['seed'],'macro_f1':float(f1(counts).mean())})
    for shot in [5,10]:
        mean=100*np.mean([r['macro_f1'] for r in exclusion_rows if r['shot']==shot])
        check(f'C5/excluded/{shot}/mean',mean,target['conflict_exclusion_mean_percent'][str(shot)],3)
    cases=read('docs/paper/revision_v4_20261013/figures/case_provenance.json')
    for row in cases['entries']:
        check('Figure4/anchor/'+str(row['original_index']),int(np.argmax(row['anchor_scores'])),CLASSES.index(row['anchor_prediction']))
        check('Figure4/fused/'+str(row['original_index']),int(np.argmax(row['fused_scores'])),CLASSES.index(row['residual_prediction']))
        original=raw['llm_structured',5,0]
        check('Figure4/saved_prediction/'+str(row['original_index']),original['predictions'][row['query_position_zero_based']],CLASSES.index(row['residual_prediction']))

    for name,data in [('source_comparison',source_rows),('coefficients',weight_rows),('activation',diagnostics),('validation_grid',grid_rows),('query_exclusion',exclusion_rows)]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
    failures=[x for x in checks if not x['passed']]
    report={'all_passed':not failures,'checks':len(checks),'failures':failures,'integrity':integrity,
        'scope':'Original supporting branch counts/coefficients/OOF decisions/grid/query exclusion/case scores only; no encoder or model training.',
        'historical_test_independence_restored':False,'full_image_to_result_reproduction':False}
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
