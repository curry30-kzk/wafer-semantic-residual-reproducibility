"""Standard-library replay of sparse counts, independent selections and paired intervals.

Does not require original caches, images, numpy, torch or a GPU.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics

T975_DF9 = 2.2621571628540993
TOL = 1e-8


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def metrics(sparse):
    n = sparse['class_count']
    if not isinstance(n, int) or n < 2:
        raise ValueError('invalid class count')
    rows, cols, diagonal = [0] * n, [0] * n, [0] * n
    seen = set()
    for i, j, count in sparse['cells']:
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (i, j, count)):
            raise ValueError('integer counts and indices required')
        if not (0 <= i < n and 0 <= j < n and count > 0) or (i, j) in seen:
            raise ValueError('invalid or repeated cell')
        seen.add((i, j))
        rows[i] += count
        cols[j] += count
        if i == j:
            diagonal[i] = count
    if sum(rows) == 0:
        raise ValueError('empty confusion counts')
    return {'macro_f1': statistics.mean(2*d/(r+c) if r+c else 0. for d,r,c in zip(diagonal,rows,cols)),
            'accuracy': sum(diagonal)/sum(rows),
            'balanced_accuracy': statistics.mean(d/r if r else 0. for d,r in zip(diagonal,rows))}


def choose(rows):
    peak = max(r['mean_macro_f1'] for r in rows)
    return next(r for r in rows if r['mean_macro_f1'] >= peak - 1e-12)


def paired(a, b):
    if len(a) != 10 or len(b) != 10:
        raise ValueError('exactly ten paired support seeds required')
    d = [(x-y)*100 for x,y in zip(a,b)]
    mean = statistics.mean(d)
    sd = statistics.stdev(d)
    half = T975_DF9*sd/math.sqrt(10)
    p = sum(abs(statistics.mean(s*x for s,x in zip(signs,d))) >= abs(mean)-1e-12
            for signs in itertools.product((-1,1),repeat=10))/1024
    return {'difference_pp': mean, 'sd_difference_pp': sd,
            'ci95_low_pp': mean-half, 'ci95_high_pp': mean+half,
            'nominal_exact_signflip_p': p,
            'wins': sum(x > 1e-12 for x in d),
            'ties': sum(abs(x) <= 1e-12 for x in d),
            'losses': sum(x < -1e-12 for x in d),
            'multiple_comparisons_corrected': False}


def recompute(root):
    protocol = read(root/'protocol.json')
    errors = []
    def verified(record):
        actual = metrics(record['metrics']['confusion_matrix_sparse'])
        for key,value in actual.items():
            error=abs(value-record['metrics'][key]); errors.append(error)
            if error>TOL: raise ValueError('saved metric differs from counts')
        return actual
    selections={}
    for method in ('classwise','global'):
        rows=[]
        for candidate,config in enumerate(protocol['grid']):
            vals=[]; enabled=0
            for shot in (5,10):
                for seed in range(10):
                    r=read(root/'validation'/method/f'candidate_{candidate:02d}'/f'100way_{shot}shot_seed{seed}.json')
                    if (r['method'],r['candidate'],r['shot'],r['seed']) != (method,candidate,shot,seed):
                        raise ValueError('Validation identity differs')
                    if r['config']!=config or r['test_labels_used_for_selection']:
                        raise ValueError('Validation configuration/scope differs')
                    vals.append(verified(r)['macro_f1']); enabled+=r['enabled']
            rows.append({'candidate':candidate,'config':config,'mean_macro_f1':statistics.mean(vals),
                         'task_count':20,'enabled_count':enabled})
        choice=choose(rows)
        frozen=read(root/'validation'/f'{method}_selection.json')
        if choice['candidate']!=frozen['selected']['candidate']:
            raise ValueError('independent selected candidate differs')
        errors.extend(abs(a['mean_macro_f1']-b['mean_macro_f1']) for a,b in zip(rows,frozen['grid']))
        selections[method]={'grid':rows,'selected':choice}
    output=[]; task_count=0
    for ds in ('fgvc','wm'):
        for shot in (5,10):
            for activation in ('original_activation','unconditional'):
                comparisons=[('visual_oof','classwise_oof','classwise_insample')]
                if ds=='fgvc': comparisons.insert(0,('independent_global','classwise_oof','global_oof'))
                for comparison,first,second in comparisons:
                    method_metrics={}
                    for method in (first,second):
                        metrics_list=[]
                        for seed in range(10):
                            r=read(root/'test'/ds/f'{method}_{activation}'/f'{100 if ds=="fgvc" else 8}way_{shot}shot_seed{seed}.json')
                            if (r['dataset'],r['method'],r['shot'],r['seed']) != (ds,f'{method}_{activation}',shot,seed):
                                raise ValueError('Test identity differs')
                            metrics_list.append(verified(r))
                        method_metrics[method]=metrics_list
                    row={'dataset':ds,'shot':shot,'comparison':comparison,'activation':activation,
                         'first_method':first,'second_method':second,'n_support_seeds':10,
                         'fixed_query_set':True,'historically_unseen_test':False}
                    for method,values in method_metrics.items():
                        for key in ('macro_f1','accuracy','balanced_accuracy'):
                            vals=[v[key] for v in values]
                            row[f'{method}_{key}_mean_percent']=100*statistics.mean(vals)
                            row[f'{method}_{key}_sd_percent']=100*statistics.stdev(vals)
                    for metric in ('macro_f1','accuracy','balanced_accuracy'):
                        row[metric+'_paired']=paired([r[metric] for r in method_metrics[first]],
                                                    [r[metric] for r in method_metrics[second]])
                    output.append(row)
    # Count unique method-task records rather than appearances across comparisons.
    task_count=len(list((root/'test').rglob('*.json')))
    if task_count!=200: raise ValueError('expected 200 unique method-task records')
    summary_path=root/'summary.json'
    if summary_path.exists():
        saved=read(summary_path)
        def check(a,b):
            if isinstance(b,dict):
                for k,v in b.items():check(a[k],v)
            elif isinstance(b,list):
                if len(a)!=len(b):raise ValueError('summary length differs')
                for x,y in zip(a,b):check(x,y)
            elif isinstance(b,(int,float)) and not isinstance(b,bool):
                error=abs(a-b); errors.append(error)
                if error>TOL:raise ValueError('summary replay differs')
            elif a!=b:raise ValueError('summary metadata differs')
        check(output,saved)
    if max(errors)>TOL:raise ValueError('selection replay differs')
    return output,selections,{'all_passed':True,'standard_library_only':True,
        'validation_method_task_count':640,'test_method_task_count':task_count,
        'summary_row_count':len(output),'numeric_comparison_count':len(errors),
        'max_absolute_error':max(errors),'tolerance':TOL,
        'paired_unit':'ten support-sampling seeds with fixed query set',
        'historically_unseen_test':False,'multiple_comparisons_corrected':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]/'results/teacher_review_v11_20261017/global_control')
    args=p.parse_args()
    _,_,audit=recompute(args.root)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
