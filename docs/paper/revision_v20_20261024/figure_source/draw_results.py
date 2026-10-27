# v20 typography revision; prior figure labels and scientific calculations retained.
"""Render archived results only; no model execution or selection.

The formal seven-method table remains the original single-prompt comparison.
All plotted Macro-F1 values are recalculated from saved confusion counts.  The
post-hoc, uniformly disabled fallback comparison is kept in separate panels.
Run with: python draw_results.py [--project-root PATH] [--output-dir PATH]
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import t
HERE = Path(__file__).resolve().parent
BLUE, ORANGE = ('#24658B', '#C47639')
INK, GRAY, GRID = ('#263540', '#63717B', '#E5EAEE')
METHODS = [('classwise_reliability', 'Classwise residual (ours)', BLUE, 'o', '-', 2.1), ('visual_anchor', 'Visual anchor', ORANGE, 's', '--', 1.5), ('proker', 'ProKeR (single prompt)', '#59676F', 'D', '-', 1.05), ('lp_plus_plus_fair', 'LP++', '#8F9BA3', '^', '--', 1.0), ('zs_lp', 'ZS-LP', '#A8B0B6', 'v', '-.', 1.0), ('clap', 'CLAP', '#778B99', 'P', ':', 1.05), ('rla', 'RLA', '#BAC0C5', 'X', '--', 1.0)]
SEEDS, SHOTS = (list(range(10)), (1, 5, 10))

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def macro_f1(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    assert np.all(matrix >= 0) and matrix.sum() > 0
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(matrix.shape[0], dtype=float), where=denominator > 0)
    return float(f1.mean())

def describe(values):
    values = np.asarray(values, dtype=float)
    assert len(values) == 10 and np.isfinite(values).all()
    mean, sd = (float(values.mean()), float(values.std(ddof=1)))
    half = float(t.ppf(0.975, 9) * sd / np.sqrt(10))
    return dict(mean=mean, sample_std=sd, ci95_low=mean - half, ci95_high=mean + half, seed_values=values.tolist(), seed_count=10)

def paired(left, right):
    assert sorted(left) == sorted(right) == SEEDS
    values = np.array([left[s] - right[s] for s in SEEDS]) * 100
    result = describe(values)
    result.update(seed_ids=SEEDS, unit='percentage points (pp)', wins=int(sum(values > 1e-10)), ties=int(sum(abs(values) <= 1e-10)), losses=int(sum(values < -1e-10)))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=HERE.parents[3])
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    root, out = (args.project_root.resolve(), args.output_dir.resolve())
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def source(path):
        path = root / path
        rel = path.relative_to(root).as_posix()
        sources[rel] = {'path': rel, 'sha256': sha256(path)}
        return path

    def read(path):
        return json.loads(source(path).read_text(encoding='utf-8-sig'))
    base = 'docs/paper/revision_v6_20261015/reproduction'
    payload = read(f'{base}/metrics_inputs.json')
    reference = read('paper_evidence/figures/reference_statistics.json')
    groups, records = (defaultdict(dict), defaultdict(list))
    selected = {m[0] for m in METHODS}
    for record in payload['records']:
        uses = [u for u in record['uses'] if u['scope'] == 'main' and u['method'] in selected or (u['scope'] == 'ablation' and u['method'] in {'classwise_reliability', 'global_reliability'})]
        if not uses:
            continue
        cm = np.zeros((record['n_classes'], record['n_classes']), dtype=np.int64)
        for row, col, count in record['confusion_entries']:
            cm[row, col] = count
        assert int(cm.sum()) == record['n_query']
        score = macro_f1(cm)
        for use in uses:
            key = (use['scope'], use['dataset'], use['method'], record['shot'])
            assert record['seed'] not in groups[key]
            groups[key][record['seed']] = score
            records[key].append({'seed': record['seed'], 'record_id': record['record_id'], 'source_id': record['source_id'], 'source_sha256': record['source_sha256']})
    summary_index = {(r['scope'], r['dataset'], r['method'], r['shot']): r for r in reference['summary']}
    paired_index = {(r['scope'], r['dataset'], r['shot'], r['right']): r for r in reference['paired_comparisons']}
    summaries, checked = ({}, [])
    for key, values in groups.items():
        assert sorted(values) == SEEDS
        result = describe([values[s] for s in SEEDS])
        ref = summary_index[key]['macro_f1']
        error = max((abs(result[field] - ref[field]) for field in ('mean', 'sample_std', 'ci95_low', 'ci95_high')))
        assert error < 1e-12, (key, error)
        summaries[key] = result
        checked.append({'kind': 'summary_from_confusion_counts', 'key': list(key), 'max_absolute_error': error})
    for dataset in ('WM_v2', 'FGVC_Test'):
        classwise = groups['main', dataset, 'classwise_reliability', 1]
        anchor = groups['main', dataset, 'visual_anchor', 1]
        assert classwise == anchor
        checked.append({'kind': 'one_shot_recovers_prescribed_anchor', 'dataset': dataset, 'ten_seed_values_exactly_equal': True})
    original, uniform = ({}, {})
    no_fallback = 'results/revision_v4_20261013/without_fallback'
    archived_pairs = read(f'{no_fallback}/paired_comparisons.json')
    with source(f'{no_fallback}/seed_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        archived_seeds = {(r['dataset'], int(r['shot']), int(r['seed']), r['granularity']): float(r['macro_f1_percent']) / 100 for r in csv.DictReader(stream)}
    mechanism_records = []
    for ds, ablation_ds in (('wm', 'WM_structured_v2'), ('fgvc', 'FGVC_Test')):
        for shot in (5, 10):
            left = groups['ablation', ablation_ds, 'classwise_reliability', shot]
            right = groups['ablation', ablation_ds, 'global_reliability', shot]
            original[ds, shot] = paired(left, right)
            ref = paired_index['ablation', ablation_ds, shot, 'global_reliability']['macro_f1']
            error = max((abs(original[ds, shot][field] / 100 - ref[field]) for field in ('mean', 'sample_std', 'ci95_low', 'ci95_high')))
            assert error < 1e-12
            checked.append({'kind': 'original_mechanism_paired_interval', 'dataset': ds, 'shot': shot, 'max_absolute_error': error})
            no_fallback_values = {'classwise': {}, 'global': {}}
            for granularity in ('classwise', 'global'):
                for seed in SEEDS:
                    way = 8 if ds == 'wm' else 100
                    path = f'{no_fallback}/{ds}/{granularity}_{way}way_{shot}shot_seed{seed}.json'
                    record = read(path)
                    assert record['shot'] == shot and record['seed'] == seed
                    assert record['safe_fallback'] is False
                    assert record['query_labels_used_for_selection'] is False
                    value = macro_f1(record['metrics']['confusion_matrix'])
                    assert abs(value - record['metrics']['macro_f1']) < 1e-12
                    assert abs(value - archived_seeds[ds, shot, seed, granularity]) < 1e-12
                    no_fallback_values[granularity][seed] = value
                    mechanism_records.append({'path': path, 'dataset': ds, 'shot': shot, 'seed': seed, 'granularity': granularity, 'macro_f1': value})
            uniform[ds, shot] = paired(no_fallback_values['classwise'], no_fallback_values['global'])
            archive = next((r for r in archived_pairs if r['dataset'] == ds and r['shot'] == shot))
            error = max((abs(uniform[ds, shot][field] - archive[archive_field]) for field, archive_field in (('mean', 'mean_difference_pp'), ('sample_std', 'sample_sd_difference_pp'), ('ci95_low', 'ci95_low_pp'), ('ci95_high', 'ci95_high_pp'))))
            assert error < 1e-10
            checked.append({'kind': 'uniform_no_fallback_paired_interval_from_confusions', 'dataset': ds, 'shot': shot, 'max_absolute_error_pp': error})
    cjk = 'DejaVu Sans (installed fallback)'
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans', 'sans-serif'], 'font.size': 9.5, 'axes.labelsize': 10.0, 'axes.titlesize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False, 'axes.edgecolor': '#9BA8B0', 'axes.linewidth': 0.6, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': GRAY, 'ytick.color': GRAY, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    fig = plt.figure(figsize=(7.2, 7.6))
    lefts, width = ([0.085, 0.575], 0.395)
    axes = [fig.add_axes([left, 0.565, width, 0.27]) for left in lefts]
    lower = [fig.add_axes([left, 0.115, width, 0.235]) for left in lefts]

    def polish(ax):
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', width=0.55, length=3, pad=3)
        ax.set_axisbelow(True)
    handles = {}
    for panel, (ax, dataset, title) in enumerate(zip(axes, ('WM_v2', 'FGVC_Test'), ('(a) WM-811K: retrospective\nfrozen-config evaluation', '(b) FGVC: Test after\nValidation selection'))):
        for method, label, color, marker, style, line_width in METHODS[::-1]:
            rows = [summaries['main', dataset, method, shot] for shot in SHOTS]
            mean = 100 * np.array([r['mean'] for r in rows])
            sd = 100 * np.array([r['sample_std'] for r in rows])
            highlighted = method in {'classwise_reliability', 'visual_anchor'}
            result = ax.errorbar(range(3), mean, yerr=sd, color=color, linestyle=style, marker=marker, ms=5.0 if highlighted else 4.0, lw=line_width, elinewidth=0.85 if highlighted else 0.6, capsize=2, capthick=0.65, zorder=6 if method == 'classwise_reliability' else 3, alpha=1.0 if highlighted else 0.9)
            handles[method] = Line2D([], [], color=color, linestyle=style, marker=marker, markersize=5, linewidth=line_width, label=label)
            if method == 'classwise_reliability':
                ax.plot(0, mean[0], marker='o', ms=6.5, markeredgewidth=1.6, markerfacecolor='white', markeredgecolor=BLUE, zorder=8)
                ax.plot(0, mean[0], marker='s', ms=2.5, color=ORANGE, zorder=9)
        ax.set(xlim=(-0.12, 2.12), ylim=(8, 50), xticks=range(3), xticklabels=['1-shot', '5-shot', '10-shot'], yticks=[10, 20, 30, 40, 50])
        ax.set_title(title, loc='left', pad=10, fontsize=10.5)
        ax.set_xlabel('Support images per class', labelpad=5)
        if panel == 0:
            ax.set_ylabel('Macro-F1(%)', labelpad=5)
        ax.grid(axis='y', color=GRID, linewidth=0.55)
        polish(ax)
    fig.legend([handles[m[0]] for m in METHODS], [m[1] for m in METHODS], loc='upper center', bbox_to_anchor=(0.52, 0.994), ncol=4, frameon=False, fontsize=9.5, handlelength=2.1, columnspacing=1.25, handletextpad=0.55, labelspacing=0.65)
    fig.text(0.085, 0.917, 'Main comparison: mean ± sample SD (10 support seeds)', fontsize=9.5, color=GRAY)
    fig.text(0.085, 0.484, 'Hollow circle: 1-shot uses the same anchor; no residual is fitted', fontsize=9.5, color=GRAY)
    fig.text(0.085, 0.451, 'Classwise − global: paired differences and nominal 95% t intervals', fontsize=10.5)
    mechanism_handles = [Line2D([], [], color=BLUE, marker='o', lw=1.5, ms=5, label='Original empirical activation rule'), Line2D([], [], color=ORANGE, marker='s', lw=1.5, ms=5, label='Fallback disabled for both (post hoc)')]
    fig.legend(handles=mechanism_handles, loc='center', bbox_to_anchor=(0.52, 0.416), ncol=2, frameon=False, fontsize=9.5, handlelength=1.9, handletextpad=0.5, columnspacing=2.0, labelspacing=0.65)
    for panel, (ax, ds, title) in enumerate(zip(lower, ('wm', 'fgvc'), ('(c) WM-811K: paired differences', '(d) FGVC Test: paired differences'))):
        for shot, center in ((5, 1.0), (10, 0.0)):
            for results, color, marker, offset, label_dy, label_va in ((original, BLUE, 'o', 0.15, 7, 'bottom'), (uniform, ORANGE, 's', -0.15, -7, 'top')):
                row = results[ds, shot]
                ax.errorbar(row['mean'], center + offset, xerr=[[row['mean'] - row['ci95_low']], [row['ci95_high'] - row['mean']]], fmt=marker, color=color, ms=5, elinewidth=1.4, capsize=3, capthick=1, markeredgecolor='white', markeredgewidth=0.45, zorder=4)
                ax.annotate(f"+{row['mean']:.2f}", (row['mean'], center + offset), xytext=(2, label_dy), textcoords='offset points', color=color, fontsize=9.5, ha='left', va=label_va)
        ax.axvline(0, color='#9DAAB3', lw=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.set(xlim=(-0.3, 5.1), ylim=(-0.55, 1.55), yticks=[0, 1], yticklabels=['10-shot', '5-shot'], xticks=[0, 1, 2, 3, 4, 5])
        ax.set_xlabel('Paired Macro-F1 difference (pp)', labelpad=6)
        ax.set_title(title, loc='left', pad=10, fontsize=10.5)
        ax.grid(False)
        ax.spines['left'].set_visible(False)
        polish(ax)
        ax.tick_params(axis='y', length=0, pad=4)
    fig.text(0.085, 0.025, 'Fixed queries; 10 support draws; no multiple-comparison adjustment', fontsize=9.5, color=GRAY)
    fig.canvas.draw()
    from audit_panel_alignment import require_matplotlib_panel_alignment
    alignment = require_matplotlib_panel_alignment(fig, axes=axes + lower, panel_ids=['a', 'b', 'c', 'd'], row_groups=[['a', 'b'], ['c', 'd']], column_groups=[['a', 'c'], ['b', 'd']], json_out=out / 'fig02_panel_alignment.json', overlay_svg=out / 'fig02_panel_alignment.svg')
    all_sizes = [obj.get_fontsize() for obj in fig.findobj(plt.Text) if obj.get_text()]
    minimum_font_pt = float(min(all_sizes))
    assert minimum_font_pt >= 9.5
    fig.savefig(out / 'fig02_results.pdf')
    fig.savefig(out / 'fig02_results.svg')
    fig.savefig(out / 'fig02_results.png', dpi=300)
    plt.close(fig)
    plotted_main = []
    for dataset in ('WM_v2', 'FGVC_Test'):
        for method, *_ in METHODS:
            for shot in SHOTS:
                key = ('main', dataset, method, shot)
                plotted_main.append({'scope': 'main', 'dataset': dataset, 'method': method, 'shot': shot, **summaries[key], 'source_records': records[key]})
    mechanisms = [{'dataset': ds, 'shot': shot, 'condition': condition, **result[ds, shot]} for ds in ('wm', 'fgvc') for shot in (5, 10) for condition, result in (('original_empirical_enable', original), ('uniform_no_fallback_post_hoc', uniform))]
    provenance = {'figure': 'fig02_results', 'creation': "Original layout of this study's saved results; no external figure or empirical value reused.", 'data_rule': 'Macro-F1 recalculated from saved confusion matrices; original main table never replaced by new same-text values.', 'protocols': {'WM_v2': 'frozen deduplicated remigration; not an independent untouched test', 'FGVC_Test': 'Test after Validation hyperparameter selection', 'mechanism_original': 'same semantics and visual cache; classwise versus global; original empirical enable rule', 'mechanism_uniform': 'same cached scores, both classwise/global disable fallback; post-hoc robustness analysis'}, 'uncertainty': {'top_panels': 'mean ± sample SD, ddof=1, ten support seeds; percent', 'lower_panels': 'mean of seed-paired differences ± unadjusted two-sided 95% Student t interval, df=9; pp', 'scope': 'Fixed query sets; support-sampling uncertainty only; no multiple-comparison adjustment'}, 'same_text_control': 'Not plotted. New same-text/re-encoded ProKeR belongs to a separate comparison in the manuscript.', 'one_shot': 'No residual estimation; classwise method exactly restores the same prescribed anchor, shown by hollow blue circle and orange center. WM 1-shot uses a visual prototype; FGVC 1-shot uses Tip-Adapter-F, which includes generic CLIP prompts.', 'geometry': {'native_inches': [7.2, 7.6], 'recommended_width_cm': 16.5, 'shared_scales': 'Top panels share the Macro-F1 axis; bottom panels share the pp axis.', 'minimum_native_font_pt': minimum_font_pt, 'minimum_font_pt_at_16_5cm': minimum_font_pt * 16.5 / (7.2 * 2.54), 'fonts': ['DejaVu Sans', str(cjk)], 'pdf_fonttype': 42, 'svg_fonttype': 'none', 'raster_dpi': 300}, 'verification': {'all_passed': True, 'checked_summary_cells': len(summaries), 'main_cells': len(plotted_main), 'paired_comparisons': len(mechanisms), 'no_fallback_raw_tasks': len(mechanism_records), 'checks': checked}, 'inputs': list(sources.values()), 'main_series': plotted_main, 'mechanism_pairs': mechanisms, 'no_fallback_raw_records': mechanism_records}
    provenance['outputs'] = [{'path': f'fig02_results.{extension}', 'sha256': sha256(out / f'fig02_results.{extension}')} for extension in ('pdf', 'png', 'svg')]
    (out / 'fig02_sources.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'outputs': str(out), 'main_cells': len(plotted_main), 'mechanism_pairs': len(mechanisms), 'raw_no_fallback_tasks': len(mechanism_records), 'minimum_native_font_pt': minimum_font_pt, 'minimum_font_pt_at_16_5cm': minimum_font_pt * 16.5 / (7.2 * 2.54), 'all_checks_passed': True}, ensure_ascii=False))
if __name__ == '__main__':
    main()
