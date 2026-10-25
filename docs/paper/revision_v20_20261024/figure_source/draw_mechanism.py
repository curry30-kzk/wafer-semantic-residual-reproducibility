# v20 typography revision; prior figure labels and scientific calculations retained.
"""Rebuild the v6 mechanism diagnostic from saved WM-811K v2 task results.

No model fitting or query-driven selection is performed. The source JSON keeps
every plotted number, source hashes, and the exact comparison definitions.
If the raw project results are absent, this script can redraw from that JSON.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment
CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
BLUE = '#276B9F'
ORANGE = '#C86D32'
INK = '#26323B'
MID = '#6D7780'
LIGHT = '#E9EDF0'

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def collect_sources(root: Path) -> dict:
    raw = root / 'results/wm811k_dedup_v2/llm_structured/raw'
    source_files = []

    def read_task(variant: str, shot: int, seed: int) -> dict:
        path = raw / variant / f'8way_{shot}shot_seed{seed}.json'
        source_files.append({'path': path.relative_to(root).as_posix(), 'sha256': digest(path)})
        return json.loads(path.read_text(encoding='utf-8'))
    five_shot = []
    task_diagnostics = []
    for shot in (5, 10):
        for seed in range(10):
            full = read_task('classwise_reliability', shot, seed)
            candidate = read_task('without_safe_fallback', shot, seed)
            selection = full['selection']
            assert np.isclose(full['anchor_metrics']['macro_f1'], candidate['anchor_metrics']['macro_f1'], atol=1e-14)
            assert np.allclose(selection['learned_reliability'], candidate['selection']['learned_reliability'], atol=1e-14, rtol=0)
            expected_selected = candidate['metrics']['macro_f1'] if selection['enabled'] else full['anchor_metrics']['macro_f1']
            assert np.isclose(full['metrics']['macro_f1'], expected_selected, atol=1e-14, rtol=0)
            if shot == 5:
                five_shot.append({'shot': shot, 'seed': seed, 'enabled': selection['enabled'], 'effective_coefficients': selection['effective_reliability'], 'proposed_per_class_f1': [full['metrics']['per_class'][c]['f1'] for c in CLASSES], 'internal_anchor_per_class_f1': [full['anchor_metrics']['per_class'][c]['f1'] for c in CLASSES], 'proposed_confusion': full['metrics']['confusion_matrix'], 'internal_anchor_confusion': full['anchor_metrics']['confusion_matrix']})
            task_diagnostics.append({'shot': shot, 'seed': seed, 'enabled': bool(selection['enabled']), 'reason': selection['reason'], 'support_oof_base_macro_f1': selection['base_support_macro_f1'], 'support_oof_candidate_macro_f1': selection['fused_support_macro_f1'], 'support_oof_delta_pp': 100 * (selection['fused_support_macro_f1'] - selection['base_support_macro_f1']), 'internal_anchor_test_macro_f1': full['anchor_metrics']['macro_f1'], 'candidate_without_fallback_test_macro_f1': candidate['metrics']['macro_f1'], 'selected_test_macro_f1': full['metrics']['macro_f1'], 'candidate_test_delta_pp': 100 * (candidate['metrics']['macro_f1'] - full['anchor_metrics']['macro_f1']), 'selected_test_delta_pp': 100 * (full['metrics']['macro_f1'] - full['anchor_metrics']['macro_f1'])})
    weights = np.array([r['effective_coefficients'] for r in five_shot])
    delta = 100 * (np.array([r['proposed_per_class_f1'] for r in five_shot]) - np.array([r['internal_anchor_per_class_f1'] for r in five_shot]))
    summary = []
    for idx, name in enumerate(CLASSES):
        summary.append({'class_index': idx, 'class_name': name, 'mean_effective_coefficient': float(weights[:, idx].mean()), 'sample_sd_effective_coefficient': float(weights[:, idx].std(ddof=1)), 'mean_delta_f1_pp': float(delta[:, idx].mean()), 'sample_sd_delta_f1_pp': float(delta[:, idx].std(ddof=1))})
    for filename, field, actual in [('classwise_f1.csv', 'mean_delta', delta.mean(axis=0) / 100), ('reliability_weights.csv', 'mean_weight', weights.mean(axis=0))]:
        path = root / 'results/wm811k_dedup_v2/analysis' / filename
        with path.open(encoding='utf-8-sig', newline='') as stream:
            records = [r for r in csv.DictReader(stream) if int(r['shot']) == 5]
        expected = np.array([float(r[field]) for r in records])
        assert np.allclose(actual, expected, atol=1e-13, rtol=0), filename
        source_files.append({'path': path.relative_to(root).as_posix(), 'sha256': digest(path)})
    rejected = [r for r in task_diagnostics if not r['enabled']]
    enabled = [r for r in task_diagnostics if r['enabled']]
    return {'schema_version': 'v6_fig03_mechanism_sources_v1', 'dataset': 'WM-811K deduplicated v2', 'semantic_source': 'llm_structured', 'figure_dimensions_inches': [7.2, 6.55], 'class_order': CLASSES, 'units': {'stored_f1': 'fraction', 'figure_f1_differences': 'percentage points (pp)', 'effective_coefficient': 'dimensionless'}, 'definitions': {'panels_a_b': '5-shot, seeds 0-9; final effective coefficient includes task-level fallback zeros; per-class F1 difference is selected method minus same-task internal anchor_metrics', 'panel_c_x': 'support OOF candidate macro-F1 minus support OOF internal-anchor macro-F1, in pp', 'panel_c_y': 'saved without_safe_fallback test macro-F1 minus same-task internal anchor_metrics test macro-F1, in pp; 5/10-shot, seeds 0-9', 'uncertainty': 'diamonds show arithmetic mean; panel b whiskers are sample standard deviation with ddof=1 across 10 seeds, not confidence intervals; panel a shows all ten raw coefficients without whiskers', 'points': 'each point is one saved seed-task; deterministic vertical offsets in a/b improve visibility and encode no additional variable', 'interpretation': 'post-hoc paired diagnostics only; effective coefficients are not semantic-correctness probabilities; no causal correlation or classwise safety claim', 'zero_coefficients': 'Random and Near-full coefficients are identically zero in these ten tasks; their F1 may change because the multiclass argmax also depends on competing-class scores', 'history': 'historical WM test exposure is not restored to independent confirmation by content deduplication'}, 'five_shot_seed_records': five_shot, 'class_summary': summary, 'task_diagnostics_5_and_10_shot': task_diagnostics, 'diagnostic_counts': {'rejected': len(rejected), 'rejected_with_positive_candidate_test_delta': sum((r['candidate_test_delta_pp'] > 0 for r in rejected)), 'enabled': len(enabled), 'enabled_with_negative_candidate_test_delta': sum((r['candidate_test_delta_pp'] < 0 for r in enabled))}, 'csv_mean_checks': 'pass: both 5-shot CSV means agree with raw task-derived values, absolute tolerance 1e-13', 'source_files': source_files}

def configure_fonts() -> dict:
    selected = None
    family = 'DejaVu Sans'
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [family, 'Arial', 'DejaVu Sans'], 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.8, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.edgecolor': '#A8B0B6', 'axes.linewidth': 0.65, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MID, 'ytick.color': INK, 'savefig.facecolor': 'white'})
    return {'family': family, 'file': str(selected) if selected else None, 'minimum_ordinary_font_pt': 9.5, 'text_editability': 'PDF Type42 embedded font; SVG text nodes'}

def clean_axes(ax, grid_axis='x'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.6, width=0.65, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=LIGHT, linewidth=0.6)

def panel_title(ax, letter, title):
    """Keep the original parenthesized label and audit its visible letter."""
    for text, x_offset in (('(', 0), (letter, 4.6), (') ' + title, 12.2)):
        ax.annotate(text, xy=(0, 1), xycoords='axes fraction', xytext=(x_offset, 12), textcoords='offset points', fontsize=11, weight='bold', ha='left', va='baseline')

def draw(data: dict, output_dir: Path) -> None:
    font = configure_fonts()
    records = data['five_shot_seed_records']
    weights = np.array([r['effective_coefficients'] for r in records])
    delta = 100 * (np.array([r['proposed_per_class_f1'] for r in records]) - np.array([r['internal_anchor_per_class_f1'] for r in records]))
    mean_w, sd_w = (weights.mean(0), weights.std(0, ddof=1))
    mean_d, sd_d = (delta.mean(0), delta.std(0, ddof=1))
    old_height, new_height = (7.7, 8.5)

    def top_anchor(old_y):
        return 1 - (1 - old_y) * old_height / new_height
    fig = plt.figure(figsize=(7.2, new_height), facecolor='white')
    top_grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], left=0.137, right=0.978, bottom=top_anchor(0.54), top=top_anchor(0.86), wspace=0.17)
    c_top_inches, c_height_inches = (5.43, 1.848)
    c_top = 1 - c_top_inches / new_height
    c_bottom = c_top - c_height_inches / new_height
    bottom_grid = fig.add_gridspec(1, 1, left=0.137, right=0.978, bottom=c_bottom, top=c_top)
    axa = fig.add_subplot(top_grid[0, 0])
    axb = fig.add_subplot(top_grid[0, 1], sharey=axa)
    axc = fig.add_subplot(bottom_grid[0, 0])
    fig.text(0.137, top_anchor(0.975), 'Classwise coefficients, gains and fallback costs', fontsize=12.5, weight='bold', va='top')
    fig.text(0.137, top_anchor(0.932), 'WM-811K v2 · structured semantics; (a,b) 5-shot, 10 seeds', fontsize=9.7, color=MID, va='top')
    for class_ax in (axa, axb):
        class_ax.axhspan(5.48, 6.52, color='#F1F6F9', zorder=-1)
    offsets = np.linspace(-0.17, 0.17, 10)
    y = np.arange(len(CLASSES))
    for idx in range(len(CLASSES)):
        axa.axhline(idx, color='#F1F3F5', linewidth=0.7, zorder=0)
        axb.axhline(idx, xmin=0, xmax=0.78, color='#F1F3F5', linewidth=0.7, zorder=0)
        axa.scatter(weights[:, idx], idx + offsets, s=11, color='#A7B0B8', alpha=0.82, linewidths=0, zorder=3, clip_on=False)
        axa.scatter(mean_w[idx], idx, color=INK, marker='D', s=24, linewidths=0, zorder=5)
        colors = [BLUE if v >= 0 else ORANGE for v in delta[:, idx]]
        axb.scatter(delta[:, idx], idx + offsets, s=11, color=colors, alpha=0.42, linewidths=0, zorder=3)
        color = BLUE if mean_d[idx] >= 0 else ORANGE
        axb.errorbar(mean_d[idx], idx, xerr=sd_d[idx], color=color, marker='D', markersize=4.2, capsize=2.6, elinewidth=1.3, capthick=0.9, zorder=5)
        axb.text(33.7, idx, f'{mean_d[idx]:+.2f}', va='center', ha='right', fontsize=10, color=color, weight='bold' if idx == 6 else 'normal')
    axa.set_ylim(7.5, -0.75)
    axa.set_yticks(y, CLASSES)
    axa.get_yticklabels()[6].set_fontweight('bold')
    axa.set_xlim(-0.025, 1.06)
    axa.set_xticks([0, 0.5, 1.0], ['0', '0.5', '1.0'])
    axa.set_xlabel('Effective coefficient\n(unitless; fallback zeros included)', labelpad=6)
    panel_title(axa, 'a', 'Coefficient distribution')
    axa.tick_params(axis='y', length=0, pad=7)
    axa.spines['left'].set_visible(False)
    clean_axes(axa)
    axb.set_xlim(-12, 34.5)
    axb.set_xticks([-10, 0, 10, 20])
    axb.set_xlabel('Paired ΔF1 (pp)\n(proposed − internal anchor)', labelpad=6)
    panel_title(axb, 'b', 'Per-class F1 change')
    axb.tick_params(axis='y', left=False, labelleft=False)
    axb.spines['left'].set_visible(False)
    clean_axes(axb)
    axb.axvline(0, color='#A8B0B6', linewidth=0.9)
    axb.text(33.7, -0.82, 'Mean', ha='right', va='bottom', fontsize=9.5, color=MID)
    fig.text(0.137, top_anchor(0.43), 'Dots: seeds; Diamonds: means; (b) bars: ± sample SD, not CIs', fontsize=9.5, color=MID)
    fig.text(0.137, top_anchor(0.405), 'Random / Near-full: coefficients are 0; competing class scores can still alter F1', fontsize=9.5, color=INK)
    tasks = data['task_diagnostics_5_and_10_shot']
    for text, x_offset in (('(', 0), ('c', 4.6), (') Task-level diagnosis: support fallback and Test gains', 12.2)):
        axc.annotate(text, xy=(0, 1), xycoords='axes fraction', xytext=(x_offset, 37), textcoords='offset points', fontsize=11, weight='bold', ha='left', va='baseline')
    axc.set_xlim(-5.2, 22.5)
    axc.set_ylim(-1.45, 8.5)
    axc.axvspan(-5.2, 0, color='#F3F5F6', zorder=0)
    clean_axes(axc, 'both')
    axc.grid(False)
    axc.axvline(0, color='#9EA8B0', linewidth=0.85)
    axc.axhline(0, color='#9EA8B0', linewidth=0.85)
    for r in tasks:
        color = BLUE if r['candidate_test_delta_pp'] >= 0 else ORANGE
        marker = 'o' if r['shot'] == 5 else 's'
        axc.scatter(r['support_oof_delta_pp'], r['candidate_test_delta_pp'], marker=marker, s=34, facecolors=color if r['enabled'] else 'white', edgecolors=color, linewidths=1.15, zorder=4)
    annotations = [((5, 1), '1', (-8, 9), BLUE, 'right'), ((10, 8), '2', (7, 8), MID, 'left'), ((5, 2), '3', (9, 0), ORANGE, 'left')]
    for key, label, offset, color, horizontal in annotations:
        r = next((v for v in tasks if (v['shot'], v['seed']) == key))
        axc.annotate(label, xy=(r['support_oof_delta_pp'], r['candidate_test_delta_pp']), xytext=offset, textcoords='offset points', fontsize=9.5, color=color, weight='bold', ha=horizontal, va='center', zorder=5)
    axc.set_xlabel('Support OOF ΔMacro-F1 (pp)', labelpad=5)
    axc.set_ylabel('Candidate Test ΔMacro-F1 (pp)\n(no fallback − internal anchor)', fontsize=9.5, labelpad=8, linespacing=1.55)
    axc.xaxis.set_major_locator(MultipleLocator(5))
    axc.yaxis.set_major_locator(MultipleLocator(2))
    axc.legend(handles=[Line2D([], [], marker='o', color='none', markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=5, label='5-shot'), Line2D([], [], marker='s', color='none', markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=5, label='10-shot'), Line2D([], [], marker='o', color='none', markerfacecolor='white', markeredgecolor=MID, markersize=5, label='Hollow: fallback')], loc='lower left', bbox_to_anchor=(0, 1.075), ncol=3, frameon=False, fontsize=9.5, handletextpad=0.4, columnspacing=1.4, borderaxespad=0, handlelength=0.8)
    axc.text(1, 1.1, 'Fallback 5/20 · Enabled 15/20', transform=axc.transAxes, ha='right', va='bottom', fontsize=9.5, color=MID)
    band_y = 0.64 / new_height
    for x, heading, body, color in [(0.137, '1  5-shot seed 1', 'Fallback; candidate +1.56 pp', BLUE), (0.423, '2  10-shot seed 8', 'Fallback; candidate +4.49 pp', MID), (0.709, '3  5-shot seed 2', 'Enabled: −0.62 pp', ORANGE)]:
        fig.text(x, band_y, heading, fontsize=9.5, color=color, va='top')
        fig.text(x, band_y - 0.22 / new_height, body, fontsize=9.5, color=color, va='top')
    fig.text(0.137, 0.04 / new_height, '(c) 5/10-shot, 10 seeds each; post hoc paired diagnosis of saved tasks only', fontsize=9.5, color=MID, va='bottom')
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment = require_matplotlib_panel_alignment(fig, json_out=output_dir / 'fig03_alignment_audit.json', overlay_svg=output_dir / 'fig03_alignment_audit.svg', tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True, panel_ids=['a', 'b', 'c'], row_groups=[['a', 'b']], column_groups=[], exemptions=[{'panels': ['b'], 'checks': ['panel-width'], 'reason': 'Panel b intentionally retains the 1.35:1 width ratio to fit signed F1 differences and the right-hand mean column; its top/bottom, height and label anchors remain aligned with a.'}])
    data['figure_dimensions_inches'] = [7.2, new_height]
    data['rendering'] = font
    data['rendering']['layout_revision'] = 'v15: pale Scratch row cue and bold class name; a/b/c geometry, raw points, means, sample SDs and coordinate limits unchanged'
    data['rendering']['alignment_status'] = alignment['verdict']
    data['rendering']['designed_width_cm'] = 16.5
    data['rendering']['minimum_font_pt_at_designed_width'] = round(9.5 * (16.5 / 2.54) / 7.2, 3)
    exports = {'pdf': 'fig03_mechanism.pdf', 'svg': 'fig03_mechanism.svg', 'png': 'fig03_mechanism.png'}
    for suffix, filename in exports.items():
        fig.savefig(output_dir / filename, dpi=300, facecolor='white', metadata={'Creator': 'draw_mechanism.py'} if suffix == 'pdf' else None)
    plt.close(fig)
    (output_dir / 'fig03_sources.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--source-json', type=Path)
    args = parser.parse_args()
    if args.source_json:
        data = json.loads(args.source_json.read_text(encoding='utf-8'))
    elif (args.root / 'results/wm811k_dedup_v2/llm_structured/raw').is_dir():
        data = collect_sources(args.root)
    else:
        data = json.loads((args.output_dir / 'fig03_sources.json').read_text(encoding='utf-8'))
    draw(data, args.output_dir)
    print(json.dumps({'figure': 'fig03_mechanism', 'csv_mean_checks': data['csv_mean_checks'], 'diagnostic_counts': data['diagnostic_counts'], 'font': data['rendering']}, ensure_ascii=False))
if __name__ == '__main__':
    main()
