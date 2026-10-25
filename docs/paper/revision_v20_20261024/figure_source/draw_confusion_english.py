# English labels translated using actual DeepL browser output.
"""Spacing-only revision of the existing paired WM confusion matrices."""
from pathlib import Path
import os
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

def vector_heatmap(ax, matrix, **kwargs):
    kwargs.pop('interpolation', None)
    kwargs.pop('aspect', None)
    rows, cols = matrix.shape
    image = ax.pcolormesh(np.arange(cols + 1) - 0.5, np.arange(rows + 1) - 0.5,
                         matrix, shading='flat', edgecolors='none',
                         antialiased=False, rasterized=False, **kwargs)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect('equal')
    return image


from audit_panel_alignment import require_matplotlib_panel_alignment
HERE = Path(os.environ['PAPER_FIGURE_OUT'])
ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'docs/paper/revision_v4_20261013/figures/confusion_recomputed.json'
data = json.loads(SOURCE.read_text(encoding='utf-8'))
# English plots use installed Arial with DejaVu Sans fallback.
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Arial', 'DejaVu Sans'], 'font.size': 9.5, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
fig = plt.figure(figsize=(7.2, 4.8), facecolor='white')
lefts = [0.135, 0.635]
width = 0.345
height = width * 7.2 / 4.8
axes = [fig.add_axes([left, 0.32, width, height], label=label) for left, label in zip(lefts, ['a', 'b'])]
fig.text(0.135, 0.962, 'WM-811K v2, 5-shot, ten support seeds', fontsize=10.2, color='#6B7882', va='top')
payload = {}
for ax, left, key, title in zip(axes, lefts, ['anchor_metrics', 'metrics'], ['(a) Same internal visual anchor', '(b) Classwise semantic residual']):
    rates = np.asarray(data['matrices'][key]['row_percent'], dtype=float)
    counts = np.asarray(data['matrices'][key]['mean_counts'], dtype=float)
    assert rates.shape == (8, 8) and np.allclose(rates.sum(1), 100)
    assert np.array_equal(rates, 100 * counts / counts.sum(1, keepdims=True))
    image = vector_heatmap(ax, rates, cmap='Blues', vmin=0, vmax=100, interpolation='none', aspect='equal')
    ax.set_xticks(np.arange(8), data['class_names'], rotation=90, ha='right', va='center', rotation_mode='anchor', fontsize=9.5)
    ax.set_yticks(np.arange(8), data['class_names'], fontsize=9.5)
    ax.tick_params(axis='both', length=0, pad=5)
    ax.set_ylabel('True class', fontsize=10, labelpad=9)
    ax.set_xlabel('Predicted class', fontsize=10, labelpad=7)
    ax.set_title(title, loc='left', fontsize=10.5, pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f'{rates[i, j]:.1f}', ha='center', va='center', fontsize=9.5, color='white' if rates[i, j] >= 55 else '#203A4A')
    cax = fig.add_axes([left + 0.0125, 0.088, width - 0.025, 0.022], label='colorbar_' + key)
    cb = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[0, 50, 100])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.ax.tick_params(labelsize=9.5, length=2.3, pad=3)
    payload[key] = {'row_percent': rates.tolist(), 'mean_counts': counts.tolist()}
fig.text(0.56, 0.018, 'Row-normalized mean confusion counts (%)', ha='center', va='bottom', fontsize=9.5, color='#203A4A')
fig.canvas.draw()
report = require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=['a', 'b'], row_groups=[['a', 'b']], column_groups=[], tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, json_out=HERE / 'fig06_alignment.json')
renderer = fig.canvas.get_renderer()
title_y = [ax._left_title.get_window_extent(renderer).y0 * 72 / fig.dpi for ax in axes]
assert abs(title_y[0] - title_y[1]) < 1.5, title_y
outside = []
for t in fig.findobj(plt.Text):
    if not t.get_text():
        continue
    box = t.get_window_extent(renderer)
    if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or (box.y1 > fig.bbox.height + 0.5):
        outside.append(t.get_text())
assert not outside, outside
fig.savefig(HERE / 'fig06_confusion.pdf')
fig.savefig(HERE / 'fig06_confusion.svg')
fig.savefig(HERE / 'fig06_confusion.png', dpi=600)
plt.close(fig)
(HERE / 'fig06_sources.json').write_text(json.dumps({'source': SOURCE.relative_to(ROOT).as_posix(), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'class_names': data['class_names'], 'matrices': payload, 'data_unchanged': True, 'native_inches': [7.2, 4.8], 'placed_width_cm': 16.5, 'ordinary_minimum_font_at_16_5cm': 9.5 * 16.5 / (7.2 * 2.54), 'changes': 'Separate shared metadata, panel titles, y-axis labels, tick labels and colorbar; keep two matrices and two 0-100 colorbars.', 'alignment_passed': report.get('verdict') == 'PASS', 'panel_title_y_positions_pt': title_y}, ensure_ascii=False, indent=2), encoding='utf-8')
print('Figure 6: 128 cells unchanged; panel alignment and text bounds checked.')
