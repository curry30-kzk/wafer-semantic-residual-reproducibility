# English labels translated using actual DeepL browser output.
"""Plot only the archived FGVC Validation grid, never the Test results."""
from pathlib import Path
import os
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
from audit_panel_alignment import require_matplotlib_panel_alignment
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


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(os.environ['PAPER_FIGURE_OUT'])
SOURCE = ROOT / 'results/revision_v4_20261013/validation_sensitivity/per_shot_grid.json'
ROWS = json.loads(SOURCE.read_text(encoding='utf-8'))
FONT = FontProperties(family='DejaVu Sans')
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 11, 'axes.unicode_minus': False, 'svg.fonttype': 'none', 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Arial']})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), layout='constrained')
lambdas, rhos = ([0.25, 0.5, 0.75, 1.0], [0.01, 0.1, 1.0, 10.0])
for shot, ax in zip([5, 10], axes):
    grid = np.empty((4, 4))
    for i, rho in enumerate(rhos):
        for j, strength in enumerate(lambdas):
            matches = [r for r in ROWS if r['shot'] == shot and r['lambda'] == strength and (r['rho'] == rho)]
            assert len(matches) == 1 and matches[0]['n'] == 10 and (matches[0]['split'] == 'Validation')
            grid[i, j] = matches[0]['mean_macro_f1_percent']
    image = vector_heatmap(ax, grid, cmap='Blues', vmin=grid.min() - 0.25, vmax=grid.max() + 0.25)
    for i in range(4):
        for j in range(4):
            rgb = image.cmap(image.norm(grid[i, j]))[:3]
            luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            color = 'white' if luminance < 0.53 else '#203A4A'
            ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', color=color, fontsize=11)
    ax.add_patch(Polygon([(3.5, 0.5), (3.5, 0.83), (3.17, 0.5)], closed=True, facecolor='#C86D32', edgecolor='none', zorder=5))
    ax.set_xticks(range(4), ['0.25', '0.5', '0.75', '1.0'])
    ax.set_yticks(range(4), ['0.01', '0.1', '1', '10'])
    ax.set_xlabel('Injection strength $\\lambda$', fontproperties=FONT, fontsize=11)
    ax.set_ylabel('Shrinkage strength $\\rho$', fontproperties=FONT, fontsize=11)
    ax.set_title(f'{shot}-shot Validation', fontsize=12)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor('face')
    colorbar.ax.tick_params(labelsize=10)
fig.suptitle('Mean Macro-F1 (%); 10 support seeds/cell\nOrange corner: original frozen configuration', fontproperties=FONT, fontsize=11.5)
fig.canvas.draw()
alignment = require_matplotlib_panel_alignment(fig, axes=list(axes), panel_ids=['a', 'b'], row_groups=[['a', 'b']], column_groups=[], json_out=OUT / 'validation_sensitivity_alignment.json', tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(OUT / 'validation_sensitivity.pdf', facecolor='white')
fig.savefig(OUT / 'validation_sensitivity.svg', facecolor='white')
fig.savefig(OUT / 'validation_sensitivity.png', dpi=600, facecolor='white')
plt.close(fig)
payload = {'data_source': str(SOURCE.relative_to(ROOT)), 'data_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'original_input_provenance': json.loads((SOURCE.parent / 'provenance.json').read_text(encoding='utf-8')), 'split': 'Validation only', 'cells': 32, 'seeds_per_cell': 10, 'frozen_configuration': {'lambda': 1.0, 'rho': 0.1}, 'note': 'Panels use separate color scales; numbers are percentage means.', 'alignment_verdict': alignment['verdict'], 'style_revision': 'v15: shared blue sequential palette, luminance-based number contrast, small orange corner markers at the unchanged frozen cells; original geometry and all 32 values unchanged'}
(OUT / 'validation_sensitivity_sources.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print('Validation sensitivity: 32 cells checked; PDF/PNG/SVG written')
