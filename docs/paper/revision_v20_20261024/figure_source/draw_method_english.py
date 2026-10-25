# English labels translated using actual DeepL browser output.
"""Original architecture diagram; real manifest-bound inputs, vector structure only."""
from pathlib import Path
import os
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Arc
from PIL import Image
from audit_panel_alignment import require_matplotlib_panel_alignment
import fitz
INPUT_IMAGE_RENDER_DPI = 600
OUT = Path(os.environ['PAPER_FIGURE_OUT'])
ROOT = Path(__file__).resolve().parents[4]
# English plots use installed Arial with DejaVu Sans fallback.
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Arial', 'DejaVu Sans'], 'font.size': 9.0, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': False})
BLUE, ORANGE, INK, GREY = ('#236486', '#B9642F', '#203A4A', '#687A86')
PALEBLUE, PALEORANGE = ('#EDF4F8', '#FCF2E9')
sources = {}

def source(path):
    path = Path(path)
    sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path
inputs = json.loads(source(ROOT / 'paper_evidence/figures/method_inputs.json').read_text(encoding='utf-8'))
records = inputs['support_records']
query_record = inputs['query_record']
assert all(r['source_split']=='Training' for r in records)
assert query_record['source_split']=='Test'
assert inputs['original_manifest_sha256']==inputs['cache_manifest_sha256']
sentence = inputs['semantic_sentence']
fig, ax = plt.subplots(figsize=(6.5, 5.25))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, 100), ylim=(0, 80))
ax.axis('off')

def text(x, y, s, size=9, color=INK, **kwargs):
    return ax.text(x, y, s, fontsize=size, color=color, va='center', **kwargs)

def box(x, y, w, h, face='white', edge='#C8D4DB', radius=0.8, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=face, edgecolor=edge, lw=lw, zorder=0))

def arrow(a, b, color=GREY, ls='-', rad=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, lw=1, color=color, linestyle=ls, connectionstyle=f'arc3,rad={rad}'))

def lock(x, y, color=BLUE):
    ax.add_patch(Arc((x, y + 1), 1.4, 1.5, theta1=0, theta2=180, color=color, lw=1))
    ax.add_patch(Rectangle((x - 0.85, y - 0.45), 1.7, 1.35, facecolor=color, lw=0))
    ax.plot(x, y + 0.1, 'o', ms=1.1, color='white')

def vector(x, y, color, label):
    for j in range(5):
        ax.add_patch(Rectangle((x, y + 1.6 * j), 6, 1.45, facecolor=color, alpha=0.18, edgecolor='white', lw=0.3))
    text(x + 3, y - 2.1, label, size=10, ha='center')

def wafer(record, x, y, w=7.1, h=None):
    if h is None:
        h = w * (6.5 / 100) / (5.25 / 80)
    path = Path(os.environ['WM_IMAGE_ROOT']) / record['image_path']
    ax.imshow(Image.open(path).convert('RGB'), extent=(x, x + w, y, y + h), interpolation='nearest', zorder=2, aspect='auto')
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='#C8D4DB', lw=0.5, zorder=3))
    return path
box(1, 36.5, 62.5, 42.5, face='#F8FBFD', edge='#D6E2E9', radius=1.1)
box(65.5, 36.5, 33.5, 42.5, face='#FFFBF7', edge='#E8D7C7', radius=1.1)
box(1, 1, 98, 33.5, face='white', edge='#D1DCE3', radius=1.1)
text(3, 76.2, '(a) Frozen image–text representations', size=10.4, weight='bold')
text(67.5, 76.2, '(b) Residual: C parameters', size=10.4, weight='bold')
for x, r in zip((3, 11.4, 19.8), records):
    wafer(r, x, 61.5)
    text(x + 3.55, 59.2, r['label'], size=8.5, ha='center')
text(15.3, 56.7, 'Support: K labeled images/class', size=8.5, ha='center')
box(32, 61, 16, 10, face=PALEBLUE, edge=BLUE)
lock(34.3, 68.6)
text(40, 66.6, 'Image encoder', size=9.3, ha='center')
text(40, 63.3, '$f_\\theta$ (frozen)', size=9.0, ha='center')
arrow((27.9, 65.7), (31.7, 65.7), BLUE)
arrow((48.2, 65.7), (53, 65.7), BLUE)
vector(54, 62, BLUE, '$\\mathbf{H}_{\\mathcal{S}}$')
box(3, 39.8, 25, 13.7, face='white', edge='#D6C0AA')
text(4.3, 51.1, 'Frozen class text', size=9.0, color=ORANGE, weight='bold')
text(4.3, 47.5, '... geometric center ...', size=8.5)
text(4.3, 44.5, 'Position / shape', size=8.5)
text(4.3, 41.6, 'Extent / confusions', size=8.5)
box(32, 42.1, 16, 10, face=PALEORANGE, edge=ORANGE)
lock(34.3, 49.7, ORANGE)
text(40, 47.8, 'Text encoder', size=9.3, ha='center')
text(40, 44.5, '$g_\\phi$ (frozen)', size=9.0, ha='center')
arrow((28.2, 47.1), (31.7, 47.1), ORANGE)
arrow((48.2, 47.1), (53, 47.1), ORANGE)
vector(54, 43.5, ORANGE, '$\\mathbf{P}$')
box(69, 63.8, 26.5, 8.4, face=PALEBLUE, edge=BLUE)
text(82.25, 69.5, 'Fit visual anchor', size=9.2, ha='center', weight='bold')
text(82.25, 66.1, 'Five-fold OOF scores', size=9, ha='center')
box(69, 52.4, 26.5, 8.4, face=PALEORANGE, edge=ORANGE)
text(82.25, 58.7, 'Dot product · debias · align', size=8.5, ha='center')
text(82.25, 54.9, 'Aligned OOF residuals', size=9, ha='center')
arrow((60.2, 66.5), (68.7, 68.0), BLUE)
arrow((60.2, 47.8), (68.7, 56.7), ORANGE)
ax.plot([60.2, 66.1, 66.1], [64.2, 64.2, 60.5], color=BLUE, lw=1)
arrow((66.1, 60.5), (68.7, 60.5), BLUE)
arrow((82.25, 63.5), (82.25, 61.1), GREY)
text(82.25, 49.0, 'Cross-entropy +\nshrinkage toward zero', size=8.5, ha='center', weight='bold', linespacing=1.3)
text(82.25, 45.5, '$\\mathbf{v}^{\\mathrm{oof}}+\\lambda\\mathbf{a}\\odot\\mathbf{r}^{\\mathrm{oof}}$', size=9.5, ha='center')
text(82.25, 41.2, 'Empirical activation $\\longrightarrow\\;\\mathbf{a}_{\\mathrm{eff}}$', size=9, ha='center')
arrow((82.25, 52.2), (82.25, 51.1), ORANGE)
text(3, 31.2, '(c) Query after full support-set fitting', size=10.4, weight='bold')
text(68.5, 28.6, 'Task-fixed coefficients', size=9, color=GREY, ha='center')
wafer(query_record, 3, 16.1, w=8.0)
text(7, 11.5, 'Query image', size=8.5, ha='center')
box(16, 15.1, 16.8, 10, face='#F8FBFD', edge=BLUE)
text(24.4, 21.8, 'Frozen features', size=9.3, ha='center')
text(24.4, 18.1, '$\\mathbf{h}(x)$', size=10, ha='center')
arrow((11.2, 20.1), (15.7, 20.1), BLUE)
text(44.5, 24.5, '$\\mathbf{v}(x)$', size=11, ha='center', color=BLUE)
text(44.5, 15.8, '$\\mathbf{r}(x)$', size=11, ha='center', color=ORANGE)
arrow((33, 21.6), (39, 24.3), BLUE)
arrow((33, 18.2), (39, 16.0), ORANGE)
box(55, 13.1, 41, 12.8, face='#FFFBF7', edge=ORANGE, lw=1.0)
text(75.5, 21.9, '$\\mathbf{z}=\\mathbf{v}+\\lambda\\,\\mathbf{a}_{\\mathrm{eff}}\\odot\\mathbf{r}$', size=12.0, ha='center')
text(75.5, 16.2, 'Predict the largest-score class', size=10, ha='center')
arrow((49.4, 24.5), (54.7, 21.9), BLUE)
arrow((49.4, 15.8), (54.7, 17.1), ORANGE)
arrow((87.5, 38.8), (87.5, 26.3), ORANGE, '--')
text(50, 7.1, 'Zero residual = same internal anchor; 1-shot fits no residual', size=8.8, ha='center')
text(50, 3.4, 'Support labels: fit and enable; query labels: evaluation only', size=8.8, color=GREY, ha='center')
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside = []
for t in fig.findobj(matplotlib.text.Text):
    if not t.get_text():
        continue
    b = t.get_window_extent(renderer)
    if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > fig.bbox.width + 0.5 or (b.y1 > fig.bbox.height + 0.5):
        outside.append(t.get_text())
assert not outside, outside
require_matplotlib_panel_alignment(fig, json_out=OUT / 'fig01_alignment.json')
fig.savefig(OUT / 'fig01_method.pdf', dpi=INPUT_IMAGE_RENDER_DPI, facecolor='white')
fig.savefig(OUT / 'fig01_method.svg', dpi=INPUT_IMAGE_RENDER_DPI, facecolor='white')
fig.savefig(OUT / 'fig01_method.png', dpi=600, facecolor='white')
plt.close(fig)
with fitz.open(OUT / 'fig01_method.pdf') as doc:
    spans = [s for b in doc[0].get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
    minimum = min((s['size'] for s in spans))
record = {'name': 'fig01_method', 'type': 'original computational schematic', 'native_width_cm': 16.51, 'recommended_latex_width_cm': 16.5, 'minimum_ordinary_design_font_pt': 8.5, 'minimum_pdf_span_pt_including_math_subscripts': minimum, 'support_selection_rule': 'First stored support record in seed0 5-shot manifest for each of Center/Edge-Loc/Scratch; Training split only', 'support_original_indices': [r['original_index'] for r in records], 'query_selection_rule': 'First stored query record in same manifest; no true label displayed or used in fitting', 'query_original_index': query_record['original_index'], 'exact_frozen_prompt_source': sentence, 'feature_vector_strips': 'structural icons only; no measured or invented numerical values', 'scope': 'WM images illustrate the general support-driven procedure; text/prompt count is dataset-specific', 'sources': sources, 'checks': {'training_support_only': True, 'historical_manifest_cache_identity_match': True, 'no_text_outside_canvas': True}}
(OUT / 'fig01_sources.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'minimum_ordinary_font_pt': 8.5, 'support_indices': record['support_original_indices'], 'checks': record['checks']}, ensure_ascii=False))
