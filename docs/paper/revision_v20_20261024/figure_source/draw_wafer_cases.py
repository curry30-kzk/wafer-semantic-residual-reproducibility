"""Draw the same four recorded wafer arrays as deterministic vector cells."""
from pathlib import Path
import os
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import font_manager
from audit_panel_alignment import require_matplotlib_panel_alignment

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=Path(os.environ['PAPER_FIGURE_OUT'])
records=json.loads((ROOT/'docs/paper/revision_v4_20261013/figures/case_provenance.json').read_text(encoding='utf-8'))['entries']
arrays=np.load(Path(os.environ['WM_CASE_ARRAYS']),allow_pickle=False)
assert [r['original_index'] for r in records]==[642104,640493,654285,640920]
palette=ListedColormap(['#000000','#D2DAE2','#E63946'])
norm=BoundaryNorm([-.5,.5,1.5,2.5],palette.N)
BLUE,ORANGE,INK,GREY='#236486','#B9642F','#203A4A','#6B7882'
# English plots use installed Arial with DejaVu Sans fallback.
for language in ['english']:
    zh=language=='chinese'
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['SimHei','Arial','DejaVu Sans'] if zh else ['Arial','DejaVu Sans'],'font.size':9.5,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':True,'axes.spines.right':True})
    fig=plt.figure(figsize=(6.5,3.5),facecolor='white')
    title='同一任务中的纠正与误改：WM-811K v2，5-shot，seed 0' if zh else 'Corrected and spoiled decisions: WM-811K v2, 5-shot, seed 0'
    fig.text(.02,.968,title,fontsize=10,weight='bold',color=INK,va='top')
    w,h=468.,252.
    axes=[]
    for i,record in enumerate(records):
        center=(.025+.1085+i*.245)*w
        size,top=95.4,30.
        ax=fig.add_axes([(center-size/2)/w,1-(top+size)/h,size/w,size/h],label=f'case{i+1}')
        a=arrays[f"wafer_{record['original_index']}"]
        assert set(np.unique(a)).issubset({0,1,2})
        # Native cells; square display matches the study's square image export.
        ax.pcolormesh(np.arange(a.shape[1]+1),np.arange(a.shape[0]+1),a,cmap=palette,norm=norm,shading='flat',edgecolors='none',antialiased=False,rasterized=False)
        ax.set_xlim(0,a.shape[1]);ax.set_ylim(a.shape[0],0);ax.set_aspect('auto')
        ax.set_xticks([]);ax.set_yticks([])
        color=BLUE if record['selection_kind']=='corrected' else ORANGE
        for spine in ax.spines.values():spine.set_color(color);spine.set_linewidth(1.5)
        labels=[('由错改对' if record['selection_kind']=='corrected' else '由对改错',color,'bold'),(f"真值：{record['true_class']}",INK,'normal'),(f"锚点：{record['anchor_prediction']}",INK,'normal'),(f"残差：{record['residual_prediction']}",INK,'normal'),(f"原始索引：{record['original_index']}",GREY,'normal')] if zh else [('Wrong → correct' if record['selection_kind']=='corrected' else 'Correct → wrong',color,'bold'),(f"Ground truth: {record['true_class']}",INK,'normal'),(f"Anchor: {record['anchor_prediction']}",INK,'normal'),(f"Residual: {record['residual_prediction']}",INK,'normal'),(f"Original index: {record['original_index']}",GREY,'normal')]
        for y,(label,tcolor,weight) in zip([136,152,168,184,200],labels):
            fig.text(center/w,1-y/h,label,ha='center',va='top',fontsize=9 if y==200 else 9.5,color=tcolor,weight=weight)
        axes.append(ax)
    foot='事后选择：指定类别/决策类型内的最小原始索引。\n仅说明单图决策，不估计成功或失败比例；阵列按固定规则矢量绘制。' if zh else 'Post hoc: lowest original index within each specified class/decision type.\nSingle-image decisions only; native arrays rendered as deterministic vector cells.'
    fig.text(.02,1-228/h,foot,fontsize=8.8,color=GREY,va='top',linespacing=1.3)
    fig.canvas.draw()
    alignment=require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=['case1','case2','case3','case4'],row_groups=[['case1','case2','case3','case4']],column_groups=[],tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,json_out=OUT/f'wafer_cases_{language}.alignment.json')
    target=OUT/'fig05_cases.pdf'
    fig.savefig(target)
    fig.savefig(OUT/f'wafer_cases_{language}.svg')
    fig.savefig(OUT/f'wafer_cases_{language}.png',dpi=600)
    plt.close(fig)
    print(language,alignment['verdict'],target.name,flush=True)
