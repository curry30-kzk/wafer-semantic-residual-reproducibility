"""Redraw the existing v20 figures from archived original inputs.

Four statistical figures require only the evidence package. Figures1/4 require
lawfully obtained provider wafer data, which are not redistributed here.
"""
from pathlib import Path
import argparse,os,subprocess,sys,json
from release_integrity import verify_release

ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/'docs/paper/revision_v20_20261024/figure_source'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--wm-images',type=Path)
    p.add_argument('--wafer-observations',type=Path)
    args=p.parse_args();verify_release(ROOT)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PAPER_FIGURE_OUT':str(out),'PYTHONIOENCODING':'utf-8'}
    if args.wm_images:env['WM_IMAGE_ROOT']=str(args.wm_images.resolve())
    if args.wafer_observations:env['WM_CASE_ARRAYS']=str(args.wafer_observations.resolve())
    jobs=[('2','draw_results.py',['--project-root',str(ROOT),'--output-dir',str(out)]),
          ('3','draw_mechanism.py',['--source-json',str(ROOT/'docs/paper/revision_v6_20261015/figures/fig03_sources.json'),'--output-dir',str(out)]),
          ('5','draw_confusion_english.py',[]),('C.1','draw_validation_sensitivity_english.py',[])]
    skipped=[]
    if args.wm_images:jobs.append(('1','draw_method_english.py',[]))
    else:skipped.append({'figure':'1','reason':'Provider-derived wafer images must be obtained separately'})
    if args.wafer_observations:jobs.append(('4','draw_wafer_cases.py',[]))
    else:skipped.append({'figure':'4','reason':'Native provider wafer arrays must be obtained separately'})
    for label,name,extra in jobs:
        proc=subprocess.run([sys.executable,str(SCRIPTS/name),*extra],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
        (out/(name+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        if proc.returncode:raise RuntimeError(f'Figure {label} failed; see {name}.log')
        print('Rendered figure',label,flush=True)
    report={'all_requested_figures_rendered':True,'rendered':[x[0] for x in jobs],'skipped':skipped,
        'numeric_inputs':'Original archived branches; no re-encoded substitute grid',
        'font_note':'Rendering uses installed fonts; vector pixels/font metrics need not be byte-identical across systems.',
        'observed_wafer_images_redistributed':False}
    (out/'render_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
