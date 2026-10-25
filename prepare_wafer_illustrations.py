"""Rebuild only the identified wafer illustrations from the provider pickle.

Obtain the original WM-811K source lawfully. No observed pixel arrays are shipped
with the public evidence package. The known provider-file SHA is checked before
reading the pickle. Derived outputs remain local unless rights permit sharing.
"""
from pathlib import Path
import argparse,hashlib,json

ROOT=Path(__file__).resolve().parent
EXPECTED='1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wm-pickle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();h=hashlib.sha256()
    with args.wm_pickle.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=EXPECTED:raise ValueError('Provider source SHA-256 differs; do not substitute an unverified pickle')
    import pandas as pd
    import numpy as np
    from PIL import Image
    method=json.loads((ROOT/'paper_evidence/figures/method_inputs.json').read_text(encoding='utf-8'))
    cases=json.loads((ROOT/'docs/paper/revision_v4_20261013/figures/case_provenance.json').read_text(encoding='utf-8'))['entries']
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    data=pd.read_pickle(args.wm_pickle)
    images=out/'images';arrays={}
    palette=np.array([[0,0,0],[210,218,226],[230,57,70]],dtype=np.uint8)
    for record in method['support_records']+[method['query_record']]:
        i=int(record['original_index']);a=np.asarray(data.loc[i,'waferMap'],dtype=np.uint8)
        path=images/record['image_path'];path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(palette[a]).resize((224,224),Image.Resampling.NEAREST).save(path)
    for record in cases:
        i=int(record['original_index']);arrays[f'wafer_{i}']=np.asarray(data.loc[i,'waferMap'],dtype=np.uint8)
    np.savez_compressed(out/'wafer_observations.npz',**arrays)
    print('Prepared provider-derived images/ and wafer_observations.npz for figures1/4.')

if __name__=='__main__':main()
