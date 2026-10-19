# Rights and upstream notices

No blanket open-source or open-data licence is assigned to this repository.
Author-created code, configurations, prompts and derived numerical records are
made publicly inspectable for reproducibility; all rights not already granted by
an applicable existing licence are reserved. Public access is not a new licence
to third-party data, model weights or software.

The following existing upstream notices are preserved:

- The ProKeR port in `src/wm811k/proker.py` identifies the official implementation
  at commit `f2ecb3d2a0d7e0993f977e550286b34f06015edb`. The upstream Apache-2.0
  licence is preserved in `third_party/licenses/ProKeR-Apache-2.0.txt`.
  Source: https://github.com/ybendou/ProKeR . The local interface and explicit
  support/validation separation are a port, not the upstream training program.
- CLAP is referenced at commit `dbdf4095b3d824546474c725539b9f4efff3ca0e`.
  Its MIT licence (copyright Julio Silva-Rodríguez, 2023) is retained verbatim at
  `docs/paper/revision_v9_20261015/bridge/reference/LICENSE`.
  Source: https://github.com/jusiro/CLAP . The local code is a protocol adaptation;
  see the bridge protocol notes for its differences from the official setting.
- The LP++ and RLA ports record their source URLs and configurations in
  `src/wm811k/lp_plus_plus.py`, `src/wm811k/rla.py` and `configs/`. Their presence
  does not relicense the upstream projects. Consult their upstream terms before
  reusing third-party code: https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline
  and https://github.com/astra-vision/ProLIP .
- Python dependencies and downloaded OpenCLIP/Qwen weights remain governed by
  their respective licences. Weights are not distributed here.

WM-811K and FGVC-Aircraft images are not distributed. Obtain them from their
original providers and comply with their access and use terms. Task manifests
contain source image identifiers/labels and derived content hashes, not images.
No licence to the original images or annotations is granted by this repository.

Dataset sources:

- WM-811K: http://mirlab.org/dataSet/public/ (original provider's directory).
- FGVC-Aircraft: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ .

An explicit reusable licence for author-owned additions can be supplied later by
the rights holder. No author name, affiliation, DOI or publication acceptance is
invented in this release.
