# Support-driven classwise semantic residual adaptation

## v1.2.0 reproduction materials for manuscript v20

Original v1.1.1 remains unchanged. This version adds missing original evidence
and portable data access,
without changing the manuscript's numerical results or creating an independent Test.

### 1. Saved statistics and original supporting evidence

Use Python3.10 or newer. In a clean checkout/extracted source package:

```sh
python -m pip install -r docs/paper/revision_v6_20261015/reproduction/requirements.txt
python release_integrity.py
python replay_all.py --output-dir recomputed/main_run
python reproduce_paper_evidence.py --output-dir recomputed/paper_evidence_run
```

Every output directory must be new. The root replay now delegates to the strict
portable wrapper. Generated input copies must match historical freeze SHA values;
scientific count files and original protocols are not edited or bypassed.

The first command group covers the original222/400/3128 saved-statistic checks.
The added evidence command directly checks original semantic-source Validation,
classwise coefficients, OOF activation, the historical grid, case scores and
three-query exclusion. See [all13 tables and6 figures](docs/manuscript_v20/artifact_map.md).
The main/public source file names retain original historical version identifiers.

### 2. Render figures

```sh
python -m pip install -r requirements_figures.txt
python draw_paper_figures.py --output-dir recomputed/figures
```

This renders statistical figures2,3,5,C.1 from original records. Figures1/4 need
provider images/arrays. Obtain the original WM-811K source lawfully, then:

```sh
python prepare_wafer_illustrations.py --wm-pickle PATH_TO_LSWMD.pkl --output-dir local_wafer_inputs
python draw_paper_figures.py --output-dir recomputed/all_figures --wm-images local_wafer_inputs/images --wafer-observations local_wafer_inputs/wafer_observations.npz
```

No observed wafer pixels, FGVC photos or third-party model weights are shipped.
Installed fonts can alter layout metrics slightly; the numeric source branch is unchanged.

### 3. Optional cache-to-result residual fitting

Extract the separate cache ZIP over this source root, preserving relative paths.
Install the model dependencies (requirements.txt), then run:

```sh
python release_integrity.py --with-caches
python reproduce_cached_residual.py --datasets wm_structured wm_qwen fgvc --shots 5 10 --seeds 0 1 2 3 4 5 6 7 8 9 --output-dir recomputed/cached_residual
```

This refits the frozen classwise residual using already saved visual/OOF/semantic
scores. It does not rebuild encoder features, visual LP heads or OOF folds, and
does not retrain every baseline or recreate all historical model selection.
Archived strict CLAP/global model runners are historical code: their original
private-byte receipts are retained; the saved-count replay for these controls works,
but this release does not claim their model training has been publicly rerun.

### Identities and source provenance

RELEASE_MANIFEST.json binds current public bytes. original_sha256 is a separate
historical identity, not a claim that compact/path-redacted JSON has identical bytes.
UTF-8 LF/CRLF transport may be accepted only if a candidate exactly matches the
declared public SHA. Changed numeric values, missing required caches, binary
changes and escaping paths are rejected. Source files are never rewritten to pass.

The original/export ledger is paper_evidence/ORIGINAL_EXPORT_PROVENANCE.json.
Original v1.1.1 inventory/scan records are at archive/v1_1_1/; use release_integrity.py
for the new candidate. No new license, author identity, DOI or live data-repository
record is invented. RIGHTS.md and all upstream notices remain applicable.

### Evidence boundaries

WM Test was observed during early development; deduplication does not restore
independence. CLAP Macro-F1 superiority is not established. Ten support seeds
share fixed queries. Extra C parameters refer only to the residual stage.
These boundaries remain unchanged by exporting more evidence.

### Provider data

WM-811K: http://mirlab.org/dataSet/public/
FGVC-Aircraft: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

Full original-image execution has not been rerun here. New provider-side preparation
can acquire a new content/run identity; it must not be mislabeled as an identical
historical cache. The added preparation CLIs have portable defaults but retain the
original algorithm; historical preparation receipts are evidence, not a new image audit.

### Release assets and tested environment

Download both ZIPs from the [v1.2.0 release](https://github.com/curry30-kzk/wafer-semantic-residual-reproducibility/releases/tag/v1.2.0).
Supplementary_Data_S1.zip is a complete source/evidence tree; extract it first.
Reproduction_Caches_S2.zip overlays the 310 optional cache files on that tree.
The cache archive alone is not executable. The Git checkout contains the source/
evidence tree and the full release manifest but excludes these optional caches.
No archive download is needed for saved-statistics recalculation.

VALIDATION_SUMMARY.json records the checks actually executed and the new CPU
environment. requirements_tested_statistics.txt and
requirements_tested_cache_figures.txt record tested package versions separately
from the original broad installation requirements. They do not assert cross-platform
bitwise identity or recover unrecorded historical dependencies.

The two ZIPs contain research data/code, not a separate supplementary manuscript
PDF. Their journal upload item type and file-size limits must be checked in the
actual submission system; a public release link is an alternative access route.
