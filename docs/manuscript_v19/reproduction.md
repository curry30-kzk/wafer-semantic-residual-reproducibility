# Reproduction scope and validation steps

This document replaces the detailed reproduction ledger formerly in English
manuscript v18 Appendix E. It describes existing evidence and inputs; v19 adds no
new experiments and no independent test data.

## Three reproduction levels

| Level | Required inputs | Publicly provided and verified scope |
| --- | --- | --- |
| Saved-output statistical recalculation | Integer confusion counts, source manifest, NumPy/SciPy or standard-library replay scripts | Lightweight inputs are public. All three existing count-level replay bundles pass. |
| Adaptation reruns from cache | Complete visual arrays, main-method score cache, text, manifests and matched numerical environment | Runners, settings and saved evaluation records are public; the complete historical cache is absent. These runners cannot reproduce the experiments from this lightweight repository alone. |
| End-to-end experiments from images | Provider datasets, CLIP weights, preprocessing/adaptation code, task manifests and complete configurations | Code snapshots and frozen settings are public; original images, weights and some historical preparation artifacts are absent. No third-party image-to-result rerun is claimed. |

Macro-F1 averages F1 across all declared categories and assigns zero when a category
denominator is zero. Sample standard deviations use ddof=1. Paired comparisons align
support seeds. Ten support seeds reuse a fixed query set; they are not ten independent
test datasets or production batches.

## Public saved-count bundles

| Bundle | Inputs and coverage | Verification performed |
| --- | --- | --- |
| Original main results, ablations and shared-text controls | 870 deduplicated count records; 114 summary cells | 222 arithmetic checks |
| Shared-text CLAP comparison | 80 Test method/task records; the saved 640 Validation candidate records are also retained | 400 numeric comparisons |
| Independently selected global/classwise and visual-OOF controls | 640 Validation records and 200 Test method/task records | 3,128 numeric comparisons |

For the third bundle, 320 classwise Validation records and 40 Test records from the
original main configuration reproduce the existing bridging outputs entry by entry.
All counts above describe archived evidence. The v18 manuscript also describes a
separate 20-task Scratch decomposition check in the original project; a dedicated
Scratch-decomposition replay is not included in the three public replay commands.
Do not equate this repository's explicitly tested coverage with every analysis or
every later manuscript paragraph.

## Run the public replay

From a fresh clone, using Python 3.10 or newer:

```sh
python -m pip install -r docs/paper/revision_v6_20261015/reproduction/requirements.txt
python docs/manuscript_v19/replay_portable.py --output-dir recomputed/my_run
```

Use a new output directory. No original images, encoder, GPU or cache is required
for this count-level calculation. Confusion counts are the metric inputs; saved
summary values are validation targets. The scripts do not reconstruct missing
counts or features from summaries.

Existing input/error handling checks can be run with:

```sh
python -m unittest discover -s docs/paper/revision_v6_20261015/reproduction -p test_recompute.py
python -m unittest discover -s docs/paper/revision_v9_20261015/bridge -p test_recompute.py
python -m unittest discover -s docs/paper/revision_v11_20261017/global_control -p test_recompute_control.py
python audit_public_release.py
```

The wrapper handles Git line-ending conversion by preparing two generated input
copies. It accepts only original, LF or CRLF bytes whose SHA-256 exactly matches
the original frozen manifest. Source scientific inputs and original freeze checks
remain unchanged. Any substantive input modification still fails.

The last command validates the released-file inventory and performs a bounded
private-path/credential-pattern scan. This is not a guarantee that every possible
secret format is detectable. It does not contact GitHub or a credential store. The inventory hashes refer to
Git-stored file bytes, with LF text line endings. If a checkout automatically
converts text to CRLF, validate the inventory from a clean LF export, for example:

```sh
git -c core.autocrlf=false archive --format=tar HEAD -o release-tree.tar
mkdir release-tree
tar -xf release-tree.tar -C release-tree
cd release-tree
python audit_public_release.py
```

The archive and extracted folder are temporary verification artifacts, not new
scientific inputs. Exact frozen CRLF input copies are generated under the replay
output directory and remain distinct from Git release-byte inventory hashes.

## Original-data access and restrictions

Obtain [WM-811K](http://mirlab.org/dataSet/public/) and
[FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) from their
providers, following their terms. This repository grants no new rights to those
datasets. See [RIGHTS.md](../../RIGHTS.md) for third-party notices and the status
of author-owned additions.

The full feature/score cache, model weights, raw images, complete WM deduplication
audit/candidate pools and some 1-shot manifests are not distributed. Missing
historical caches cannot be inferred or fabricated from summary metrics. Rebuilt
caches may differ across numerical environments. Do not bypass frozen cache-hash
checks or claim an approximate rebuild is byte-identical to a missing cache.

## Evidence boundaries

WM Test was observed in early development. Content-level deduplication removes
identical-content overlap but does not restore historical test independence.
Additional CLAP/global controls are post hoc comparisons. Their freeze records do
not make them newly unseen external tests. The shared-text FGVC comparison does
not establish Macro-F1 superiority over CLAP, and crossing-zero intervals do not
establish equivalence. Residual coefficient counts exclude visual-head/OOF/encoder
costs. These boundaries are unchanged by moving documentation out of the paper.
