# Support-driven classwise semantic residual adaptation

Reproducibility materials for **面向小样本晶圆缺陷分类的支持集驱动类别级语义残差适配**
(Support-Driven Classwise Semantic Residual Adaptation for Few-Shot Wafer Defect
Classification).

The scientific evidence corresponds to the current Chinese v14 manuscript. The
private manuscript draft is not included; historical version numbers in folders
identify the original evidence snapshots and are retained to keep relative paths
and provenance understandable.

This repository contains the frozen implementation/configuration snapshot,
semantic descriptions, support/query task manifests and saved confusion-count
evidence used in the study. It supports a **standalone replay of saved-result
statistics**. The public package does not include raw images, model weights or
feature/score arrays and has not been verified as an image-to-result rerun.

## Quick start: reproduce the saved statistics

Use Python 3.10 or newer. Run these commands from the repository root:

```sh
python -m pip install -r docs/paper/revision_v6_20261015/reproduction/requirements.txt
python replay_all.py --output-dir recomputed/first_run
```

Choose a new output directory on each run. No GPU, PyTorch, original dataset,
private directory or feature cache is needed for this replay. The command
recomputes each task's metrics directly from saved integer confusion counts,
then checks means, sample standard deviations, paired differences and intervals
against the saved summaries. The summaries are comparison targets, not inputs
from which task counts are reconstructed.

The verified scope is:

| Evidence | Saved task records | Verification |
|---|---:|---:|
| Main results, ablations and shared-text controls | 870 deduplicated count records; 114 summary cells | 222 checks |
| Shared-text CLAP comparison | 80 method/task records | 400 numeric comparisons |
| Independently selected global/classwise and visual-OOF controls | 640 Validation + 200 Test records | 3,128 numeric comparisons |

The first block requires NumPy/SciPy. The latter two replay scripts use only the
Python standard library. The commands below run the existing arithmetic and
malformed-input checks; they do not run model training:

```sh
python -m unittest discover -s docs/paper/revision_v6_20261015/reproduction -p test_recompute.py
python -m unittest discover -s docs/paper/revision_v9_20261015/bridge -p test_recompute.py
python -m unittest discover -s docs/paper/revision_v11_20261017/global_control -p test_recompute_control.py
```

## What is included

- `src/` and root `run_*.py`: the archived method implementations and experiment
  entry points. The frozen vision-language encoder is CLIP ViT-B/32; the residual
  stage learns one bounded, zero-shrunk coefficient per class while keeping the
  already fitted visual classifier fixed.
- `configs/`: frozen task/method settings, grids and provenance metadata.
- `data/processed/*/semantics/`: frozen Qwen semantic bundles, including the
  generation metadata retained in the source snapshot. WM's structured prompts
  are also defined in `src/wm811k/llm_descriptions.py`.
- `data/processed/*/splits/`: the supplied 5/10-shot task manifests, source image
  identifiers/labels and ordering. These manifests contain no image pixels.
- `docs/paper/revision_v6_20261015/reproduction/`: compact saved count inputs and
  their arithmetic replay. The source manifest is historical provenance; not
  every upstream raw result it names is included here.
- `docs/paper/revision_v9_20261015/bridge/` and
  `docs/paper/revision_v11_20261017/global_control/`: supplementary comparison
  runners and arithmetic replay scripts at their original relative locations.
- `results/teacher_review_v*/`: numerical records only. The directory names are
  historical experiment identifiers; no review reports or correspondence are
  included. Validation grids are retained even when a comparison is unfavorable.
- `verification_outputs/`: actual local replay results from the public export.
- `EXPORT_PROVENANCE.json`: original snapshot hashes and public export hashes.
  `PUBLIC_MANIFEST.json`: the released-file hash inventory.

## Image-level reruns require additional inputs

The full experiment dependencies are listed in `requirements.txt`. After legally
obtaining the datasets and installing the model dependencies, inspect the real
entry points with:

```sh
python run_wm811k_dedup_v2.py --help
python run_wm811k_dedup_v2.py cache --help
python run_class_reliability_residual.py --help
python run_fgvc_aircraft_final_test.py --help
python run_proker_baselines.py --help
```

These are executable command-line entry points, not a claim that `--help`
reproduces an experiment. Raw-data preprocessing/image export, the complete WM
deduplication-audit/candidate-pool files, some 1-shot manifests, historical
visual/OOF score caches, encoded text prototypes and checkpoint files are not
included. Frozen CLAP/global runners explicitly check the original cache hashes;
they cannot be run against this lightweight package alone. Reconstructed image
caches can differ across software/hardware versions. Do not bypass their checks
or call a newly generated cache bit-identical to a missing historical cache.

For raw data use the original providers:
[WM-811K](http://mirlab.org/dataSet/public/) and
[FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).
This repository grants no new rights to these datasets. See [RIGHTS.md](RIGHTS.md)
for retained third-party notices and the status of author-owned additions.

## Interpretation and provenance

- WM Test was observed during earlier method development. Content deduplication
  addresses identical-content support/query leakage but does not restore
  historical test independence. WM results are retrospective evidence.
- The added shared-text CLAP and global/OOF comparisons are post-hoc controls.
  Their new Validation selections constrain those comparisons; a freeze receipt
  does not turn them into an independently unseen external test.
- Ten support seeds share a fixed query set. The paired intervals describe
  variation across those support samples; they are not ten independent test
  datasets, fabs or future production batches.
- FGVC's shared-text CLAP comparison does not establish a Macro-F1 advantage for
  the proposed method; crossing-zero intervals also do not establish equivalence.
- The claimed `C` extra parameters concern the residual stage only. They exclude
  prior visual-head fitting, OOF construction and encoder feature extraction.
- Personal machine paths in a small number of JSON metadata fields were
  relativized for this export. Numerical results, class names, prompts and core
  Python code were not changed. The original arithmetic input/manifest trio is
  byte-identical, so its original internal SHA checks remain active. Original
  freeze receipts retain upstream hashes; they are provenance records, while
  `PUBLIC_MANIFEST.json` verifies the actual public files.

## 中文说明

本仓库公开代码、冻结配置、语义描述、支持/查询清单和保存的统计证据，
可直接重算论文已保存的混淆计数及配对统计。它不是新的独立晶圆测试，
也不包含从原始图像到全部最终结果的一键复现实验。原图、模型权重和特征
缓存需按原提供方规则另行获取；缺少的历史缓存不能用汇总数字反推或伪造。
WM 历史测试暴露和 CLAP 对照的结论边界保持公开。
