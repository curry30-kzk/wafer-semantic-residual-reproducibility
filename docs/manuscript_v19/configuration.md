# Dataset composition and frozen implementation settings

Readable configuration tables moved from English manuscript v18 Appendix A during
the v19 organization revision. The reported values and original selection records
are unchanged. The manuscript retains a short Appendix A for indispensable
implementation and selection constraints; Appendices B–D remain in the manuscript.

## WM content-deduplicated composition

| Defect category | Training support candidates | Test queries |
| --- | --- | --- |
| Center | 3,456 | 832 |
| Donut | 409 | 146 |
| Edge-Loc | 2,388 | 2,772 |
| Edge-Ring | 8,546 | 1,126 |
| Loc | 1,602 | 1,973 |
| Random | 595 | 257 |
| Scratch | 489 | 693 |
| Near-full | 46 | 95 |
| Total | 17,531 | 7,894 |

Each task uses K original support images per class; the Test query counts remain
fixed across ten support seeds. Equal support counts do not imply equal morphological
diversity. Accuracy weights images equally, whereas Macro-F1 weights the declared
defect categories equally. The eight known-defect setting does not establish normal
wafer screening or unknown-defect detection performance.

## Main implementation settings

| Item | WM v2 | FGVC Test |
| --- | --- | --- |
| Encoder and features | Frozen CLIP ViT-B/32; 512-dimensional | Same as left |
| Classes and support budget | 8 categories; 8 × K annotated original images per task | 100 classes, 100 × K annotated original images per task |
| Tasks and fixed queries | Three shot settings, ten seeds each; 7,894 images | Three shot settings, ten seeds each; 3,333 images |
| 1-shot anchor | Visual prototype | Tip-Adapter-F, 20 cache-key training rounds |
| 5/10-shot anchor | LP regularization selected by five-fold support-set cross-validation | Same as left |
| LP solver configuration | LBFGS, up to 2,000 steps; random_state set to 0 | Same as left |
| Support augmentation | 10 views per original image; original and augmented views assigned to folds by group | Same as left; bottom strip removed before transformation |
| Residual hyperparameters | (lambda,rho)=(1,0.01) | (lambda,rho)=(1,0.1) |
| Coefficient optimization | CPU float64; Adam, learning rate 0.05, 200 steps | Same as left |
| Initialization and constraints | Initialized to zero vectors; projected onto [0, 1] for each of C classes at each step | Same as left |
| Stabilization term and scale | epsilon=1e-12; cross-class population standard deviation | Same as left |
| 1-shot Tip-F parameter | Not applicable | alpha=1.17,beta=1, learning rate 0.001 |
| Additional residual parameters | 8 coefficients fixed within each task | 100 task-internal fixed coefficients |

Augmented views do not add independently labeled images. The image encoder, prompts
and semantic prototypes remain frozen. Normalization follows OpenAI CLIP. The LP
regularizer is selected using support-set cross-validation. Support-derived visual
weights, semantic biases, alignment scales and residual coefficients are fitted under
the documented protocol. Matching label budgets does not match optimization budgets,
augmentation counts or runtime across methods. The C extra parameters refer only to
the residual stage; feature extraction, visual-head fitting and OOF construction
are additional work.

The LP++ port uses deterministic zero-bias initialization and 300 visual-weight
updates; text coefficients are updated every 10 iterations, with coefficient-gradient
accumulation retained from the ported implementation and one augmented view.
ZS-LP and CLAP use SGD, learning rate 0.1, momentum 0.9 and no weight decay, with one
warm-up epoch at 1e-5 followed by cosine annealing and batch size 256. RLA uses identity
initialization, Adam at 1e-4, stabilizer 1e-4 and regularization 1/K. These three ports
use 10 views and 300 optimization rounds. These settings are documented comparison
conditions, not a claim of full reproduction of the original papers' budgets.

## ProKeR selection and final settings

Original single-prompt and additional same-text ProKeR select and freeze parameters
separately. Both use the same candidate grid but different text priors. The additional
same-text run is a post hoc robustness comparison and does not replace the original
main-table entry. Text and selected kernel parameters can both change, so the contrast
does not isolate a causal text effect.

FGVC selects a single pair per shot on Validation, shared across all ten Test support
seeds. WM 5/10-shot selects per task with five-fold cross-validation grouped by
original support image. WM 1-shot fixes (3, 0.05) for both text variants.

| Shot | Original single-prompt ProKeR (beta, eta) | Additional same-text ProKeR (beta, eta) |
| --- | --- | --- |
| 1 | (0.5,0.1) | (0.5,0.1) |
| 5 | (2,0.1) | (2,0.1) |
| 10 | (3,1) | (2,0.1) |

| Seed | Original 5-shot | Original 10-shot | Same-text 5-shot | Same-text 10-shot |
| --- | --- | --- | --- | --- |
| 0 | (3,0.1) | (1,0.1) | (3,0.1) | (0.5,0.1) |
| 1 | (3,0.1) | (3,0.1) | (2,0.1) | (2,0.1) |
| 2 | (1,0.1) | (3,0.1) | (3,0.1) | (3,0.1) |
| 3 | (1,0.1) | (2,0.1) | (1,0.1) | (1,0.1) |
| 4 | (3,0.1) | (3,0.1) | (3,0.1) | (2,0.1) |
| 5 | (2,0.1) | (1,0.1) | (3,0.1) | (0.5,0.1) |
| 6 | (2,0.1) | (1,0.1) | (2,0.1) | (0.5,0.1) |
| 7 | (2,0.1) | (0.5,0.1) | (2,0.1) | (0.5,0.1) |
| 8 | (3,0.1) | (3,1) | (1,0.1) | (1,0.1) |
| 9 | (2,0.1) | (2,0.1) | (1,0.1) | (2,0.1) |

The manuscript denotes the second parameter by eta; the archived protocol uses the
JSON key `lambda` for the same kernel-regularization parameter. The candidate grid,
tie breaking, augmentation, equations and solver are preserved in
[`configs/proker_protocol_20261013.json`](../../configs/proker_protocol_20261013.json).
Additional selected parameters are preserved in the saved task records and replay
inputs. These are same-protocol ports, not official numerical reproductions.

## Source locations

- [`configs/`](../../configs/): frozen protocol/configuration JSON files.
- [`data/processed/WM811K_dedup_v2/splits/`](../../data/processed/WM811K_dedup_v2/splits/): supplied WM 5/10-shot manifests.
- [`data/processed/FGVC-Aircraft/splits/`](../../data/processed/FGVC-Aircraft/splits/): supplied FGVC manifests.
- [`docs/paper/revision_v6_20261015/reproduction/`](../paper/revision_v6_20261015/reproduction/): original lightweight count inputs and arithmetic replay.

The complete historical preprocessing artifacts, some 1-shot manifests and complete
image-feature/score caches are not included. These tables document the original
experiments; their presence does not make this package an image-to-result rerun.
