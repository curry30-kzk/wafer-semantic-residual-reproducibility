# Materials index

## Version correspondence

- Current manuscript at preparation: JVCIR v34, English and Chinese, 27 October 2026.
- Materials v1.2.0: the frozen source/evidence snapshot originally cited by v34.
- Materials v1.2.2: the same scientific records with the Figure 2 legend corrected
  to `Visual anchor`, clearer navigation, citation metadata, and deduplicated receipts.
- The materials update does not change experiments or restore WM Test independence.

## Inputs and entry points

| Material | Location | Use |
|---|---|---|
| Frozen method/protocol settings | `configs/` | Fix method and comparison settings |
| Fixed tasks and semantic descriptions | `data/processed/` | 1/5/10-shot task manifests, WM/FGVC semantic inputs and audit metadata |
| Saved task records and selection grids | `results/` | Original per-task outcomes and Validation selection evidence |
| Additional source evidence | `paper_evidence/` | Semantic-source, case, activation, query-exclusion and grid evidence |
| Saved-statistics recalculation | `replay_all.py` | 222 main, 400 CLAP and 3,128 global numeric checks |
| Additional evidence recalculation | `reproduce_paper_evidence.py` | 174 additional checks |
| Figure generation | `draw_paper_figures.py` | Figures 2, 3, 5 and C.1 without provider pixels; Figures 1/4 with provider inputs |
| Optional residual refitting | `reproduce_cached_residual.py` | Requires the separately downloaded S2 cache |
| File integrity and provenance | `RELEASE_MANIFEST.json`, `EXPORT_PROVENANCE.json` | Public identities distinct from historical freeze receipts |

The [table/figure mapping](docs/manuscript_v20/artifact_map.md) retains the original
13-table and 6-figure correspondence. Historical source folder names v6/v9/v11/v20
are still used by the portable runners; they are not obsolete manuscript drafts.

## Downloads and rights

Use source/S1 for statistical replay. Only cache refitting needs the unchanged
[S2 archive](https://github.com/curry30-kzk/wafer-semantic-residual-reproducibility/releases/download/v1.2.0/Reproduction_Caches_S2.zip).
See [README](README.md) for exact commands and the verified scope, and
[RIGHTS.md](RIGHTS.md) for author and upstream terms. Provider images and model
weights are not redistributed. No DOI or blanket reuse licence is newly assigned.
