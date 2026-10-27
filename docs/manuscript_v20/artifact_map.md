# Evidence and reproduction map (historical v20 paths)

These scientific outputs are retained in manuscript v34. Material version v1.2.1 preserves the historical paths used by the portable entry points.
The English/Chinese manuscript retains its necessary appendices. These files are data/code, not a replacement supplementary manuscript PDF.

| Output | Evidence and actual entrypoint |
|---|---|
| 1: 适配方法定位 | Descriptive comparison supported by the papers cited in the manuscript; it is not an experimental result requiring a training replay. |
| 2: 数据划分与环境 | Frozen configurations, task manifests, preparation_report.json and WM audit tables document the archived protocol. The reported historical environment is distinct from the newly tested environment in VALIDATION_SUMMARY.json. No new image-level preparation was executed in this repair. |
| 3: 比较条件 | Frozen configs and the archived CLAP/global protocols document text, candidate budgets and parameter selection. This is a protocol table rather than a metric output. |
| 4: 原始主表 | Original published integer counts/configurations/protocols, replay_all.py; this is statistical recalculation, not model retraining. |
| 5: 锚点配对及Holm | Original published integer counts/configurations/protocols, replay_all.py; this is statistical recalculation, not model retraining. |
| 6: 同缓存消融 | Original published integer counts/configurations/protocols, replay_all.py; this is statistical recalculation, not model retraining. |
| 7: Scratch分解 | Original published integer counts/configurations/protocols, replay_all.py; this is statistical recalculation, not model retraining. |
| 8: 补充大表 | Original published integer counts/configurations/protocols, replay_all.py; this is statistical recalculation, not model retraining. |
| B.1: 原始ProKeR配对 | Original published integer counts/configurations/protocols, replay_all.py; this is statistical recalculation, not model retraining. |
| B.2: F1与Accuracy双指标 | Original published integer counts/configurations/protocols, replay_all.py; this is statistical recalculation, not model retraining. |
| C.1: FGVC语义来源对照 | paper_evidence/semantic_sources/*.json; reproduce_paper_evidence.py recomputes the 60 original matched-source Validation tasks. |
| C.2: WM逐类F1与系数 | Original results/wm811k_dedup_v2/llm_structured/raw/ contains exact coefficients and OOF decisions; reproduce_paper_evidence.py checks16 class/shot rows. |
| C.3: 激活/回退诊断 | Original structured/Qwen raw task records and paper_evidence/fgvc_original_test.json; reproduce_paper_evidence.py checks six original activation rows. |
| 1: 方法示意图 | paper_evidence/figures/method_inputs.json fixes support/query indices and prompt; provider-derived images required for rendering. |
| 2: 主结果与机制对照 | Original count-level replay and no-fallback records; draw_paper_figures.py. |
| 3: 逐类系数/OOF-查询诊断 | Original branch coefficients/OOF data; draw_paper_figures.py. No later re-encoded branch substitutes these records. |
| 4: 四幅纠正/误改案例 | Original case_provenance.json provides scores/indices/predictions; images are obtained from the provider and reconstructed with prepare_wafer_illustrations.py. |
| 5: 两幅混淆矩阵 | Original confusion_recomputed.json and public integer counts; draw_paper_figures.py. |
| C.1: 原FGVC完整Validation网格 | paper_evidence/original_validation_grid_tasks.json plus original grid summaries; reproduce_paper_evidence.py and draw_paper_figures.py. |

Appendix C.5: paper_evidence/conflict_query_exclusion.json records the exact three removed query rows and original confusion counts; reproduce_paper_evidence.py rescoring does not retrain or reselect.

Rendering uses the original Python layouts with relocated paths and installed-font fallback. Four statistical figures need no original images. Figures1/4 require lawfully obtained provider data; no observed-image pixels are redistributed.
