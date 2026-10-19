"""Build a portable count-level input bundle from archived predictions/CMs.

This extracts no image or model feature data and never trains or selects a model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from recompute import metrics_from_confusion, read_json, sha256

SHOTS = (1, 5, 10)
SEEDS = tuple(range(10))
WM_ABLATIONS = ("visual_anchor", "fixed_all_one", "global_reliability", "classwise_reliability",
                "direct_weighted_fusion", "without_calibration", "platt_calibration", "without_safe_fallback")
FGVC_ABLATIONS = ("visual_anchor", "fixed_all_one", "global_reliability", "classwise_reliability",
                  "without_calibration", "without_safe_fallback")


def cm_from_labels(true, predicted, n_classes):
    true, predicted = np.asarray(true), np.asarray(predicted)
    if (true.ndim != 1 or predicted.shape != true.shape or not len(true)
            or not np.issubdtype(true.dtype, np.integer) or not np.issubdtype(predicted.dtype, np.integer)
            or np.any(true < 0) or np.any(true >= n_classes)
            or np.any(predicted < 0) or np.any(predicted >= n_classes)):
        raise ValueError("Prediction label vectors must have equal positive length and valid integer class indices.")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (true, predicted), 1)
    return cm


def cm_from_csv(path, n_classes):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not {"true_index", "predicted_index"}.issubset(fields):
            raise ValueError(f"Missing true_index/predicted_index in {path.name}")
        rows = list(reader)
    return cm_from_labels([int(row["true_index"]) for row in rows],
                          [int(row["predicted_index"]) for row in rows], n_classes), fields


def completed_result_state(root, folder, completion_name):
    """Require a complete 60-task count/prediction bundle, never a partial slice."""
    root, folder = Path(root).resolve(), Path(folder).resolve()
    expected_paths = []
    for ds, n_way in (("wm", 8), ("fgvc", 100)):
        for shot in SHOTS:
            for seed in SEEDS:
                stem = f"{n_way}way_{shot}shot_seed{seed}"
                expected_paths.extend([folder / ds / "raw" / (stem + ".json"),
                                       folder / ds / "scores" / (stem + ".npz"),
                                       folder / ds / "predictions" / (stem + ".csv")])
    expected_paths.extend(folder / name for name in
                          (completion_name, "summary.json", "paired_comparisons.json"))
    missing = [path.relative_to(root).as_posix() for path in expected_paths if not path.is_file()]
    state = {"status": "not_complete" if missing else "complete",
             "relative_directory": folder.relative_to(root).as_posix(),
             "missing_expected_files": missing, "expected_tasks": 60,
             "included_tasks": 0 if missing else 60}
    if not missing:
        receipt = read_json(folder / completion_name)
        if receipt.get("tasks") != 60 or isinstance(receipt.get("tasks"), bool):
            raise ValueError(f"Completion receipt must declare exactly 60 tasks: {completion_name}")
    return state


class Builder:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources = {}
        self.loaded = {}
        self.records = {}
        self.audit = []

    def source(self, path, fields):
        path = Path(path).resolve()
        relative = path.relative_to(self.root).as_posix()
        if relative not in self.sources:
            self.sources[relative] = {"source_id": f"s{len(self.sources):04d}", "relative_path": relative,
                                      "sha256": sha256(path), "format": path.suffix.lstrip("."), "fields_used": []}
        item = self.sources[relative]
        item["fields_used"] = sorted(set(item["fields_used"]) | set(fields))
        return item["source_id"]

    def read(self, path):
        path = Path(path).resolve()
        if path not in self.loaded:
            self.loaded[path] = read_json(path)
        return self.loaded[path]

    def add(self, path, raw, field, use, *, csv_path=None, npz_path=None):
        shot, seed = int(raw["shot"]), int(raw["seed"])
        cm = np.asarray(raw["metrics"]["confusion_matrix"])
        values = metrics_from_confusion(cm)
        n_classes = len(cm)
        expected_n = 7894 if n_classes == 8 else 3333 if n_classes == 100 else None
        if (shot not in SHOTS or seed not in SEEDS or expected_n is None or int(cm.sum()) != expected_n
                or raw.get("debug_run", False) or raw.get("query_source", "Test") != "Test"
                or raw.get("query_labels_used_for_selection", False)):
            raise ValueError(f"Ineligible original record: {path.name}, {field}")
        scalar_errors = {name: abs(values[name] - float(raw["metrics"][name])) for name in values}
        if max(scalar_errors.values()) > 1e-12:
            raise ValueError(f"CM/scalar mismatch: {path.name}, {field}")
        prefix = field[:-len("metrics.confusion_matrix")]
        source_id = self.source(path, [field, prefix + "shot", prefix + "seed",
                                      prefix + "metrics.macro_f1", prefix + "metrics.accuracy"])
        key = (source_id, field, shot, seed)
        if key in self.records:
            record = self.records[key]
            if use not in record["uses"]:
                record["uses"].append(use)
            return record
        triples = [[int(r), int(c), int(cm[r, c])] for r, c in zip(*np.nonzero(cm))]
        record = {"record_id": f"cm{len(self.records):04d}", "uses": [use], "shot": shot, "seed": seed,
                  "n_classes": n_classes, "n_query": int(cm.sum()), "source_id": source_id,
                  "source_sha256": self.sources[Path(path).resolve().relative_to(self.root).as_posix()]["sha256"],
                  "source_field": field, "confusion_entries": triples}
        origins = [{"source_id": source_id, "field": field, "role": "original_confusion_counts"}]
        if csv_path is not None:
            csv_cm, fields = cm_from_csv(csv_path, n_classes)
            if not np.array_equal(csv_cm, cm):
                raise ValueError(f"Saved prediction CSV/CM mismatch: {csv_path.name}")
            origins.append({"source_id": self.source(csv_path, ["true_index", "predicted_index"]),
                            "fields": ["true_index", "predicted_index"], "role": "prediction_count_crosscheck",
                            "actual_csv_header": fields})
        if npz_path is not None:
            with np.load(npz_path, allow_pickle=False) as archive:
                npz_cm = cm_from_labels(archive["y_true"], archive["predictions"], n_classes)
                if not np.array_equal(archive["predictions"], archive["scores"].argmax(axis=1)):
                    raise ValueError(f"Saved score argmax/prediction mismatch: {npz_path.name}")
            if not np.array_equal(npz_cm, cm):
                raise ValueError(f"NPZ true/pred/CM mismatch: {npz_path.name}")
            if "score_sha256" in raw and sha256(npz_path) != raw["score_sha256"]:
                raise ValueError(f"Original score_sha256 mismatch: {npz_path.name}")
            origins.append({"source_id": self.source(npz_path, ["y_true", "predictions", "scores"]),
                            "fields": ["y_true", "predictions", "scores"], "role": "prediction_count_and_argmax_crosscheck"})
        record["origins"] = origins
        by_id = {item["source_id"]: item for item in self.sources.values()}
        for origin in origins:
            origin["sha256"] = by_id[origin["source_id"]]["sha256"]
        record["original_record_method"] = raw.get("method")
        record["original_record_dataset"] = raw.get("dataset")
        self.records[key] = record
        self.audit.append({"record_id": record["record_id"], "scalar_errors": scalar_errors,
                           "prediction_csv_crosscheck": csv_path is not None,
                           "prediction_npz_crosscheck": npz_path is not None})
        return record

    def raw_folder(self, folder, scope, dataset, method, shots=SHOTS):
        n_way = 8 if dataset.startswith("WM") else 100
        added = []
        for shot in shots:
            for seed in SEEDS:
                path = folder / f"{n_way}way_{shot}shot_seed{seed}.json"
                raw = self.read(path)
                if (int(raw["shot"]), int(raw["seed"])) != (shot, seed):
                    raise ValueError(f"Record identity mismatch: {path.name}")
                added.append(self.add(path, raw, "metrics.confusion_matrix",
                                      {"scope": scope, "dataset": dataset, "method": method},
                                      csv_path=path.with_name(path.stem + "_predictions.csv")))
        return added


def build(root, output, proker_dir, same_text_dir, main_table, require_same_text=False,
          shared_semantics_dir=None, require_shared_semantics=False):
    b = Builder(root)
    root = b.root
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    wm = root / "results/wm811k_dedup_v2"
    fg = root / "results/fgvc_aircraft_test"
    for method in ("lp_plus_plus_fair", "zs_lp", "clap", "rla"):
        b.raw_folder(wm / "baselines" / method / "raw", "main", "WM_v2", method)
        b.raw_folder(fg / ("clip_" + method) / "raw", "main", "FGVC_Test", method)
    b.raw_folder(wm / "baselines/visual_prototype/raw", "main", "WM_v2", "visual_anchor", shots=(1,))
    b.raw_folder(wm / "baselines/linear_probe/raw", "main", "WM_v2", "visual_anchor", shots=(5, 10))
    for method in WM_ABLATIONS:
        records = b.raw_folder(wm / "llm_structured/raw" / method, "ablation", "WM_structured_v2", method)
        if method == "classwise_reliability":
            for record in records:
                record["uses"].append({"scope": "main", "dataset": "WM_v2", "method": method})
    fg_path = fg / "class_reliability_residual/test_results.json"
    fg_payload = b.read(fg_path)
    for method in FGVC_ABLATIONS:
        originals = fg_payload["variants"][method]
        if {(int(row["shot"]), int(row["seed"])) for row in originals} != {(shot, seed) for shot in SHOTS for seed in SEEDS}:
            raise ValueError(f"FGVC variant does not contain the expected 30 unique tasks: {method}")
        for index, raw in enumerate(originals):
            record = b.add(fg_path, raw, f"variants.{method}[{index}].metrics.confusion_matrix",
                           {"scope": "ablation", "dataset": "FGVC_Test", "method": method})
            if method in ("visual_anchor", "classwise_reliability"):
                record["uses"].append({"scope": "main", "dataset": "FGVC_Test", "method": method})
    proker_state = {"status": "not_requested"}
    if proker_dir is not None:
        proker_dir = Path(proker_dir)
        proker_dir = proker_dir if proker_dir.is_absolute() else root / proker_dir
        for folder, dataset in (("wm", "WM_v2"), ("fgvc", "FGVC_Test")):
            b.raw_folder(proker_dir / folder / "raw", "main", dataset, "proker")
        proker_state = {"status": "complete", "relative_directory": proker_dir.relative_to(root).as_posix(), "tasks": 60}
    same_text_dir = Path(same_text_dir)
    same_text_dir = same_text_dir if same_text_dir.is_absolute() else root / same_text_dir
    same_text_state = completed_result_state(root, same_text_dir, "test_evaluation_completed.json")
    missing = same_text_state["missing_expected_files"]
    if require_same_text and missing:
        raise ValueError(f"Same-text ProKeR is incomplete: {len(missing)} missing files/receipt; no values imputed.")
    if not missing:
        b.source(same_text_dir / "test_evaluation_completed.json", ["tasks", "completed_utc"])
        for ds, n_way, dataset in (("wm", 8, "WM_v2"), ("fgvc", 100, "FGVC_Test")):
            for shot in SHOTS:
                for seed in SEEDS:
                    stem = f"{n_way}way_{shot}shot_seed{seed}"
                    path = same_text_dir / ds / "raw" / (stem + ".json")
                    raw = b.read(path)
                    if (raw["method"], raw["dataset"], int(raw["shot"]), int(raw["seed"])) != ("same_text_proker", ds, shot, seed):
                        raise ValueError(f"Same-text task identity mismatch: {path.name}")
                    b.add(path, raw, "metrics.confusion_matrix",
                          {"scope": "same_text_control", "dataset": dataset, "method": "same_text_proker"},
                          csv_path=same_text_dir / ds / "predictions" / (stem + ".csv"),
                          npz_path=same_text_dir / ds / "scores" / (stem + ".npz"))
        for record in b.records.values():
            for use in list(record["uses"]):
                if use["scope"] == "main" and use["method"] == "classwise_reliability":
                    record["uses"].append({"scope": "same_text_control", "dataset": use["dataset"], "method": use["method"]})
    shared_semantics_state = {"status": "not_requested", "included_tasks": 0}
    if shared_semantics_dir is not None:
        shared_semantics_dir = Path(shared_semantics_dir)
        shared_semantics_dir = shared_semantics_dir if shared_semantics_dir.is_absolute() else root / shared_semantics_dir
        shared_semantics_state = completed_result_state(root, shared_semantics_dir, "evaluation_completed.json")
        shared_missing = shared_semantics_state["missing_expected_files"]
        if require_shared_semantics and shared_missing:
            raise ValueError(f"Shared semantic control is incomplete: {len(shared_missing)} missing files/receipt; no values imputed.")
        if not shared_missing:
            if same_text_state["status"] != "complete":
                raise ValueError("Shared semantic control requires the complete matching 60-task ProKeR result bundle.")
            b.source(shared_semantics_dir / "evaluation_completed.json", ["tasks", "completed_utc"])
            for ds, n_way, dataset in (("wm", 8, "WM_v2"), ("fgvc", 100, "FGVC_Test")):
                for shot in SHOTS:
                    for seed in SEEDS:
                        stem = f"{n_way}way_{shot}shot_seed{seed}"
                        path = shared_semantics_dir / ds / "raw" / (stem + ".json")
                        raw = b.read(path)
                        if (raw["method"], raw["dataset"], int(raw["shot"]), int(raw["seed"])) != ("shared_reencoded_classwise", ds, shot, seed):
                            raise ValueError(f"Shared semantic task identity mismatch: {path.name}")
                        if (raw.get("semantic_raw_exactly_shared_with_proker_prior") is not True
                                or raw.get("visual_scores_unchanged_from_original_main_cache") is not True):
                            raise ValueError(f"Missing explicit shared-semantic control boundaries: {path.name}")
                        b.add(path, raw, "metrics.confusion_matrix",
                              {"scope": "shared_text_control", "dataset": dataset, "method": "shared_reencoded_classwise"},
                              csv_path=shared_semantics_dir / ds / "predictions" / (stem + ".csv"),
                              npz_path=shared_semantics_dir / ds / "scores" / (stem + ".npz"))
            for record in b.records.values():
                for use in list(record["uses"]):
                    if use["scope"] == "same_text_control" and use["method"] == "same_text_proker":
                        record["uses"].append({"scope": "shared_text_control", "dataset": use["dataset"], "method": use["method"]})
                    if use["scope"] == "main" and use["method"] == "classwise_reliability":
                        record["uses"].append({"scope": "shared_text_control", "dataset": use["dataset"], "method": "legacy_classwise_reliability"})
    legacy_path = root / "docs/paper/revision_20261008/verified_results.json"
    old = b.read(legacy_path)
    old_source = b.source(legacy_path, ["main", "ablations", "six_anchor_comparisons", "all_baseline_comparisons", "ablation_direct_comparisons"])
    references = {"purpose": "Verification comparators only; never used to construct the metric inputs.", "summary": [], "comparisons": []}
    for scope, section, datasets in (("main", "main", ("WM_v2", "FGVC_Test")),
                                      ("ablation", "ablations", ("WM_structured_v2", "FGVC_Test"))):
        for dataset in datasets:
            methods = list(old[section][dataset]) if scope == "main" else WM_ABLATIONS if dataset.startswith("WM") else FGVC_ABLATIONS
            for method in methods:
                for shot in SHOTS:
                    values = old[section][dataset][method][str(shot)]
                    references["summary"].append({"scope": scope, "dataset": dataset, "method": method, "shot": shot,
                                                   "mean": values["mean"], "sample_std": values["std"],
                                                   "seed_values": values["seed_values"], "reference_source": old_source})
    def add_pair(scope, dataset, shot, right, values):
        references["comparisons"].append({"scope": scope, "dataset": dataset, "shot": int(shot), "right": right,
                                           **{name: values[name] for name in ("mean", "ci95_low", "ci95_high")},
                                           "sample_std": values["std"], "reference_source": old_source})
    for row in old["six_anchor_comparisons"]:
        add_pair("main", row["dataset"], row["shot"], "visual_anchor", row)
    for dataset, methods in old["all_baseline_comparisons"].items():
        for method, shots in methods.items():
            for shot, values in shots.items():
                add_pair("main", dataset, shot, method, values)
    for dataset, methods in old["ablation_direct_comparisons"].items():
        if dataset not in ("WM_structured_v2", "FGVC_Test"):
            continue
        allowed = WM_ABLATIONS if dataset.startswith("WM") else FGVC_ABLATIONS
        for method, shots in methods.items():
            if method in allowed:
                for shot, values in shots.items():
                    add_pair("ablation", dataset, shot, method, values)
    if main_table is not None:
        table = Path(main_table)
        table = table if table.is_absolute() else root / table
        table_source = b.source(table, ["dataset", "shot", "method", "macro_f1_mean", "macro_f1_sample_std"])
        with table.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["method"] == "proker" and proker_dir is None:
                    continue
                references["summary"].append({"scope": "main", "dataset": row["dataset"], "method": row["method"],
                                               "shot": int(row["shot"]), "mean": float(row["macro_f1_mean"]),
                                               "sample_std": float(row["macro_f1_sample_std"]), "reference_source": table_source})
    dataset_names = {"wm": "WM_v2", "fgvc": "FGVC_Test"}
    if same_text_state["status"] == "complete":
        path = same_text_dir / "summary.json"
        source = b.source(path, ["dataset", "shot", "metric", "mean_percent", "sample_sd_percent"])
        for row in b.read(path):
            if row["metric"] == "macro_f1":
                references["summary"].append({"scope": "same_text_control", "dataset": dataset_names[row["dataset"]],
                    "method": "same_text_proker", "shot": int(row["shot"]), "mean": row["mean_percent"] / 100,
                    "sample_std": row["sample_sd_percent"] / 100, "reference_source": source})
        path = same_text_dir / "paired_comparisons.json"
        source = b.source(path, ["dataset", "shot", "mean_difference_pp", "sample_sd_difference_pp", "ci95_low_pp", "ci95_high_pp"])
        for row in b.read(path):
            references["comparisons"].append({"scope": "same_text_control", "dataset": dataset_names[row["dataset"]],
                "shot": int(row["shot"]), "right": "same_text_proker", "mean": row["mean_difference_pp"] / 100,
                "sample_std": row["sample_sd_difference_pp"] / 100, "ci95_low": row["ci95_low_pp"] / 100,
                "ci95_high": row["ci95_high_pp"] / 100, "reference_source": source})
    if shared_semantics_state["status"] == "complete":
        path = shared_semantics_dir / "summary.json"
        source = b.source(path, ["dataset", "shot", "mean_macro_f1_percent", "sample_sd_percent", "proker_mean_percent", "proker_sd_percent"])
        for row in b.read(path):
            for method, mean_key, sd_key in (("shared_reencoded_classwise", "mean_macro_f1_percent", "sample_sd_percent"),
                                            ("same_text_proker", "proker_mean_percent", "proker_sd_percent")):
                references["summary"].append({"scope": "shared_text_control", "dataset": dataset_names[row["dataset"]],
                    "method": method, "shot": int(row["shot"]), "mean": row[mean_key] / 100,
                    "sample_std": row[sd_key] / 100, "reference_source": source})
        path = shared_semantics_dir / "paired_comparisons.json"
        source = b.source(path, ["dataset", "shot", "comparison", "mean_difference_pp", "sample_sd_difference_pp", "ci95_low_pp", "ci95_high_pp"])
        comparison_methods = {"shared reencoded classwise minus shared-text ProKeR": "same_text_proker",
                              "shared reencoded classwise minus legacy frozen classwise": "legacy_classwise_reliability"}
        for row in b.read(path):
            references["comparisons"].append({"scope": "shared_text_control", "dataset": dataset_names[row["dataset"]],
                "shot": int(row["shot"]), "right": comparison_methods[row["comparison"]],
                "mean": row["mean_difference_pp"] / 100, "sample_std": row["sample_sd_difference_pp"] / 100,
                "ci95_low": row["ci95_low_pp"] / 100, "ci95_high": row["ci95_high_pp"] / 100, "reference_source": source})
    payload = {"schema_version": 1, "reproduction_level": "saved confusion-count statistics",
               "encoding": "confusion_entries=[true_class_index,predicted_class_index,count]; omitted cells are zero",
               "expected_seed_ids": list(SEEDS), "shots": list(SHOTS),
               "records": list(b.records.values())}
    input_path = output / "metrics_inputs.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    reference_path = output / "reference_checks.json"
    reference_path.write_text(json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_records = []
    for record in b.records.values():
        for use in record["uses"]:
            manifest_records.append({**use, **{name: record[name] for name in
                                              ("record_id", "shot", "seed", "source_id", "source_sha256", "source_field", "n_classes", "n_query", "origins", "original_record_method", "original_record_dataset")}})
    manifest = {"schema_version": 1, "relative_path_base": "repository root; paths are provenance, not a runtime requirement",
                "main_proker": proker_state, "same_text_proker": same_text_state,
                "shared_reencoded_semantics": shared_semantics_state,
                "scope_notes": {"main": "Frozen formal main-table anchor; WM 1shot visual prototype, 5/10shot independent linear probe.",
                                "ablation": "WM structured semantic branch internal-cache anchor; FGVC frozen Test variants.",
                                "same_text_control": "Post-hoc frozen legacy main versus ProKeR with matching frozen prompt groups, newly encoded text; not exact legacy semantic-array identity.",
                                "shared_text_control": "Post-hoc control using exactly shared reencoded raw semantic scores and unchanged cached main visual scores; distinct from frozen original results."},
                "bundle_files": {"metrics_inputs.json": {"relative_path": "metrics_inputs.json", "sha256": sha256(input_path)},
                                 "reference_checks.json": {"relative_path": "reference_checks.json", "sha256": sha256(reference_path)}},
                "sources": list(b.sources.values()), "records": manifest_records,
                "extraction_checks": {"original_scalar_recalculation_max_error": max(max(row["scalar_errors"].values()) for row in b.audit),
                                      "prediction_csv_count_crosschecks": sum(row["prediction_csv_crosscheck"] for row in b.audit),
                                      "prediction_npz_count_crosschecks": sum(row["prediction_npz_crosscheck"] for row in b.audit),
                                      "records_checked": len(b.audit)}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=here.parents[3])
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--proker-dir", type=Path, default=Path("results/proker_20261013"))
    parser.add_argument("--skip-proker", action="store_true", help="Build only legacy six methods; no placeholder values.")
    parser.add_argument("--same-text-proker-dir", type=Path, default=Path("results/revision_v4_20261013/same_text_proker"))
    parser.add_argument("--require-same-text-proker", action="store_true")
    parser.add_argument("--shared-semantics-dir", type=Path, default=Path("results/revision_v4_20261013/shared_reencoded_semantics"))
    parser.add_argument("--require-shared-semantics", action="store_true")
    parser.add_argument("--main-table", type=Path, default=Path("results/proker_20261013/analysis/main_table.csv"))
    args = parser.parse_args()
    manifest = build(args.repo_root, args.output_dir, None if args.skip_proker else args.proker_dir,
                     args.same_text_proker_dir, args.main_table, args.require_same_text_proker,
                     args.shared_semantics_dir, args.require_shared_semantics)
    print(json.dumps({name: manifest[name] for name in ("main_proker", "extraction_checks")}, ensure_ascii=False))
    print(json.dumps({"same_text_proker_status": manifest["same_text_proker"]["status"],
                      "same_text_missing_files": len(manifest["same_text_proker"]["missing_expected_files"])}, ensure_ascii=False))
    print(json.dumps({"shared_semantics_status": manifest["shared_reencoded_semantics"]["status"],
                      "shared_semantics_missing_files": len(manifest["shared_reencoded_semantics"]["missing_expected_files"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
