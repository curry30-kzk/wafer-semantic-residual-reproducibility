"""Recalculate metrics from per-seed confusion counts. Only NumPy/SciPy needed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def metrics_from_confusion(matrix):
    cm = np.asarray(matrix, dtype=float)
    if (cm.ndim != 2 or cm.shape[0] != cm.shape[1] or not cm.size
            or not np.isfinite(cm).all() or np.any(cm < 0)
            or not np.equal(cm, np.floor(cm)).all() or cm.sum() == 0):
        raise ValueError("A nonempty square confusion matrix of nonnegative integer counts is required.")
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * np.diag(cm), denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(np.trace(cm) / cm.sum())}


def decode_confusion(record):
    count = int(record["n_classes"])
    if count <= 0:
        raise ValueError("n_classes must be positive.")
    cm = np.zeros((count, count), dtype=np.int64)
    used = set()
    for row, column, value in record["confusion_entries"]:
        if (not all(isinstance(v, int) and not isinstance(v, bool) for v in (row, column, value))
                or not 0 <= row < count or not 0 <= column < count or value <= 0
                or (row, column) in used):
            raise ValueError("Sparse confusion entries must be unique [true, predicted, positive integer count].")
        used.add((row, column))
        cm[row, column] = value
    return cm


def describe(values):
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("At least two finite seed values are needed for a sample SD and t interval.")
    mean = float(data.mean())
    std = float(data.std(ddof=1))
    half = float(t.ppf(0.975, len(data) - 1) * std / math.sqrt(len(data)))
    return {"mean": mean, "sample_std": std, "ci95_low": mean - half,
            "ci95_high": mean + half, "seed_count": len(data), "seed_values": data.tolist()}


def paired(left, right):
    if set(left) != set(right):
        raise ValueError("Paired comparisons require exactly matching seed IDs.")
    seeds = sorted(left)
    differences = np.asarray([left[seed] - right[seed] for seed in seeds], dtype=float)
    result = describe(differences)
    result.update(seed_ids=seeds, wins=int(np.sum(differences > 1e-12)),
                  ties=int(np.sum(np.abs(differences) <= 1e-12)), losses=int(np.sum(differences < -1e-12)))
    return result


def calculate(payload):
    groups = defaultdict(dict)
    per_seed = []
    expected = payload["expected_seed_ids"]
    for record in payload["records"]:
        cm = decode_confusion(record)
        if int(cm.sum()) != record["n_query"]:
            raise ValueError(f"Query count mismatch: {record['record_id']}")
        values = metrics_from_confusion(cm)
        for use in record["uses"]:
            key = (use["scope"], use["dataset"], use["method"], record["shot"])
            if record["seed"] in groups[key]:
                raise ValueError(f"Duplicate seed: {key}, {record['seed']}")
            groups[key][record["seed"]] = values
            per_seed.append({"scope": key[0], "dataset": key[1], "method": key[2], "shot": key[3],
                             "seed": record["seed"], **values, "record_id": record["record_id"]})
    summary = []
    for key, seeds in sorted(groups.items()):
        if sorted(seeds) != expected:
            raise ValueError(f"Incomplete seed set: {key}; got {sorted(seeds)}, expected {expected}")
        summary.append({"scope": key[0], "dataset": key[1], "method": key[2], "shot": key[3],
                        "seed_ids": expected,
                        **{metric: describe([seeds[seed][metric] for seed in expected])
                           for metric in ("macro_f1", "accuracy")}})
    comparisons = []
    for key, seeds in sorted(groups.items()):
        scope, dataset, method, shot = key
        left_method = "shared_reencoded_classwise" if scope == "shared_text_control" else "classwise_reliability"
        if method == left_method:
            continue
        main = groups.get((scope, dataset, left_method, shot))
        if main is None:
            raise ValueError(f"Missing main method for {scope}, {dataset}, {shot}")
        comparisons.append({"scope": scope, "dataset": dataset, "shot": shot,
                            "left": left_method, "right": method,
                            **{metric: paired({s: main[s][metric] for s in expected},
                                              {s: seeds[s][metric] for s in expected})
                               for metric in ("macro_f1", "accuracy")}})
    return per_seed, summary, comparisons


def check_references(summary, comparisons, references, tolerance=1e-12):
    index = {(row["scope"], row["dataset"], row["method"], row["shot"]): row for row in summary}
    pair_index = {(r["scope"], r["dataset"], r["shot"], r["right"]): r for r in comparisons}
    checks = []
    for reference in references["summary"]:
        key = tuple(reference[name] for name in ("scope", "dataset", "method", "shot"))
        if key not in index:
            checks.append({"kind": "summary", "key": list(key), "reference_source": reference["reference_source"],
                           "passed": False, "reason": "missing_group"})
            continue
        actual = index[key]["macro_f1"]
        errors = {field: abs(actual[field] - reference[field]) for field in ("mean", "sample_std")}
        if "seed_values" in reference:
            errors["max_seed_error"] = float(np.max(np.abs(np.asarray(actual["seed_values"]) - reference["seed_values"])))
        checks.append({"kind": "summary", "key": list(key), "reference_source": reference["reference_source"],
                       "absolute_errors": errors, "passed": max(errors.values()) <= tolerance})
    for reference in references["comparisons"]:
        key = tuple(reference[name] for name in ("scope", "dataset", "shot", "right"))
        actual = pair_index[key]["macro_f1"]
        errors = {field: abs(actual[field] - reference[field])
                  for field in ("mean", "sample_std", "ci95_low", "ci95_high")}
        checks.append({"kind": "paired_t_interval", "key": list(key), "reference_source": reference["reference_source"],
                       "absolute_errors": errors, "passed": max(errors.values()) <= tolerance})
    return {"tolerance": tolerance, "checks_count": len(checks),
            "all_passed": all(row["passed"] for row in checks), "checks": checks}


def write_csv(path, rows):
    if not rows:
        Path(path).unlink(missing_ok=True)
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(inputs, manifest, references, output, source_root=None):
    payload = read_json(inputs)
    inventory = read_json(manifest)
    if sha256(inputs) != inventory["bundle_files"]["metrics_inputs.json"]["sha256"]:
        raise ValueError("Bundled metric input SHA-256 does not match the manifest.")
    if sha256(references) != inventory["bundle_files"]["reference_checks.json"]["sha256"]:
        raise ValueError("Bundled reference SHA-256 does not match the manifest.")
    source_audit = {"status": "not_requested", "note": "Original files are optional for standalone count-level reproduction."}
    if source_root:
        failures = [entry["relative_path"] for entry in inventory["sources"]
                    if not (Path(source_root) / entry["relative_path"]).is_file()
                    or sha256(Path(source_root) / entry["relative_path"]) != entry["sha256"]]
        source_audit = {"status": "passed" if not failures else "failed", "files_checked": len(inventory["sources"]),
                        "failed_relative_paths": failures}
        if failures:
            raise ValueError(f"Original source checksum audit failed for {len(failures)} files.")
    per_seed, summary, comparisons = calculate(payload)
    verification = check_references(summary, comparisons, read_json(references))
    verification.update(input_sha256=sha256(inputs), manifest_sha256=sha256(manifest),
                        input_record_count=len(payload["records"]), summary_cell_count=len(summary),
                        numpy_version=np.__version__, scipy_version=scipy.__version__,
                        source_audit=source_audit, reproduction_level="saved predictions/confusion counts; no image-level training")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "statistics.json").write_text(json.dumps({"summary": summary, "paired_comparisons": comparisons},
                                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "per_seed.csv", sorted(per_seed, key=lambda r: (r["scope"], r["dataset"], r["method"], r["shot"], r["seed"])))
    table = []
    for row in summary:
        item = {name: row[name] for name in ("scope", "dataset", "method", "shot")}
        item["seed_count"] = row["macro_f1"]["seed_count"]
        for metric in ("macro_f1", "accuracy"):
            item[metric + "_mean"] = row[metric]["mean"]
            item[metric + "_sample_std"] = row[metric]["sample_std"]
            item[metric + "_mean_sd_percent"] = f"{100 * row[metric]['mean']:.2f}±{100 * row[metric]['sample_std']:.2f}"
        table.append(item)
    write_csv(output / "main_table.csv", [r for r in table if r["scope"] == "main"])
    write_csv(output / "ablation_table.csv", [r for r in table if r["scope"] == "ablation"])
    write_csv(output / "all_summary_cells.csv", table)
    write_csv(output / "same_text_control.csv", [r for r in table if r["scope"] == "same_text_control"])
    write_csv(output / "shared_text_control.csv", [r for r in table if r["scope"] == "shared_text_control"])
    paired_rows = []
    for row in comparisons:
        for metric in ("macro_f1", "accuracy"):
            values = row[metric]
            paired_rows.append({**{name: row[name] for name in ("scope", "dataset", "shot", "left", "right")},
                                "metric": metric, "seed_count": values["seed_count"],
                                "mean_difference_pp": 100 * values["mean"],
                                "sample_std_pp": 100 * values["sample_std"],
                                "ci95_low_pp": 100 * values["ci95_low"],
                                "ci95_high_pp": 100 * values["ci95_high"],
                                **{name: values[name] for name in ("wins", "ties", "losses")}})
    write_csv(output / "paired_comparisons.csv", paired_rows)
    return verification


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=folder / "metrics_inputs.json")
    parser.add_argument("--manifest", type=Path, default=folder / "manifest.json")
    parser.add_argument("--references", type=Path, default=folder / "reference_checks.json")
    parser.add_argument("--output-dir", type=Path, default=folder / "recomputed")
    parser.add_argument("--source-root", type=Path, help="Optional repository/raw-artifact root for all original SHA-256 checks.")
    args = parser.parse_args()
    result = run(args.inputs, args.manifest, args.references, args.output_dir, args.source_root)
    print(json.dumps({name: result[name] for name in ("all_passed", "checks_count", "input_record_count", "summary_cell_count", "source_audit")}, ensure_ascii=False))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
