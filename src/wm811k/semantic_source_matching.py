"""Reproducible semantic-source matching audits for the paper experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHOTS = (1, 5, 10)
SEEDS = tuple(range(10))
VISUAL_ARRAY_KEYS = (
    "support_labels",
    "query_labels",
    "visual_support_scores",
    "visual_query_scores",
    "visual_oof_scores",
)
REPORT_FILENAMES = {
    "summary": "source_summary.csv",
    "anchor_summary": "anchor_summary.csv",
    "paired_comparisons": "paired_source_comparisons.csv",
    "semantic_vs_anchor_comparisons": "semantic_vs_anchor_comparisons.csv",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_set_sha256(paths: Sequence[Path], *, root: Path) -> str:
    """Hash manifest identities and bytes in stable relative-path order."""
    digest = hashlib.sha256()
    resolved_root = root.resolve()
    for path in sorted((Path(item).resolve() for item in paths), key=lambda item: item.as_posix()):
        relative = path.relative_to(resolved_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_and_validate_frozen_config(
    config_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Load a frozen config and fail if any paper-facing input has drifted."""
    path = Path(config_path)
    root = Path(project_root).resolve()
    config = _read_json(path)
    if config.get("schema_version") != "semantic_source_matching_frozen_v1":
        raise ValueError("unsupported semantic-source frozen config schema")
    if not config.get("shots") or not config.get("seeds"):
        raise ValueError("frozen config must declare non-empty shots and seeds")
    common = config.get("common_config")
    if not isinstance(common, Mapping) or not {"lambda", "rho"} <= set(common):
        raise ValueError("frozen config must declare common lambda and rho")

    semantic_checks: list[dict[str, Any]] = []
    for item in config.get("semantic_bundles", []):
        bundle_path = _resolve(root, str(item["path"])).resolve()
        if not bundle_path.is_file():
            raise FileNotFoundError(f"semantic bundle is missing: {bundle_path}")
        observed = sha256_file(bundle_path)
        expected = str(item["sha256"]).lower()
        if observed != expected:
            raise ValueError(f"semantic bundle hash mismatch: {item['name']}")
        semantic_checks.append(
            {"name": str(item["name"]), "path": str(item["path"]), "sha256": observed}
        )
    if not semantic_checks:
        raise ValueError("frozen config must declare semantic bundles")

    manifest_checks: list[dict[str, Any]] = []
    for item in config.get("manifest_sets", []):
        manifest_root = _resolve(root, str(item["root"])).resolve()
        paths = tuple(path for path in manifest_root.glob(str(item["glob"])) if path.is_file())
        expected_count = int(item["count"])
        if len(paths) != expected_count:
            raise ValueError(
                f"manifest set count mismatch: {item['name']} expected {expected_count}, found {len(paths)}"
            )
        observed = manifest_set_sha256(paths, root=root)
        if observed != str(item["sha256"]).lower():
            raise ValueError(f"manifest set hash mismatch: {item['name']}")
        manifest_checks.append(
            {
                "name": str(item["name"]),
                "root": str(item["root"]),
                "glob": str(item["glob"]),
                "count": len(paths),
                "sha256": observed,
            }
        )
    if not manifest_checks:
        raise ValueError("frozen config must declare manifest sets")

    config["validation"] = {
        "semantic_bundles": semantic_checks,
        "manifest_sets": manifest_checks,
        "passed": True,
    }
    return config


def _array_path(metadata_path: Path, payload: Mapping[str, Any]) -> Path:
    declared = Path(str(payload.get("array_file", "")))
    return declared if declared.is_absolute() else metadata_path.parent / declared


def audit_visual_cache_pair(first_path: Path, second_path: Path) -> dict[str, Any]:
    """Compare task identity and visual arrays while ignoring semantic arrays."""
    first = _read_json(first_path)
    second = _read_json(second_path)
    failures: list[str] = []
    for key in ("shot", "seed", "manifest_sha256"):
        if first.get(key) != second.get(key):
            failures.append(key)
    first_arrays_path = _array_path(first_path, first)
    second_arrays_path = _array_path(second_path, second)
    if not first_arrays_path.is_file() or not second_arrays_path.is_file():
        raise FileNotFoundError("cache array file is missing")
    max_difference = 0.0
    with np.load(first_arrays_path) as first_arrays, np.load(second_arrays_path) as second_arrays:
        for key in VISUAL_ARRAY_KEYS:
            first_has = key in first_arrays
            second_has = key in second_arrays
            if not first_has and not second_has:
                continue
            if first_has != second_has:
                failures.append(f"array:{key}")
                continue
            left = np.asarray(first_arrays[key])
            right = np.asarray(second_arrays[key])
            if left.shape != right.shape:
                failures.append(f"shape:{key}")
                continue
            difference = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            max_difference = max(max_difference, difference)
            if difference != 0.0:
                failures.append(f"values:{key}")
    return {
        "first": first_path.as_posix(),
        "second": second_path.as_posix(),
        "shot": int(first.get("shot", -1)),
        "seed": int(first.get("seed", -1)),
        "manifest_match": first.get("manifest_sha256") == second.get("manifest_sha256"),
        "max_abs_visual_difference": max_difference,
        "passed": not failures,
        "failures": failures,
    }


def _keyed(records: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], float]:
    output: dict[tuple[int, int], float] = {}
    for record in records:
        key = (int(record["shot"]), int(record["seed"]))
        if key in output:
            raise ValueError(f"duplicate shot/seed record: {key}")
        output[key] = float(record["macro_f1"])
    return output


def anchor_metric_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        if "anchor_macro_f1" in record:
            value = record["anchor_macro_f1"]
        else:
            anchor = record.get("anchor_metrics")
            if not isinstance(anchor, Mapping) or "macro_f1" not in anchor:
                raise ValueError("task row does not contain anchor Macro-F1")
            value = anchor["macro_f1"]
        output.append(
            {"shot": int(record["shot"]), "seed": int(record["seed"]), "macro_f1": float(value)}
        )
    return output


def _ci(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return float(values[0]), float(values[0])
    margin = float(t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
    mean = float(values.mean())
    return mean - margin, mean + margin


def summarize_metric_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    source: str,
    policy: str = "observed",
) -> list[dict[str, Any]]:
    values = _keyed(records)
    rows: list[dict[str, Any]] = []
    for shot in sorted({key[0] for key in values}):
        selected = np.asarray(
            [value for (current_shot, _), value in sorted(values.items()) if current_shot == shot]
        )
        low, high = _ci(selected)
        rows.append(
            {
                "dataset": dataset,
                "source": source,
                "config_policy": policy,
                "shot": shot,
                "seed_count": int(selected.size),
                "mean": float(selected.mean()),
                "std": float(selected.std(ddof=1)) if selected.size > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return rows


def paired_delta_rows(
    records: Sequence[Mapping[str, Any]],
    reference_records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    source: str,
    reference: str,
    policy: str = "observed",
) -> list[dict[str, Any]]:
    left = _keyed(records)
    right = _keyed(reference_records)
    if set(left) != set(right):
        raise ValueError("paired source records must have identical shot/seed keys")
    rows: list[dict[str, Any]] = []
    for shot in sorted({key[0] for key in left}):
        deltas = np.asarray([left[key] - right[key] for key in sorted(left) if key[0] == shot])
        low, high = _ci(deltas)
        rows.append(
            {
                "dataset": dataset,
                "source": source,
                "reference": reference,
                "config_policy": policy,
                "shot": shot,
                "seed_count": int(deltas.size),
                "mean_delta": float(deltas.mean()),
                "delta_std": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "wins": int(np.count_nonzero(deltas > 0.0)),
                "ties": int(np.count_nonzero(deltas == 0.0)),
                "losses": int(np.count_nonzero(deltas < 0.0)),
            }
        )
    return rows


def _load_raw_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*shot_seed*.json")):
        payload = _read_json(path)
        if payload.get("debug_run") or "debug" in path.stem:
            continue
        rows.append(
            {
                "shot": int(payload["shot"]),
                "seed": int(payload["seed"]),
                "macro_f1": float(payload["metrics"]["macro_f1"]),
                "anchor_macro_f1": float(payload["anchor_metrics"]["macro_f1"]),
                "path": path.as_posix(),
            }
        )
    if len(rows) != 30:
        raise ValueError(f"expected 30 formal raw records under {root}, found {len(rows)}")
    return rows


def _load_fgvc_grid(root: Path, lambda_value: float, rho: float) -> list[dict[str, Any]]:
    payload = _read_json(root / "development" / "all_config_results.json")
    matches = [row for row in payload if float(row["lambda"]) == lambda_value and float(row["rho"]) == rho]
    if len(matches) != 1:
        raise ValueError(f"expected one FGVC config match for {lambda_value}/{rho}, found {len(matches)}")
    return [
        {
            "shot": int(row["shot"]),
            "seed": int(row["seed"]),
            "macro_f1": float(row["metrics"]["macro_f1"]),
            "anchor_macro_f1": float(row["anchor_metrics"]["macro_f1"]),
        }
        for row in matches[0]["task_results"]
    ]


def _load_selected_config(root: Path) -> tuple[float, float]:
    selected = _read_json(root / "development" / "selected_config.json")["selected"]
    return float(selected["lambda"]), float(selected["rho"])


def _bundle_metadata(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    metadata = payload.get("metadata", {})
    generation = metadata.get("generation", {}) if isinstance(metadata, dict) else {}
    fallback = generation.get("fallback_classes", [])
    return {
        "source": payload.get("source"),
        "class_count": len(payload.get("classes", [])),
        "fallback_count": len(fallback),
        "fallback_classes": list(fallback),
        "path": path.as_posix(),
        "sha256": sha256_file(path),
    }


def _cache_audits(
    source_roots: Mapping[str, Path],
    reference: str,
    *,
    expected_count: int = 30,
) -> list[dict[str, Any]]:
    reference_files = {
        path.name: path
        for path in (source_roots[reference] / "cache").glob("*shot_seed*.json")
        if "debug" not in path.stem
    }
    if len(reference_files) != expected_count:
        raise ValueError("reference visual cache grid is incomplete")
    audits: list[dict[str, Any]] = []
    for source, root in source_roots.items():
        if source == reference:
            continue
        current_files = {
            path.name: path
            for path in (root / "cache").glob("*shot_seed*.json")
            if "debug" not in path.stem
        }
        if set(current_files) != set(reference_files):
            raise ValueError(f"visual cache file grid mismatch for {source}")
        audits.extend(
            audit_visual_cache_pair(reference_files[name], current_files[name])
            for name in sorted(reference_files)
        )
    return audits


def build_report(*, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    wm_roots = {
        "llm_structured": root / "results" / "wm811k_class_reliability" / "llm_structured",
        "qwen2_5_7b_instruct": root / "results" / "wm811k_class_reliability" / "qwen2_5_7b_instruct",
    }
    fgvc_root = root / "results" / "semantic_source_matching" / "fgvc_validation_runs"
    fgvc_roots = {
        source: fgvc_root / source
        for source in ("variant_name", "hierarchy_deterministic", "qwen2_5_7b_instruct")
    }
    wm_records = {
        source: _load_raw_records(source_root / "raw" / "classwise_reliability")
        for source, source_root in wm_roots.items()
    }
    wm_reference = wm_records["llm_structured"]
    wm_summary = [
        row
        for source, records in wm_records.items()
        for row in summarize_metric_rows(records, dataset="WM-811K", source=source)
    ]
    wm_pairs = paired_delta_rows(
        wm_records["qwen2_5_7b_instruct"],
        wm_reference,
        dataset="WM-811K",
        source="qwen2_5_7b_instruct",
        reference="llm_structured",
    )
    wm_anchor_records = anchor_metric_records(wm_reference)
    wm_anchor_summary = summarize_metric_rows(
        wm_anchor_records, dataset="WM-811K", source="visual_anchor"
    )
    wm_semantic_anchor_pairs = [
        row
        for source, records in wm_records.items()
        for row in paired_delta_rows(
            records,
            anchor_metric_records(records),
            dataset="WM-811K",
            source=source,
            reference="visual_anchor",
        )
    ]

    common_lambda, common_rho = _load_selected_config(fgvc_roots["hierarchy_deterministic"])
    fgvc_common = {
        source: _load_fgvc_grid(source_root, common_lambda, common_rho)
        for source, source_root in fgvc_roots.items()
    }
    common_policy = f"common_lambda_{common_lambda:g}_rho_{common_rho:g}"
    fgvc_summary = [
        row
        for source, records in fgvc_common.items()
        for row in summarize_metric_rows(
            records,
            dataset="FGVC-Aircraft Validation",
            source=source,
            policy=common_policy,
        )
    ]
    fgvc_reference = fgvc_common["hierarchy_deterministic"]
    fgvc_pairs = [
        row
        for source in ("variant_name", "qwen2_5_7b_instruct")
        for row in paired_delta_rows(
            fgvc_common[source],
            fgvc_reference,
            dataset="FGVC-Aircraft Validation",
            source=source,
            reference="hierarchy_deterministic",
            policy=common_policy,
        )
    ]
    fgvc_anchor_records = anchor_metric_records(fgvc_reference)
    fgvc_anchor_summary = summarize_metric_rows(
        fgvc_anchor_records,
        dataset="FGVC-Aircraft Validation",
        source="visual_anchor",
        policy=common_policy,
    )
    fgvc_semantic_anchor_pairs = [
        row
        for source in fgvc_roots
        for row in paired_delta_rows(
            fgvc_common[source],
            anchor_metric_records(fgvc_common[source]),
            dataset="FGVC-Aircraft Validation",
            source=source,
            reference="visual_anchor",
            policy=common_policy,
        )
    ]
    fgvc_selected: list[dict[str, Any]] = []
    fgvc_selected_semantic_anchor_pairs: list[dict[str, Any]] = []
    for source, source_root in fgvc_roots.items():
        lambda_value, rho = _load_selected_config(source_root)
        selected_records = _load_fgvc_grid(source_root, lambda_value, rho)
        policy = f"source_selected_lambda_{lambda_value:g}_rho_{rho:g}"
        fgvc_selected.extend(
            summarize_metric_rows(
                selected_records,
                dataset="FGVC-Aircraft Validation",
                source=source,
                policy=policy,
            )
        )
        fgvc_selected_semantic_anchor_pairs.extend(
            paired_delta_rows(
                selected_records,
                anchor_metric_records(selected_records),
                dataset="FGVC-Aircraft Validation",
                source=source,
                reference="visual_anchor",
                policy=policy,
            )
        )

    semantics_root = root / "data" / "processed" / "FGVC-Aircraft" / "semantics"
    bundles = {
        "variant_name": _bundle_metadata(semantics_root / "variant_name_v1.json"),
        "hierarchy_deterministic": _bundle_metadata(
            semantics_root / "hierarchy_deterministic_v1.json"
        ),
        "qwen2_5_7b_instruct": _bundle_metadata(
            semantics_root / "qwen2_5_7b_instruct_v1.json"
        ),
    }
    audits = {
        "WM-811K": _cache_audits(wm_roots, "llm_structured"),
        "FGVC-Aircraft Validation": _cache_audits(fgvc_roots, "hierarchy_deterministic"),
    }
    if not all(item["passed"] for values in audits.values() for item in values):
        raise ValueError("one or more semantic sources do not share the frozen visual cache")
    return {
        "schema_version": "semantic_source_matching_report_v1",
        "protocol": {
            "shared_visual_cache": True,
            "support_only_selection": True,
            "query_labels_used_for_selection": False,
            "seed_count": 10,
        },
        "visual_cache_audits": audits,
        "semantic_bundle_metadata": bundles,
        "common_config": {
            "dataset": "FGVC-Aircraft Validation",
            "reference_source": "hierarchy_deterministic",
            "lambda": common_lambda,
            "rho": common_rho,
        },
        "summary": wm_summary + fgvc_summary + fgvc_selected,
        "anchor_summary": wm_anchor_summary + fgvc_anchor_summary,
        "paired_comparisons": wm_pairs + fgvc_pairs,
        "semantic_vs_anchor_comparisons": (
            wm_semantic_anchor_pairs
            + fgvc_semantic_anchor_pairs
            + fgvc_selected_semantic_anchor_pairs
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: Mapping[str, Any], output: str | Path) -> Path:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audits = [item for dataset in report["visual_cache_audits"].values() for item in dataset]
    (target / "visual_cache_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for key, filename in REPORT_FILENAMES.items():
        _write_csv(target / filename, report[key])
    return target


def _portable_report(report: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(report))
    for rows in output.get("visual_cache_audits", {}).values():
        for row in rows:
            row.pop("first", None)
            row.pop("second", None)
    for metadata in output.get("semantic_bundle_metadata", {}).values():
        metadata.pop("path", None)
    return output


def export_compact_artifacts(report_path: str | Path, output: str | Path) -> list[Path]:
    """Export portable JSON/CSV evidence without feature arrays or absolute paths."""
    source = Path(report_path)
    report = _portable_report(_read_json(source))
    target = Path(output)
    if target.suffix.lower() == ".npz":
        raise ValueError("compact artifact output must be a directory, not an NPZ file")
    target.mkdir(parents=True, exist_ok=True)
    report_output = target / "report.json"
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audits = [item for values in report["visual_cache_audits"].values() for item in values]
    audit_output = target / "visual_cache_audit.json"
    audit_output.write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    exported = [report_output, audit_output]
    for key, filename in REPORT_FILENAMES.items():
        path = target / filename
        _write_csv(path, report[key])
        exported.append(path)
    if list(target.rglob("*.npz")):
        raise ValueError("compact artifact directory contains forbidden NPZ files")
    return exported
