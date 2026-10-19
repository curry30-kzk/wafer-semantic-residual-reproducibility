"""Method-neutral, auditable experiment artifact writing."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .metrics import PRIMARY_METRICS, summarize_seed_metrics
from .v2_contract import load_v2_contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WM_V2_CONFIG = PROJECT_ROOT / "configs" / "wm811k_dedup_v2_frozen.json"


PREDICTION_FIELDS = (
    "original_index",
    "image_path",
    "true_index",
    "true_label",
    "predicted_index",
    "predicted_label",
    "cosine_score",
    "correct",
)

SCHEMA_METHODS = {
    "clip_linear_probe_result_v1": "linear_probe",
    "clip_tip_adapter_result_v1": "tip_adapter",
    "clip_tip_adapter_f_result_v1": "tip_adapter_f",
    "clip_lp_plus_plus_result_v1": "lp_plus_plus",
    "clip_lp_plus_plus_fair_result_v1": "lp_plus_plus_fair",
    "clip_zs_lp_result_v1": "zs_lp",
    "clip_clap_result_v1": "clap",
    "clip_rla_result_v1": "rla",
    "clip_domain_geometry_fusion_result_v1": "domain_geometry_fusion",
    "clip_fixed_fusion_result_v1": "fixed_fusion",
    "clip_llm_reliability_fusion_result_v1": "llm_reliability_fusion",
    "clip_llm_structured_fixed_fusion_result_v1": "llm_structured_fixed_fusion",
    "clip_zero_shot_result_v1": "zero_shot",
    "clip_visual_prototype_result_v1": "visual_prototype",
    "resnet18_prototype_result_v1": "resnet18_prototype",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_result_provenance(
    task: Any,
    *,
    frozen_config_path: str | Path = DEFAULT_WM_V2_CONFIG,
) -> dict[str, Any]:
    """Return auditable manifest identity, plus the full v2 contract when required."""

    manifest_path = Path(task.manifest_path)
    if not manifest_path.is_file():
        if str(task.protocol_name) == "wm811k_dedup_8way_v2":
            raise FileNotFoundError(f"v2 task manifest does not exist: {manifest_path}")
        return {}
    fields: dict[str, Any] = {"manifest_sha256": _sha256_file(manifest_path)}
    if str(task.protocol_name) != "wm811k_dedup_8way_v2":
        return fields
    contract = load_v2_contract(frozen_config_path)
    if getattr(task, "dedup_audit_hash", None) != contract.dedup_audit_sha256:
        raise ValueError("v2 task audit hash does not match frozen result provenance")
    if getattr(task, "candidate_pool_hash", None) != contract.candidate_pool_sha256:
        raise ValueError("v2 task candidate pool hash does not match frozen result provenance")
    fields.update(
        {
            "dedup_audit_sha256": contract.dedup_audit_sha256,
            "candidate_pool_sha256": contract.candidate_pool_sha256,
            "frozen_config_sha256": contract.file_sha256,
            "visual_encoder": contract.visual_encoder,
            "preprocess": contract.preprocess,
            "augmentation": contract.augmentation,
        }
    )
    return fields


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, buffer.getvalue())


def claim_output_root(output_root: str | Path, method: str) -> None:
    """Atomically bind a result directory to one experiment method."""

    if method not in {
        "linear_probe",
        "tip_adapter",
        "tip_adapter_f",
        "lp_plus_plus",
        "lp_plus_plus_fair",
        "zs_lp",
        "clap",
        "rla",
        "zero_shot",
        "visual_prototype",
        "fixed_fusion",
        "domain_geometry_fusion",
        "llm_structured_fixed_fusion",
        "llm_reliability_fusion",
    }:
        raise ValueError(f"unsupported CLIP experiment method: {method}")
    output_root = Path(output_root)
    marker_path = output_root / ".experiment_method.json"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid experiment method marker: {marker_path}") from exc
        existing_method = str(marker.get("method", ""))
        if existing_method != method:
            raise ValueError(
                f"output root already belongs to {existing_method!r}, "
                f"not {method!r}: {output_root}"
            )
        return

    existing_methods: set[str] = set()
    raw_root = output_root / "raw"
    if raw_root.is_dir():
        for result_path in raw_root.glob("*.json"):
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid existing result JSON: {result_path}") from exc
            schema = str(payload.get("schema_version", ""))
            if schema:
                existing_methods.add(SCHEMA_METHODS.get(schema, f"unknown:{schema}"))
    if existing_methods and existing_methods != {method}:
        raise ValueError(
            f"output root already belongs to {sorted(existing_methods)!r}, "
            f"not {method!r}: {output_root}"
        )
    _write_json(marker_path, {"method": method})


def _confusion_rows(
    result: dict[str, Any], class_names: Sequence[str]
) -> list[dict[str, Any]]:
    matrix = result["metrics"]["confusion_matrix"]
    rows = []
    for true_index, true_name in enumerate(class_names):
        row: dict[str, Any] = {"true_label": true_name}
        row.update(
            {
                predicted_name: int(matrix[true_index][predicted_index])
                for predicted_index, predicted_name in enumerate(class_names)
            }
        )
        rows.append(row)
    return rows


def write_run_artifacts(
    output_root: str | Path,
    result: dict[str, Any],
    prediction_rows: Sequence[dict[str, Any]],
    *,
    stem: str,
    extra_prediction_fields: Sequence[str] = (),
) -> None:
    """Write one run without mutating the caller's result dictionary."""

    all_prediction_fields = (*PREDICTION_FIELDS, *extra_prediction_fields)
    if len(set(all_prediction_fields)) != len(all_prediction_fields):
        raise ValueError("prediction field names must be unique")
    output_root = Path(output_root)
    raw_root = output_root / "raw"
    confusion_root = output_root / "confusion_matrices"
    predictions_name = f"{stem}_predictions.csv"
    persisted_result = dict(result)
    persisted_result["artifacts"] = {
        "predictions_csv": f"raw/{predictions_name}",
        "confusion_matrix_csv": f"confusion_matrices/{stem}.csv",
    }
    _write_json(raw_root / f"{stem}.json", persisted_result)
    _write_csv(
        raw_root / predictions_name,
        all_prediction_fields,
        prediction_rows,
    )
    class_names = tuple(str(name) for name in result["class_names"])
    _write_csv(
        confusion_root / f"{stem}.csv",
        ("true_label", *class_names),
        _confusion_rows(result, class_names),
    )


def write_fewshot_summaries(
    output_root: str | Path,
    results: Sequence[dict[str, Any]],
    expected_seeds: Sequence[int],
) -> None:
    """Write shot-wise mean, sample standard deviation, and Student-t CI."""

    output_root = Path(output_root)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[int(result["shot"])].append(result)

    summary_rows: list[dict[str, Any]] = []
    for shot in sorted(grouped):
        summary = summarize_seed_metrics(
            grouped[shot], expected_seeds=expected_seeds
        )
        _write_json(
            output_root / "summaries" / f"{shot}shot_summary.json", summary
        )
        for metric_name in PRIMARY_METRICS:
            metric = summary["metrics"][metric_name]
            summary_rows.append(
                {
                    "shot": shot,
                    "metric": metric_name,
                    "mean": metric["mean"],
                    "std": metric["std"],
                    "ci95_low": metric["ci95_low"],
                    "ci95_high": metric["ci95_high"],
                    "seeds": ";".join(str(seed) for seed in summary["seeds"]),
                }
            )
    _write_csv(
        output_root / "summaries" / "all_shots_summary.csv",
        ("shot", "metric", "mean", "std", "ci95_low", "ci95_high", "seeds"),
        summary_rows,
    )


def write_zero_shot_summary(
    output_root: str | Path, result: dict[str, Any]
) -> None:
    """Write one deterministic result without invented seed statistics."""

    output_root = Path(output_root)
    metrics = {
        metric_name: float(result["metrics"][metric_name])
        for metric_name in PRIMARY_METRICS
    }
    payload = {
        "aggregation": "deterministic single evaluation",
        "metrics": metrics,
    }
    rows = [
        {"metric": metric_name, "value": metrics[metric_name]}
        for metric_name in PRIMARY_METRICS
    ]
    _write_json(output_root / "summary" / "zero_shot_summary.json", payload)
    _write_csv(
        output_root / "summary" / "zero_shot_summary.csv",
        ("metric", "value"),
        rows,
    )
