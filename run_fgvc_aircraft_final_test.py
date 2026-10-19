"""Run the once-only frozen FGVC-Aircraft Test evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fgvc_aircraft.semantics import load_semantic_bundle
from run_class_reliability_residual import _load_audited_linear_selections
from wm811k.baseline_experiment import discover_tasks
from wm811k.class_reliability_experiment import (
    evaluate_cached_task,
    load_task_cache,
    run_reliability_cache_tasks,
)
from wm811k.clip_features import FrozenClipEncoder
from wm811k.experiment_reporting import claim_output_root
from wm811k.metrics import summarize_seed_metrics
from wm811k.resnet18_features import resolve_device


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "FGVC-Aircraft"
DEFAULT_IMAGE_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "FGVC-Aircraft" / "fgvc-aircraft-2013b" / "data"
)
DEFAULT_FROZEN_CONFIG = PROJECT_ROOT / "configs" / "fgvc_aircraft_class_reliability_frozen_v1.json"
DEFAULT_VALIDATION_ROOT = (
    PROJECT_ROOT / "results" / "fgvc_aircraft_validation" / "class_reliability_residual"
)
DEFAULT_LINEAR_BASELINE_ROOT = (
    PROJECT_ROOT / "results" / "fgvc_aircraft_validation" / "clip_linear_probe" / "raw"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "fgvc_aircraft_test" / "class_reliability_residual"
)
FORMAL_SEEDS = tuple(range(10))
DIAGNOSTIC_ABLATIONS = ("without_calibration", "without_safe_fallback")


def _claim_final_test_output_root(output_root: Path) -> None:
    """Claim the dedicated Test directory without broadening CLIP method schemas."""
    marker = output_root / ".experiment_method.json"
    output_root.mkdir(parents=True, exist_ok=True)
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("method") != "fgvc_aircraft_final_test":
            raise ValueError("Test output root belongs to another method")
        return
    marker.write_text(
        json.dumps({"method": "fgvc_aircraft_final_test"}, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FGVC-Aircraft 冻结配置的最终 Test 评价")
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--confirm-test-once", action="store_true", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--linear-baseline-root", type=Path, default=DEFAULT_LINEAR_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "fgvc_aircraft_class_reliability_frozen_v1",
        "development_split": "Validation",
        "intended_query_split": "Test",
        "test_run_before_freeze": False,
        "semantic_source": "qwen2_5_7b_instruct",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"frozen config has invalid {key!r}")
    if not path.is_file():
        raise FileNotFoundError(f"frozen config does not exist: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "shot", "mean", "std", "ci95_low", "ci95_high"))
        writer.writeheader()
        writer.writerows(rows)


def add_declared_diagnostic_ablations(
    results_path: Path,
    *,
    caches: Sequence[Any],
    selected_lambda: float,
    selected_rho: float,
) -> dict[str, Any]:
    """Append predeclared diagnostic ablations to an existing frozen Test result.

    Existing variants are treated as immutable. The supplied caches must already
    have been produced by the frozen Test protocol; this function only performs
    cheap score re-evaluation and never selects parameters from Test labels.
    """
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    variants = payload.setdefault("variants", {})
    required_existing = {
        "visual_anchor",
        "fixed_all_one",
        "global_reliability",
        "classwise_reliability",
    }
    missing = sorted(required_existing.difference(variants))
    if missing:
        raise ValueError(f"existing Test result is missing required variants: {missing}")
    for name in DIAGNOSTIC_ABLATIONS:
        if name in variants:
            raise FileExistsError(f"diagnostic ablation already exists: {name}")

    settings = {
        "without_calibration": {"calibration": False, "safe_fallback": True},
        "without_safe_fallback": {"calibration": True, "safe_fallback": False},
    }
    for name, options in settings.items():
        variants[name] = [
            evaluate_cached_task(
                cache,
                lambda_value=float(selected_lambda),
                rho=float(selected_rho),
                granularity="classwise",
                **options,
            )
            for cache in caches
        ]

    summaries = [row for row in payload.get("summary", []) if row.get("variant") not in DIAGNOSTIC_ABLATIONS]
    for name in DIAGNOSTIC_ABLATIONS:
        results = variants[name]
        for shot in (1, 5, 10):
            current = [result for result in results if int(result["shot"]) == shot]
            if not current:
                continue
            summary = summarize_seed_metrics(current, expected_seeds=FORMAL_SEEDS)["metrics"]["macro_f1"]
            summaries.append({
                "variant": name,
                "shot": shot,
                "mean": summary["mean"],
                "std": summary["std"],
                "ci95_low": summary["ci95_low"],
                "ci95_high": summary["ci95_high"],
            })
    payload["summary"] = summaries
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _payload_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_diagnostic_ablation_artifacts(results_path: Path, *, cache_count: int) -> dict[str, Any]:
    """Write the frozen-cache supplement table and its provenance audit."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    variants = payload.get("variants", {})
    required = {
        "visual_anchor",
        "fixed_all_one",
        "global_reliability",
        "classwise_reliability",
        *DIAGNOSTIC_ABLATIONS,
    }
    missing = sorted(required.difference(variants))
    if missing:
        raise ValueError(f"cannot write diagnostic artifacts; missing variants: {missing}")
    _write_summary_csv(results_path.parent / "final_macro_f1_table.csv", payload["summary"])
    existing_names = (
        "visual_anchor",
        "fixed_all_one",
        "global_reliability",
        "classwise_reliability",
    )
    config = payload.get("frozen_config", {})
    audit = {
        "schema_version": "fgvc_aircraft_test_diagnostic_ablation_audit_v1",
        "added_variants": list(DIAGNOSTIC_ABLATIONS),
        "evaluation_source": "immutable_frozen_test_score_cache",
        "cache_count": int(cache_count),
        "selected_lambda": float(config.get("selected_lambda", 1.0)),
        "selected_rho": float(config.get("selected_rho", 0.1)),
        "query_labels_used_for_selection": False,
        "parameters_reselected_on_test": False,
        "existing_variant_sha256": {
            name: _payload_sha256(variants[name]) for name in existing_names
        },
    }
    _write_json(results_path.parent / "diagnostic_ablation_audit.json", audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = _load_frozen_config(args.frozen_config)
    output_root = args.output_root.resolve()
    _claim_final_test_output_root(output_root)
    if (output_root / "test_audit.json").exists():
        raise FileExistsError("final Test result already exists; refusing a second evaluation")

    tasks = discover_tasks(
        args.data_root,
        run_all=True,
        query_split="test",
        n_way=100,
        image_root=args.image_root,
        formal_seeds=FORMAL_SEEDS,
    )
    validation_tasks = discover_tasks(
        args.data_root,
        run_all=True,
        query_split="validation",
        n_way=100,
        image_root=args.image_root,
        formal_seeds=FORMAL_SEEDS,
    )
    bundle_path = args.data_root / "semantics" / "qwen2_5_7b_instruct_v1.json"
    if _sha256(bundle_path) != config["semantic_bundle_sha256"]:
        raise ValueError("Qwen semantic bundle hash does not match frozen config")
    bundle = load_semantic_bundle(bundle_path, tasks[0].class_names)
    selections = _load_audited_linear_selections(
        validation_tasks,
        args.linear_baseline_root,
        support_augment_views=10,
    )
    test_support = {
        (task.shot, task.seed): [record.relative_path for record in task.support]
        for task in tasks
    }
    validation_support = {
        (task.shot, task.seed): [record.relative_path for record in task.support]
        for task in validation_tasks
    }
    if test_support != validation_support:
        raise ValueError("Test support samples do not match the audited Validation support")
    encoder = FrozenClipEncoder.create(resolve_device(args.device), dataset="fgvc_aircraft")
    cache_root = output_root / "cache"
    paths = run_reliability_cache_tasks(
        tasks,
        encoder=encoder,
        semantic_prompt_groups=bundle.prompts_by_class,
        semantic_source=bundle.source,
        semantic_bundle_sha256=_sha256(bundle_path),
        output_root=cache_root,
        batch_size=args.batch_size,
        limit_query=None,
        aggregate=True,
        formal_seeds=FORMAL_SEEDS,
        show_progress=not args.no_progress,
        linear_selection_by_task=selections,
        required_query_source="Test",
    )
    from wm811k.class_reliability_experiment import load_task_cache

    caches = [load_task_cache(path) for path in paths]
    settings = {
        "visual_anchor": {"lambda_value": 0.0, "rho": float(config["selected_rho"]), "granularity": "classwise"},
        "fixed_all_one": {"lambda_value": float(config["selected_lambda"]), "rho": float(config["selected_rho"]), "granularity": "fixed"},
        "global_reliability": {"lambda_value": float(config["selected_lambda"]), "rho": float(config["selected_rho"]), "granularity": "global"},
        "classwise_reliability": {"lambda_value": float(config["selected_lambda"]), "rho": float(config["selected_rho"]), "granularity": "classwise"},
    }
    variants = {
        name: [evaluate_cached_task(cache, calibration=True, safe_fallback=True, **setting) for cache in caches]
        for name, setting in settings.items()
    }
    summaries = []
    for name, results in variants.items():
        for shot in (1, 5, 10):
            current = [result for result in results if result["shot"] == shot]
            summary = summarize_seed_metrics(current, expected_seeds=FORMAL_SEEDS)["metrics"]["macro_f1"]
            summaries.append({"variant": name, "shot": shot, "mean": summary["mean"], "std": summary["std"], "ci95_low": summary["ci95_low"], "ci95_high": summary["ci95_high"]})
    _write_json(output_root / "test_results.json", {"frozen_config": config, "variants": variants, "summary": summaries})
    _write_summary_csv(output_root / "final_macro_f1_table.csv", summaries)
    _write_json(output_root / "test_audit.json", {"schema_version": "fgvc_aircraft_final_test_audit_v1", "query_split": "Test", "test_run_after_freeze": True, "frozen_config_sha256": _sha256(args.frozen_config), "semantic_bundle_sha256": _sha256(bundle_path), "cache_count": len(caches), "seeds": list(FORMAL_SEEDS), "query_labels_used_for_selection": False})
    print(json.dumps({"cache_count": len(caches), "output_root": output_root.as_posix(), "summary": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
