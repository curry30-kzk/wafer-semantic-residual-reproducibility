"""WM-811K main-dataset orchestration for classwise semantic reliability."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from scipy.stats import t

from .baseline_experiment import discover_tasks
from .class_reliability_experiment import (
    TaskReliabilityCache,
    evaluate_cached_task,
    load_task_cache,
    run_reliability_cache_tasks,
)
from .clip_adaptation import (
    build_group_stratified_support_folds,
    build_stratified_support_folds,
)
from .clip_features import FrozenClipEncoder, MODEL_IDENTIFIER
from .class_reliability_residual import apply_classwise_residual
from .llm_descriptions import build_llm_structured_prompt_ensemble
from .qwen_descriptions import load_qwen_semantic_bundle
from .labels import DEFECT_CLASSES
from .metrics import evaluate_predictions, summarize_seed_metrics
from .experiment_reporting import task_result_provenance
from .resnet18_features import resolve_device
from .v2_contract import WM811KDedupV2Contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "WM811K"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "wm811k_class_reliability"
DEFAULT_LINEAR_ROOT = PROJECT_ROOT / "results" / "clip_linear_probe" / "raw"
DEFAULT_TIP_F_ROOT = PROJECT_ROOT / "results" / "clip_tip_adapter_f" / "raw"
SEMANTIC_SOURCE = "llm_structured"
SEMANTIC_VERSION = "wm811k_codex_gpt5_structured_v1"
QWEN_SEMANTIC_SOURCE = "qwen2_5_7b_instruct"
QWEN_SEMANTIC_VERSION = "wm811k_qwen_semantic_bundle_v1"
SUPPORTED_SEMANTIC_SOURCES = (SEMANTIC_SOURCE, QWEN_SEMANTIC_SOURCE)
DEFAULT_QWEN_SEMANTICS_PATH = (
    DEFAULT_DATA_ROOT / "semantics" / "qwen2_5_7b_instruct_v1.json"
)
FROZEN_LAMBDA = 1.0
FROZEN_RHO = 0.01
EXPECTED_SHOTS = (1, 5, 10)
EXPECTED_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class WMRunContext:
    """Explicitly bind one WM protocol to its data and result namespaces."""

    protocol: str
    data_root: Path
    image_root: Path
    output_root: Path
    linear_root: Path
    frozen_config_path: Path | None = None


V1_RUN_CONTEXT = WMRunContext(
    protocol="wm811k_official_8way_v1",
    data_root=DEFAULT_DATA_ROOT,
    image_root=DEFAULT_DATA_ROOT / "images",
    output_root=DEFAULT_OUTPUT_ROOT,
    linear_root=DEFAULT_LINEAR_ROOT,
)


def normalize_formal_seeds(seeds: Sequence[int] | None = None) -> tuple[int, ...]:
    values = EXPECTED_SEEDS if seeds is None else tuple(int(seed) for seed in seeds)
    if not values or len(set(values)) != len(values) or any(seed < 0 for seed in values):
        raise ValueError("formal seeds must be a non-empty sequence of unique non-negative integers")
    return values


def paired_seed_statistics(
    method_values: Sequence[float],
    anchor_values: Sequence[float],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize paired per-seed improvements without using query labels for selection."""

    method = np.asarray(tuple(float(value) for value in method_values), dtype=np.float64)
    anchor = np.asarray(tuple(float(value) for value in anchor_values), dtype=np.float64)
    if method.ndim != 1 or anchor.ndim != 1 or method.size != anchor.size or method.size < 2:
        raise ValueError("paired seed values must be equal-length 1D sequences with at least two seeds")
    if not np.isfinite(method).all() or not np.isfinite(anchor).all():
        raise ValueError("paired seed values must be finite")
    confidence_value = float(confidence)
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be between zero and one")
    deltas = method - anchor
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1))
    critical = float(t.ppf((1.0 + confidence_value) / 2.0, deltas.size - 1))
    margin = critical * std_delta / np.sqrt(deltas.size)
    if std_delta == 0.0:
        p_value = 1.0 if mean_delta == 0.0 else 0.0
    else:
        statistic = mean_delta / (std_delta / np.sqrt(deltas.size))
        p_value = float(2.0 * t.sf(abs(statistic), deltas.size - 1))
    return {
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci95_low": float(mean_delta - margin),
        "ci95_high": float(mean_delta + margin),
        "p_value_two_sided": p_value,
        "nondecline_seed_count": int(np.sum(deltas >= 0.0)),
        "improved_seed_count": int(np.sum(deltas > 0.0)),
        "seed_count": int(deltas.size),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_source_hash(
    semantic_source: str = SEMANTIC_SOURCE,
    *,
    qwen_bundle_path: str | Path = DEFAULT_QWEN_SEMANTICS_PATH,
) -> str:
    """Hash the frozen semantic artifact used by a cache."""

    source = str(semantic_source).strip()
    if source == SEMANTIC_SOURCE:
        return _sha256_file(Path(__file__).resolve().parents[0] / "llm_descriptions.py")
    if source == QWEN_SEMANTIC_SOURCE:
        return _sha256_file(Path(qwen_bundle_path))
    raise ValueError(f"unsupported WM semantic source: {source}")


def build_semantic_prompt_groups(
    class_names: Sequence[str],
    *,
    semantic_source: str = SEMANTIC_SOURCE,
    qwen_bundle_path: str | Path = DEFAULT_QWEN_SEMANTICS_PATH,
) -> tuple[tuple[str, ...], ...]:
    """Load one frozen prompt ensemble for the selected semantic source."""

    source = str(semantic_source).strip()
    if source == SEMANTIC_SOURCE:
        return build_llm_structured_prompt_ensemble(class_names)
    if source == QWEN_SEMANTIC_SOURCE:
        bundle = load_qwen_semantic_bundle(qwen_bundle_path)
        if tuple(class_names) != bundle.class_names:
            raise ValueError("Qwen semantic class order does not match WM task")
        return bundle.prompts_by_class
    raise ValueError(f"unsupported WM semantic source: {source}")


def discover_wm_tasks(
    data_root: str | Path | None = None,
    *,
    shot: int = 5,
    seed: int = 0,
    run_all: bool = False,
    formal_seeds: Sequence[int] | None = None,
    image_root: str | Path | None = None,
    context: WMRunContext = V1_RUN_CONTEXT,
) -> list[Any]:
    """Load the frozen WM-811K Training-support/Test-query task grid."""

    seeds = normalize_formal_seeds(formal_seeds)
    tasks = discover_tasks(
        context.data_root if data_root is None else data_root,
        shot=shot,
        seed=seed,
        run_all=run_all,
        formal_seeds=seeds,
        query_split=None,
        n_way=8,
        image_root=context.image_root if image_root is None else image_root,
    )
    for task in tasks:
        if task.protocol_name != context.protocol:
            raise ValueError(f"WM tasks must use {context.protocol}")
        if task.class_names != DEFECT_CLASSES:
            raise ValueError("WM tasks must use the official class order")
        if task.support_source != "Training" or task.query_source != "Test":
            raise ValueError("WM reliability requires Training support and Test query")
    return tasks


def validate_wm_cache_contract(
    cache: TaskReliabilityCache,
    *,
    formal_seeds: Sequence[int] | None = None,
    context: WMRunContext = V1_RUN_CONTEXT,
    v2_contract: WM811KDedupV2Contract | None = None,
) -> None:
    """Validate the paper-facing split and provenance contract."""

    if cache.protocol != context.protocol:
        raise ValueError(f"cache protocol must be {context.protocol}")
    if cache.class_names != DEFECT_CLASSES:
        raise ValueError("cache class order must be the official WM order")
    if cache.support_source != "Training" or cache.query_source != "Test":
        raise ValueError("WM cache must use Training support and Test query")
    if cache.shot not in EXPECTED_SHOTS or cache.seed not in normalize_formal_seeds(formal_seeds):
        raise ValueError("cache shot/seed is outside the formal WM grid")
    runtime = cache.runtime_metadata or {}
    if bool(runtime.get("query_labels_used_for_selection", True)):
        raise ValueError("cache reports query-label selection leakage")
    expected_mode = "visual_prototype" if cache.shot == 1 else "linear_probe"
    if cache.anchor_mode != expected_mode:
        raise ValueError(f"{cache.shot}-shot cache must use {expected_mode} anchor")
    if not cache.manifest_sha256 or not cache.semantic_bundle_sha256:
        raise ValueError("cache provenance hashes must not be blank")
    if context.protocol == "wm811k_dedup_8way_v2":
        if v2_contract is None:
            raise ValueError("WM v2 cache validation requires the frozen v2 contract")
        v2_contract.validate_cache_metadata(
            {
                "protocol": cache.protocol,
                "shot": cache.shot,
                **dict(cache.provenance or {}),
            }
        )


def _load_selection(
    path: Path,
    task: Any,
    *,
    expected_schema: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"audited visual baseline result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != expected_schema:
        raise ValueError(f"unexpected visual baseline schema: {path}")
    if payload.get("protocol") != task.protocol_name:
        raise ValueError("visual baseline protocol does not match WM task")
    if int(payload.get("shot", -1)) != task.shot or int(payload.get("seed", -1)) != task.seed:
        raise ValueError("visual baseline shot/seed does not match WM task")
    if payload.get("query_source", "Test") not in (None, "Test"):
        raise ValueError("WM visual baseline must use Test query")
    if payload.get("query_labels_used_for_selection") is not False:
        raise ValueError("visual baseline selection used query labels")
    if tuple(payload.get("class_names", [])) != task.class_names:
        raise ValueError("visual baseline class order does not match WM task")
    if payload.get("model_identifier") != MODEL_IDENTIFIER:
        raise ValueError("visual baseline CLIP model identifier does not match")
    if task.protocol_name == "wm811k_dedup_8way_v2":
        expected_provenance = task_result_provenance(task)
        for name, value in expected_provenance.items():
            if payload.get(name) != value:
                raise ValueError(
                    f"v2 visual baseline {name} does not match the frozen task"
                )
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("visual baseline selection metadata is missing")
    return selection


def load_wm_linear_selections(
    tasks: Sequence[Any],
    baseline_root: str | Path | None = None,
    *,
    context: WMRunContext = V1_RUN_CONTEXT,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Load only Training-support C selections from audited linear probes."""

    root = context.linear_root if baseline_root is None else Path(baseline_root)
    output: dict[tuple[int, int], dict[str, Any]] = {}
    for task in tasks:
        if task.shot == 1:
            continue
        path = root / f"8way_{task.shot}shot_seed{task.seed}.json"
        selection = _load_selection(path, task, expected_schema="clip_linear_probe_result_v1")
        labels = np.asarray(
            [record.label_index for record in task.support], dtype=np.int64
        )
        if context.protocol == "wm811k_dedup_8way_v2":
            groups = [str(record.content_hash) for record in task.support]
            folds = [
                fold.tolist()
                for fold in build_group_stratified_support_folds(
                    labels, groups, num_classes=task.n_way
                )
            ]
        else:
            folds = [
                fold.tolist()
                for fold in build_stratified_support_folds(
                    labels, num_classes=task.n_way
                )
            ]
        if selection.get("folds") != folds:
            raise ValueError("linear baseline support folds do not match WM manifest")
        c_value = float(selection.get("C", float("nan")))
        if not np.isfinite(c_value) or c_value <= 0.0:
            raise ValueError("linear baseline C is invalid")
        output[(task.shot, task.seed)] = {
            **selection,
            "baseline_result_path": path.resolve().as_posix(),
            "selection_source": "Training support only",
        }
    return output


def build_wm_caches(
    *,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
    shot: int = 5,
    seed: int = 0,
    run_all: bool = False,
    formal_seeds: Sequence[int] | None = None,
    image_root: str | Path | None = None,
    batch_size: int = 32,
    device: str = "auto",
    limit_query: int | None = None,
    no_progress: bool = False,
    linear_baseline_root: str | Path | None = None,
    semantic_source: str = SEMANTIC_SOURCE,
    qwen_bundle_path: str | Path = DEFAULT_QWEN_SEMANTICS_PATH,
    reference_visual_cache_root: str | Path | None = None,
    context: WMRunContext = V1_RUN_CONTEXT,
    v2_contract: WM811KDedupV2Contract | None = None,
) -> list[Path]:
    seeds = normalize_formal_seeds(formal_seeds)
    if context.protocol == "wm811k_dedup_8way_v2":
        if v2_contract is None or v2_contract.protocol != context.protocol:
            raise ValueError("WM v2 cache construction requires its frozen contract")
        if tuple(seeds) != tuple(v2_contract.payload["formal_seeds"]) and run_all:
            raise ValueError("formal WM v2 cache runs require frozen seeds 0-9")
    tasks = discover_wm_tasks(
        data_root,
        shot=shot,
        seed=seed,
        run_all=run_all,
        formal_seeds=seeds,
        image_root=image_root,
        context=context,
    )
    selection = load_wm_linear_selections(
        tasks, linear_baseline_root, context=context
    )
    encoder = FrozenClipEncoder.create(resolve_device(device), dataset="wm811k")
    groups = build_semantic_prompt_groups(
        tasks[0].class_names,
        semantic_source=semantic_source,
        qwen_bundle_path=qwen_bundle_path,
    )
    semantic_hash = semantic_source_hash(
        semantic_source, qwen_bundle_path=qwen_bundle_path
    )
    if v2_contract is not None:
        expected_semantic_hash = (
            v2_contract.qwen_semantic_sha256
            if semantic_source == QWEN_SEMANTIC_SOURCE
            else v2_contract.structured_semantic_sha256
        )
        if semantic_hash != expected_semantic_hash:
            raise ValueError("WM v2 semantic bundle hash does not match the frozen contract")
    return run_reliability_cache_tasks(
        tasks,
        encoder=encoder,
        semantic_prompt_groups=groups,
        semantic_source=semantic_source,
        semantic_bundle_sha256=semantic_hash,
        output_root=(context.output_root if output_root is None else Path(output_root))
        / semantic_source
        / "cache",
        batch_size=batch_size,
        limit_query=limit_query,
        aggregate=run_all,
        formal_seeds=seeds,
        show_progress=not no_progress,
        support_augment_views=10,
        tip_f_max_epochs=20,
        reference_visual_cache_root=reference_visual_cache_root,
        linear_selection_by_task=selection,
        required_query_source="Test",
        anchor_prompt_source="frozen_wm811k_generic_class_prompt",
        dataset="wm811k",
        one_shot_anchor_mode="visual_prototype",
        cache_provenance=(
            None
            if v2_contract is None
            else {
                "dedup_audit_sha256": v2_contract.dedup_audit_sha256,
                "candidate_pool_sha256": v2_contract.candidate_pool_sha256,
                "frozen_config_sha256": v2_contract.file_sha256,
                "visual_encoder": v2_contract.visual_encoder,
                "preprocess": v2_contract.preprocess,
                "augmentation": v2_contract.augmentation,
            }
        ),
    )


def _with_wm_metadata(result: dict[str, Any], cache: TaskReliabilityCache) -> dict[str, Any]:
    result = dict(result)
    result["schema_version"] = "wm811k_class_reliability_task_result_v1"
    result["method"] = "WM-811K classwise semantic reliability residual"
    result["debug_run"] = bool((cache.runtime_metadata or {}).get("debug_run", False))
    result["paper_eligible"] = not result["debug_run"]
    source = str((cache.runtime_metadata or {}).get("semantic_source", SEMANTIC_SOURCE))
    result["semantic_source"] = source
    result["semantic_version"] = (
        QWEN_SEMANTIC_VERSION if source == QWEN_SEMANTIC_SOURCE else SEMANTIC_VERSION
    )
    # Bind every v2 Test result to the same auditable contract as its cache.
    # These fields are deliberately copied rather than inferred from a result
    # directory name, so later analysis can reject stale or mixed protocols.
    if cache.provenance:
        for name in (
            "dedup_audit_sha256",
            "candidate_pool_sha256",
            "frozen_config_sha256",
            "visual_encoder",
            "preprocess",
            "augmentation",
        ):
            if name in cache.provenance:
                result[name] = cache.provenance[name]
    result["manifest_sha256"] = cache.manifest_sha256
    result["semantic_bundle_sha256"] = cache.semantic_bundle_sha256
    if int(cache.shot) == 1:
        result["selection"] = dict(result["selection"])
        result["selection"]["reason"] = "one_shot_frozen_visual_prototype_fallback"
    return result


def evaluate_wm_cache(
    cache: TaskReliabilityCache,
    *,
    granularity: str = "classwise",
    calibration: bool | Literal["support_bias_zscore", "none", "platt"] = True,
    safe_fallback: bool = True,
    fusion_mode: Literal["residual", "direct_weighted"] = "residual",
    lambda_value: float = FROZEN_LAMBDA,
    rho: float = FROZEN_RHO,
    formal_seeds: Sequence[int] | None = None,
    context: WMRunContext = V1_RUN_CONTEXT,
    v2_contract: WM811KDedupV2Contract | None = None,
) -> dict[str, Any]:
    validate_wm_cache_contract(
        cache,
        formal_seeds=formal_seeds,
        context=context,
        v2_contract=v2_contract,
    )
    lambda_float = float(lambda_value)
    if lambda_float not in (0.0, FROZEN_LAMBDA) or float(rho) != FROZEN_RHO:
        raise ValueError("WM Test evaluation only permits lambda=0 or the frozen lambda, with frozen rho")
    return _with_wm_metadata(
        evaluate_cached_task(
            cache,
            lambda_value=lambda_float,
            rho=FROZEN_RHO,
            granularity=granularity,  # type: ignore[arg-type]
            calibration=calibration,
            safe_fallback=safe_fallback,
            fusion_mode=fusion_mode,
        ),
        cache,
    )


def _cache_paths(
    output_root: Path,
    semantic_source: str = SEMANTIC_SOURCE,
    formal_seeds: Sequence[int] | None = None,
) -> list[Path]:
    seeds = normalize_formal_seeds(formal_seeds)
    cache_root = output_root / semantic_source / "cache"
    return [
        cache_root / f"8way_{shot}shot_seed{seed}.json"
        for shot in EXPECTED_SHOTS
        for seed in seeds
    ]


def load_wm_formal_caches(
    output_root: str | Path | None = None,
    *,
    semantic_source: str = SEMANTIC_SOURCE,
    formal_seeds: Sequence[int] | None = None,
    context: WMRunContext = V1_RUN_CONTEXT,
    v2_contract: WM811KDedupV2Contract | None = None,
) -> list[TaskReliabilityCache]:
    seeds = normalize_formal_seeds(formal_seeds)
    root = context.output_root if output_root is None else Path(output_root)
    paths = _cache_paths(root, semantic_source, seeds)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} WM formal caches; first={missing[0]}")
    caches = [load_task_cache(path) for path in paths]
    for cache in caches:
        validate_wm_cache_contract(
            cache,
            formal_seeds=seeds,
            context=context,
            v2_contract=v2_contract,
        )
        if bool((cache.runtime_metadata or {}).get("debug_run")):
            raise ValueError("debug WM cache cannot enter formal report")
    return caches


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


VARIANTS = {
    "visual_anchor": {"granularity": "classwise", "calibration": True, "safe_fallback": True, "lambda_value": 0.0},
    "fixed_all_one": {"granularity": "fixed", "calibration": True, "safe_fallback": True},
    "global_reliability": {"granularity": "global", "calibration": True, "safe_fallback": True},
    "classwise_reliability": {"granularity": "classwise", "calibration": True, "safe_fallback": True},
    "without_calibration": {"granularity": "classwise", "calibration": False, "safe_fallback": True},
    "without_safe_fallback": {"granularity": "classwise", "calibration": True, "safe_fallback": False},
    "direct_weighted_fusion": {
        "granularity": "classwise",
        "calibration": True,
        "safe_fallback": True,
        "fusion_mode": "direct_weighted",
    },
    "platt_calibration": {
        "granularity": "classwise",
        "calibration": "platt",
        "safe_fallback": True,
        "fusion_mode": "residual",
    },
    "temperature_scaling": {
        "granularity": "classwise",
        "calibration": "temperature",
        "safe_fallback": True,
        "fusion_mode": "residual",
    },
    "grid_reliability": {
        "granularity": "grid",
        "calibration": True,
        "safe_fallback": True,
        "fusion_mode": "residual",
    },
}


def run_wm_formal_evaluations(
    output_root: str | Path | None = None,
    *,
    variants: Sequence[str] = tuple(VARIANTS),
    semantic_source: str = SEMANTIC_SOURCE,
    formal_seeds: Sequence[int] | None = None,
    context: WMRunContext = V1_RUN_CONTEXT,
    v2_contract: WM811KDedupV2Contract | None = None,
) -> dict[str, list[dict[str, Any]]]:
    root = context.output_root if output_root is None else Path(output_root)
    seeds = normalize_formal_seeds(formal_seeds)
    caches = load_wm_formal_caches(
        root,
        semantic_source=semantic_source,
        formal_seeds=seeds,
        context=context,
        v2_contract=v2_contract,
    )
    unknown = sorted(set(variants).difference(VARIANTS))
    if unknown:
        raise ValueError(f"unknown WM reliability variants: {unknown}")
    all_results: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        setting = dict(VARIANTS[variant])
        if variant == "visual_anchor":
            # lambda=0 is the exact frozen anchor; no semantic selection is active.
            setting["lambda_value"] = 0.0
        else:
            setting["lambda_value"] = FROZEN_LAMBDA
            setting["rho"] = FROZEN_RHO
        results = [
            evaluate_wm_cache(
                cache,
                formal_seeds=seeds,
                context=context,
                v2_contract=v2_contract,
                **setting,
            )
            for cache in caches
        ]
        all_results[variant] = results
        raw_root = root / semantic_source / "raw" / variant
        for result in results:
            stem = f"8way_{result['shot']}shot_seed{result['seed']}"
            _write_json(raw_root / f"{stem}.json", result)
            rows = []
            records = (next(cache for cache in caches if cache.shot == result["shot"] and cache.seed == result["seed"]).runtime_metadata or {}).get("query_records", [])
            for index, prediction in enumerate(result["predictions"]):
                record = records[index] if index < len(records) else {}
                rows.append({
                    "original_index": record.get("original_index"),
                    "image_path": record.get("image_path"),
                    "true_index": record.get("true_index"),
                    "predicted_index": int(prediction),
                    "correct": bool(record.get("true_index") == int(prediction)),
                })
            if rows:
                _write_csv(raw_root / f"{stem}_predictions.csv", rows)
    return all_results


def summarize_wm_results(
    results_by_variant: dict[str, Sequence[dict[str, Any]]],
    output_root: str | Path | None = None,
    *,
    semantic_source: str = SEMANTIC_SOURCE,
    formal_seeds: Sequence[int] | None = None,
    context: WMRunContext = V1_RUN_CONTEXT,
) -> list[dict[str, Any]]:
    seeds = normalize_formal_seeds(formal_seeds)
    output = context.output_root if output_root is None else Path(output_root)
    root = output / semantic_source / "summaries"
    rows: list[dict[str, Any]] = []
    for variant, results in results_by_variant.items():
        for shot in EXPECTED_SHOTS:
            current = [result for result in results if int(result["shot"]) == shot]
            summary = summarize_seed_metrics(current, expected_seeds=seeds)
            metric = summary["metrics"]["macro_f1"]
            anchor_values = [float(result["anchor_metrics"]["macro_f1"]) for result in current]
            paired = paired_seed_statistics(
                [float(result["metrics"]["macro_f1"]) for result in current],
                [float(result["anchor_metrics"]["macro_f1"]) for result in current],
            )
            rows.append({
                "variant": variant,
                "shot": shot,
                "macro_f1_mean": metric["mean"],
                "macro_f1_std": metric["std"],
                "macro_f1_ci95_low": metric["ci95_low"],
                "macro_f1_ci95_high": metric["ci95_high"],
                "anchor_macro_f1_mean": float(np.mean(anchor_values)),
                "mean_improvement": float(metric["mean"] - np.mean(anchor_values)),
                "paired_delta_std": paired["std_delta"],
                "paired_delta_ci95_low": paired["ci95_low"],
                "paired_delta_ci95_high": paired["ci95_high"],
                "paired_p_value_two_sided": paired["p_value_two_sided"],
                "nondecline_seed_count": paired["nondecline_seed_count"],
                "improved_seed_count": paired["improved_seed_count"],
            })
            _write_json(root / f"{variant}_{shot}shot_summary.json", summary)
    _write_csv(root / "all_variants_summary.csv", rows)
    _write_json(root / "all_variants_summary.json", rows)
    return rows


def write_wm_report(
    results_by_variant: dict[str, Sequence[dict[str, Any]]],
    output_root: str | Path | None = None,
    *,
    semantic_source: str = SEMANTIC_SOURCE,
    formal_seeds: Sequence[int] | None = None,
    context: WMRunContext = V1_RUN_CONTEXT,
) -> dict[str, Any]:
    seeds = normalize_formal_seeds(formal_seeds)
    output = context.output_root if output_root is None else Path(output_root)
    root = output / semantic_source / "report"
    rows = summarize_wm_results(
        results_by_variant,
        output,
        semantic_source=semantic_source,
        formal_seeds=seeds,
        context=context,
    )
    payload = {
        "schema_version": "wm811k_class_reliability_report_v1",
        "protocol": context.protocol,
        "support_source": "Training",
        "query_source": "Test",
        "semantic_source": semantic_source,
        "semantic_version": (
            QWEN_SEMANTIC_VERSION
            if semantic_source == QWEN_SEMANTIC_SOURCE
            else SEMANTIC_VERSION
        ),
        "lambda": FROZEN_LAMBDA,
        "rho": FROZEN_RHO,
        "formal_seeds": list(seeds),
        "query_labels_used_for_selection": False,
        "variant_count": len(results_by_variant),
        "summary_rows": rows,
    }
    _write_json(root / "wm811k_test_audit.json", payload)
    return payload
