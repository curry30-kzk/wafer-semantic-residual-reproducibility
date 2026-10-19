"""Develop classwise semantic reliability on FGVC-Aircraft Validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fgvc_aircraft.semantics import load_semantic_bundle
from wm811k.baseline_experiment import discover_tasks
from wm811k.clip_adaptation import build_stratified_support_folds
from wm811k.class_reliability_experiment import (
    TaskReliabilityCache,
    evaluate_cached_task,
    load_task_cache,
    run_reliability_cache_tasks,
)
from wm811k.clip_features import FrozenClipEncoder, MODEL_IDENTIFIER
from wm811k.resnet18_features import resolve_device


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "FGVC-Aircraft"
DEFAULT_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "FGVC-Aircraft"
    / "fgvc-aircraft-2013b"
    / "data"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fgvc_aircraft_validation"
    / "class_reliability_residual"
)
DEFAULT_LINEAR_BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fgvc_aircraft_validation"
    / "clip_linear_probe"
    / "raw"
)
SEMANTIC_SOURCES = (
    "variant_name",
    "hierarchy_deterministic",
    "qwen2_5_7b_instruct",
)
LAMBDA_GRID = (0.25, 0.5, 0.75, 1.0)
RHO_GRID = (0.01, 0.1, 1.0, 10.0)
SUPPORTED_SEEDS = tuple(range(10))
DEFAULT_FORMAL_SEEDS = tuple(range(5))


class ReliabilityArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        if parsed.command == "cache":
            if parsed.run_all and (
                parsed.shot is not None or parsed.seed is not None
            ):
                self.error("--all cannot be combined with --shot or --seed")
            if parsed.seeds is not None and len(set(parsed.seeds)) != len(parsed.seeds):
                self.error("--seeds must contain unique values")
            if parsed.seeds is not None and not parsed.run_all:
                self.error("--seeds requires --all")
            if parsed.run_all and parsed.limit_query is not None:
                self.error(
                    "--limit-query is debug-only and cannot be combined with --all"
                )
            if (
                parsed.reuse_visual_cache_from is not None
                and parsed.reuse_visual_cache_from == parsed.semantic_source
            ):
                self.error("visual cache reuse source must differ from semantic source")
            if parsed.shot is None:
                parsed.shot = 5
            if parsed.seed is None:
                parsed.seed = 0
            parsed.formal_seeds = (
                tuple(parsed.seeds)
                if parsed.seeds is not None
                else DEFAULT_FORMAL_SEEDS
            )
        elif parsed.command in {"develop", "ablate", "report", "freeze"}:
            if parsed.seeds is not None and len(set(parsed.seeds)) != len(parsed.seeds):
                self.error("--seeds must contain unique values")
            parsed.formal_seeds = (
                tuple(parsed.seeds)
                if parsed.seeds is not None
                else DEFAULT_FORMAL_SEEDS
            )
        return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-split", choices=("validation",), default="validation")
    parser.add_argument(
        "--semantic-source",
        choices=SEMANTIC_SOURCES,
        default="hierarchy_deterministic",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = ReliabilityArgumentParser(
        description="FGVC-Aircraft 类别级语义可靠性残差开发实验"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="构建冻结视觉和语义任务缓存")
    _add_common_arguments(cache)
    cache.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    cache.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    cache.add_argument("--shot", type=int, choices=(1, 5, 10), default=None)
    cache.add_argument("--seed", type=int, choices=SUPPORTED_SEEDS, default=None)
    cache.add_argument(
        "--seeds",
        type=int,
        choices=SUPPORTED_SEEDS,
        nargs="+",
        default=None,
        help="正式运行的随机种子列表；默认兼容历史 seed 0-4",
    )
    cache.add_argument("--all", dest="run_all", action="store_true")
    cache.add_argument("--batch-size", type=_positive_int, default=32)
    cache.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cache.add_argument("--limit-query", type=_positive_int, default=None)
    cache.add_argument("--no-progress", action="store_true")
    cache.add_argument("--support-augment-views", type=_positive_int, default=10)
    cache.add_argument("--tip-f-max-epochs", type=_positive_int, default=20)
    cache.add_argument(
        "--reuse-visual-cache-from",
        choices=SEMANTIC_SOURCES,
        default=None,
        help="复用另一语义源中逐元素相同的冻结视觉锚点分数",
    )
    cache.add_argument(
        "--reuse-baseline-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="复用已审计正式线性探针结果中的支持集所选 C",
    )
    cache.add_argument(
        "--linear-baseline-root",
        type=Path,
        default=DEFAULT_LINEAR_BASELINE_ROOT,
    )

    for command, help_text in (
        ("develop", "评价完整的 lambda/rho Validation 网格"),
        ("ablate", "运行可靠性粒度和安全机制消融"),
        ("report", "审计并汇总 Validation 产物"),
        ("freeze", "冻结 Test 前的方法配置和产物哈希"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_common_arguments(child)
        child.add_argument(
            "--seeds",
            type=int,
            choices=SUPPORTED_SEEDS,
            nargs="+",
            default=None,
            help="正式运行的随机种子列表；默认兼容历史 seed 0-4",
        )
        if command == "freeze":
            child.add_argument(
                "--config-output",
                type=Path,
                default=(
                    PROJECT_ROOT
                    / "configs"
                    / "fgvc_aircraft_class_reliability_frozen_v1.json"
                ),
            )
    return parser


def cache_directory(output_root: str | Path, semantic_source: str) -> Path:
    source = str(semantic_source)
    if source not in SEMANTIC_SOURCES:
        raise ValueError(f"unsupported semantic source: {source}")
    return Path(output_root) / source / "cache"


def formal_cache_paths(
    cache_root: str | Path,
    formal_seeds: Sequence[int] = DEFAULT_FORMAL_SEEDS,
) -> tuple[Path, ...]:
    root = Path(cache_root)
    return tuple(
        root / f"100way_{shot}shot_seed{seed}.json"
        for shot in (1, 5, 10)
        for seed in formal_seeds
    )


def development_configurations() -> tuple[tuple[float, float], ...]:
    return tuple(
        (float(lambda_value), float(rho))
        for lambda_value in LAMBDA_GRID
        for rho in RHO_GRID
    )


def summarize_development_results(
    config_results: Sequence[dict[str, Any]],
    formal_seeds: Sequence[int] = DEFAULT_FORMAL_SEEDS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in config_results:
        included = [
            result
            for result in config["task_results"]
            if int(result["shot"]) in (5, 10)
        ]
        keys = {(int(result["shot"]), int(result["seed"])) for result in included}
        expected = {(shot, seed) for shot in (5, 10) for seed in formal_seeds}
        if keys != expected or len(included) != len(expected):
            raise ValueError(
                "development summary requires 5/10-shot x the requested formal seeds"
            )
        macro_values = np.asarray(
            [float(result["metrics"]["macro_f1"]) for result in included],
            dtype=np.float64,
        )
        anchor_values = np.asarray(
            [float(result["anchor_metrics"]["macro_f1"]) for result in included],
            dtype=np.float64,
        )
        if not np.isfinite(macro_values).all() or not np.isfinite(anchor_values).all():
            raise ValueError("development metrics must be finite")
        differences = macro_values - anchor_values
        rows.append(
            {
                "lambda": float(config["lambda"]),
                "rho": float(config["rho"]),
                "shots": [5, 10],
                "task_count": len(included),
                "mean_macro_f1": float(macro_values.mean()),
                "mean_anchor_macro_f1": float(anchor_values.mean()),
                "mean_improvement": float(differences.mean()),
                "nondecline_task_count": int(np.count_nonzero(differences >= 0.0)),
                "improved_task_count": int(np.count_nonzero(differences > 0.0)),
            }
        )
    return rows


def select_global_config(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select best mean Macro-F1; tie-break by larger rho then smaller lambda."""

    candidates = list(rows)
    if not candidates:
        raise ValueError("configuration rows must not be empty")
    for row in candidates:
        for key in ("lambda", "rho", "mean_macro_f1"):
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f"configuration {key} must be finite")
    return min(
        candidates,
        key=lambda row: (
            -float(row["mean_macro_f1"]),
            -float(row["rho"]),
            float(row["lambda"]),
        ),
    )


def _semantic_bundle_path(data_root: Path, semantic_source: str) -> Path:
    filenames = {
        "variant_name": "variant_name_v1.json",
        "hierarchy_deterministic": "hierarchy_deterministic_v1.json",
        "qwen2_5_7b_instruct": "qwen2_5_7b_instruct_v1.json",
    }
    return data_root / "semantics" / filenames[semantic_source]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_formal_cache_grid(
    cache_root: str | Path,
    formal_seeds: Sequence[int] = DEFAULT_FORMAL_SEEDS,
) -> list[TaskReliabilityCache]:
    paths = formal_cache_paths(cache_root, formal_seeds)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(paths)} formal task caches are required; missing "
            f"{len(missing)} files, first={missing[0]}"
        )
    caches = [load_task_cache(path) for path in paths]
    expected_keys = {(shot, seed) for shot in (1, 5, 10) for seed in formal_seeds}
    actual_keys = {(int(cache.shot), int(cache.seed)) for cache in caches}
    if actual_keys != expected_keys or len(caches) != len(expected_keys):
        raise ValueError("formal task cache grid is incomplete or duplicated")
    protocols = {cache.protocol for cache in caches}
    class_orders = {cache.class_names for cache in caches}
    semantic_hashes = {cache.semantic_bundle_sha256 for cache in caches}
    semantic_sources = {
        str((cache.runtime_metadata or {}).get("semantic_source", ""))
        for cache in caches
    }
    if len(protocols) != 1 or len(class_orders) != 1:
        raise ValueError("formal caches must share one protocol and class order")
    if len(next(iter(class_orders))) != 100:
        raise ValueError("formal aircraft caches must contain exactly 100 classes")
    if len(semantic_hashes) != 1 or len(semantic_sources) != 1:
        raise ValueError("formal caches must share one frozen semantic bundle")
    for cache in caches:
        runtime = cache.runtime_metadata or {}
        if cache.support_source != "Training" or cache.query_source != "Validation":
            raise ValueError("formal caches must use Training support and Validation query")
        if bool(runtime.get("debug_run")):
            raise ValueError("debug caches cannot enter formal development")
        if bool(runtime.get("query_labels_used_for_selection", True)):
            raise ValueError("formal cache reports query-label selection leakage")
    return caches


def _source_root(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / args.semantic_source


def _configuration_id(lambda_value: float, rho: float) -> str:
    def token(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    return f"lambda_{token(lambda_value)}_rho_{token(rho)}"


def _load_audited_linear_selections(
    tasks: Sequence[Any],
    baseline_root: Path,
    *,
    support_augment_views: int,
) -> dict[tuple[int, int], dict[str, Any]]:
    selections: dict[tuple[int, int], dict[str, Any]] = {}
    for task in tasks:
        if int(task.shot) == 1:
            continue
        path = baseline_root / f"{task.n_way}way_{task.shot}shot_seed{task.seed}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"audited linear-probe baseline result does not exist: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "clip_linear_probe_result_v1":
            raise ValueError("linear baseline result schema is invalid")
        if payload.get("protocol") != task.protocol_name:
            raise ValueError("linear baseline protocol does not match task")
        if int(payload.get("shot", -1)) != task.shot or int(
            payload.get("seed", -1)
        ) != task.seed:
            raise ValueError("linear baseline shot/seed does not match task")
        if payload.get("query_source") != "Validation":
            raise ValueError("linear baseline selection must come from Validation run")
        if payload.get("query_labels_used_for_selection") is not False:
            raise ValueError("linear baseline reports query-label selection leakage")
        if tuple(payload.get("class_names", [])) != task.class_names:
            raise ValueError("linear baseline class order does not match task")
        if payload.get("model_identifier") != MODEL_IDENTIFIER:
            raise ValueError("linear baseline CLIP model identifier does not match")
        manifest_path = Path(str(payload.get("manifest_path", "")))
        if manifest_path.resolve() != task.manifest_path.resolve():
            raise ValueError("linear baseline manifest path does not match task")
        augmentation = payload.get("support_augmentation")
        if not isinstance(augmentation, dict) or int(
            augmentation.get("view_count", -1)
        ) != int(support_augment_views):
            raise ValueError("linear baseline support augmentation does not match")
        expected_seed = 20260816 + task.shot * 100 + task.seed
        if int(augmentation.get("seed", -1)) != expected_seed:
            raise ValueError("linear baseline augmentation seed does not match")
        selection = payload.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("linear baseline selection metadata is missing")
        support_labels = np.asarray(
            [record.label_index for record in task.support], dtype=np.int64
        )
        expected_folds = [
            fold.tolist()
            for fold in build_stratified_support_folds(
                support_labels,
                num_classes=task.n_way,
            )
        ]
        if selection.get("folds") != expected_folds:
            raise ValueError("linear baseline support folds do not match manifest")
        c_value = float(selection.get("C", float("nan")))
        if not np.isfinite(c_value) or c_value <= 0.0:
            raise ValueError("linear baseline selected C is invalid")
        selections[(task.shot, task.seed)] = {
            **selection,
            "baseline_result_path": path.resolve().as_posix(),
            "baseline_result_macro_f1": float(payload["metrics"]["macro_f1"]),
        }
    return selections


def _run_cache(args: argparse.Namespace) -> dict[str, Any]:
    tasks = discover_tasks(
        args.data_root,
        shot=args.shot,
        seed=args.seed,
        run_all=args.run_all,
        formal_seeds=args.formal_seeds,
        query_split="validation",
        n_way=100,
        image_root=args.image_root,
    )
    bundle_path = _semantic_bundle_path(args.data_root, args.semantic_source)
    bundle = load_semantic_bundle(bundle_path, tasks[0].class_names)
    reference_root = (
        None
        if args.reuse_visual_cache_from is None
        else cache_directory(args.output_root, args.reuse_visual_cache_from)
    )
    linear_selections = (
        _load_audited_linear_selections(
            tasks,
            args.linear_baseline_root,
            support_augment_views=args.support_augment_views,
        )
        if args.reuse_baseline_selection and reference_root is None
        else None
    )
    device = resolve_device(args.device)
    encoder = FrozenClipEncoder.create(device, dataset="fgvc_aircraft")
    paths = run_reliability_cache_tasks(
        tasks,
        encoder=encoder,
        semantic_prompt_groups=bundle.prompts_by_class,
        semantic_source=bundle.source,
        semantic_bundle_sha256=_sha256_path(bundle_path),
        output_root=cache_directory(args.output_root, args.semantic_source),
        batch_size=args.batch_size,
        limit_query=args.limit_query,
        aggregate=args.run_all,
        formal_seeds=args.formal_seeds,
        show_progress=not args.no_progress,
        support_augment_views=args.support_augment_views,
        tip_f_max_epochs=args.tip_f_max_epochs,
        reference_visual_cache_root=(
            reference_root
        ),
        linear_selection_by_task=linear_selections,
    )
    return {
        "command": "cache",
        "semantic_source": args.semantic_source,
        "query_split": "validation",
        "formal": bool(args.run_all),
        "debug_run": args.limit_query is not None,
        "cache_count": len(paths),
        "visual_anchor_reused_from": args.reuse_visual_cache_from,
        "baseline_selection_reused": bool(linear_selections),
        "cache_files": [path.as_posix() for path in paths],
    }


def _evaluate_grid(caches: Sequence[TaskReliabilityCache]) -> list[dict[str, Any]]:
    config_results: list[dict[str, Any]] = []
    for lambda_value, rho in development_configurations():
        task_results = [
            evaluate_cached_task(
                cache,
                lambda_value=lambda_value,
                rho=rho,
                granularity="classwise",
                calibration=True,
                safe_fallback=True,
            )
            for cache in caches
        ]
        config_results.append(
            {
                "lambda": lambda_value,
                "rho": rho,
                "task_results": task_results,
            }
        )
    return config_results


def _run_develop(args: argparse.Namespace) -> dict[str, Any]:
    caches = load_formal_cache_grid(
        cache_directory(args.output_root, args.semantic_source),
        args.formal_seeds,
    )
    config_results = _evaluate_grid(caches)
    rows = summarize_development_results(config_results, args.formal_seeds)
    selected = select_global_config(rows)
    output = _source_root(args) / "development"
    _write_json_atomic(output / "all_config_results.json", config_results)
    _write_json_atomic(output / "development_grid.json", rows)
    _write_csv_atomic(output / "development_grid.csv", rows)
    _write_json_atomic(
        output / "selected_config.json",
        {
            "schema_version": "class_reliability_selected_config_v1",
            "selection_split": "Validation",
            "selection_tasks": (
                "5-shot and 10-shot, seeds "
                + ",".join(str(seed) for seed in args.formal_seeds)
            ),
            "selection_metric": (
                "mean Macro-F1 over "
                + str(2 * len(args.formal_seeds))
                + " tasks"
            ),
            "tie_break": "larger rho, then smaller lambda",
            "semantic_source": args.semantic_source,
            "selected": selected,
            "all_configuration_count": len(rows),
        },
    )
    return {
        "command": "develop",
        "semantic_source": args.semantic_source,
        "configuration_count": len(rows),
        "selected": selected,
        "output_directory": output.as_posix(),
    }


def _load_selected_config(args: argparse.Namespace) -> dict[str, Any]:
    path = _source_root(args) / "development" / "selected_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"selected Validation config does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("selected config payload is invalid")
    return selected


def _summarize_ablation(
    variants: dict[str, list[dict[str, Any]]],
    formal_seeds: Sequence[int] = DEFAULT_FORMAL_SEEDS,
) -> list[dict[str, Any]]:
    rows = []
    for variant, results in variants.items():
        for shot in (1, 5, 10):
            current = [result for result in results if int(result["shot"]) == shot]
            if {int(result["seed"]) for result in current} != set(formal_seeds):
                raise ValueError("each ablation needs the requested formal seeds per shot")
            values = np.asarray(
                [float(result["metrics"]["macro_f1"]) for result in current]
            )
            anchors = np.asarray(
                [float(result["anchor_metrics"]["macro_f1"]) for result in current]
            )
            differences = values - anchors
            rows.append(
                {
                    "variant": variant,
                    "shot": shot,
                    "mean_macro_f1": float(values.mean()),
                    "mean_anchor_macro_f1": float(anchors.mean()),
                    "mean_improvement": float(differences.mean()),
                    "nondecline_seed_count": int(np.count_nonzero(differences >= 0.0)),
                    "improved_seed_count": int(np.count_nonzero(differences > 0.0)),
                    "enabled_seed_count": int(
                        sum(bool(result["selection"].get("enabled")) for result in current)
                    ),
                }
            )
    return rows


def _run_ablate(args: argparse.Namespace) -> dict[str, Any]:
    caches = load_formal_cache_grid(
        cache_directory(args.output_root, args.semantic_source),
        args.formal_seeds,
    )
    selected = _load_selected_config(args)
    lambda_value = float(selected["lambda"])
    rho = float(selected["rho"])
    settings = {
        "visual_anchor": {
            "lambda_value": 0.0,
            "rho": rho,
            "granularity": "classwise",
            "calibration": True,
            "safe_fallback": True,
        },
        "fixed_all_one": {
            "lambda_value": lambda_value,
            "rho": rho,
            "granularity": "fixed",
            "calibration": True,
            "safe_fallback": True,
        },
        "global_reliability": {
            "lambda_value": lambda_value,
            "rho": rho,
            "granularity": "global",
            "calibration": True,
            "safe_fallback": True,
        },
        "classwise_reliability": {
            "lambda_value": lambda_value,
            "rho": rho,
            "granularity": "classwise",
            "calibration": True,
            "safe_fallback": True,
        },
        "without_calibration": {
            "lambda_value": lambda_value,
            "rho": rho,
            "granularity": "classwise",
            "calibration": False,
            "safe_fallback": True,
        },
        "without_safe_fallback": {
            "lambda_value": lambda_value,
            "rho": rho,
            "granularity": "classwise",
            "calibration": True,
            "safe_fallback": False,
        },
    }
    variants = {
        name: [evaluate_cached_task(cache, **setting) for cache in caches]
        for name, setting in settings.items()
    }
    rows = _summarize_ablation(variants, args.formal_seeds)
    output = _source_root(args) / "ablation"
    _write_json_atomic(
        output / "ablation_results.json",
        {
            "selected_config": selected,
            "settings": settings,
            "variants": variants,
        },
    )
    _write_json_atomic(output / "ablation_summary.json", rows)
    _write_csv_atomic(output / "ablation_summary.csv", rows)
    return {
        "command": "ablate",
        "semantic_source": args.semantic_source,
        "variant_count": len(variants),
        "selected": selected,
        "output_directory": output.as_posix(),
    }


def _run_report(args: argparse.Namespace) -> dict[str, Any]:
    caches = load_formal_cache_grid(
        cache_directory(args.output_root, args.semantic_source),
        args.formal_seeds,
    )
    source_root = _source_root(args)
    grid_path = source_root / "development" / "development_grid.json"
    selected_path = source_root / "development" / "selected_config.json"
    ablation_path = source_root / "ablation" / "ablation_results.json"
    for path in (grid_path, selected_path, ablation_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Validation artifact is missing: {path}")
    rows = json.loads(grid_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("development grid must preserve all 16 configurations")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected"]
    audit = {
        "schema_version": "class_reliability_validation_audit_v1",
        "semantic_source": args.semantic_source,
        "query_split": "Validation",
        "test_run": False,
        "cache_count": len(caches),
        "class_count": len(caches[0].class_names),
        "configuration_count": len(rows),
        "selected": selected,
        "semantic_bundle_sha256": caches[0].semantic_bundle_sha256,
        "query_labels_used_for_selection": False,
        "all_cache_query_sources": sorted({cache.query_source for cache in caches}),
    }
    output = source_root / "report" / "validation_audit.json"
    _write_json_atomic(output, audit)
    return {"command": "report", **audit, "output": output.as_posix()}


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _run_freeze(args: argparse.Namespace) -> dict[str, Any]:
    if args.semantic_source != "qwen2_5_7b_instruct":
        raise ValueError("final Test freeze requires the frozen Qwen semantic source")
    if _git_output("status", "--porcelain"):
        raise ValueError("git working tree must be clean before freezing Test config")
    report = _run_report(args)
    source_root = _source_root(args)
    selected_path = source_root / "development" / "selected_config.json"
    ablation_path = source_root / "ablation" / "ablation_results.json"
    cache_grid = load_formal_cache_grid(
        cache_directory(args.output_root, args.semantic_source),
        args.formal_seeds,
    )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected"]
    payload = {
        "schema_version": "fgvc_aircraft_class_reliability_frozen_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "protocol": cache_grid[0].protocol,
        "class_count": len(cache_grid[0].class_names),
        "development_split": "Validation",
        "intended_query_split": "Test",
        "test_run_before_freeze": False,
        "semantic_source": args.semantic_source,
        "semantic_bundle_sha256": cache_grid[0].semantic_bundle_sha256,
        "selected_lambda": float(selected["lambda"]),
        "selected_rho": float(selected["rho"]),
        "anchor_policy": {
            "1-shot": "Tip-Adapter-F exact fallback",
            "5-shot": "CLIP linear probe",
            "10-shot": "CLIP linear probe",
        },
        "optimization": {
            "dtype": "CPU float64",
            "optimizer": "Adam",
            "steps": 200,
            "learning_rate": 0.05,
            "constraint": "project to [0,1] after every step",
        },
        "calibration": "support-only class bias, row standardization, visual scale match",
        "safe_fallback": "enable only after strict support OOF Macro-F1 improvement",
        "selected_config_sha256": _sha256_path(selected_path),
        "ablation_results_sha256": _sha256_path(ablation_path),
        "validation_audit": report,
    }
    _write_json_atomic(args.config_output, payload)
    return {
        "command": "freeze",
        "config_output": args.config_output.as_posix(),
        "git_commit": payload["git_commit"],
        "test_run_before_freeze": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "cache": _run_cache,
        "develop": _run_develop,
        "ablate": _run_ablate,
        "report": _run_report,
        "freeze": _run_freeze,
    }
    result = handlers[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
