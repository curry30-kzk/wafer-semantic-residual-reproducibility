"""Orchestration and auditable artifact writing for the ResNet18 baseline."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .fewshot_dataset import FewShotTask, ImageRecord, load_fewshot_task
from .experiment_reporting import write_fewshot_summaries, write_run_artifacts
from .metrics import evaluate_predictions
from .prototype import build_class_prototypes, predict_with_prototypes
from .resnet18_features import (
    MODEL_IDENTIFIER,
    PREPROCESS_IDENTIFIER,
    extract_features,
    resnet18_preprocess_metadata,
)


FeatureExtractor = Callable[..., dict[str, np.ndarray]]


def discover_tasks(
    data_root: str | Path,
    *,
    shot: int = 5,
    seed: int = 0,
    run_all: bool = False,
    formal_seeds: Sequence[int] | None = None,
    query_split: str | None = None,
    n_way: int = 8,
    image_root: str | Path | None = None,
) -> list[FewShotTask]:
    """Load exactly one frozen task or the complete 15-task paper grid."""

    data_root = Path(data_root)
    selected_seeds = tuple(range(5) if formal_seeds is None else formal_seeds)
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("formal_seeds must be a non-empty sequence of unique seeds")
    pairs = (
        [(current_shot, current_seed) for current_shot in (1, 5, 10) for current_seed in selected_seeds]
        if run_all
        else [(shot, seed)]
    )
    if n_way <= 0:
        raise ValueError("n_way must be positive")
    split_root = data_root / "splits"
    if query_split is not None:
        normalized_split = str(query_split).strip()
        if not normalized_split:
            raise ValueError("query_split must not be blank")
        split_root = split_root / normalized_split

    tasks = []
    for current_shot, current_seed in pairs:
        manifest_path = (
            split_root
            / f"{n_way}way_{current_shot}shot_seed{current_seed}.json"
        )
        tasks.append(
            load_fewshot_task(
                manifest_path,
                data_root,
                image_root=image_root,
            )
        )
    return tasks


def select_debug_query_subset(
    records: Sequence[ImageRecord], *, limit: int, num_classes: int
) -> tuple[ImageRecord, ...]:
    """Select a deterministic round-robin subset so smoke tests include all classes."""

    if limit <= 0:
        raise ValueError("query limit must be positive")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if limit >= len(records):
        return tuple(records)

    by_class = [deque() for _ in range(num_classes)]
    for record in records:
        if record.label_index < 0 or record.label_index >= num_classes:
            raise ValueError("query record label index is outside the class range")
        by_class[record.label_index].append(record)

    selected: list[ImageRecord] = []
    while len(selected) < limit:
        added = False
        for class_records in by_class:
            if class_records and len(selected) < limit:
                selected.append(class_records.popleft())
                added = True
        if not added:
            break
    return tuple(selected)


def _validate_task_grid(tasks: Sequence[FewShotTask]) -> None:
    if not tasks:
        raise ValueError("at least one few-shot task is required")
    first = tasks[0]
    first_query = tuple(
        (record.relative_path, record.label_index, record.original_index)
        for record in first.query
    )
    for task in tasks[1:]:
        if task.class_names != first.class_names:
            raise ValueError("all tasks must use the same manifest class order")
        if task.protocol_name != first.protocol_name:
            raise ValueError("all tasks must use the same protocol name")
        current_query = tuple(
            (record.relative_path, record.label_index, record.original_index)
            for record in task.query
        )
        if current_query != first_query:
            raise ValueError("all tasks must use the identical fixed query set")


def _deduplicated_records(
    tasks: Sequence[FewShotTask], query_by_task: dict[tuple[int, int], tuple[ImageRecord, ...]]
) -> list[ImageRecord]:
    records: OrderedDict[str, ImageRecord] = OrderedDict()
    for task in tasks:
        for record in task.support + query_by_task[(task.shot, task.seed)]:
            existing = records.get(record.relative_path)
            if existing is not None and existing.label_index != record.label_index:
                raise ValueError(
                    f"image path has conflicting labels: {record.relative_path}"
                )
            records.setdefault(record.relative_path, record)
    return list(records.values())


def run_tasks(
    tasks: Sequence[FewShotTask],
    *,
    output_root: str | Path,
    batch_size: int,
    device: str,
    limit_query: int | None,
    aggregate: bool,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    feature_extractor: FeatureExtractor = extract_features,
    preprocess_identifier: str = PREPROCESS_IDENTIFIER,
    preprocess_metadata: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Extract deduplicated features, evaluate tasks, and write raw artifacts."""

    _validate_task_grid(tasks)
    output_root = Path(output_root)
    selected_preprocess = (
        resnet18_preprocess_metadata()
        if preprocess_metadata is None
        else dict(preprocess_metadata)
    )
    debug_run = limit_query is not None
    if aggregate and debug_run:
        raise ValueError("debug query limits cannot be used for formal aggregation")

    query_by_task: dict[tuple[int, int], tuple[ImageRecord, ...]] = {}
    for task in tasks:
        query_records = (
            select_debug_query_subset(
                task.query, limit=limit_query, num_classes=task.n_way
            )
            if limit_query is not None
            else task.query
        )
        query_by_task[(task.shot, task.seed)] = query_records

    unique_records = _deduplicated_records(tasks, query_by_task)
    feature_started = time.perf_counter()
    features_by_path = feature_extractor(
        unique_records,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
    )
    feature_seconds = time.perf_counter() - feature_started

    expected_paths = {record.relative_path for record in unique_records}
    if set(features_by_path) != expected_paths:
        missing = sorted(expected_paths - set(features_by_path))
        extra = sorted(set(features_by_path) - expected_paths)
        raise ValueError(f"feature extractor path mismatch; missing={missing}, extra={extra}")

    results: list[dict[str, Any]] = []
    for task in tasks:
        query_records = query_by_task[(task.shot, task.seed)]
        run_started = time.perf_counter()
        support_features = np.stack(
            [features_by_path[record.relative_path] for record in task.support]
        )
        support_labels = np.asarray(
            [record.label_index for record in task.support], dtype=np.int64
        )
        query_features = np.stack(
            [features_by_path[record.relative_path] for record in query_records]
        )
        y_true = np.asarray(
            [record.label_index for record in query_records], dtype=np.int64
        )
        prototypes = build_class_prototypes(
            support_features, support_labels, task.n_way
        )
        predictions, scores = predict_with_prototypes(query_features, prototypes)
        metrics = evaluate_predictions(y_true, predictions, task.class_names)
        classification_seconds = time.perf_counter() - run_started

        prediction_rows = []
        for row_index, record in enumerate(query_records):
            predicted_index = int(predictions[row_index])
            prediction_rows.append(
                {
                    "original_index": record.original_index,
                    "image_path": record.relative_path,
                    "true_index": record.label_index,
                    "true_label": record.label,
                    "predicted_index": predicted_index,
                    "predicted_label": task.class_names[predicted_index],
                    "cosine_score": float(scores[row_index, predicted_index]),
                    "correct": bool(predicted_index == record.label_index),
                }
            )

        result: dict[str, Any] = {
            "schema_version": "resnet18_prototype_result_v1",
            "method": "ResNet18 frozen visual prototype",
            "model_identifier": MODEL_IDENTIFIER,
            "preprocess_identifier": str(preprocess_identifier),
            "preprocess": selected_preprocess,
            "protocol": task.protocol_name,
            "support_source": task.support_source,
            "query_source": task.query_source,
            "manifest_path": task.manifest_path.as_posix(),
            "class_names": list(task.class_names),
            "n_way": task.n_way,
            "shot": task.shot,
            "seed": task.seed,
            "device": device,
            "batch_size": batch_size,
            "support_count": len(task.support),
            "query_count": len(query_records),
            "full_query_count": len(task.query),
            "debug_run": debug_run,
            "query_limit": limit_query,
            "timing_seconds": {
                "shared_feature_extraction": feature_seconds,
                "prototype_and_evaluation": classification_seconds,
            },
            "metrics": metrics,
        }
        stem = f"{result['n_way']}way_{result['shot']}shot_seed{result['seed']}"
        if result["debug_run"]:
            stem += f"_debug_q{result['query_count']}"
        write_run_artifacts(
            output_root, result, prediction_rows, stem=stem
        )
        results.append(result)

    if aggregate:
        write_fewshot_summaries(output_root, results, expected_seeds)
    return results
