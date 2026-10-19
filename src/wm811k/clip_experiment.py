"""Paper-auditable CLIP evaluation on frozen WM-811K tasks."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_experiment import select_debug_query_subset
from .clip_features import FrozenClipEncoder, PROMPT_TEMPLATE
from .domain_descriptions import (
    HUMAN_GEOMETRY_PROMPT_VERSION,
    build_human_geometry_prompts,
)
from .experiment_reporting import (
    task_result_provenance,
    write_fewshot_summaries,
    write_run_artifacts,
    write_zero_shot_summary,
)
from .fewshot_dataset import FewShotTask, ImageRecord
from .labels import DEFECT_CLASSES
from .llm_descriptions import (
    LLM_STRUCTURED_PROMPT_PROVENANCE,
    LLM_STRUCTURED_PROMPT_SOURCE,
    LLM_STRUCTURED_PROMPT_VERSION,
    build_llm_structured_prompt_ensemble,
)
from .metrics import evaluate_predictions
from .prototype import build_class_prototypes, predict_with_prototypes
from .protocols import WM811K_PROTOCOL_NAME, get_protocol_spec
from .reliability_fusion import (
    DEFAULT_TEXT_WEIGHT_GRID,
    aggregate_prompt_embeddings,
    calibrate_text_scores,
    fuse_normalized_scores,
    select_text_weight_leave_one_out,
)
def validate_task_grid(
    tasks: Sequence[FewShotTask],
    *,
    require_formal_grid: bool = False,
    formal_seeds: Sequence[int] | None = None,
) -> None:
    """Require one protocol, class order, image size, and fixed query grid."""

    if not tasks:
        raise ValueError("at least one few-shot task is required")
    try:
        spec = get_protocol_spec(tasks[0].protocol_name)
    except ValueError as exc:
        raise ValueError("tasks must use a registered official protocol") from exc
    allowed_seeds = tuple(spec.seeds if formal_seeds is None else formal_seeds)
    if not allowed_seeds or len(set(allowed_seeds)) != len(allowed_seeds):
        raise ValueError("formal_seeds must be a non-empty sequence of unique seeds")
    allowed_keys = frozenset(
        (shot, seed) for shot in spec.shots for seed in allowed_seeds
    )
    task_keys: list[tuple[int, int]] = []
    for task in tasks:
        if task.protocol_name != spec.name:
            raise ValueError(
                f"tasks must use the same official protocol {spec.name!r}"
            )
        if task.n_way != spec.n_way:
            raise ValueError(
                f"tasks must use the official {spec.n_way}-class protocol"
            )
        if (
            spec.name == WM811K_PROTOCOL_NAME
            and task.class_names != DEFECT_CLASSES
        ):
            raise ValueError("tasks must use the official 8-class WM-811K order")
        if task.image_size != spec.image_size:
            raise ValueError(
                f"tasks must use prepared {spec.image_size}x{spec.image_size} images"
            )
        key = (int(task.shot), int(task.seed))
        if key not in allowed_keys:
            raise ValueError(
                f"task shot/seed must be in {spec.shots} x {allowed_seeds}"
            )
        expected_name = spec.manifest_name(task.shot, task.seed)
        if task.manifest_path.name != expected_name:
            raise ValueError(
                f"manifest filename must match task metadata: {expected_name}"
            )
        task_keys.append(key)
    if len(set(task_keys)) != len(task_keys):
        raise ValueError("task grid contains a duplicate shot/seed pair")
    if require_formal_grid:
        expected_count = len(allowed_keys)
        if len(tasks) != expected_count:
            raise ValueError(
                f"formal task grid must contain exactly {expected_count} tasks"
            )
        if set(task_keys) != allowed_keys:
            raise ValueError(
                "formal task grid does not cover the declared shot/seed grid exactly"
            )
    first = tasks[0]
    first_query = tuple(
        (record.relative_path, record.label_index, record.original_index)
        for record in first.query
    )
    for task in tasks[1:]:
        if task.protocol_name != first.protocol_name:
            raise ValueError("all tasks must use the same protocol name")
        if task.class_names != first.class_names:
            raise ValueError("all tasks must use the same manifest class order")
        if task.image_size != first.image_size:
            raise ValueError("all tasks must use the same prepared image size")
        current_query = tuple(
            (record.relative_path, record.label_index, record.original_index)
            for record in task.query
        )
        if current_query != first_query:
            raise ValueError("all tasks must use the identical fixed query set")


def _validate_feature_paths(
    records: Sequence[ImageRecord], features_by_path: dict[str, np.ndarray]
) -> None:
    expected = {record.relative_path for record in records}
    actual = set(features_by_path)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"CLIP feature path mismatch; missing={missing}, extra={extra}"
        )


def _prediction_rows(
    records: Sequence[ImageRecord],
    class_names: Sequence[str],
    predictions: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for row_index, record in enumerate(records):
        predicted_index = int(predictions[row_index])
        rows.append(
            {
                "original_index": record.original_index,
                "image_path": record.relative_path,
                "true_index": record.label_index,
                "true_label": record.label,
                "predicted_index": predicted_index,
                "predicted_label": class_names[predicted_index],
                "cosine_score": float(scores[row_index, predicted_index]),
                "correct": bool(predicted_index == record.label_index),
            }
        )
    return rows


def _fusion_prediction_rows(
    records: Sequence[ImageRecord],
    class_names: Sequence[str],
    predictions: np.ndarray,
    visual_scores: np.ndarray,
    text_scores: np.ndarray,
    fused_scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows = _prediction_rows(records, class_names, predictions, fused_scores)
    for row_index, row in enumerate(rows):
        predicted_index = int(predictions[row_index])
        row.update(
            {
                "visual_score": float(visual_scores[row_index, predicted_index]),
                "text_score": float(text_scores[row_index, predicted_index]),
                "fused_score": float(fused_scores[row_index, predicted_index]),
            }
        )
    return rows


def _deduplicated_visual_records(
    tasks: Sequence[FewShotTask],
    query_by_task: dict[tuple[int, int], tuple[ImageRecord, ...]],
) -> list[ImageRecord]:
    records: OrderedDict[str, ImageRecord] = OrderedDict()
    for task in tasks:
        task_records = task.support + query_by_task[(task.shot, task.seed)]
        for record in task_records:
            existing = records.get(record.relative_path)
            if existing is not None and existing.label_index != record.label_index:
                raise ValueError(
                    f"image path has conflicting labels: {record.relative_path}"
                )
            records.setdefault(record.relative_path, record)
    return list(records.values())


def run_zero_shot_tasks(
    tasks: Sequence[FewShotTask],
    *,
    encoder: FrozenClipEncoder,
    output_root: str | Path,
    batch_size: int,
    limit_query: int | None,
    formal: bool,
    formal_seeds: Sequence[int] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Validate the task grid and evaluate its fixed query exactly once."""

    if formal and limit_query is not None:
        raise ValueError("formal Zero-shot evaluation cannot use a debug query limit")
    validate_task_grid(
        tasks, require_formal_grid=formal, formal_seeds=formal_seeds
    )
    task = tasks[0]
    output_root = Path(output_root)
    debug_run = limit_query is not None
    paper_eligible = bool(formal and not debug_run)
    query_records = (
        select_debug_query_subset(
            task.query, limit=limit_query, num_classes=task.n_way
        )
        if limit_query is not None
        else task.query
    )

    feature_started = time.perf_counter()
    features_by_path = encoder.encode_images(
        query_records,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    feature_seconds = time.perf_counter() - feature_started
    _validate_feature_paths(query_records, features_by_path)
    query_features = np.stack(
        [features_by_path[record.relative_path] for record in query_records]
    )

    classification_started = time.perf_counter()
    text_prototypes, prompts = encoder.encode_class_text_prototypes(task.class_names)
    if text_prototypes.shape[0] != task.n_way:
        raise ValueError("CLIP text prototype count must match manifest classes")
    predictions, scores = predict_with_prototypes(query_features, text_prototypes)
    y_true = np.asarray(
        [record.label_index for record in query_records], dtype=np.int64
    )
    metrics = evaluate_predictions(y_true, predictions, task.class_names)
    classification_seconds = time.perf_counter() - classification_started

    runtime_metadata = encoder.runtime_metadata(
        feature_dimension=int(query_features.shape[1])
    )
    result: dict[str, Any] = {
        "schema_version": "clip_zero_shot_result_v1",
        "method": "OpenAI CLIP ViT-B/32 zero-shot",
        **runtime_metadata,
        "prompt_template": PROMPT_TEMPLATE,
        "prompts": list(prompts),
        "similarity": "cosine",
        "protocol": task.protocol_name,
        **task_result_provenance(task),
        "support_source": task.support_source,
        "query_source": task.query_source,
        "validated_manifest_count": len(tasks),
        "validated_manifest_paths": [
            current.manifest_path.as_posix() for current in tasks
        ],
        "context_manifest_path": task.manifest_path.as_posix(),
        "class_names": list(task.class_names),
        "n_way": task.n_way,
        "shot": None,
        "seed": None,
        "context_shot": task.shot,
        "context_seed": task.seed,
        "device": encoder.device,
        "batch_size": batch_size,
        "support_count": 0,
        "support_used": False,
        "support_dependent": False,
        "query_count": len(query_records),
        "full_query_count": len(task.query),
        "debug_run": debug_run,
        "paper_eligible": paper_eligible,
        "formal_grid_validated": bool(formal),
        "query_limit": limit_query,
        "deterministic": True,
        "timing_seconds": {
            "image_feature_extraction": feature_seconds,
            "text_encoding_and_evaluation": classification_seconds,
        },
        "metrics": metrics,
    }
    stem = f"{task.n_way}way_zero_shot"
    if not paper_eligible:
        stem += f"_context_{task.shot}shot_seed{task.seed}"
    if debug_run:
        stem += f"_debug_q{len(query_records)}"
    write_run_artifacts(
        output_root,
        result,
        _prediction_rows(query_records, task.class_names, predictions, scores),
        stem=stem,
    )
    if paper_eligible:
        write_zero_shot_summary(output_root, result)
    return result


def run_visual_prototype_tasks(
    tasks: Sequence[FewShotTask],
    *,
    encoder: FrozenClipEncoder,
    output_root: str | Path,
    batch_size: int,
    limit_query: int | None,
    aggregate: bool,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate K-shot visual prototypes from one deduplicated CLIP pass."""

    validate_task_grid(
        tasks, require_formal_grid=aggregate, formal_seeds=expected_seeds
    )
    output_root = Path(output_root)
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

    unique_records = _deduplicated_visual_records(tasks, query_by_task)
    feature_started = time.perf_counter()
    features_by_path = encoder.encode_images(
        unique_records,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    feature_seconds = time.perf_counter() - feature_started
    _validate_feature_paths(unique_records, features_by_path)
    first_feature = next(iter(features_by_path.values()))
    if np.asarray(first_feature).ndim != 1:
        raise ValueError("each CLIP image feature must be a 1D vector")
    runtime_metadata = encoder.runtime_metadata(
        feature_dimension=int(np.asarray(first_feature).shape[0])
    )

    results: list[dict[str, Any]] = []
    for task in tasks:
        query_records = query_by_task[(task.shot, task.seed)]
        classification_started = time.perf_counter()
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
        classification_seconds = time.perf_counter() - classification_started

        result: dict[str, Any] = {
            "schema_version": "clip_visual_prototype_result_v1",
            "method": "OpenAI CLIP ViT-B/32 frozen visual prototype",
            **runtime_metadata,
            "similarity": "cosine",
            "protocol": task.protocol_name,
            **task_result_provenance(task),
            "support_source": task.support_source,
            "query_source": task.query_source,
            "manifest_path": task.manifest_path.as_posix(),
            "class_names": list(task.class_names),
            "n_way": task.n_way,
            "shot": task.shot,
            "seed": task.seed,
            "device": encoder.device,
            "batch_size": batch_size,
            "support_count": len(task.support),
            "support_used": True,
            "support_dependent": True,
            "query_count": len(query_records),
            "full_query_count": len(task.query),
            "debug_run": debug_run,
            "formal_grid_validated": bool(aggregate),
            "query_limit": limit_query,
            "timing_seconds": {
                "shared_image_feature_extraction": feature_seconds,
                "prototype_and_evaluation": classification_seconds,
            },
            "metrics": metrics,
        }
        stem = f"{task.n_way}way_{task.shot}shot_seed{task.seed}"
        if debug_run:
            stem += f"_debug_q{len(query_records)}"
        write_run_artifacts(
            output_root,
            result,
            _prediction_rows(query_records, task.class_names, predictions, scores),
            stem=stem,
        )
        results.append(result)

    if aggregate:
        write_fewshot_summaries(output_root, results, expected_seeds)
    return results


def _run_score_fusion_tasks(
    tasks: Sequence[FewShotTask],
    *,
    encoder: FrozenClipEncoder,
    text_encoder: Callable[[], tuple[np.ndarray, Sequence[str]]],
    schema_version: str,
    method: str,
    text_metadata: dict[str, Any],
    prompts_per_class: int = 1,
    output_root: str | Path,
    batch_size: int,
    limit_query: int | None,
    aggregate: bool,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Run one fixed visual-text score fusion with an immutable text source."""

    validate_task_grid(
        tasks, require_formal_grid=aggregate, formal_seeds=expected_seeds
    )
    output_root = Path(output_root)
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

    unique_records = _deduplicated_visual_records(tasks, query_by_task)
    feature_started = time.perf_counter()
    features_by_path = encoder.encode_images(
        unique_records,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    feature_seconds = time.perf_counter() - feature_started
    _validate_feature_paths(unique_records, features_by_path)
    first_feature = np.asarray(next(iter(features_by_path.values())))
    if first_feature.ndim != 1:
        raise ValueError("each CLIP image feature must be a 1D vector")

    text_started = time.perf_counter()
    text_prototypes, prompts = text_encoder()
    text_seconds = time.perf_counter() - text_started
    text_prototypes = np.asarray(text_prototypes)
    if text_prototypes.ndim != 2:
        raise ValueError("CLIP text prototypes must be a 2D feature matrix")
    if text_prototypes.shape[0] != tasks[0].n_way:
        raise ValueError("CLIP text prototype count must match manifest classes")
    if prompts_per_class <= 0:
        raise ValueError("prompts_per_class must be positive")
    if len(prompts) != tasks[0].n_way * prompts_per_class:
        raise ValueError("CLIP prompt count must match manifest classes")
    if text_prototypes.shape[1] != first_feature.shape[0]:
        raise ValueError(
            "CLIP text feature dimension must match image feature dimension"
        )
    runtime_metadata = encoder.runtime_metadata(
        feature_dimension=int(first_feature.shape[0])
    )

    results: list[dict[str, Any]] = []
    for task in tasks:
        query_records = query_by_task[(task.shot, task.seed)]
        classification_started = time.perf_counter()
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
        visual_prototypes = build_class_prototypes(
            support_features, support_labels, task.n_way
        )
        _, visual_scores = predict_with_prototypes(
            query_features, visual_prototypes
        )
        _, text_scores = predict_with_prototypes(query_features, text_prototypes)
        if visual_scores.shape != text_scores.shape:
            raise ValueError("visual and text score matrices must have identical shapes")
        fused_scores = 0.5 * visual_scores + 0.5 * text_scores
        predictions = np.argmax(fused_scores, axis=1).astype(np.int64)
        metrics = evaluate_predictions(y_true, predictions, task.class_names)
        classification_seconds = time.perf_counter() - classification_started

        result: dict[str, Any] = {
            "schema_version": schema_version,
            "method": method,
            **runtime_metadata,
            **text_metadata,
            "prompts": list(prompts),
            "prompts_per_class": prompts_per_class,
            "similarity": "fixed weighted sum of cosine scores",
            "score_field_note": (
                "cosine_score stores the final fused score for reporting compatibility"
            ),
            "fusion_level": "score",
            "visual_weight": 0.5,
            "text_weight": 0.5,
            "weight_selection": "fixed a priori; no test tuning",
            "protocol": task.protocol_name,
            **task_result_provenance(task),
            "support_source": task.support_source,
            "query_source": task.query_source,
            "manifest_path": task.manifest_path.as_posix(),
            "class_names": list(task.class_names),
            "n_way": task.n_way,
            "shot": task.shot,
            "seed": task.seed,
            "device": encoder.device,
            "batch_size": batch_size,
            "support_count": len(task.support),
            "support_used": True,
            "support_dependent": True,
            "query_count": len(query_records),
            "full_query_count": len(task.query),
            "debug_run": debug_run,
            "formal_grid_validated": bool(aggregate),
            "query_limit": limit_query,
            "timing_seconds": {
                "shared_image_feature_extraction": feature_seconds,
                "shared_text_prototype_encoding": text_seconds,
                "fusion_and_evaluation": classification_seconds,
            },
            "metrics": metrics,
        }
        stem = f"{task.n_way}way_{task.shot}shot_seed{task.seed}"
        if debug_run:
            stem += f"_debug_q{len(query_records)}"
        write_run_artifacts(
            output_root,
            result,
            _fusion_prediction_rows(
                query_records,
                task.class_names,
                predictions,
                visual_scores,
                text_scores,
                fused_scores,
            ),
            stem=stem,
            extra_prediction_fields=(
                "visual_score",
                "text_score",
                "fused_score",
            ),
        )
        results.append(result)

    if aggregate:
        write_fewshot_summaries(output_root, results, expected_seeds)
    return results


def run_fixed_fusion_tasks(
    tasks: Sequence[FewShotTask],
    *,
    encoder: FrozenClipEncoder,
    output_root: str | Path,
    batch_size: int,
    limit_query: int | None,
    aggregate: bool,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Fuse frozen visual and generic class-name scores at fixed equal weight."""

    return _run_score_fusion_tasks(
        tasks,
        encoder=encoder,
        text_encoder=lambda: encoder.encode_class_text_prototypes(
            tasks[0].class_names
        ),
        schema_version="clip_fixed_fusion_result_v1",
        method="OpenAI CLIP ViT-B/32 fixed visual-text score fusion",
        text_metadata={"prompt_template": PROMPT_TEMPLATE},
        output_root=output_root,
        batch_size=batch_size,
        limit_query=limit_query,
        aggregate=aggregate,
        expected_seeds=expected_seeds,
        show_progress=show_progress,
    )


def run_domain_geometry_fusion_tasks(
    tasks: Sequence[FewShotTask],
    *,
    encoder: FrozenClipEncoder,
    output_root: str | Path,
    batch_size: int,
    limit_query: int | None,
    aggregate: bool,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Fuse visual scores with frozen human wafer-geometry text scores."""

    def encode_domain_text() -> tuple[np.ndarray, Sequence[str]]:
        prompts = build_human_geometry_prompts(tasks[0].class_names)
        return encoder.encode_text_prototypes(prompts), prompts

    return _run_score_fusion_tasks(
        tasks,
        encoder=encoder,
        text_encoder=encode_domain_text,
        schema_version="clip_domain_geometry_fusion_result_v1",
        method=(
            "OpenAI CLIP ViT-B/32 fixed human-geometry visual-text score fusion"
        ),
        text_metadata={
            "text_prompt_source": "human_written_domain_geometry",
            "text_prompt_version": HUMAN_GEOMETRY_PROMPT_VERSION,
            "description_selection": (
                "fixed a priori from class geometry definitions; no test tuning"
            ),
        },
        output_root=output_root,
        batch_size=batch_size,
        limit_query=limit_query,
        aggregate=aggregate,
        expected_seeds=expected_seeds,
        show_progress=show_progress,
    )


def _encode_llm_structured_text(
    encoder: FrozenClipEncoder, class_names: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    ensemble = build_llm_structured_prompt_ensemble(class_names)
    flattened = tuple(prompt for class_prompts in ensemble for prompt in class_prompts)
    prompt_embeddings = encoder.encode_text_prototypes(flattened)
    text_prototypes = aggregate_prompt_embeddings(
        prompt_embeddings, prompts_per_class=4
    )
    return text_prototypes, flattened


def _llm_text_metadata() -> dict[str, Any]:
    return {
        "text_prompt_source": LLM_STRUCTURED_PROMPT_SOURCE,
        "text_prompt_version": LLM_STRUCTURED_PROMPT_VERSION,
        "text_prompt_provenance": LLM_STRUCTURED_PROMPT_PROVENANCE,
        "description_selection": (
            "generated once and frozen before formal evaluation; no Test tuning"
        ),
        "prompt_attributes": [
            "location",
            "shape",
            "coverage_or_distribution",
            "distinguishing_relation",
        ],
        "prompt_aggregation": (
            "mean of four individually L2-normalized embeddings per class, "
            "followed by class-prototype L2 normalization"
        ),
    }


def run_llm_structured_fixed_fusion_tasks(
    tasks: Sequence[FewShotTask],
    *,
    encoder: FrozenClipEncoder,
    output_root: str | Path,
    batch_size: int,
    limit_query: int | None,
    aggregate: bool,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Ablate structured LLM semantics under the prior fixed 0.5 fusion."""

    return _run_score_fusion_tasks(
        tasks,
        encoder=encoder,
        text_encoder=lambda: _encode_llm_structured_text(
            encoder, tasks[0].class_names
        ),
        schema_version="clip_llm_structured_fixed_fusion_result_v1",
        method=(
            "OpenAI CLIP ViT-B/32 fixed structured-LLM visual-text score fusion"
        ),
        text_metadata=_llm_text_metadata(),
        prompts_per_class=4,
        output_root=output_root,
        batch_size=batch_size,
        limit_query=limit_query,
        aggregate=aggregate,
        expected_seeds=expected_seeds,
        show_progress=show_progress,
    )


def run_llm_reliability_fusion_tasks(
    tasks: Sequence[FewShotTask],
    *,
    encoder: FrozenClipEncoder,
    output_root: str | Path,
    batch_size: int,
    limit_query: int | None,
    aggregate: bool,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Fuse structured text only when support leave-one-out validates it."""

    validate_task_grid(
        tasks, require_formal_grid=aggregate, formal_seeds=expected_seeds
    )
    output_root = Path(output_root)
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

    unique_records = _deduplicated_visual_records(tasks, query_by_task)
    feature_started = time.perf_counter()
    features_by_path = encoder.encode_images(
        unique_records,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    feature_seconds = time.perf_counter() - feature_started
    _validate_feature_paths(unique_records, features_by_path)
    first_feature = np.asarray(next(iter(features_by_path.values())))
    if first_feature.ndim != 1:
        raise ValueError("each CLIP image feature must be a 1D vector")

    text_started = time.perf_counter()
    text_prototypes, prompts = _encode_llm_structured_text(
        encoder, tasks[0].class_names
    )
    text_seconds = time.perf_counter() - text_started
    if text_prototypes.shape != (tasks[0].n_way, first_feature.shape[0]):
        raise ValueError(
            "structured text prototypes must match class and feature dimensions"
        )
    runtime_metadata = encoder.runtime_metadata(
        feature_dimension=int(first_feature.shape[0])
    )

    results: list[dict[str, Any]] = []
    for task in tasks:
        query_records = query_by_task[(task.shot, task.seed)]
        classification_started = time.perf_counter()
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

        if task.shot == 1:
            selection: dict[str, Any] = {
                "selected_text_weight": 0.0,
                "selected_visual_weight": 1.0,
                "selection_source": "one_shot_no_valid_leave_one_out",
                "tie_break": "smallest_text_weight",
                "weight_grid": list(DEFAULT_TEXT_WEIGHT_GRID),
                "macro_f1_by_weight": {},
                "selected_support_macro_f1": None,
                "leave_one_out_count": 0,
            }
        else:
            selection = select_text_weight_leave_one_out(
                support_features,
                support_labels,
                text_prototypes,
                class_names=task.class_names,
                weight_grid=DEFAULT_TEXT_WEIGHT_GRID,
            )
        text_weight = float(selection["selected_text_weight"])

        visual_prototypes = build_class_prototypes(
            support_features, support_labels, task.n_way
        )
        _, visual_scores = predict_with_prototypes(
            query_features, visual_prototypes
        )
        _, query_text_scores = predict_with_prototypes(
            query_features, text_prototypes
        )
        _, support_text_scores = predict_with_prototypes(
            support_features, text_prototypes
        )
        calibrated_text_scores, text_bias = calibrate_text_scores(
            query_text_scores, support_text_scores
        )
        fused_scores = fuse_normalized_scores(
            visual_scores,
            calibrated_text_scores,
            text_weight=text_weight,
        )
        predictions = np.argmax(fused_scores, axis=1).astype(np.int64)
        metrics = evaluate_predictions(y_true, predictions, task.class_names)
        classification_seconds = time.perf_counter() - classification_started

        result: dict[str, Any] = {
            "schema_version": "clip_llm_reliability_fusion_result_v1",
            "method": (
                "OpenAI CLIP ViT-B/32 structured LLM support-reliability fusion"
            ),
            **runtime_metadata,
            **_llm_text_metadata(),
            "prompts": list(prompts),
            "prompts_per_class": 4,
            "similarity": (
                "support-selected weighted sum of row-standardized cosine scores"
            ),
            "score_field_note": (
                "cosine_score stores the final fused standardized score; "
                "text_score stores support-bias-corrected text score"
            ),
            "fusion_level": "score",
            "visual_weight": float(1.0 - text_weight),
            "text_weight": text_weight,
            "weight_selection": (
                "support-only leave-one-out Macro-F1; smallest-weight tie break; "
                "1-shot fixed to zero; no Test tuning"
            ),
            "support_weight_selection": selection,
            "text_weight_grid": list(DEFAULT_TEXT_WEIGHT_GRID),
            "text_bias_source": "complete_support_set_classwise_mean",
            "text_bias": [float(value) for value in text_bias],
            "text_bias_formula": "T_cal(x,c) = T(x,c) - mean_support T(x,c)",
            "score_normalization": (
                "per-sample classwise population z-score with epsilon 1e-12"
            ),
            "protocol": task.protocol_name,
            **task_result_provenance(task),
            "support_source": task.support_source,
            "query_source": task.query_source,
            "manifest_path": task.manifest_path.as_posix(),
            "class_names": list(task.class_names),
            "n_way": task.n_way,
            "shot": task.shot,
            "seed": task.seed,
            "device": encoder.device,
            "batch_size": batch_size,
            "support_count": len(task.support),
            "support_used": True,
            "support_dependent": True,
            "query_count": len(query_records),
            "full_query_count": len(task.query),
            "debug_run": debug_run,
            "formal_grid_validated": bool(aggregate),
            "query_limit": limit_query,
            "timing_seconds": {
                "shared_image_feature_extraction": feature_seconds,
                "shared_text_prototype_encoding": text_seconds,
                "support_selection_fusion_and_evaluation": classification_seconds,
            },
            "metrics": metrics,
        }
        stem = f"{task.n_way}way_{task.shot}shot_seed{task.seed}"
        if debug_run:
            stem += f"_debug_q{len(query_records)}"
        write_run_artifacts(
            output_root,
            result,
            _fusion_prediction_rows(
                query_records,
                task.class_names,
                predictions,
                visual_scores,
                calibrated_text_scores,
                fused_scores,
            ),
            stem=stem,
            extra_prediction_fields=(
                "visual_score",
                "text_score",
                "fused_score",
            ),
        )
        results.append(result)

    if aggregate:
        write_fewshot_summaries(output_root, results, expected_seeds)
    return results
