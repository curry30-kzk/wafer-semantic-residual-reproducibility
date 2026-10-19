"""Support-only calibration and conservative visual-text score fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .metrics import evaluate_predictions
from .prototype import build_class_prototypes, l2_normalize, predict_with_prototypes


DEFAULT_TEXT_WEIGHT_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
STANDARDIZATION_EPSILON = 1e-12


def _score_matrix(scores: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def aggregate_prompt_embeddings(
    prompt_embeddings: np.ndarray, *, prompts_per_class: int
) -> np.ndarray:
    """Mean normalized prompt rows class-wise, then normalize each class mean."""

    embeddings = _score_matrix(prompt_embeddings, "prompt_embeddings")
    if prompts_per_class <= 0:
        raise ValueError("prompts_per_class must be positive")
    if embeddings.shape[0] % prompts_per_class != 0:
        raise ValueError(
            "prompt embedding rows must be divisible by prompts_per_class"
        )
    normalized = l2_normalize(embeddings)
    grouped = normalized.reshape(
        embeddings.shape[0] // prompts_per_class,
        prompts_per_class,
        embeddings.shape[1],
    )
    return l2_normalize(grouped.mean(axis=1)).astype(np.float32, copy=False)


def calibrate_text_scores(
    text_scores: np.ndarray, support_text_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract class-specific text bias estimated from support images only."""

    query = _score_matrix(text_scores, "text_scores")
    support = _score_matrix(support_text_scores, "support_text_scores")
    if query.shape[1] != support.shape[1]:
        raise ValueError("text and support score matrices need equal class counts")
    bias = support.mean(axis=0)
    return query - bias[None, :], bias


def row_standardize(scores: np.ndarray) -> np.ndarray:
    """Standardize each sample across classes without changing its argmax."""

    matrix = _score_matrix(scores, "scores")
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = matrix.std(axis=1, keepdims=True)
    safe_scale = np.maximum(scale, STANDARDIZATION_EPSILON)
    return centered / safe_scale


def fuse_normalized_scores(
    visual_scores: np.ndarray,
    calibrated_text_scores: np.ndarray,
    *,
    text_weight: float,
) -> np.ndarray:
    """Fuse row-standardized score sources at one declared text weight."""

    visual = _score_matrix(visual_scores, "visual_scores")
    text = _score_matrix(calibrated_text_scores, "calibrated_text_scores")
    if visual.shape != text.shape:
        raise ValueError("visual and text score matrices must have identical shapes")
    weight = float(text_weight)
    if not np.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ValueError("text_weight must be finite and in [0, 1]")
    return (1.0 - weight) * row_standardize(visual) + weight * row_standardize(text)


def _validate_weight_grid(weight_grid: Sequence[float]) -> tuple[float, ...]:
    weights = tuple(float(weight) for weight in weight_grid)
    if not weights:
        raise ValueError("weight_grid must not be empty")
    if any(not np.isfinite(weight) or weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("weight_grid values must be finite and in [0, 1]")
    if tuple(sorted(set(weights))) != weights:
        raise ValueError("weight_grid must be strictly increasing and unique")
    if weights[0] != 0.0:
        raise ValueError("weight_grid must start with zero for safe fallback")
    return weights


def select_text_weight_leave_one_out(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    text_prototypes: np.ndarray,
    *,
    class_names: Sequence[str],
    weight_grid: Sequence[float] = DEFAULT_TEXT_WEIGHT_GRID,
) -> dict[str, Any]:
    """Choose a global text weight using support-only leave-one-out Macro-F1."""

    features = _score_matrix(support_features, "support_features")
    labels = np.asarray(support_labels)
    prototypes = _score_matrix(text_prototypes, "text_prototypes")
    names = tuple(str(name) for name in class_names)
    weights = _validate_weight_grid(weight_grid)
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("support_labels must match support feature rows")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("support_labels must contain integer class indices")
    labels = labels.astype(np.int64, copy=False)
    num_classes = len(names)
    if num_classes == 0 or prototypes.shape[0] != num_classes:
        raise ValueError("text prototypes and class_names must have equal rows")
    if prototypes.shape[1] != features.shape[1]:
        raise ValueError("text and support feature dimensions must match")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("support label is outside the class range")
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts < 2):
        raise ValueError("leave-one-out requires at least two support examples per class")

    predictions_by_weight = {
        weight: np.empty(labels.shape[0], dtype=np.int64) for weight in weights
    }
    normalized_support = l2_normalize(features)
    normalized_text = l2_normalize(prototypes)
    all_text_scores = normalized_support @ normalized_text.T

    for held_index in range(features.shape[0]):
        keep = np.ones(features.shape[0], dtype=bool)
        keep[held_index] = False
        visual_prototypes = build_class_prototypes(
            features[keep], labels[keep], num_classes
        )
        _, visual_scores = predict_with_prototypes(
            features[held_index : held_index + 1], visual_prototypes
        )
        calibrated_text, _ = calibrate_text_scores(
            all_text_scores[held_index : held_index + 1], all_text_scores[keep]
        )
        for weight in weights:
            fused = fuse_normalized_scores(
                visual_scores, calibrated_text, text_weight=weight
            )
            predictions_by_weight[weight][held_index] = int(np.argmax(fused[0]))

    macro_f1_by_weight: dict[str, float] = {}
    best_weight = weights[0]
    best_macro_f1 = -1.0
    for weight in weights:
        metrics = evaluate_predictions(labels, predictions_by_weight[weight], names)
        macro_f1 = float(metrics["macro_f1"])
        macro_f1_by_weight[f"{weight:.1f}"] = macro_f1
        if macro_f1 > best_macro_f1 + 1e-12:
            best_macro_f1 = macro_f1
            best_weight = weight

    return {
        "selected_text_weight": float(best_weight),
        "selected_visual_weight": float(1.0 - best_weight),
        "selection_source": "support_leave_one_out_macro_f1",
        "tie_break": "smallest_text_weight",
        "weight_grid": list(weights),
        "macro_f1_by_weight": macro_f1_by_weight,
        "selected_support_macro_f1": best_macro_f1,
        "leave_one_out_count": int(labels.shape[0]),
    }
