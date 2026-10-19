"""Support-only CLIP adaptation mathematics for paper baselines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .metrics import evaluate_predictions
from .prototype import l2_normalize


LINEAR_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
TIP_BETA_GRID = (0.5, 1.0, 2.0, 4.0, 6.0)
TIP_ALPHA_GRID = (0.25, 0.5, 1.0, 2.0, 3.0)
TIP_DEFAULT_BETA = 1.0
TIP_DEFAULT_ALPHA = 1.17
SUPPORT_FOLD_COUNT = 5
SUPPORT_AUGMENT_VIEWS = 10
TIP_F_MAX_EPOCHS = 20


def _validated_feature_matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _validated_labels(
    values: Any, *, expected_count: int, num_classes: int
) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.shape[0] != expected_count:
        raise ValueError("labels must be 1D and match the sample count")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integers")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("labels contain an out-of-range class index")
    return labels


def build_stratified_support_folds(
    labels: Any, *, num_classes: int, fold_count: int = SUPPORT_FOLD_COUNT
) -> tuple[np.ndarray, ...]:
    """Partition balanced K-shot support in deterministic manifest order."""

    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if fold_count <= 1:
        raise ValueError("fold_count must be greater than one")
    raw_labels = np.asarray(labels)
    validated = _validated_labels(
        raw_labels, expected_count=raw_labels.size, num_classes=num_classes
    )
    class_indices = [np.flatnonzero(validated == index) for index in range(num_classes)]
    class_counts = {indices.size for indices in class_indices}
    if len(class_counts) != 1:
        raise ValueError("support folds require the same shot count for every class")
    shot = class_counts.pop()
    if shot < 2:
        raise ValueError("support validation requires at least two samples per class")
    if shot % fold_count != 0:
        raise ValueError("shot count must be divisible by fold_count")
    held_out_per_class = shot // fold_count
    folds = []
    for fold_index in range(fold_count):
        start = fold_index * held_out_per_class
        stop = start + held_out_per_class
        fold = np.concatenate(
            [indices[start:stop] for indices in class_indices]
        ).astype(np.int64, copy=False)
        folds.append(fold)
    return tuple(folds)


def build_group_stratified_support_folds(
    labels: Any,
    group_ids: Any,
    *,
    num_classes: int,
    fold_count: int = SUPPORT_FOLD_COUNT,
) -> tuple[np.ndarray, ...]:
    """Partition support by class while keeping every content group intact."""

    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if fold_count <= 1:
        raise ValueError("fold_count must be greater than one")
    raw_labels = np.asarray(labels)
    validated = _validated_labels(
        raw_labels, expected_count=raw_labels.size, num_classes=num_classes
    )
    groups = np.asarray(group_ids, dtype=str)
    if groups.ndim != 1 or groups.shape != validated.shape:
        raise ValueError("group_ids must be 1D and match labels")
    if np.any(np.char.str_len(groups) == 0):
        raise ValueError("group_ids must not contain blanks")

    for group_id in np.unique(groups):
        if np.unique(validated[groups == group_id]).size != 1:
            raise ValueError("one content group has multiple labels")

    class_groups: list[list[str]] = []
    for class_index in range(num_classes):
        ordered_groups = list(dict.fromkeys(groups[validated == class_index].tolist()))
        class_groups.append(ordered_groups)
    unique_counts = {len(values) for values in class_groups}
    if len(unique_counts) != 1:
        raise ValueError("support folds require equal unique groups per class")
    group_shot = unique_counts.pop()
    if group_shot < 2:
        raise ValueError("support validation requires at least two groups per class")
    if group_shot % fold_count != 0:
        raise ValueError("unique group count must be divisible by fold_count")

    held_out_per_class = group_shot // fold_count
    folds: list[np.ndarray] = []
    for fold_index in range(fold_count):
        start = fold_index * held_out_per_class
        stop = start + held_out_per_class
        selected = {
            group_id
            for values in class_groups
            for group_id in values[start:stop]
        }
        folds.append(
            np.flatnonzero(np.isin(groups, tuple(selected))).astype(
                np.int64, copy=False
            )
        )
    return tuple(folds)


def build_cache_model(
    augmented_features: Any, labels: Any, *, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Average augmentation views into normalized cache keys and one-hot values."""

    views = np.asarray(augmented_features, dtype=np.float64)
    if views.ndim != 3 or min(views.shape) <= 0:
        raise ValueError(
            "augmented_features must have shape (views, samples, dimension)"
        )
    if not np.isfinite(views).all():
        raise ValueError("augmented_features contains non-finite values")
    support_labels = _validated_labels(
        labels, expected_count=views.shape[1], num_classes=num_classes
    )
    keys = l2_normalize(views.mean(axis=0))
    values = np.eye(num_classes, dtype=np.float64)[support_labels]
    return keys, values


def tip_adapter_logits(
    features: Any,
    text_prototypes: Any,
    cache_keys: Any,
    cache_values: Any,
    *,
    beta: float,
    alpha: float,
    normalize_cache_keys: bool = True,
) -> np.ndarray:
    """Apply the published Tip-Adapter CLIP-plus-cache logit equation."""

    if beta <= 0 or alpha <= 0:
        raise ValueError("beta and alpha must be positive")
    query = l2_normalize(_validated_feature_matrix(features, name="features"))
    text = l2_normalize(
        _validated_feature_matrix(text_prototypes, name="text_prototypes")
    )
    keys = _validated_feature_matrix(cache_keys, name="cache_keys")
    if normalize_cache_keys:
        keys = l2_normalize(keys)
    values = _validated_feature_matrix(cache_values, name="cache_values")
    if query.shape[1] != text.shape[1] or query.shape[1] != keys.shape[1]:
        raise ValueError("feature dimensions must match")
    if keys.shape[0] != values.shape[0]:
        raise ValueError("cache key and value counts must match")
    if text.shape[0] != values.shape[1]:
        raise ValueError("text and cache class counts must match")
    clip_logits = 100.0 * query @ text.T
    affinity = query @ keys.T
    cache_logits = np.exp(-float(beta) * (1.0 - affinity)) @ values
    return clip_logits + float(alpha) * cache_logits


def select_tip_hyperparameters(
    validation_features: Any,
    cache_features: Any,
    labels: Any,
    text_prototypes: Any,
    folds: Sequence[np.ndarray],
    *,
    num_classes: int,
    beta_grid: Sequence[float] = TIP_BETA_GRID,
    alpha_grid: Sequence[float] = TIP_ALPHA_GRID,
    default_beta: float = TIP_DEFAULT_BETA,
    default_alpha: float = TIP_DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Select Tip-Adapter settings using support folds only."""

    queries = _validated_feature_matrix(
        validation_features, name="validation_features"
    )
    caches = _validated_feature_matrix(cache_features, name="cache_features")
    if queries.shape != caches.shape:
        raise ValueError("validation and cache feature matrices must match")
    support_labels = _validated_labels(
        labels, expected_count=queries.shape[0], num_classes=num_classes
    )
    text = _validated_feature_matrix(text_prototypes, name="text_prototypes")
    if text.shape != (num_classes, queries.shape[1]):
        raise ValueError("text_prototypes has the wrong class or feature dimension")
    fold_arrays = tuple(np.asarray(fold, dtype=np.int64) for fold in folds)
    held_out = np.concatenate(fold_arrays)
    if sorted(held_out.tolist()) != list(range(queries.shape[0])):
        raise ValueError("folds must hold out every support sample exactly once")

    beta_values = tuple(float(value) for value in beta_grid)
    alpha_values = tuple(float(value) for value in alpha_grid)
    if not beta_values or not alpha_values:
        raise ValueError("hyperparameter grids must not be empty")

    candidate_rows: list[dict[str, float]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_row: dict[str, float] | None = None
    all_indices = np.arange(queries.shape[0])
    class_names = tuple(str(index) for index in range(num_classes))
    for beta in beta_values:
        for alpha in alpha_values:
            true_parts = []
            prediction_parts = []
            for fold in fold_arrays:
                train_mask = np.ones(queries.shape[0], dtype=bool)
                train_mask[fold] = False
                train_indices = all_indices[train_mask]
                train_views = caches[train_indices][None, :, :]
                keys, values = build_cache_model(
                    train_views,
                    support_labels[train_indices],
                    num_classes=num_classes,
                )
                logits = tip_adapter_logits(
                    queries[fold],
                    text,
                    keys,
                    values,
                    beta=beta,
                    alpha=alpha,
                )
                true_parts.append(support_labels[fold])
                prediction_parts.append(np.argmax(logits, axis=1))
            y_true = np.concatenate(true_parts)
            y_pred = np.concatenate(prediction_parts)
            macro_f1 = float(
                evaluate_predictions(y_true, y_pred, class_names)["macro_f1"]
            )
            row = {"beta": beta, "alpha": alpha, "macro_f1": macro_f1}
            candidate_rows.append(row)
            tie_key = (
                -macro_f1,
                (beta - default_beta) ** 2 + (alpha - default_alpha) ** 2,
                beta,
                alpha,
            )
            if best_key is None or tie_key < best_key:
                best_key = tie_key
                best_row = row
    assert best_row is not None
    return {
        **best_row,
        "validation_count": int(queries.shape[0]),
        "candidates": candidate_rows,
    }
