"""Encoder-independent cosine prototype classification."""

from __future__ import annotations

import numpy as np


def _as_feature_matrix(features: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """L2-normalize every feature row, rejecting undefined zero vectors."""

    matrix = _as_feature_matrix(features, "features")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("cannot L2-normalize a zero-norm feature row")
    return matrix / norms


def build_class_prototypes(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Mean normalized support features by explicit class index."""

    features = _as_feature_matrix(support_features, "support_features")
    labels = np.asarray(support_labels)
    if labels.ndim != 1:
        raise ValueError("support_labels must be a 1D array")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("support_labels length must match support feature rows")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError("support_labels must contain integer class indices")
        labels = labels.astype(np.int64)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"support label is outside valid range [0, {num_classes})")

    normalized_support = l2_normalize(features)
    prototypes: list[np.ndarray] = []
    for class_index in range(num_classes):
        class_rows = normalized_support[labels == class_index]
        if class_rows.shape[0] == 0:
            raise ValueError(f"support is missing class {class_index}")
        prototypes.append(class_rows.mean(axis=0))
    return l2_normalize(np.stack(prototypes, axis=0))


def predict_with_prototypes(
    query_features: np.ndarray, prototypes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic cosine-nearest predictions and score matrix."""

    query = _as_feature_matrix(query_features, "query_features")
    prototype_matrix = _as_feature_matrix(prototypes, "prototypes")
    if query.shape[1] != prototype_matrix.shape[1]:
        raise ValueError(
            "query feature dimension must match prototype feature dimension"
        )
    normalized_query = l2_normalize(query)
    normalized_prototypes = l2_normalize(prototype_matrix)
    scores = normalized_query @ normalized_prototypes.T
    predictions = np.argmax(scores, axis=1).astype(np.int64)
    return predictions, scores
