"""Shared paper metrics for WM-811K few-shot experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.stats import t
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


PRIMARY_METRICS = ("accuracy", "macro_f1", "balanced_accuracy")


def _validated_labels(
    values: Iterable[int], *, name: str, num_classes: int
) -> np.ndarray:
    labels = np.asarray(list(values))
    if labels.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if labels.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(labels.dtype, np.integer):
        try:
            numeric = labels.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain integer labels") from exc
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{name} must contain integer labels")
        labels = numeric.astype(np.int64)
    else:
        labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"{name} contains a label outside valid range [0, {num_classes})")
    return labels


def evaluate_predictions(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute JSON-serializable imbalanced classification metrics."""

    names = tuple(str(name) for name in class_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("class_names must be non-empty and unique")
    true_labels = _validated_labels(y_true, name="y_true", num_classes=len(names))
    pred_labels = _validated_labels(y_pred, name="y_pred", num_classes=len(names))
    if true_labels.shape[0] != pred_labels.shape[0]:
        raise ValueError("y_true and y_pred lengths must match")

    label_indices = np.arange(len(names), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=label_indices,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, pred_labels, labels=label_indices)
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(names)
    }
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def summarize_seed_metrics(
    run_results: Sequence[dict[str, Any]],
    *,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Summarize one shot across exact seeds using a 95% Student-t interval."""

    if not run_results:
        raise ValueError("run_results must not be empty")
    if any(bool(result.get("debug_run")) for result in run_results):
        raise ValueError("debug runs cannot be included in formal seed summaries")

    expected = tuple(int(seed) for seed in expected_seeds)
    actual = tuple(int(result["seed"]) for result in run_results)
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError(
            f"run seeds must match expected seeds {list(expected)}, got {list(actual)}"
        )

    shots = {int(result["shot"]) for result in run_results}
    if len(shots) != 1:
        raise ValueError("all summarized runs must have the same shot")

    by_seed = {int(result["seed"]): result for result in run_results}
    ordered_results = [by_seed[seed] for seed in expected]
    metric_summaries: dict[str, dict[str, Any]] = {}
    for metric_name in PRIMARY_METRICS:
        values = np.asarray(
            [float(result["metrics"][metric_name]) for result in ordered_results],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"metric {metric_name} contains a non-finite value")
        mean = float(np.mean(values))
        if values.size == 1:
            sample_std = 0.0
            margin = 0.0
        else:
            sample_std = float(np.std(values, ddof=1))
            critical = float(t.ppf(0.975, df=values.size - 1))
            margin = critical * sample_std / math.sqrt(values.size)
        metric_summaries[metric_name] = {
            "mean": mean,
            "std": sample_std,
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
            "values": values.tolist(),
        }

    return {
        "shot": next(iter(shots)),
        "seeds": list(expected),
        "confidence_interval": "two-sided Student-t 95%",
        "metrics": metric_summaries,
    }
