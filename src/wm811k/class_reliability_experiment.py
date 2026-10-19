"""Leakage-free experiment utilities for classwise semantic reliability."""

from __future__ import annotations

import json
import hashlib
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .clip_adaptation import (
    LINEAR_C_GRID,
    build_group_stratified_support_folds,
    build_stratified_support_folds,
)
from .clip_adaptation_experiment import (
    FittedLinearProbe,
    FittedTipAdapter,
    _fit_logistic,
    _validate_support_inputs,
    fit_linear_probe_support_only,
    fit_tip_adapter_f_support_only,
)
from .baseline_experiment import select_debug_query_subset
from .class_reliability_residual import (
    SemanticCalibration,
    apply_classwise_residual,
    apply_direct_weighted_fusion,
    apply_semantic_calibration,
    fit_reliability,
    fit_reliability_closed_form,
    fit_semantic_calibration,
)
from .clip_experiment import validate_task_grid
from .fewshot_dataset import FewShotTask, ImageRecord
from .metrics import evaluate_predictions
from .prototype import build_class_prototypes, l2_normalize, predict_with_prototypes
from .v2_contract import augmented_feature_identity, clean_feature_identity


ONE_SHOT_RELIABILITY_REASON = "one_shot_frozen_visual_prototype_fallback"


@dataclass(frozen=True)
class VisualAnchorFit:
    """A frozen visual anchor plus support-only out-of-fold evidence."""

    mode: str
    final_model: FittedLinearProbe | FittedTipAdapter | "FittedVisualPrototype"
    oof_scores: np.ndarray | None
    held_out_indices: np.ndarray
    final_scores_on_support: np.ndarray
    selection: dict[str, Any]

    def decision_scores(self, features: Any) -> np.ndarray:
        return np.asarray(self.final_model.decision_scores(features), dtype=np.float64)


@dataclass(frozen=True)
class FittedVisualPrototype:
    """Frozen CLIP support prototype used for the one-shot anchor."""

    prototypes: np.ndarray
    selection: dict[str, Any]
    trainable_parameter_count: int = 0

    def decision_scores(self, features: Any) -> np.ndarray:
        query = l2_normalize(np.asarray(features, dtype=np.float64))
        normalized = l2_normalize(self.prototypes)
        return np.asarray(query @ normalized.T, dtype=np.float64)

    def predict(self, features: Any) -> np.ndarray:
        predictions, _ = predict_with_prototypes(features, self.prototypes)
        return predictions


@dataclass(frozen=True)
class SemanticScoreCalibration:
    """Fold-wise support residuals and full-support query calibration."""

    oof_residual: np.ndarray
    query_residual: np.ndarray
    fold_class_bias: tuple[np.ndarray, ...]
    final_calibration: SemanticCalibration
    metadata: dict[str, Any]


def calibrate_semantic_scores_temperature_support_only(
    support_text_scores: Any,
    query_text_scores: Any,
    *,
    support_labels: Any,
    visual_support_scores: Any,
    visual_oof_scores: Any,
    folds: Sequence[Any],
    temperature_grid: Sequence[float] = tuple(np.geomspace(0.25, 4.0, 33)),
) -> SemanticScoreCalibration:
    """Cross-fit one scalar temperature using support labels only."""
    support = _finite_matrix(support_text_scores, name="support_text_scores")
    query = _finite_matrix(query_text_scores, name="query_text_scores")
    visual_support = _finite_matrix(visual_support_scores, name="visual_support_scores")
    visual_oof = _finite_matrix(visual_oof_scores, name="visual_oof_scores")
    labels = np.asarray(support_labels, dtype=np.int64)
    if support.shape != visual_support.shape or support.shape != visual_oof.shape:
        raise ValueError("support text and visual scores must share one shape")
    if query.shape[1] != support.shape[1] or labels.shape != (support.shape[0],):
        raise ValueError("temperature calibration inputs have incompatible shapes")
    fold_arrays = tuple(np.asarray(fold, dtype=np.int64) for fold in folds)
    if not fold_arrays:
        raise ValueError("folds must not be empty")
    grid = tuple(sorted(set(float(t) for t in temperature_grid)))
    if not grid or any((not np.isfinite(t) or t <= 0.0) for t in grid):
        raise ValueError("temperature_grid must contain positive finite values")

    def nll(scores: np.ndarray, y: np.ndarray, temperature: float) -> float:
        logits = scores / temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        return float(np.mean(-logits[np.arange(y.size), y] + np.log(np.exp(logits).sum(axis=1))))

    def choose(indices: np.ndarray) -> float:
        return min(grid, key=lambda t: (nll(support[indices], labels[indices], t), t))

    oof_logits = np.empty_like(support)
    temperatures: list[float] = []
    all_indices = np.arange(support.shape[0], dtype=np.int64)
    for held_out in fold_arrays:
        mask = np.ones(support.shape[0], dtype=bool)
        mask[held_out] = False
        train_indices = all_indices[mask]
        temperature = choose(train_indices)
        temperatures.append(temperature)
        oof_logits[held_out] = support[held_out] / temperature
    final_temperature = choose(all_indices)
    visual_scale = max(float(np.median(np.std(visual_oof, axis=1, ddof=0))), 1e-12)
    oof_residual = _row_standardize(oof_logits, epsilon=1e-12) * visual_scale
    query_logits = query / final_temperature
    query_residual = _row_standardize(query_logits, epsilon=1e-12) * max(float(np.median(np.std(visual_support, axis=1, ddof=0))), 1e-12)
    return SemanticScoreCalibration(
        oof_residual=oof_residual,
        query_residual=query_residual,
        fold_class_bias=tuple(np.zeros(support.shape[1], dtype=np.float64) for _ in fold_arrays),
        final_calibration=fit_semantic_calibration(support / final_temperature, visual_support),
        metadata={
            "strategy": "support_foldwise_scalar_temperature",
            "query_statistics_used": False,
            "fold_temperatures": temperatures,
            "final_temperature": final_temperature,
            "temperature_grid": list(grid),
        },
    )


@dataclass(frozen=True)
class TaskReliabilityCache:
    """All immutable scores needed to evaluate one reliability task cheaply."""

    protocol: str
    shot: int
    seed: int
    manifest_sha256: str
    semantic_bundle_sha256: str
    class_names: tuple[str, ...]
    support_source: str
    query_source: str
    support_labels: np.ndarray
    query_labels: np.ndarray
    visual_support_scores: np.ndarray
    visual_oof_scores: np.ndarray | None
    visual_query_scores: np.ndarray
    semantic_oof_raw: np.ndarray | None
    semantic_oof_residual: np.ndarray | None
    semantic_query_raw: np.ndarray
    semantic_query_residual: np.ndarray
    anchor_mode: str
    anchor_selection: dict[str, Any]
    calibration_metadata: dict[str, Any]
    runtime_metadata: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None


def anchor_mode_for_shot(shot: int) -> str:
    """Return the Validation-frozen strongest visual anchor for a shot count."""

    shot_value = int(shot)
    if shot_value == 1:
        return "tip_adapter_f"
    if shot_value in (5, 10):
        return "linear_probe"
    raise ValueError("shot must be one of 1, 5, or 10")


def _logistic_decision_scores(model: Any, features: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(features), dtype=np.float64)
    if scores.ndim == 1:
        scores = np.stack((-scores, scores), axis=1)
    return scores


def fit_linear_anchor_with_oof(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    *,
    num_classes: int,
    shot: int,
    c_grid: Sequence[float] = LINEAR_C_GRID,
    selected_c: float | None = None,
    selection_provenance: Mapping[str, Any] | None = None,
    support_group_ids: Sequence[str] | None = None,
) -> VisualAnchorFit:
    """Fit the frozen linear anchor and score every support row out of fold."""

    if int(shot) not in (5, 10):
        raise ValueError("linear anchor OOF scores require 5-shot or 10-shot support")
    clean, augmented, support_labels = _validate_support_inputs(
        clean_features,
        augmented_features,
        labels,
        num_classes=int(num_classes),
        shot=int(shot),
    )
    group_ids: np.ndarray | None = None
    if support_group_ids is None:
        folds = build_stratified_support_folds(
            support_labels,
            num_classes=int(num_classes),
        )
    else:
        group_ids = np.asarray(support_group_ids, dtype=str)
        if group_ids.shape != support_labels.shape:
            raise ValueError("support_group_ids must match support labels")
        folds = build_group_stratified_support_folds(
            support_labels,
            group_ids,
            num_classes=int(num_classes),
        )
    if selected_c is None:
        final_model = fit_linear_probe_support_only(
            clean,
            augmented,
            support_labels,
            num_classes=int(num_classes),
            shot=int(shot),
            c_grid=tuple(float(value) for value in c_grid),
        )
        selected_c_value = float(final_model.selection["C"])
    else:
        selected_c_value = float(selected_c)
        if not np.isfinite(selected_c_value) or selected_c_value <= 0.0:
            raise ValueError("selected_c must be finite and positive")
        if selection_provenance is None:
            raise ValueError("selected_c reuse requires selection_provenance")
        provenance = dict(selection_provenance)
        if float(provenance.get("C", float("nan"))) != selected_c_value:
            raise ValueError("selection provenance C does not match selected_c")
        expected_folds = [fold.tolist() for fold in folds]
        if provenance.get("folds") != expected_folds:
            raise ValueError("selection provenance folds do not match support order")
        selection = {
            **provenance,
            "selection_reused": True,
            "selection_reuse_source": "audited_baseline_result",
        }
        final_logistic = _fit_logistic(
            augmented,
            support_labels,
            c_value=selected_c_value,
        )
        trainable_parameters = clean.shape[1] * int(num_classes) + int(num_classes)
        final_model = FittedLinearProbe(
            final_logistic,
            selection,
            trainable_parameters,
        )
    oof_scores = np.empty((clean.shape[0], int(num_classes)), dtype=np.float64)
    held_out_parts: list[np.ndarray] = []
    all_indices = np.arange(clean.shape[0], dtype=np.int64)
    for fold in folds:
        held_out = np.asarray(fold, dtype=np.int64)
        train_mask = np.ones(clean.shape[0], dtype=bool)
        train_mask[held_out] = False
        train_indices = all_indices[train_mask]
        fold_model = _fit_logistic(
            augmented[:, train_indices, :],
            support_labels[train_indices],
            c_value=selected_c_value,
        )
        fold_scores = _logistic_decision_scores(fold_model, clean[held_out])
        if fold_scores.shape != (held_out.size, int(num_classes)):
            raise ValueError("fold decision scores have the wrong class dimension")
        oof_scores[held_out] = fold_scores
        held_out_parts.append(held_out)

    held_out_indices = np.concatenate(held_out_parts)
    if sorted(held_out_indices.tolist()) != list(range(clean.shape[0])):
        raise ValueError("support folds must hold out every row exactly once")
    final_scores = np.asarray(final_model.decision_scores(clean), dtype=np.float64)
    selection = dict(final_model.selection)
    selection.update(
        {
            "oof_count": int(held_out_indices.size),
            "oof_folds": [fold.tolist() for fold in folds],
            "oof_uses_selected_C": True,
            "query_labels_used_for_selection": False,
        }
    )
    if group_ids is not None:
        fold_group_ids = [group_ids[fold].tolist() for fold in folds]
        fold_payload = {
            "group_ids": group_ids.tolist(),
            "fold_group_ids": fold_group_ids,
        }
        selection.update(
            {
                "oof_group_aware": True,
                "oof_group_ids": group_ids.tolist(),
                "oof_fold_group_ids": fold_group_ids,
                "oof_fold_assignment_sha256": hashlib.sha256(
                    json.dumps(
                        fold_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "oof_each_support_held_out_once": True,
            }
        )
    return VisualAnchorFit(
        mode="linear_probe",
        final_model=final_model,
        oof_scores=oof_scores,
        held_out_indices=held_out_indices,
        final_scores_on_support=final_scores,
        selection=selection,
    )


def fit_visual_anchor(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    num_classes: int,
    shot: int,
    device: str,
    c_grid: Sequence[float] = LINEAR_C_GRID,
    tip_f_max_epochs: int = 20,
    selected_c: float | None = None,
    selection_provenance: Mapping[str, Any] | None = None,
    anchor_mode: str | None = None,
    support_group_ids: Sequence[str] | None = None,
) -> VisualAnchorFit:
    """Fit the predeclared visual anchor without consulting query labels."""

    mode = anchor_mode_for_shot(shot) if anchor_mode is None else str(anchor_mode)
    if mode == "linear_probe":
        return fit_linear_anchor_with_oof(
            clean_features,
            augmented_features,
            labels,
            num_classes=num_classes,
            shot=shot,
            c_grid=c_grid,
            selected_c=selected_c,
            selection_provenance=selection_provenance,
            support_group_ids=support_group_ids,
        )

    clean, augmented, support_labels = _validate_support_inputs(
        clean_features,
        augmented_features,
        labels,
        num_classes=int(num_classes),
        shot=int(shot),
    )
    if mode == "visual_prototype":
        prototypes = build_class_prototypes(
            clean,
            support_labels,
            int(num_classes),
        )
        selection = {
            "strategy": "frozen_support_visual_prototype",
            "reason": ONE_SHOT_RELIABILITY_REASON,
            "query_labels_used_for_selection": False,
            "support_count": int(clean.shape[0]),
        }
        final_model = FittedVisualPrototype(prototypes, selection)
        return VisualAnchorFit(
            mode="visual_prototype",
            final_model=final_model,
            oof_scores=None,
            held_out_indices=np.empty(0, dtype=np.int64),
            final_scores_on_support=final_model.decision_scores(clean),
            selection=selection,
        )
    if mode != "tip_adapter_f":
        raise ValueError(f"unsupported visual anchor mode: {mode}")
    final_model = fit_tip_adapter_f_support_only(
        clean,
        augmented,
        support_labels,
        text_prototypes,
        num_classes=int(num_classes),
        shot=int(shot),
        device=device,
        max_epochs=int(tip_f_max_epochs),
    )
    selection = dict(final_model.selection)
    selection.update(
        {
            "reliability_reason": ONE_SHOT_RELIABILITY_REASON,
            "query_labels_used_for_selection": False,
        }
    )
    return VisualAnchorFit(
        mode="tip_adapter_f",
        final_model=final_model,
        oof_scores=None,
        held_out_indices=np.empty(0, dtype=np.int64),
        final_scores_on_support=np.asarray(
            final_model.decision_scores(clean), dtype=np.float64
        ),
        selection=selection,
    )


def _finite_matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _row_standardize(centered: np.ndarray, *, epsilon: float) -> np.ndarray:
    rows = centered - centered.mean(axis=1, keepdims=True)
    scales = np.maximum(rows.std(axis=1, keepdims=True, ddof=0), epsilon)
    return rows / scales


def calibrate_semantic_scores_support_only(
    support_text_scores: Any,
    query_text_scores: Any,
    *,
    visual_support_scores: Any,
    visual_oof_scores: Any,
    folds: Sequence[Any],
    epsilon: float = 1e-12,
) -> SemanticScoreCalibration:
    """Calibrate OOF/query semantics without using any query statistics."""

    support_text = _finite_matrix(
        support_text_scores, name="support_text_scores"
    )
    query_text = _finite_matrix(query_text_scores, name="query_text_scores")
    visual_support = _finite_matrix(
        visual_support_scores, name="visual_support_scores"
    )
    visual_oof = _finite_matrix(visual_oof_scores, name="visual_oof_scores")
    if support_text.shape != visual_support.shape or support_text.shape != visual_oof.shape:
        raise ValueError("support text and visual scores must share one shape")
    if query_text.shape[1] != support_text.shape[1]:
        raise ValueError("query text scores have the wrong class dimension")
    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    fold_arrays = tuple(np.asarray(fold, dtype=np.int64) for fold in folds)
    if not fold_arrays or any(fold.ndim != 1 or fold.size == 0 for fold in fold_arrays):
        raise ValueError("folds must contain non-empty 1D index arrays")
    held_out_indices = np.concatenate(fold_arrays)
    if sorted(held_out_indices.tolist()) != list(range(support_text.shape[0])):
        raise ValueError("folds must hold out every support row exactly once")

    normalized_oof = np.empty_like(support_text, dtype=np.float64)
    fold_biases: list[np.ndarray] = []
    all_indices = np.arange(support_text.shape[0], dtype=np.int64)
    for held_out in fold_arrays:
        if np.any(held_out < 0) or np.any(held_out >= support_text.shape[0]):
            raise ValueError("fold contains an out-of-range support index")
        train_mask = np.ones(support_text.shape[0], dtype=bool)
        train_mask[held_out] = False
        train_indices = all_indices[train_mask]
        if train_indices.size == 0:
            raise ValueError("every fold needs at least one support training row")
        class_bias = support_text[train_indices].mean(axis=0)
        fold_biases.append(class_bias)
        normalized_oof[held_out] = _row_standardize(
            support_text[held_out] - class_bias[None, :],
            epsilon=epsilon_value,
        )

    visual_scale = max(
        float(np.median(np.std(visual_oof, axis=1, ddof=0))),
        epsilon_value,
    )
    oof_residual = normalized_oof * visual_scale
    final_calibration = fit_semantic_calibration(
        support_text,
        visual_support,
        epsilon=epsilon_value,
    )
    query_residual = apply_semantic_calibration(query_text, final_calibration)
    return SemanticScoreCalibration(
        oof_residual=oof_residual,
        query_residual=query_residual,
        fold_class_bias=tuple(bias.copy() for bias in fold_biases),
        final_calibration=final_calibration,
        metadata={
            "strategy": "support_foldwise_bias_and_oof_visual_scale",
            "query_statistics_used": False,
            "oof_count": int(held_out_indices.size),
            "folds": [fold.tolist() for fold in fold_arrays],
            "fold_class_bias": [bias.tolist() for bias in fold_biases],
            "oof_visual_scale": float(visual_scale),
            "final_class_bias": final_calibration.class_bias.tolist(),
            "final_visual_scale": float(final_calibration.visual_scale),
            "epsilon": epsilon_value,
        },
    )


def _platt_decision_scores(
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    evaluation_scores: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Fit one balanced one-vs-rest Platt model per class."""

    from sklearn.linear_model import LogisticRegression

    class_count = train_scores.shape[1]
    output = np.empty((evaluation_scores.shape[0], class_count), dtype=np.float64)
    parameters: list[dict[str, float]] = []
    for class_index in range(class_count):
        binary = (train_labels == class_index).astype(np.int64)
        if np.unique(binary).size != 2:
            raise ValueError("Platt calibration needs positive and negative support rows")
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
            solver="lbfgs",
        )
        model.fit(train_scores[:, class_index : class_index + 1], binary)
        output[:, class_index] = model.decision_function(
            evaluation_scores[:, class_index : class_index + 1]
        )
        parameters.append(
            {
                "coefficient": float(model.coef_[0, 0]),
                "intercept": float(model.intercept_[0]),
            }
        )
    return output, parameters


def calibrate_semantic_scores_platt_support_only(
    support_text_scores: Any,
    query_text_scores: Any,
    *,
    support_labels: Any,
    visual_support_scores: Any,
    visual_oof_scores: Any,
    folds: Sequence[Any],
    epsilon: float = 1e-12,
) -> SemanticScoreCalibration:
    """Cross-fit per-class Platt maps without consulting query statistics."""

    support_text = _finite_matrix(support_text_scores, name="support_text_scores")
    query_text = _finite_matrix(query_text_scores, name="query_text_scores")
    visual_support = _finite_matrix(visual_support_scores, name="visual_support_scores")
    visual_oof = _finite_matrix(visual_oof_scores, name="visual_oof_scores")
    if support_text.shape != visual_support.shape or support_text.shape != visual_oof.shape:
        raise ValueError("support text and visual scores must share one shape")
    if query_text.shape[1] != support_text.shape[1]:
        raise ValueError("query text scores have the wrong class dimension")
    labels = np.asarray(support_labels, dtype=np.int64)
    if labels.shape != (support_text.shape[0],):
        raise ValueError("support_labels must match support score rows")
    if np.any(labels < 0) or np.any(labels >= support_text.shape[1]):
        raise ValueError("support_labels contain an out-of-range class")
    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    fold_arrays = tuple(np.asarray(fold, dtype=np.int64) for fold in folds)
    if not fold_arrays or any(fold.ndim != 1 or fold.size == 0 for fold in fold_arrays):
        raise ValueError("folds must contain non-empty 1D index arrays")
    held_out_indices = np.concatenate(fold_arrays)
    if sorted(held_out_indices.tolist()) != list(range(support_text.shape[0])):
        raise ValueError("folds must hold out every support row exactly once")

    all_indices = np.arange(support_text.shape[0], dtype=np.int64)
    oof_logits = np.empty_like(support_text, dtype=np.float64)
    fold_parameters: list[list[dict[str, float]]] = []
    for held_out in fold_arrays:
        train_mask = np.ones(support_text.shape[0], dtype=bool)
        train_mask[held_out] = False
        fold_logits, parameters = _platt_decision_scores(
            support_text[all_indices[train_mask]],
            labels[all_indices[train_mask]],
            support_text[held_out],
        )
        oof_logits[held_out] = fold_logits
        fold_parameters.append(parameters)

    oof_visual_scale = max(
        float(np.median(np.std(visual_oof, axis=1, ddof=0))), epsilon_value
    )
    oof_residual = _row_standardize(oof_logits, epsilon=epsilon_value) * oof_visual_scale
    query_logits, final_parameters = _platt_decision_scores(
        support_text, labels, query_text
    )
    final_visual_scale = max(
        float(np.median(np.std(visual_support, axis=1, ddof=0))), epsilon_value
    )
    query_residual = _row_standardize(query_logits, epsilon=epsilon_value) * final_visual_scale
    final_calibration = SemanticCalibration(
        class_bias=np.zeros(support_text.shape[1], dtype=np.float64),
        visual_scale=final_visual_scale,
        epsilon=epsilon_value,
    )
    return SemanticScoreCalibration(
        oof_residual=oof_residual,
        query_residual=query_residual,
        fold_class_bias=tuple(
            np.asarray([item["intercept"] for item in parameters], dtype=np.float64)
            for parameters in fold_parameters
        ),
        final_calibration=final_calibration,
        metadata={
            "strategy": "support_crossfit_per_class_platt",
            "query_statistics_used": False,
            "oof_count": int(held_out_indices.size),
            "folds": [fold.tolist() for fold in fold_arrays],
            "fold_parameters": fold_parameters,
            "final_parameters": final_parameters,
            "oof_visual_scale": oof_visual_scale,
            "final_visual_scale": final_visual_scale,
            "epsilon": epsilon_value,
        },
    )


def _validated_cache(cache: TaskReliabilityCache) -> TaskReliabilityCache:
    class_names = tuple(str(name) for name in cache.class_names)
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("cache class_names must be non-empty and unique")
    class_count = len(class_names)
    support_labels = np.asarray(cache.support_labels, dtype=np.int64)
    query_labels = np.asarray(cache.query_labels, dtype=np.int64)
    visual_support = _finite_matrix(
        cache.visual_support_scores, name="visual_support_scores"
    )
    visual_query = _finite_matrix(
        cache.visual_query_scores, name="visual_query_scores"
    )
    semantic_query_raw = _finite_matrix(
        cache.semantic_query_raw, name="semantic_query_raw"
    )
    semantic_query_residual = _finite_matrix(
        cache.semantic_query_residual, name="semantic_query_residual"
    )
    if visual_support.shape[1] != class_count or visual_query.shape[1] != class_count:
        raise ValueError("cache visual score class dimension does not match class_names")
    if support_labels.shape != (visual_support.shape[0],):
        raise ValueError("support_labels do not match visual_support_scores")
    if query_labels.shape != (visual_query.shape[0],):
        raise ValueError("query_labels do not match visual_query_scores")
    if semantic_query_raw.shape != visual_query.shape:
        raise ValueError("semantic_query_raw must match visual_query_scores")
    if semantic_query_residual.shape != visual_query.shape:
        raise ValueError("semantic_query_residual must match visual_query_scores")
    if np.any(support_labels < 0) or np.any(support_labels >= class_count):
        raise ValueError("support_labels contain an out-of-range class")
    if np.any(query_labels < 0) or np.any(query_labels >= class_count):
        raise ValueError("query_labels contain an out-of-range class")
    optional = (
        ("visual_oof_scores", cache.visual_oof_scores),
        ("semantic_oof_raw", cache.semantic_oof_raw),
        ("semantic_oof_residual", cache.semantic_oof_residual),
    )
    for name, values in optional:
        if values is not None and _finite_matrix(values, name=name).shape != visual_support.shape:
            raise ValueError(f"{name} must match visual_support_scores")
    if int(cache.shot) == 1:
        if any(values is not None for _, values in optional):
            raise ValueError("1-shot cache must not contain reliability OOF scores")
    elif any(values is None for _, values in optional):
        raise ValueError("5-shot/10-shot cache requires all reliability OOF scores")
    expected_anchor_mode = (
        "visual_prototype"
        if (
            cache.protocol in {
                "wm811k_official_8way_v1",
                "wm811k_dedup_8way_v2",
            }
            and cache.query_source == "Test"
            and int(cache.shot) == 1
        )
        else anchor_mode_for_shot(int(cache.shot))
    )
    if cache.anchor_mode != expected_anchor_mode:
        raise ValueError("cache anchor_mode does not match the frozen shot policy")
    if not str(cache.protocol).strip() or not str(cache.manifest_sha256).strip():
        raise ValueError("cache protocol and manifest hash must not be blank")
    if not str(cache.semantic_bundle_sha256).strip():
        raise ValueError("cache semantic bundle hash must not be blank")
    if not str(cache.support_source).strip() or not str(cache.query_source).strip():
        raise ValueError("cache support/query sources must not be blank")
    if cache.protocol == "wm811k_dedup_8way_v2":
        provenance = dict(cache.provenance or {})
        required_provenance = (
            "dedup_audit_sha256",
            "candidate_pool_sha256",
            "frozen_config_sha256",
            "visual_encoder",
            "preprocess",
            "augmentation",
            "clean_feature_set_sha256",
            "augmented_feature_set_sha256",
        )
        missing = [name for name in required_provenance if not provenance.get(name)]
        if missing:
            raise ValueError(f"v2 cache provenance is missing: {missing}")
        fold_hash = provenance.get("fold_assignment_sha256")
        if int(cache.shot) == 1:
            if fold_hash is not None:
                raise ValueError("v2 one-shot cache must not have an OOF fold hash")
        elif not isinstance(fold_hash, str) or len(fold_hash) != 64:
            raise ValueError("v2 cache provenance has an invalid OOF fold hash")
    return cache


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_task_cache(
    output_directory: str | Path,
    cache: TaskReliabilityCache,
    *,
    stem: str | None = None,
) -> Path:
    """Atomically write one cache as JSON metadata plus compressed NPZ arrays."""

    _validated_cache(cache)
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_stem = stem or f"{int(cache.shot)}shot_seed{int(cache.seed)}"
    if not resolved_stem or Path(resolved_stem).name != resolved_stem:
        raise ValueError("cache stem must be one plain file name")
    metadata_path = output_root / f"{resolved_stem}.json"
    array_path = metadata_path.with_suffix(".npz")
    arrays = {
        "support_labels": np.asarray(cache.support_labels, dtype=np.int64),
        "query_labels": np.asarray(cache.query_labels, dtype=np.int64),
        "visual_support_scores": np.asarray(cache.visual_support_scores),
        "visual_query_scores": np.asarray(cache.visual_query_scores),
        "semantic_query_raw": np.asarray(cache.semantic_query_raw),
        "semantic_query_residual": np.asarray(cache.semantic_query_residual),
    }
    optional_arrays = {
        "visual_oof_scores": cache.visual_oof_scores,
        "semantic_oof_raw": cache.semantic_oof_raw,
        "semantic_oof_residual": cache.semantic_oof_residual,
    }
    for name, values in optional_arrays.items():
        if values is not None:
            arrays[name] = np.asarray(values)
    metadata = {
        "schema_version": "class_reliability_task_cache_v1",
        "protocol": cache.protocol,
        "shot": int(cache.shot),
        "seed": int(cache.seed),
        "manifest_sha256": cache.manifest_sha256,
        "semantic_bundle_sha256": cache.semantic_bundle_sha256,
        "class_names": list(cache.class_names),
        "support_source": cache.support_source,
        "query_source": cache.query_source,
        "anchor_mode": cache.anchor_mode,
        "anchor_selection": cache.anchor_selection,
        "calibration_metadata": cache.calibration_metadata,
        "runtime_metadata": cache.runtime_metadata or {},
        "provenance": cache.provenance or {},
        "array_file": array_path.name,
        "optional_arrays_present": {
            name: values is not None for name, values in optional_arrays.items()
        },
    }
    _atomic_write_npz(array_path, arrays)
    _atomic_write_json(metadata_path, metadata)
    return metadata_path


def load_task_cache(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_semantic_bundle_sha256: str | None = None,
) -> TaskReliabilityCache:
    """Load and validate one task cache and its declared provenance hashes."""

    metadata_path = Path(path)
    if metadata_path.suffix == ".npz":
        metadata_path = metadata_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"cache metadata does not exist: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != "class_reliability_task_cache_v1":
        raise ValueError("unsupported task cache schema")
    manifest_hash = str(metadata.get("manifest_sha256", ""))
    semantic_hash = str(metadata.get("semantic_bundle_sha256", ""))
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise ValueError("task cache manifest hash does not match")
    if expected_semantic_bundle_sha256 is not None and semantic_hash != expected_semantic_bundle_sha256:
        raise ValueError("task cache semantic bundle hash does not match")
    array_path = metadata_path.parent / str(metadata.get("array_file", ""))
    if not array_path.is_file():
        raise FileNotFoundError(f"cache array file does not exist: {array_path}")
    present = dict(metadata.get("optional_arrays_present", {}))
    with np.load(array_path, allow_pickle=False) as arrays:
        def optional(name: str) -> np.ndarray | None:
            return np.asarray(arrays[name]) if bool(present.get(name)) else None

        cache = TaskReliabilityCache(
            protocol=str(metadata.get("protocol", "")),
            shot=int(metadata["shot"]),
            seed=int(metadata["seed"]),
            manifest_sha256=manifest_hash,
            semantic_bundle_sha256=semantic_hash,
            class_names=tuple(str(name) for name in metadata.get("class_names", [])),
            support_source=str(metadata.get("support_source", "")),
            query_source=str(metadata.get("query_source", "")),
            support_labels=np.asarray(arrays["support_labels"], dtype=np.int64),
            query_labels=np.asarray(arrays["query_labels"], dtype=np.int64),
            visual_support_scores=np.asarray(arrays["visual_support_scores"]),
            visual_oof_scores=optional("visual_oof_scores"),
            visual_query_scores=np.asarray(arrays["visual_query_scores"]),
            semantic_oof_raw=optional("semantic_oof_raw"),
            semantic_oof_residual=optional("semantic_oof_residual"),
            semantic_query_raw=np.asarray(arrays["semantic_query_raw"]),
            semantic_query_residual=np.asarray(arrays["semantic_query_residual"]),
            anchor_mode=str(metadata.get("anchor_mode", "")),
            anchor_selection=dict(metadata.get("anchor_selection", {})),
            calibration_metadata=dict(metadata.get("calibration_metadata", {})),
            runtime_metadata=dict(metadata.get("runtime_metadata", {})),
            provenance=dict(metadata.get("provenance", {})),
        )
    return _validated_cache(cache)


def _reliability_selection_payload(fitted: Any, reliability: np.ndarray) -> dict[str, Any]:
    payload = {
        "learned_reliability": fitted.learned_reliability.tolist(),
        "effective_reliability": reliability.tolist(),
        "enabled": bool(fitted.enabled),
        "base_support_macro_f1": float(fitted.base_support_macro_f1),
        "fused_support_macro_f1": float(fitted.fused_support_macro_f1),
        "loss_start": fitted.loss_start,
        "loss_end": fitted.loss_end,
        "steps": int(fitted.steps),
        "numerical_fallback": bool(fitted.numerical_fallback),
        "granularity": fitted.granularity,
    }
    payload["nonzero_class_fraction"] = float(np.mean(reliability > 0.0))
    return payload


def evaluate_cached_task(
    cache: TaskReliabilityCache,
    *,
    lambda_value: float,
    rho: float,
    granularity: Literal["classwise", "global", "fixed", "grid"],
    calibration: bool | Literal["support_bias_zscore", "none", "platt", "temperature"] = True,
    safe_fallback: bool = True,
    fusion_mode: Literal["residual", "direct_weighted"] = "residual",
) -> dict[str, Any]:
    """Evaluate one declared reliability setting from immutable cached scores."""

    _validated_cache(cache)
    if granularity not in ("classwise", "global", "fixed", "grid"):
        raise ValueError("granularity must be classwise, global, or fixed")
    if fusion_mode not in ("residual", "direct_weighted"):
        raise ValueError("fusion_mode must be residual or direct_weighted")
    calibration_method = (
        "support_bias_zscore"
        if calibration is True
        else "none"
        if calibration is False
        else str(calibration)
    )
    if calibration_method not in ("support_bias_zscore", "none", "platt", "temperature"):
        raise ValueError("unsupported calibration method")
    visual = np.asarray(cache.visual_query_scores)
    class_count = len(cache.class_names)
    if int(cache.shot) == 1:
        reliability = np.zeros(class_count, dtype=np.float64)
        fused = visual.copy()
        selection: dict[str, Any] = {
            "enabled": False,
            "reason": ONE_SHOT_RELIABILITY_REASON,
            "learned_reliability": reliability.tolist(),
            "effective_reliability": reliability.tolist(),
            "nonzero_class_fraction": 0.0,
        }
    else:
        if calibration_method == "support_bias_zscore":
            oof_residual = cache.semantic_oof_residual
            query_residual = cache.semantic_query_residual
        elif calibration_method == "none":
            oof_residual = cache.semantic_oof_raw
            query_residual = cache.semantic_query_raw
        elif calibration_method == "temperature":
            if cache.semantic_oof_raw is None or cache.visual_oof_scores is None:
                raise ValueError("temperature calibration requires support OOF scores")
            folds = build_stratified_support_folds(cache.support_labels, num_classes=class_count)
            temperature = calibrate_semantic_scores_temperature_support_only(
                cache.semantic_oof_raw,
                cache.semantic_query_raw,
                support_labels=cache.support_labels,
                visual_support_scores=cache.visual_support_scores,
                visual_oof_scores=cache.visual_oof_scores,
                folds=folds,
            )
            oof_residual = temperature.oof_residual
            query_residual = temperature.query_residual
        else:
            if cache.semantic_oof_raw is None or cache.visual_oof_scores is None:
                raise ValueError("Platt calibration requires support OOF scores")
            folds = build_stratified_support_folds(
                cache.support_labels,
                num_classes=class_count,
            )
            platt = calibrate_semantic_scores_platt_support_only(
                cache.semantic_oof_raw,
                cache.semantic_query_raw,
                support_labels=cache.support_labels,
                visual_support_scores=cache.visual_support_scores,
                visual_oof_scores=cache.visual_oof_scores,
                folds=folds,
            )
            oof_residual = platt.oof_residual
            query_residual = platt.query_residual
        if oof_residual is None or query_residual is None or cache.visual_oof_scores is None:
            raise ValueError("non-one-shot evaluation requires complete OOF scores")
        if granularity == "fixed":
            reliability = np.ones(class_count, dtype=np.float64)
            selection = {
                "enabled": True,
                "reason": "fixed_all_one_ablation",
                "learned_reliability": reliability.tolist(),
                "effective_reliability": reliability.tolist(),
                "nonzero_class_fraction": 1.0,
            }
        else:
            if granularity == "grid":
                fitted = fit_reliability_closed_form(
                    cache.visual_oof_scores,
                    oof_residual,
                    cache.support_labels,
                    class_names=cache.class_names,
                    lambda_value=lambda_value,
                    rho=rho,
                    fusion_mode=fusion_mode,
                )
            else:
                fitted = fit_reliability(
                    cache.visual_oof_scores,
                    oof_residual,
                    cache.support_labels,
                    class_names=cache.class_names,
                    lambda_value=lambda_value,
                    rho=rho,
                    granularity=granularity,
                    fusion_mode=fusion_mode,
                )
            reliability = (
                fitted.effective_reliability
                if safe_fallback
                else fitted.learned_reliability
            )
            selection = _reliability_selection_payload(fitted, reliability)
            selection["reason"] = (
                "support_oof_macro_f1_strict_improvement"
                if fitted.enabled
                else "support_oof_safe_fallback"
            )
        if fusion_mode == "residual":
            fused = apply_classwise_residual(
                visual,
                query_residual,
                reliability,
                lambda_value=lambda_value,
            )
        else:
            fused = apply_direct_weighted_fusion(
                visual,
                query_residual,
                reliability,
                lambda_value=lambda_value,
            )
    predictions = np.argmax(fused, axis=1).astype(np.int64)
    anchor_predictions = np.argmax(visual, axis=1).astype(np.int64)
    return {
        "schema_version": "class_reliability_task_result_v1",
        "protocol": cache.protocol,
        "shot": int(cache.shot),
        "seed": int(cache.seed),
        "support_source": cache.support_source,
        "query_source": cache.query_source,
        "anchor_mode": cache.anchor_mode,
        "lambda": float(lambda_value),
        "rho": float(rho),
        "granularity": granularity,
        "calibration": calibration_method != "none",
        "calibration_method": calibration_method,
        "safe_fallback": bool(safe_fallback),
        "fusion_mode": fusion_mode,
        "query_labels_used_for_selection": False,
        "selection": selection,
        "anchor_metrics": evaluate_predictions(
            cache.query_labels, anchor_predictions, cache.class_names
        ),
        "metrics": evaluate_predictions(
            cache.query_labels, predictions, cache.class_names
        ),
        "predictions": predictions.tolist(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicated_task_records(
    tasks: Sequence[FewShotTask],
    query_by_task: dict[tuple[int, int], tuple[ImageRecord, ...]],
) -> list[ImageRecord]:
    records: OrderedDict[str, ImageRecord] = OrderedDict()
    for task in tasks:
        for record in (*task.support, *query_by_task[(task.shot, task.seed)]):
            existing = records.get(record.relative_path)
            if existing is not None and existing.label_index != record.label_index:
                raise ValueError(
                    f"image path has conflicting labels: {record.relative_path}"
                )
            records.setdefault(record.relative_path, record)
    return list(records.values())


def _cosine_scores(features: Any, prototypes: Any) -> np.ndarray:
    feature_rows = l2_normalize(_finite_matrix(features, name="features"))
    prototype_rows = l2_normalize(_finite_matrix(prototypes, name="prototypes"))
    if feature_rows.shape[1] != prototype_rows.shape[1]:
        raise ValueError("feature and prototype dimensions must match")
    return np.asarray(feature_rows @ prototype_rows.T, dtype=np.float64)


def _ordered_identity_sha256(values: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(values),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def run_reliability_cache_tasks(
    tasks: Sequence[FewShotTask],
    *,
    encoder: Any,
    semantic_prompt_groups: Sequence[Sequence[str]],
    semantic_source: str,
    semantic_bundle_sha256: str,
    output_root: str | Path,
    batch_size: int,
    limit_query: int | None,
    aggregate: bool,
    formal_seeds: Sequence[int] | None = None,
    show_progress: bool = True,
    support_augment_views: int = 10,
    tip_f_max_epochs: int = 20,
    reference_visual_cache_root: str | Path | None = None,
    linear_selection_by_task: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
    required_query_source: str = "Validation",
    dataset: str = "fgvc_aircraft",
    anchor_prompt_source: str = "frozen_aircraft_generic_class_prompt",
    one_shot_anchor_mode: str | None = None,
    cache_provenance: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Encode once and build immutable support-only reliability caches."""

    task_tuple = tuple(tasks)
    validate_task_grid(
        task_tuple, require_formal_grid=aggregate, formal_seeds=formal_seeds
    )
    if aggregate and limit_query is not None:
        raise ValueError("debug query limits cannot be used for formal caches")
    required_query = str(required_query_source).strip()
    if not required_query:
        raise ValueError("required_query_source must not be blank")
    if any(task.query_source != required_query for task in task_tuple):
        raise ValueError(f"reliability caches require {required_query} queries")
    groups = tuple(
        tuple(str(prompt).strip() for prompt in group)
        for group in semantic_prompt_groups
    )
    if len(groups) != task_tuple[0].n_way or any(not group for group in groups):
        raise ValueError("semantic prompt groups must match the task class count")
    if any(not prompt for group in groups for prompt in group):
        raise ValueError("semantic prompt groups must not contain blank prompts")
    if not str(semantic_source).strip() or not str(semantic_bundle_sha256).strip():
        raise ValueError("semantic source and bundle hash must not be blank")
    provenance_base = dict(cache_provenance or {})
    v2_tasks = tuple(
        task
        for task in task_tuple
        if task.protocol_name == "wm811k_dedup_8way_v2"
    )
    if v2_tasks:
        required_provenance = (
            "dedup_audit_sha256",
            "candidate_pool_sha256",
            "frozen_config_sha256",
            "visual_encoder",
            "preprocess",
            "augmentation",
        )
        missing = [name for name in required_provenance if not provenance_base.get(name)]
        if missing:
            raise ValueError(f"WM-811K v2 cache provenance is missing: {missing}")
        for task in v2_tasks:
            if task.dedup_audit_hash != provenance_base["dedup_audit_sha256"]:
                raise ValueError("WM-811K v2 task audit hash does not match cache provenance")
            if task.candidate_pool_hash != provenance_base["candidate_pool_sha256"]:
                raise ValueError("WM-811K v2 task candidate pool hash does not match cache provenance")

    query_by_task: dict[tuple[int, int], tuple[ImageRecord, ...]] = {}
    for task in task_tuple:
        query_by_task[(task.shot, task.seed)] = (
            select_debug_query_subset(
                task.query,
                limit=limit_query,
                num_classes=task.n_way,
            )
            if limit_query is not None
            else task.query
        )
    unique_records = _deduplicated_task_records(task_tuple, query_by_task)
    feature_started = time.perf_counter()
    features_by_path = encoder.encode_images(
        unique_records,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    feature_seconds = time.perf_counter() - feature_started
    expected_paths = {record.relative_path for record in unique_records}
    if set(features_by_path) != expected_paths:
        raise ValueError("clean CLIP feature paths do not match requested records")
    first_feature = np.asarray(next(iter(features_by_path.values())))
    if first_feature.ndim != 1:
        raise ValueError("each CLIP image feature must be one row vector")

    reference_root = (
        None
        if reference_visual_cache_root is None
        else Path(reference_visual_cache_root)
    )
    anchor_text_prototypes = None
    anchor_prompts: tuple[str, ...] = ()
    anchor_text_seconds = 0.0
    if reference_root is None:
        anchor_text_started = time.perf_counter()
        anchor_text_prototypes, anchor_prompts = encoder.encode_class_text_prototypes(
            task_tuple[0].class_names
        )
        anchor_text_seconds = time.perf_counter() - anchor_text_started
    semantic_text_started = time.perf_counter()
    semantic_prototypes, semantic_prompts = encoder.encode_prompt_groups(groups)
    semantic_text_seconds = time.perf_counter() - semantic_text_started
    expected_shape = (task_tuple[0].n_way, first_feature.shape[0])
    if anchor_text_prototypes is not None and np.asarray(anchor_text_prototypes).shape != expected_shape:
        raise ValueError("anchor text prototypes have the wrong shape")
    if np.asarray(semantic_prototypes).shape != expected_shape:
        raise ValueError("semantic text prototypes have the wrong shape")
    shared_runtime = dict(encoder.runtime_metadata(int(first_feature.shape[0])))

    output_directory = Path(output_root)
    paths: list[Path] = []
    for task in task_tuple:
        task_started = time.perf_counter()
        query_records = query_by_task[(task.shot, task.seed)]
        clean_support = np.stack(
            [features_by_path[record.relative_path] for record in task.support]
        )
        query_features = np.stack(
            [features_by_path[record.relative_path] for record in query_records]
        )
        support_labels = np.asarray(
            [record.label_index for record in task.support], dtype=np.int64
        )
        support_group_ids = (
            [str(record.content_hash) for record in task.support]
            if task.protocol_name == "wm811k_dedup_8way_v2"
            else None
        )
        if support_group_ids is not None and any(
            not value for value in support_group_ids
        ):
            raise ValueError("WM-811K v2 support records require content hashes")
        query_labels = np.asarray(
            [record.label_index for record in query_records], dtype=np.int64
        )
        debug_run = limit_query is not None
        stem = f"{task.n_way}way_{task.shot}shot_seed{task.seed}"
        if debug_run:
            stem += f"_debug_q{len(query_records)}"
        manifest_sha256 = _sha256_file(task.manifest_path)
        augmentation_seed = 20260816 + task.shot * 100 + task.seed
        clean_feature_set_sha256: str | None = None
        augmented_feature_set_sha256: str | None = None
        if task.protocol_name == "wm811k_dedup_8way_v2":
            ordered_clean_identities: list[dict[str, Any]] = []
            task_records = (
                ("support", task.support),
                ("query", query_records),
            )
            for role, records in task_records:
                for order, record in enumerate(records):
                    if not record.content_hash:
                        raise ValueError("WM-811K v2 feature identity requires content hashes")
                    ordered_clean_identities.append(
                        {
                            "role": role,
                            "order": order,
                            "identity": clean_feature_identity(
                                content_hash=record.content_hash,
                                model=str(provenance_base["visual_encoder"]),
                                weights=str(shared_runtime.get("pretrained", "openai")),
                                preprocess=str(provenance_base["preprocess"]),
                                dtype="float32",
                            ),
                        }
                    )
            clean_feature_set_sha256 = _ordered_identity_sha256(
                ordered_clean_identities
            )
            augmented_identities: list[dict[str, Any]] = []
            for view_index in range(int(support_augment_views)):
                for support_order, record in enumerate(task.support):
                    base_identity = clean_feature_identity(
                        content_hash=str(record.content_hash),
                        model=str(provenance_base["visual_encoder"]),
                        weights=str(shared_runtime.get("pretrained", "openai")),
                        preprocess=str(provenance_base["preprocess"]),
                        dtype="float32",
                    )
                    augmented_identities.append(
                        {
                            "view": view_index,
                            "support_order": support_order,
                            "identity": augmented_feature_identity(
                                base=base_identity,
                                implementation=str(provenance_base["augmentation"]),
                                seed=augmentation_seed,
                                view=view_index,
                                record_index=int(record.original_index),
                                order=support_order,
                            ),
                        }
                    )
            augmented_feature_set_sha256 = _ordered_identity_sha256(
                augmented_identities
            )
        if reference_root is None:
            augmentation_started = time.perf_counter()
            augmented_support = encoder.encode_augmented_images(
                task.support,
                batch_size=batch_size,
                view_count=support_augment_views,
                seed=augmentation_seed,
                show_progress=show_progress,
            )
            augmentation_seconds = time.perf_counter() - augmentation_started
            anchor_started = time.perf_counter()
            if anchor_text_prototypes is None:
                raise RuntimeError("anchor text prototypes were not initialized")
            reused_selection = (
                None
                if linear_selection_by_task is None
                else linear_selection_by_task.get((task.shot, task.seed))
            )
            anchor = fit_visual_anchor(
                clean_support,
                augmented_support,
                support_labels,
                anchor_text_prototypes,
                num_classes=task.n_way,
                shot=task.shot,
                device=encoder.device,
                tip_f_max_epochs=tip_f_max_epochs,
                selected_c=(
                    None
                    if reused_selection is None
                    else float(reused_selection["C"])
                ),
                selection_provenance=reused_selection,
                anchor_mode=(
                    one_shot_anchor_mode
                    if int(task.shot) == 1 and one_shot_anchor_mode is not None
                    else None
                ),
                support_group_ids=support_group_ids,
            )
            anchor_seconds = time.perf_counter() - anchor_started
            visual_support_scores = anchor.final_scores_on_support
            visual_oof_scores = anchor.oof_scores
            visual_query_scores = anchor.decision_scores(query_features)
            anchor_mode = anchor.mode
            anchor_selection = anchor.selection
            reference_path = None
        else:
            reference_path = reference_root / f"{stem}.json"
            reference = load_task_cache(
                reference_path,
                expected_manifest_sha256=manifest_sha256,
            )
            if reference.class_names != task.class_names:
                raise ValueError("reference visual cache class order does not match")
            if reference.support_source != task.support_source or reference.query_source != task.query_source:
                raise ValueError("reference visual cache split sources do not match")
            if not np.array_equal(reference.support_labels, support_labels):
                raise ValueError("reference visual cache support labels do not match")
            if not np.array_equal(reference.query_labels, query_labels):
                raise ValueError("reference visual cache query labels do not match")
            if bool((reference.runtime_metadata or {}).get("debug_run")) != debug_run:
                raise ValueError("reference visual cache debug status does not match")
            visual_support_scores = reference.visual_support_scores.copy()
            visual_oof_scores = (
                None
                if reference.visual_oof_scores is None
                else reference.visual_oof_scores.copy()
            )
            visual_query_scores = reference.visual_query_scores.copy()
            anchor_mode = reference.anchor_mode
            anchor_selection = dict(reference.anchor_selection)
            augmentation_seconds = 0.0
            anchor_seconds = 0.0
        task_provenance: dict[str, Any] | None = None
        if task.protocol_name == "wm811k_dedup_8way_v2":
            fold_assignment_sha256 = (
                None
                if int(task.shot) == 1
                else anchor_selection.get("oof_fold_assignment_sha256")
            )
            task_provenance = {
                **provenance_base,
                "dedup_audit_sha256": task.dedup_audit_hash,
                "candidate_pool_sha256": task.candidate_pool_hash,
                "fold_assignment_sha256": fold_assignment_sha256,
                "clean_feature_set_sha256": clean_feature_set_sha256,
                "augmented_feature_set_sha256": augmented_feature_set_sha256,
            }
            if reference_root is not None:
                reference_provenance = dict(reference.provenance or {})
                mismatched = [
                    name
                    for name, value in task_provenance.items()
                    if reference_provenance.get(name) != value
                ]
                if mismatched:
                    raise ValueError(
                        "reference visual cache provenance does not match: "
                        f"{mismatched}"
                    )
        semantic_support_raw = _cosine_scores(
            clean_support, semantic_prototypes
        )
        semantic_query_raw = _cosine_scores(query_features, semantic_prototypes)
        if task.shot == 1:
            final_calibration = fit_semantic_calibration(
                semantic_support_raw,
                visual_support_scores,
            )
            semantic_query_residual = apply_semantic_calibration(
                semantic_query_raw,
                final_calibration,
            )
            semantic_oof_raw = None
            semantic_oof_residual = None
            calibration_metadata = {
                "strategy": "one_shot_full_support_query_calibration_only",
                "query_statistics_used": False,
                "reliability_learning_disabled": True,
                "final_class_bias": final_calibration.class_bias.tolist(),
                "final_visual_scale": float(final_calibration.visual_scale),
                "epsilon": float(final_calibration.epsilon),
            }
        else:
            if visual_oof_scores is None:
                raise ValueError("non-one-shot visual anchor is missing OOF scores")
            folds = (
                build_stratified_support_folds(
                    support_labels,
                    num_classes=task.n_way,
                )
                if support_group_ids is None
                else build_group_stratified_support_folds(
                    support_labels,
                    support_group_ids,
                    num_classes=task.n_way,
                )
            )
            calibrated = calibrate_semantic_scores_support_only(
                semantic_support_raw,
                semantic_query_raw,
                visual_support_scores=visual_support_scores,
                visual_oof_scores=visual_oof_scores,
                folds=folds,
            )
            semantic_oof_raw = semantic_support_raw
            semantic_oof_residual = calibrated.oof_residual
            semantic_query_residual = calibrated.query_residual
            calibration_metadata = calibrated.metadata
            if support_group_ids is not None:
                calibration_metadata = {
                    **calibration_metadata,
                    "oof_group_aware": True,
                    "oof_group_ids": list(support_group_ids),
                    "oof_fold_group_ids": [
                        [support_group_ids[int(index)] for index in fold]
                        for fold in folds
                    ],
                    "oof_fold_assignment_sha256": anchor_selection[
                        "oof_fold_assignment_sha256"
                    ],
                    "oof_each_support_held_out_once": True,
                }

        runtime_metadata = {
            **shared_runtime,
            "semantic_source": str(semantic_source),
            "semantic_prompts": list(semantic_prompts),
            "semantic_prompts_per_class": [len(group) for group in groups],
            "anchor_prompt_source": str(anchor_prompt_source),
            "anchor_prompts": list(anchor_prompts),
            "visual_anchor_reused": reference_root is not None,
            "visual_anchor_reference_cache": (
                None if reference_path is None else reference_path.as_posix()
            ),
            "semantic_score": "cosine_similarity",
            "query_labels_used_for_selection": False,
            "debug_run": debug_run,
            "paper_eligible": bool(aggregate and not debug_run),
            "query_limit": limit_query,
            "query_count": len(query_records),
            "full_query_count": len(task.query),
            "query_records": [
                {
                    "original_index": int(record.original_index),
                    "image_path": record.relative_path,
                    "true_index": int(record.label_index),
                    "true_label": record.label,
                }
                for record in query_records
            ],
            "batch_size": int(batch_size),
            "support_augmentation": {
                "view_count": int(support_augment_views),
                "seed": int(augmentation_seed),
            },
            "timing_seconds": {
                "shared_clean_image_features": feature_seconds,
                "shared_anchor_text": anchor_text_seconds,
                "shared_semantic_text": semantic_text_seconds,
                "support_augmentation": augmentation_seconds,
                "visual_anchor_fit": anchor_seconds,
                "task_total": time.perf_counter() - task_started,
            },
        }
        cache = TaskReliabilityCache(
            protocol=task.protocol_name,
            shot=task.shot,
            seed=task.seed,
            manifest_sha256=manifest_sha256,
            semantic_bundle_sha256=str(semantic_bundle_sha256),
            class_names=task.class_names,
            support_source=task.support_source,
            query_source=task.query_source,
            support_labels=support_labels,
            query_labels=query_labels,
            visual_support_scores=visual_support_scores,
            visual_oof_scores=visual_oof_scores,
            visual_query_scores=visual_query_scores,
            semantic_oof_raw=semantic_oof_raw,
            semantic_oof_residual=semantic_oof_residual,
            semantic_query_raw=semantic_query_raw,
            semantic_query_residual=semantic_query_residual,
            anchor_mode=anchor_mode,
            anchor_selection=anchor_selection,
            calibration_metadata=calibration_metadata,
            runtime_metadata=runtime_metadata,
            provenance=task_provenance,
        )
        paths.append(write_task_cache(output_directory, cache, stem=stem))
    return paths
