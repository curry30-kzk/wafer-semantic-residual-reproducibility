"""Support-only classwise semantic reliability residual mathematics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from .metrics import evaluate_predictions


DEFAULT_EPSILON = 1e-12
DEFAULT_OPTIMIZATION_STEPS = 200
DEFAULT_LEARNING_RATE = 0.05
SUPPORT_IMPROVEMENT_EPSILON = 1e-12


@dataclass(frozen=True)
class SemanticCalibration:
    """Support-estimated text bias and visual score scale."""

    class_bias: np.ndarray
    visual_scale: float
    epsilon: float


@dataclass(frozen=True)
class ReliabilityFit:
    """Learned and safely enabled semantic reliability weights."""

    learned_reliability: np.ndarray
    effective_reliability: np.ndarray
    enabled: bool
    base_support_macro_f1: float
    fused_support_macro_f1: float
    loss_start: float | None
    loss_end: float | None
    steps: int
    numerical_fallback: bool
    granularity: str
    fusion_mode: str


def _finite_score_matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def fit_semantic_calibration(
    support_text_scores: Any,
    support_visual_scores: Any,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> SemanticCalibration:
    """Fit text bias and score scale from support scores only."""

    text = _finite_score_matrix(
        support_text_scores, name="support_text_scores"
    )
    visual = _finite_score_matrix(
        support_visual_scores, name="support_visual_scores"
    )
    if text.shape != visual.shape:
        raise ValueError(
            "support text and visual scores must have an identical shape"
        )
    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    visual_scale = float(np.median(np.std(visual, axis=1, ddof=0)))
    return SemanticCalibration(
        class_bias=text.mean(axis=0),
        visual_scale=max(visual_scale, epsilon_value),
        epsilon=epsilon_value,
    )


def apply_semantic_calibration(
    text_scores: Any, calibration: SemanticCalibration
) -> np.ndarray:
    """Apply support-fitted bias removal, row z-score, and visual scaling."""

    text = _finite_score_matrix(text_scores, name="text_scores")
    bias = np.asarray(calibration.class_bias, dtype=np.float64)
    if bias.ndim != 1 or text.shape[1] != bias.shape[0]:
        raise ValueError("text_scores has the wrong class dimension")
    if not np.isfinite(bias).all():
        raise ValueError("calibration class_bias must be finite")
    visual_scale = float(calibration.visual_scale)
    epsilon = float(calibration.epsilon)
    if not np.isfinite(visual_scale) or visual_scale <= 0.0:
        raise ValueError("calibration visual_scale must be finite and positive")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("calibration epsilon must be finite and positive")

    centered = text - bias[None, :]
    centered -= centered.mean(axis=1, keepdims=True)
    row_scale = np.maximum(
        centered.std(axis=1, keepdims=True, ddof=0), epsilon
    )
    residual = centered / row_scale * visual_scale
    if not np.isfinite(residual).all():
        raise ValueError("calibrated semantic residual must be finite")
    return residual


def apply_classwise_residual(
    visual_scores: Any,
    semantic_residual: Any,
    reliability: Any,
    *,
    lambda_value: float,
) -> np.ndarray:
    """Add one reliability-scaled semantic residual column per class."""

    original = np.asarray(visual_scores)
    visual = _finite_score_matrix(original, name="visual_scores")
    residual = _finite_score_matrix(
        semantic_residual, name="semantic_residual"
    )
    if visual.shape != residual.shape:
        raise ValueError(
            "visual_scores and semantic_residual must have an identical shape"
        )
    weights = np.asarray(reliability, dtype=np.float64)
    if weights.shape != (visual.shape[1],):
        raise ValueError("reliability must have one value per class")
    if not np.isfinite(weights).all():
        raise ValueError("reliability must contain only finite values")
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("reliability values must be in [0, 1]")
    scale = float(lambda_value)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("lambda_value must be finite and non-negative")
    if scale == 0.0 or np.count_nonzero(weights) == 0:
        return original.copy()
    fused = visual + scale * residual * weights[None, :]
    if not np.isfinite(fused).all():
        raise ValueError("fused scores must be finite")
    return fused.astype(original.dtype, copy=False)


def apply_direct_weighted_fusion(
    visual_scores: Any,
    semantic_scores: Any,
    reliability: Any,
    *,
    lambda_value: float = 1.0,
) -> np.ndarray:
    """Interpolate visual and semantic columns with classwise weights."""

    original = np.asarray(visual_scores)
    visual = _finite_score_matrix(original, name="visual_scores")
    semantic = _finite_score_matrix(semantic_scores, name="semantic_scores")
    if visual.shape != semantic.shape:
        raise ValueError(
            "visual_scores and semantic_scores must have an identical shape"
        )
    weights = np.asarray(reliability, dtype=np.float64)
    if weights.shape != (visual.shape[1],):
        raise ValueError("reliability must have one value per class")
    if not np.isfinite(weights).all():
        raise ValueError("reliability must contain only finite values")
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("reliability values must be in [0, 1]")
    scale = float(lambda_value)
    if not np.isfinite(scale) or scale < 0.0 or scale > 1.0:
        raise ValueError("direct fusion lambda_value must be in [0, 1]")
    effective = scale * weights
    if np.count_nonzero(effective) == 0:
        return original.copy()
    fused = (1.0 - effective[None, :]) * visual + effective[None, :] * semantic
    if not np.isfinite(fused).all():
        raise ValueError("fused scores must be finite")
    return fused.astype(original.dtype, copy=False)


def _validated_labels(
    values: Any, *, row_count: int, class_count: int
) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.shape[0] != row_count:
        raise ValueError("labels must match the score row count")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer class indices")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("labels contain a class index outside the score range")
    return labels


def _validated_class_names(
    class_names: Sequence[str], *, class_count: int
) -> tuple[str, ...]:
    names = tuple(str(name) for name in class_names)
    if len(names) != class_count or any(not name.strip() for name in names):
        raise ValueError("class_names must match the score columns")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    return names


def fit_reliability(
    visual_scores: Any,
    semantic_residual: Any,
    labels: Any,
    *,
    class_names: Sequence[str],
    lambda_value: float,
    rho: float,
    granularity: Literal["classwise", "global"],
    fusion_mode: Literal["residual", "direct_weighted"] = "residual",
    steps: int = DEFAULT_OPTIMIZATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> ReliabilityFit:
    """Fit projected reliability weights and apply a support-only fallback."""

    visual = _finite_score_matrix(visual_scores, name="visual_scores")
    residual = _finite_score_matrix(
        semantic_residual, name="semantic_residual"
    )
    if visual.shape != residual.shape:
        raise ValueError(
            "visual_scores and semantic_residual must have an identical shape"
        )
    names = _validated_class_names(
        class_names, class_count=visual.shape[1]
    )
    target = _validated_labels(
        labels,
        row_count=visual.shape[0],
        class_count=visual.shape[1],
    )
    if granularity not in ("classwise", "global"):
        raise ValueError("granularity must be 'classwise' or 'global'")
    if fusion_mode not in ("residual", "direct_weighted"):
        raise ValueError("fusion_mode must be 'residual' or 'direct_weighted'")
    lambda_float = float(lambda_value)
    rho_float = float(rho)
    learning_rate_float = float(learning_rate)
    steps_int = int(steps)
    if not np.isfinite(lambda_float) or lambda_float < 0.0:
        raise ValueError("lambda_value must be finite and non-negative")
    if not np.isfinite(rho_float) or rho_float < 0.0:
        raise ValueError("rho must be finite and non-negative")
    if not np.isfinite(learning_rate_float) or learning_rate_float <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if steps_int <= 0 or steps_int != steps:
        raise ValueError("steps must be a positive integer")

    parameter_count = 1 if granularity == "global" else visual.shape[1]
    parameter = torch.zeros(
        parameter_count,
        dtype=torch.float64,
        device="cpu",
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([parameter], lr=learning_rate_float)
    visual_tensor = torch.from_numpy(visual)
    residual_tensor = torch.from_numpy(residual)
    target_tensor = torch.from_numpy(target)
    loss_values: list[float] = []
    numerical_fallback = False

    for _ in range(steps_int):
        optimizer.zero_grad(set_to_none=True)
        weights = (
            parameter.expand(visual.shape[1])
            if granularity == "global"
            else parameter
        )
        if fusion_mode == "residual":
            logits = visual_tensor + lambda_float * residual_tensor * weights[None, :]
        else:
            effective = lambda_float * weights[None, :]
            logits = (1.0 - effective) * visual_tensor + effective * residual_tensor
        loss = torch.nn.functional.cross_entropy(logits, target_tensor)
        loss = loss + rho_float * torch.mean(parameter.square())
        if not bool(torch.isfinite(loss)):
            numerical_fallback = True
            break
        loss.backward()
        if parameter.grad is None or not bool(
            torch.isfinite(parameter.grad).all()
        ):
            numerical_fallback = True
            break
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(0.0, 1.0)
            if not bool(torch.isfinite(parameter).all()):
                numerical_fallback = True
                break
        loss_values.append(float(loss.detach()))

    learned = parameter.detach().cpu().numpy().astype(np.float64, copy=True)
    if granularity == "global":
        learned = np.repeat(learned, visual.shape[1])
    if numerical_fallback or not np.isfinite(learned).all():
        learned = np.zeros(visual.shape[1], dtype=np.float64)
        numerical_fallback = True

    if fusion_mode == "residual":
        learned_fused = apply_classwise_residual(
            visual,
            residual,
            learned,
            lambda_value=lambda_float,
        )
    else:
        learned_fused = apply_direct_weighted_fusion(
            visual,
            residual,
            learned,
            lambda_value=lambda_float,
        )
    base_predictions = np.argmax(visual, axis=1).astype(np.int64)
    fused_predictions = np.argmax(learned_fused, axis=1).astype(np.int64)
    base_macro_f1 = float(
        evaluate_predictions(target, base_predictions, names)["macro_f1"]
    )
    fused_macro_f1 = float(
        evaluate_predictions(target, fused_predictions, names)["macro_f1"]
    )
    enabled = bool(
        not numerical_fallback
        and fused_macro_f1 > base_macro_f1 + SUPPORT_IMPROVEMENT_EPSILON
    )
    effective = learned.copy() if enabled else np.zeros_like(learned)
    return ReliabilityFit(
        learned_reliability=learned,
        effective_reliability=effective,
        enabled=enabled,
        base_support_macro_f1=base_macro_f1,
        fused_support_macro_f1=fused_macro_f1,
        loss_start=loss_values[0] if loss_values else None,
        loss_end=loss_values[-1] if loss_values else None,
        steps=len(loss_values),
        numerical_fallback=numerical_fallback,
        granularity=granularity,
        fusion_mode=fusion_mode,
    )


def fit_reliability_closed_form(
    visual_scores: Any,
    semantic_residual: Any,
    labels: Any,
    *,
    class_names: Sequence[str],
    lambda_value: float,
    rho: float,
    grid_values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    fusion_mode: Literal["residual", "direct_weighted"] = "residual",
) -> ReliabilityFit:
    """Deterministic coordinate-grid reliability fit using support labels only.

    This is a no-gradient control for the projected Adam optimizer.  Each class
    coordinate is selected by exhaustive one-dimensional search while holding
    the remaining coordinates fixed; ties are resolved toward the smaller
    weight.  The final support-F1 safety rule is identical to ``fit_reliability``.
    """
    visual = _finite_score_matrix(visual_scores, name="visual_scores")
    residual = _finite_score_matrix(semantic_residual, name="semantic_residual")
    if visual.shape != residual.shape:
        raise ValueError("visual_scores and semantic_residual must have an identical shape")
    target = _validated_labels(labels, row_count=visual.shape[0], class_count=visual.shape[1])
    names = _validated_class_names(class_names, class_count=visual.shape[1])
    values = tuple(float(v) for v in grid_values)
    if not values or any((not np.isfinite(v) or v < 0.0 or v > 1.0) for v in values):
        raise ValueError("grid_values must be finite values in [0, 1]")
    if fusion_mode not in ("residual", "direct_weighted"):
        raise ValueError("fusion_mode must be 'residual' or 'direct_weighted'")
    lam = float(lambda_value)
    if not np.isfinite(lam) or lam < 0.0 or (fusion_mode == "direct_weighted" and lam > 1.0):
        raise ValueError("invalid lambda_value")
    penalty = float(rho)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("rho must be finite and non-negative")

    def fused(weights: np.ndarray) -> np.ndarray:
        if fusion_mode == "residual":
            return apply_classwise_residual(visual, residual, weights, lambda_value=lam)
        return apply_direct_weighted_fusion(visual, residual, weights, lambda_value=lam)

    def objective(weights: np.ndarray) -> float:
        logits = fused(weights)
        logits = logits - logits.max(axis=1, keepdims=True)
        log_z = np.log(np.exp(logits).sum(axis=1))
        nll = float(np.mean(-logits[np.arange(target.size), target] + log_z))
        return nll + penalty * float(np.mean(np.square(weights)))

    weights = np.zeros(visual.shape[1], dtype=np.float64)
    start = objective(weights)
    for class_index in range(visual.shape[1]):
        candidates = []
        for value in sorted(set(values)):
            trial = weights.copy()
            trial[class_index] = value
            candidates.append((objective(trial), value))
        _, selected = min(candidates, key=lambda item: (item[0], item[1]))
        weights[class_index] = selected
    learned_fused = fused(weights)
    base_predictions = np.argmax(visual, axis=1).astype(np.int64)
    fused_predictions = np.argmax(learned_fused, axis=1).astype(np.int64)
    base_f1 = float(evaluate_predictions(target, base_predictions, names)["macro_f1"])
    fused_f1 = float(evaluate_predictions(target, fused_predictions, names)["macro_f1"])
    enabled = bool(fused_f1 > base_f1 + SUPPORT_IMPROVEMENT_EPSILON)
    effective = weights.copy() if enabled else np.zeros_like(weights)
    return ReliabilityFit(
        learned_reliability=weights,
        effective_reliability=effective,
        enabled=enabled,
        base_support_macro_f1=base_f1,
        fused_support_macro_f1=fused_f1,
        loss_start=start,
        loss_end=objective(weights),
        steps=visual.shape[1],
        numerical_fallback=False,
        granularity="classwise",
        fusion_mode=fusion_mode,
    )
