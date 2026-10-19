"""Leakage-controlled LP++ for frozen CLIP features.

The objective and data-derived step sizes follow Huang et al., CVPR 2024.
The implementation fixes the update budget before Test and never accepts
query labels, avoiding the Test-time epoch selection present in the public
reference runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .prototype import l2_normalize


LP_PLUS_PLUS_DOI = "10.1109/CVPR52733.2024.02244"
LP_PLUS_PLUS_REPOSITORY = (
    "https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline"
)
LP_PLUS_PLUS_REFERENCE_COMMIT = "f81054f995b060f8348d20e7306e15889b70c786"
LP_PLUS_PLUS_EPOCHS = 300
LP_PLUS_PLUS_W_UPDATES_PER_ALPHA = 10


def _validated_inputs(
    features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    num_classes: int,
    shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visual = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    text = np.asarray(text_prototypes, dtype=np.float32)
    if visual.ndim != 2 or not np.isfinite(visual).all():
        raise ValueError("features must be one finite 2D matrix")
    if targets.shape != (visual.shape[0],):
        raise ValueError("labels must match feature rows")
    if text.shape != (int(num_classes), visual.shape[1]):
        raise ValueError("text_prototypes must have one row per class")
    if not np.isfinite(text).all():
        raise ValueError("text_prototypes contain non-finite values")
    if int(shot) <= 0 or visual.shape[0] != int(num_classes) * int(shot):
        raise ValueError("features must contain exactly shot rows per class")
    counts = np.bincount(targets, minlength=int(num_classes))
    if counts.shape[0] != int(num_classes) or np.any(counts != int(shot)):
        raise ValueError("labels must contain exactly shot rows per class")
    return l2_normalize(visual), targets, l2_normalize(text)


def lp_plus_plus_initialization(
    features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    num_classes: int,
    shot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the data-informed variables from Eq. (33) of LP++."""
    visual, targets, text = _validated_inputs(
        features,
        labels,
        text_prototypes,
        num_classes=num_classes,
        shot=shot,
    )
    weights = np.stack(
        [visual[targets == class_index].sum(axis=0) for class_index in range(num_classes)]
    ).astype(np.float32)
    alpha = np.asarray(
        [
            250.0
            / float(shot)
            * float(np.sum(visual[targets == class_index] @ text[class_index]))
            for class_index in range(num_classes)
        ],
        dtype=np.float32,
    )
    return weights, alpha


def lp_plus_plus_learning_rates(
    features: Any,
    text_prototypes: Any,
) -> tuple[float, float]:
    """Compute the block step sizes from Eqs. (10) and (11)."""
    visual = l2_normalize(np.asarray(features, dtype=np.float32))
    text = l2_normalize(np.asarray(text_prototypes, dtype=np.float32))
    if visual.ndim != 2 or text.ndim != 2 or visual.shape[1] != text.shape[1]:
        raise ValueError("features and text_prototypes must share one feature dimension")
    if not np.isfinite(visual).all() or not np.isfinite(text).all():
        raise ValueError("LP++ inputs must be finite")
    singular_values = np.linalg.svd(visual, compute_uv=False)
    max_eigenvalue = float(singular_values[0] ** 2)
    text_scores = visual @ text.T
    max_text_sum = float(np.max(np.sum(np.square(text_scores), axis=0)))
    epsilon = np.finfo(np.float32).eps
    lr_w = 4.0 * visual.shape[0] / max(max_eigenvalue, epsilon)
    lr_alpha = visual.shape[0] / (4.0 * max(max_text_sum, epsilon))
    return float(lr_w), float(lr_alpha)


@dataclass(frozen=True)
class FittedLPPlusPlus:
    weights: np.ndarray
    alpha: np.ndarray
    text_prototypes: np.ndarray
    selection: dict[str, Any]
    trainable_parameter_count: int

    def decision_scores(self, features: Any) -> np.ndarray:
        visual = l2_normalize(np.asarray(features, dtype=np.float32))
        if visual.ndim != 2 or visual.shape[1] != self.weights.shape[1]:
            raise ValueError("features have the wrong LP++ feature dimension")
        vision_logits = visual @ self.weights.T
        text_logits = visual @ self.text_prototypes.T
        scores = vision_logits + text_logits * self.alpha[None, :]
        if not np.isfinite(scores).all():
            raise ValueError("LP++ produced non-finite scores")
        return scores.astype(np.float32, copy=False)

    def predict(self, features: Any) -> np.ndarray:
        return np.argmax(self.decision_scores(features), axis=1).astype(np.int64)


@dataclass(frozen=True)
class FittedFairLPPlusPlus:
    """LP++ with the public implementation's bias and alpha protocol."""

    weights: np.ndarray
    bias: np.ndarray
    alpha: np.ndarray
    text_prototypes: np.ndarray
    selection: dict[str, Any]
    trainable_parameter_count: int

    def decision_scores(self, features: Any) -> np.ndarray:
        visual = l2_normalize(np.asarray(features, dtype=np.float32))
        if visual.ndim != 2 or visual.shape[1] != self.weights.shape[1]:
            raise ValueError("features have the wrong LP++ feature dimension")
        vision_logits = visual @ self.weights.T + self.bias[None, :]
        text_logits = visual @ self.text_prototypes.T
        scores = vision_logits + text_logits * self.alpha[None, :]
        if not np.isfinite(scores).all():
            raise ValueError("fair LP++ produced non-finite scores")
        return scores.astype(np.float32, copy=False)

    def predict(self, features: Any) -> np.ndarray:
        return np.argmax(self.decision_scores(features), axis=1).astype(np.int64)


def fit_fair_lp_plus_plus_support_only(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    num_classes: int,
    shot: int,
    device: str,
    epochs: int = LP_PLUS_PLUS_EPOCHS,
) -> FittedFairLPPlusPlus:
    """Fit LP++ using the public code's parameter/update protocol.

    The budget is fixed before evaluation. Validation/Test selection from the
    public runner is deliberately excluded so the result remains compatible
    with this project's support-only protocol.
    """
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    clean, targets, text = _validated_inputs(
        clean_features,
        labels,
        text_prototypes,
        num_classes=num_classes,
        shot=shot,
    )
    augmented = np.asarray(augmented_features, dtype=np.float32)
    if augmented.ndim != 3 or augmented.shape[1:] != clean.shape:
        raise ValueError("augmented_features must have shape [views, support, dimension]")
    if not np.isfinite(augmented).all():
        raise ValueError("augmented_features contain non-finite values")

    training = l2_normalize(augmented[0])
    initial_weights, per_class_alpha = lp_plus_plus_initialization(
        training,
        targets,
        text,
        num_classes=num_classes,
        shot=shot,
    )
    shared_alpha = float(np.mean(per_class_alpha))
    lr_w, lr_alpha = lp_plus_plus_learning_rates(training, text)

    import torch

    torch.manual_seed(0)
    resolved = torch.device(device)
    feature_tensor = torch.as_tensor(training, dtype=torch.float32, device=resolved)
    label_tensor = torch.as_tensor(targets, dtype=torch.long, device=resolved)
    text_tensor = torch.as_tensor(text, dtype=torch.float32, device=resolved)
    weights = torch.as_tensor(
        initial_weights, dtype=torch.float32, device=resolved
    ).clone().requires_grad_(True)
    bias = torch.zeros(num_classes, dtype=torch.float32, device=resolved, requires_grad=True)
    alpha = torch.full(
        (num_classes,), shared_alpha, dtype=torch.float32, device=resolved, requires_grad=True
    )
    velocity = torch.zeros_like(weights)
    bias_velocity = torch.zeros_like(bias)
    alpha_gradient_sum = torch.zeros_like(alpha)
    losses: list[float] = []
    for epoch_index in range(int(epochs)):
        logits = feature_tensor @ weights.T + bias[None, :]
        logits = logits + (feature_tensor @ text_tensor.T) * alpha[None, :]
        loss = torch.nn.functional.cross_entropy(logits, label_tensor)
        weight_gradient, bias_gradient, alpha_gradient = torch.autograd.grad(
            loss, (weights, bias, alpha), retain_graph=False
        )
        alpha_gradient_sum.add_(alpha_gradient)
        with torch.no_grad():
            velocity.mul_(0.9).add_(weight_gradient)
            weights.sub_(float(lr_w) * velocity)
            bias_velocity.mul_(0.9).add_(bias_gradient)
            bias.sub_(float(lr_w) * bias_velocity)
        losses.append(float(loss.detach().cpu()))
        if (epoch_index + 1) % LP_PLUS_PLUS_W_UPDATES_PER_ALPHA == 0:
            with torch.no_grad():
                alpha.sub_(float(lr_alpha) * alpha_gradient_sum)
        if not torch.isfinite(weights).all() or not torch.isfinite(bias).all() or not torch.isfinite(alpha).all():
            raise FloatingPointError("fair LP++ optimization produced non-finite parameters")

    fitted_weights = weights.detach().cpu().numpy().astype(np.float32, copy=True)
    fitted_bias = bias.detach().cpu().numpy().astype(np.float32, copy=True)
    fitted_alpha = alpha.detach().cpu().numpy().astype(np.float32, copy=True)
    final_scores = (
        training @ fitted_weights.T
        + fitted_bias[None, :]
        + (training @ text.T) * fitted_alpha[None, :]
    )
    final_predictions = np.argmax(final_scores, axis=1)
    selection = {
        "strategy": "official_protocol_fixed_budget",
        "reason": None,
        "epoch": int(epochs),
        "learning_rate_w": float(lr_w),
        "learning_rate_alpha": float(lr_alpha),
        "alpha_update_period": LP_PLUS_PLUS_W_UPDATES_PER_ALPHA,
        "training_view_index": 0,
        "bias_enabled": True,
        "bias_initialization": "zero_for_reproducible_support_only_protocol",
        "alpha_initialization": "shared_mean",
        "alpha_gradient_strategy": "cumulative_without_reset",
        "support_training_accuracy": float(np.mean(final_predictions == targets)),
        "final_support_loss": float(losses[-1]),
        "initial_alpha_shared": shared_alpha,
        "final_alpha_min": float(np.min(fitted_alpha)),
        "final_alpha_max": float(np.max(fitted_alpha)),
        "query_labels_used_for_selection": False,
        "paper_doi": LP_PLUS_PLUS_DOI,
        "reference_repository": LP_PLUS_PLUS_REPOSITORY,
        "reference_commit": LP_PLUS_PLUS_REFERENCE_COMMIT,
        "protocol_note": (
            "Official bias, shared-alpha initialization, and cumulative alpha "
            "gradients are retained; Validation/Test epoch selection is excluded."
        ),
    }
    return FittedFairLPPlusPlus(
        weights=fitted_weights,
        bias=fitted_bias,
        alpha=fitted_alpha,
        text_prototypes=text,
        selection=selection,
        trainable_parameter_count=int(num_classes * (training.shape[1] + 2)),
    )


def fit_lp_plus_plus_support_only(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    num_classes: int,
    shot: int,
    device: str,
    epochs: int = LP_PLUS_PLUS_EPOCHS,
) -> FittedLPPlusPlus:
    """Fit LP++ with a predeclared budget and support features only."""
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    clean, targets, text = _validated_inputs(
        clean_features,
        labels,
        text_prototypes,
        num_classes=num_classes,
        shot=shot,
    )
    augmented = np.asarray(augmented_features, dtype=np.float32)
    if augmented.ndim != 3 or augmented.shape[1:] != clean.shape:
        raise ValueError("augmented_features must have shape [views, support, dimension]")
    if not np.isfinite(augmented).all():
        raise ValueError("augmented_features contain non-finite values")
    # The public LP++ runner extracts one randomly augmented support view before
    # full-batch optimization. The shared runner supplies deterministic views;
    # the first one is frozen here before any Test evaluation.
    training = l2_normalize(augmented[0])
    initial_weights, initial_alpha = lp_plus_plus_initialization(
        training,
        targets,
        text,
        num_classes=num_classes,
        shot=shot,
    )
    lr_w, lr_alpha = lp_plus_plus_learning_rates(training, text)

    import torch

    resolved = torch.device(device)
    feature_tensor = torch.as_tensor(training, dtype=torch.float32, device=resolved)
    label_tensor = torch.as_tensor(targets, dtype=torch.long, device=resolved)
    text_tensor = torch.as_tensor(text, dtype=torch.float32, device=resolved)
    weights = torch.as_tensor(
        initial_weights, dtype=torch.float32, device=resolved
    ).clone().requires_grad_(True)
    alpha = torch.as_tensor(
        initial_alpha, dtype=torch.float32, device=resolved
    ).clone().requires_grad_(True)
    velocity = torch.zeros_like(weights)
    losses: list[float] = []
    for epoch_index in range(int(epochs)):
        logits = feature_tensor @ weights.T + (feature_tensor @ text_tensor.T) * alpha[None, :]
        loss = torch.nn.functional.cross_entropy(logits, label_tensor)
        weight_gradient = torch.autograd.grad(loss, weights, retain_graph=False)[0]
        with torch.no_grad():
            velocity.mul_(0.9).add_(weight_gradient)
            weights.sub_(float(lr_w) * velocity)
        losses.append(float(loss.detach().cpu()))
        if (epoch_index + 1) % LP_PLUS_PLUS_W_UPDATES_PER_ALPHA == 0:
            alpha_logits = feature_tensor @ weights.detach().T + (feature_tensor @ text_tensor.T) * alpha[None, :]
            alpha_loss = torch.nn.functional.cross_entropy(alpha_logits, label_tensor)
            alpha_gradient = torch.autograd.grad(alpha_loss, alpha, retain_graph=False)[0]
            with torch.no_grad():
                alpha.sub_(float(lr_alpha) * alpha_gradient)
        if not torch.isfinite(weights).all() or not torch.isfinite(alpha).all():
            raise FloatingPointError("LP++ optimization produced non-finite parameters")

    fitted_weights = weights.detach().cpu().numpy().astype(np.float32, copy=True)
    fitted_alpha = alpha.detach().cpu().numpy().astype(np.float32, copy=True)
    final_scores = training @ fitted_weights.T + (training @ text.T) * fitted_alpha[None, :]
    final_predictions = np.argmax(final_scores, axis=1)
    selection = {
        "strategy": "fixed_paper_budget_no_test_selection",
        "reason": None,
        "epoch": int(epochs),
        "learning_rate_w": float(lr_w),
        "learning_rate_alpha": float(lr_alpha),
        "alpha_update_period": LP_PLUS_PLUS_W_UPDATES_PER_ALPHA,
        "training_view_index": 0,
        "support_training_accuracy": float(np.mean(final_predictions == targets)),
        "final_support_loss": float(losses[-1]),
        "initial_alpha_min": float(np.min(initial_alpha)),
        "initial_alpha_max": float(np.max(initial_alpha)),
        "final_alpha_min": float(np.min(fitted_alpha)),
        "final_alpha_max": float(np.max(fitted_alpha)),
        "query_labels_used_for_selection": False,
        "paper_doi": LP_PLUS_PLUS_DOI,
        "reference_repository": LP_PLUS_PLUS_REPOSITORY,
        "reference_commit": LP_PLUS_PLUS_REFERENCE_COMMIT,
        "protocol_note": (
            "The 300-update budget is fixed before Test; the public runner's "
            "Validation-triggered repeated Test evaluation is not reproduced."
        ),
    }
    return FittedLPPlusPlus(
        weights=fitted_weights,
        alpha=fitted_alpha,
        text_prototypes=text,
        selection=selection,
        trainable_parameter_count=int(num_classes * (training.shape[1] + 1)),
    )
