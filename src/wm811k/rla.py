"""Support-only Regularized Linear Adapter for frozen CLIP features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .prototype import l2_normalize


RLA_PAPER_TITLE = "CLIP's Visual Embedding Projector is a Few-shot Cornucopia"
RLA_REPOSITORY = "https://github.com/astra-vision/ProLIP"
RLA_REFERENCE_COMMIT = "b43e795cbf0b53a9b0aa83cf4f47d9755896ba6f"
RLA_EPOCHS = 300
RLA_LEARNING_RATE = 1e-4
RLA_ADAM_EPS = 1e-4
RLA_LOGIT_SCALE = 100.0
RLA_SUPPORT_AUGMENT_VIEWS = 10


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.asarray(values, dtype=np.float32).copy()
    copied.setflags(write=False)
    return copied


def _validate_matrix(name: str, values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(np.linalg.norm(matrix, axis=1) <= 0.0):
        raise ValueError(f"{name} contains a zero-norm row")
    return matrix


def _validated_inputs(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    num_classes: int,
    shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(num_classes) <= 1:
        raise ValueError("num_classes must be greater than one")
    if int(shot) <= 0:
        raise ValueError("shot must be positive")
    clean = _validate_matrix("clean_features", clean_features)
    expected_count = int(num_classes) * int(shot)
    if clean.shape[0] != expected_count:
        raise ValueError("clean_features must contain exactly shot rows per class")
    augmented = np.asarray(augmented_features, dtype=np.float32)
    if augmented.ndim != 3 or augmented.shape[1:] != clean.shape:
        raise ValueError(
            "augmented_features must have shape [views, support, dimension]"
        )
    if augmented.shape[0] <= 0 or not np.isfinite(augmented).all():
        raise ValueError("augmented_features must contain finite support views")
    flat_augmented = augmented.reshape(-1, clean.shape[1])
    if np.any(np.linalg.norm(flat_augmented, axis=1) <= 0.0):
        raise ValueError("augmented_features contains a zero-norm row")
    targets = np.asarray(labels, dtype=np.int64)
    if targets.shape != (expected_count,):
        raise ValueError("labels must match the clean support rows")
    expected_labels = np.repeat(np.arange(int(num_classes)), int(shot))
    if sorted(targets.tolist()) != sorted(expected_labels.tolist()):
        raise ValueError("labels must contain exactly shot samples per class")
    text = _validate_matrix("text_prototypes", text_prototypes)
    if text.shape != (int(num_classes), clean.shape[1]):
        raise ValueError("text_prototypes must have one row per class")
    return (
        l2_normalize(clean),
        l2_normalize(flat_augmented).reshape(augmented.shape),
        targets,
        l2_normalize(text),
    )


@dataclass(frozen=True)
class FittedRLA:
    adapter_weights: np.ndarray
    text_prototypes: np.ndarray
    selection: dict[str, Any]
    trainable_parameter_count: int

    def decision_scores(self, features: Any) -> np.ndarray:
        query = _validate_matrix("features", features)
        dimension = self.adapter_weights.shape[0]
        if self.adapter_weights.shape != (dimension, dimension):
            raise ValueError("adapter_weights must be square")
        if query.shape[1] != dimension:
            raise ValueError("query feature dimension does not match adapter")
        if self.text_prototypes.shape[1] != dimension:
            raise ValueError("text prototype dimension does not match adapter")
        adapted = query @ self.adapter_weights.T
        return (
            RLA_LOGIT_SCALE
            * l2_normalize(adapted)
            @ l2_normalize(self.text_prototypes).T
        ).astype(np.float32, copy=False)

    def predict(self, features: Any) -> np.ndarray:
        return np.argmax(self.decision_scores(features), axis=1).astype(np.int64)


def fit_rla_support_only(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    num_classes: int,
    shot: int,
    device: str,
    seed: int,
    epochs: int = RLA_EPOCHS,
) -> FittedRLA:
    """Fit an identity-initialized visual projector using support labels only."""

    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    clean, augmented, targets, text = _validated_inputs(
        clean_features,
        augmented_features,
        labels,
        text_prototypes,
        num_classes=num_classes,
        shot=shot,
    )

    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    resolved = torch.device(device)
    dimension = int(clean.shape[1])
    augmented_tensor = torch.as_tensor(
        augmented, dtype=torch.float32, device=resolved
    )
    target_tensor = torch.as_tensor(targets, dtype=torch.long, device=resolved)
    text_tensor = torch.as_tensor(text, dtype=torch.float32, device=resolved)
    identity = torch.eye(dimension, dtype=torch.float32, device=resolved)
    adapter = torch.nn.Linear(dimension, dimension, bias=False, device=resolved)
    with torch.no_grad():
        adapter.weight.copy_(identity)
    optimizer = torch.optim.Adam(
        adapter.parameters(), lr=RLA_LEARNING_RATE, eps=RLA_ADAM_EPS
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    regularization_lambda = 1.0 / float(shot)
    epoch_losses: list[float] = []
    for epoch_index in range(int(epochs)):
        view_index = epoch_index % augmented.shape[0]
        adapted = torch.nn.functional.normalize(
            adapter(augmented_tensor[view_index]), dim=1
        )
        logits = RLA_LOGIT_SCALE * adapted @ text_tensor.T
        cross_entropy = torch.nn.functional.cross_entropy(logits, target_tensor)
        regularizer = torch.nn.functional.mse_loss(
            adapter.weight, identity, reduction="sum"
        )
        loss = cross_entropy + regularization_lambda * regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_losses.append(float(loss.detach().cpu()))
        if not torch.isfinite(adapter.weight).all():
            raise FloatingPointError("RLA optimization produced non-finite weights")

    fitted_weights = (
        adapter.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    )
    fitted = FittedRLA(
        adapter_weights=_readonly_copy(fitted_weights),
        text_prototypes=_readonly_copy(text),
        selection={},
        trainable_parameter_count=dimension * dimension,
    )
    support_predictions = fitted.predict(clean)
    selection = {
        "strategy": "fixed_support_only_training_budget",
        "reason": None,
        "epoch": int(epochs),
        "optimizer": "Adam",
        "learning_rate": RLA_LEARNING_RATE,
        "adam_eps": RLA_ADAM_EPS,
        "learning_rate_scheduler": "cosine_annealing",
        "support_batching": "full_batch_one_augmentation_view_per_epoch",
        "support_augmentation_view_count": int(augmented.shape[0]),
        "optimizer_steps": int(epochs),
        "initialization": "identity",
        "regularizer": "frobenius_distance_from_identity",
        "regularization_lambda": regularization_lambda,
        "regularization_lambda_policy": "one_over_shot",
        "logit_scale": RLA_LOGIT_SCALE,
        "training_seed": int(seed),
        "support_training_accuracy": float(
            np.mean(support_predictions == targets)
        ),
        "final_support_loss": float(epoch_losses[-1]),
        "query_labels_used_for_selection": False,
        "paper_title": RLA_PAPER_TITLE,
        "reference_repository": RLA_REPOSITORY,
        "reference_commit": RLA_REFERENCE_COMMIT,
        "protocol_note": (
            "Same-protocol port using the project's ten deterministic support "
            "augmentation views instead of the official 300-view cache."
        ),
    }
    return FittedRLA(
        adapter_weights=fitted.adapter_weights,
        text_prototypes=fitted.text_prototypes,
        selection=selection,
        trainable_parameter_count=fitted.trainable_parameter_count,
    )

