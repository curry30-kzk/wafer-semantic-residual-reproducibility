"""Support-only ZS-LP and CLAP adaptation for frozen CLIP features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .prototype import l2_normalize


CLAP_PAPER_TITLE = (
    "A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models"
)
CLAP_REPOSITORY = "https://github.com/jusiro/CLAP"
CLAP_REFERENCE_COMMIT = "dbdf4095b3d824546474c725539b9f4efff3ca0e"
CLAP_EPOCHS = 300
CLAP_BATCH_SIZE = 256
CLAP_LEARNING_RATE = 0.1
CLAP_MOMENTUM = 0.9
CLAP_WEIGHT_DECAY = 0.0
CLAP_WARMUP_EPOCHS = 1
CLAP_WARMUP_LEARNING_RATE = 1e-5
CLAP_LOGIT_SCALE = 100.0
CLAP_SUPPORT_AUGMENT_VIEWS = 10


def _readonly_copy(values: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.asarray(values, dtype=dtype).copy()
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
    augmented = np.asarray(augmented_features, dtype=np.float32)
    expected_count = int(num_classes) * int(shot)
    if clean.shape[0] != expected_count:
        raise ValueError("clean_features must contain exactly shot rows per class")
    if augmented.ndim != 3 or augmented.shape[1:] != clean.shape:
        raise ValueError(
            "augmented_features must have shape [views, support, dimension]"
        )
    if augmented.shape[0] <= 0 or not np.isfinite(augmented).all():
        raise ValueError("augmented_features must contain finite support views")
    if np.any(np.linalg.norm(augmented, axis=2) <= 0.0):
        raise ValueError("augmented_features contains a zero-norm row")
    targets = np.asarray(labels, dtype=np.int64)
    if targets.shape != (expected_count,):
        raise ValueError("labels must match the clean support rows")
    if np.any(targets < 0) or np.any(targets >= int(num_classes)):
        raise ValueError("labels contain an out-of-range class index")
    counts = np.bincount(targets, minlength=int(num_classes))
    if counts.shape != (int(num_classes),) or np.any(counts != int(shot)):
        raise ValueError("labels must contain exactly shot rows per class")
    text = _validate_matrix("text_prototypes", text_prototypes)
    if text.shape != (int(num_classes), clean.shape[1]):
        raise ValueError("text_prototypes must have one row per class")
    clean = l2_normalize(clean)
    augmented = l2_normalize(augmented.reshape(-1, clean.shape[1])).reshape(
        augmented.shape
    )
    return clean, augmented, targets, l2_normalize(text)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted.astype(np.float64, copy=False))
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def class_confidence_coefficients(
    augmented_features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    logit_scale: float = CLAP_LOGIT_SCALE,
) -> np.ndarray:
    """Return the true-class zero-shot confidence mean for every class."""
    augmented = np.asarray(augmented_features, dtype=np.float32)
    if augmented.ndim == 2:
        augmented = augmented[None, :, :]
    if augmented.ndim != 3 or augmented.shape[0] == 0:
        raise ValueError("augmented_features must be a 2D matrix or 3D view tensor")
    if not np.isfinite(augmented).all():
        raise ValueError("augmented_features contains non-finite values")
    support_labels = np.asarray(labels, dtype=np.int64)
    if support_labels.shape != (augmented.shape[1],):
        raise ValueError("labels must match each augmented support view")
    text = _validate_matrix("text_prototypes", text_prototypes)
    num_classes = int(text.shape[0])
    if np.any(support_labels < 0) or np.any(support_labels >= num_classes):
        raise ValueError("labels contain an out-of-range class index")
    if not math.isfinite(float(logit_scale)) or float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be finite and positive")
    flat = augmented.reshape(-1, augmented.shape[-1])
    if np.any(np.linalg.norm(flat, axis=1) <= 0.0):
        raise ValueError("augmented_features contains a zero-norm row")
    flat = l2_normalize(flat)
    text = l2_normalize(text)
    repeated_labels = np.tile(support_labels, augmented.shape[0])
    counts = np.bincount(repeated_labels, minlength=num_classes)
    if counts.shape != (num_classes,) or np.any(counts == 0):
        raise ValueError("every text class must occur in support labels")
    probabilities = _stable_softmax(float(logit_scale) * flat @ text.T)
    coefficients = np.asarray(
        [
            probabilities[repeated_labels == class_index, class_index].mean()
            for class_index in range(num_classes)
        ],
        dtype=np.float32,
    )
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("class confidence computation produced non-finite values")
    return coefficients


def _learning_rate(epoch_index: int, epochs: int) -> float:
    if epoch_index < CLAP_WARMUP_EPOCHS:
        return CLAP_WARMUP_LEARNING_RATE
    cosine_epochs = max(1, int(epochs) - CLAP_WARMUP_EPOCHS)
    progress = (epoch_index - CLAP_WARMUP_EPOCHS) / cosine_epochs
    return float(
        CLAP_LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


@dataclass(frozen=True)
class FittedCLAP:
    weights: np.ndarray
    class_coefficients: np.ndarray
    selection: dict[str, Any]
    trainable_parameter_count: int

    def decision_scores(self, features: Any) -> np.ndarray:
        query = _validate_matrix("features", features)
        if query.shape[1] != self.weights.shape[1]:
            raise ValueError("query feature dimension does not match fitted weights")
        return (
            CLAP_LOGIT_SCALE
            * l2_normalize(query)
            @ l2_normalize(self.weights).T
        ).astype(np.float32, copy=False)

    def predict(self, features: Any) -> np.ndarray:
        return np.argmax(self.decision_scores(features), axis=1).astype(np.int64)


def fit_clap_support_only(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    num_classes: int,
    shot: int,
    device: str,
    mode: str,
    seed: int,
    epochs: int = CLAP_EPOCHS,
    batch_size: int = CLAP_BATCH_SIZE,
) -> FittedCLAP:
    """Fit text-initialized ZS-LP or class-constrained CLAP on support only."""
    if mode not in ("zs_lp", "clap"):
        raise ValueError("mode must be either 'zs_lp' or 'clap'")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    clean, augmented, targets, text = _validated_inputs(
        clean_features,
        augmented_features,
        labels,
        text_prototypes,
        num_classes=num_classes,
        shot=shot,
    )
    coefficients = class_confidence_coefficients(
        augmented,
        targets,
        text,
        logit_scale=CLAP_LOGIT_SCALE,
    )
    flat_training = augmented.reshape(-1, augmented.shape[-1])
    repeated_targets = np.tile(targets, augmented.shape[0])
    batches_per_epoch = int(math.ceil(clean.shape[0] / int(batch_size)))

    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    resolved = torch.device(device)
    feature_tensor = torch.as_tensor(
        flat_training, dtype=torch.float32, device=resolved
    )
    target_tensor = torch.as_tensor(
        repeated_targets, dtype=torch.long, device=resolved
    )
    text_tensor = torch.as_tensor(text, dtype=torch.float32, device=resolved)
    coefficient_tensor = torch.as_tensor(
        coefficients, dtype=torch.float32, device=resolved
    )
    weights = torch.nn.Parameter(text_tensor.clone())
    optimizer = torch.optim.SGD(
        [weights],
        lr=CLAP_LEARNING_RATE,
        momentum=CLAP_MOMENTUM,
        weight_decay=CLAP_WEIGHT_DECAY,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    epoch_losses: list[float] = []
    optimizer_steps = 0
    for epoch_index in range(int(epochs)):
        learning_rate = _learning_rate(epoch_index, int(epochs))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        permutation = torch.randperm(
            flat_training.shape[0], generator=generator, device="cpu"
        )
        batch_losses: list[float] = []
        for batch_index in range(batches_per_epoch):
            start = batch_index * int(batch_size)
            stop = min(start + int(batch_size), flat_training.shape[0])
            indices = permutation[start:stop].to(resolved)
            batch_features = feature_tensor.index_select(0, indices)
            batch_targets = target_tensor.index_select(0, indices)
            normalized_weights = torch.nn.functional.normalize(weights, dim=1)
            logits = CLAP_LOGIT_SCALE * batch_features @ normalized_weights.T
            loss = torch.nn.functional.cross_entropy(logits, batch_targets)
            if mode == "clap":
                distances = (weights - text_tensor).pow(2).sum(dim=1)
                loss = loss + torch.mean(coefficient_tensor * distances)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            batch_losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(batch_losses)))
        if not torch.isfinite(weights).all():
            raise FloatingPointError("CLAP optimization produced non-finite weights")

    fitted_weights = weights.detach().cpu().numpy().astype(np.float32, copy=True)
    clean_scores = (
        CLAP_LOGIT_SCALE
        * clean
        @ l2_normalize(fitted_weights).T
    )
    clean_predictions = np.argmax(clean_scores, axis=1)
    selection = {
        "strategy": "fixed_support_only_training_budget",
        "reason": None,
        "mode": mode,
        "epoch": int(epochs),
        "optimizer": "SGD",
        "learning_rate": CLAP_LEARNING_RATE,
        "momentum": CLAP_MOMENTUM,
        "weight_decay": CLAP_WEIGHT_DECAY,
        "learning_rate_scheduler": "one_epoch_constant_warmup_then_cosine",
        "warmup_epochs": CLAP_WARMUP_EPOCHS,
        "warmup_learning_rate": CLAP_WARMUP_LEARNING_RATE,
        "first_epoch_learning_rate": _learning_rate(0, int(epochs)),
        "final_epoch_learning_rate": _learning_rate(int(epochs) - 1, int(epochs)),
        "batch_size": int(batch_size),
        "batches_per_epoch": batches_per_epoch,
        "optimizer_steps": optimizer_steps,
        "support_augmentation_view_count": int(augmented.shape[0]),
        "support_observations_per_epoch": min(
            int(flat_training.shape[0]), batches_per_epoch * int(batch_size)
        ),
        "initialization": "zero_shot_text_prototypes",
        "constraint": (
            "class_confidence_weighted_l2" if mode == "clap" else "none"
        ),
        "class_coefficients": coefficients.tolist(),
        "logit_scale": CLAP_LOGIT_SCALE,
        "training_seed": int(seed),
        "support_training_accuracy": float(
            np.mean(clean_predictions == targets)
        ),
        "final_support_loss": float(epoch_losses[-1]),
        "query_labels_used_for_selection": False,
        "paper_title": CLAP_PAPER_TITLE,
        "reference_repository": CLAP_REPOSITORY,
        "reference_commit": CLAP_REFERENCE_COMMIT,
        "protocol_note": (
            "The public main configuration's fixed support-derived class "
            "coefficients are retained. This project uses its established ten "
            "support views instead of the public implementation's twenty."
        ),
    }
    return FittedCLAP(
        weights=_readonly_copy(fitted_weights, dtype=np.float32),
        class_coefficients=_readonly_copy(coefficients, dtype=np.float32),
        selection=selection,
        trainable_parameter_count=int(num_classes * clean.shape[1]),
    )

