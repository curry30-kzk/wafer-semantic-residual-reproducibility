"""Support-only fitting for mainstream few-shot CLIP adaptation baselines."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .clip_adaptation import (
    LINEAR_C_GRID,
    TIP_ALPHA_GRID,
    TIP_BETA_GRID,
    TIP_DEFAULT_ALPHA,
    TIP_DEFAULT_BETA,
    TIP_F_MAX_EPOCHS,
    build_cache_model,
    build_group_stratified_support_folds,
    build_stratified_support_folds,
    select_tip_hyperparameters,
    tip_adapter_logits,
)
from .baseline_experiment import select_debug_query_subset
from .clip_experiment import validate_task_grid
from .clip_features import tip_adapter_augmentation_metadata
from .clap import CLAP_EPOCHS, FittedCLAP, fit_clap_support_only
from .experiment_reporting import (
    task_result_provenance,
    write_fewshot_summaries,
    write_run_artifacts,
)
from .fewshot_dataset import FewShotTask, ImageRecord
from .metrics import evaluate_predictions
from .lp_plus_plus import (
    FittedFairLPPlusPlus,
    FittedLPPlusPlus,
    LP_PLUS_PLUS_EPOCHS,
    fit_fair_lp_plus_plus_support_only,
    fit_lp_plus_plus_support_only,
)
from .prototype import l2_normalize
from .rla import RLA_EPOCHS, FittedRLA, fit_rla_support_only


ONE_SHOT_REASON = "one_shot_no_class_complete_support_validation"
ADAPTATION_MODES = (
    "linear_probe",
    "tip_adapter",
    "tip_adapter_f",
    "lp_plus_plus",
    "lp_plus_plus_fair",
    "zs_lp",
    "clap",
    "rla",
)
ADAPTATION_SCHEMAS = {
    "linear_probe": "clip_linear_probe_result_v1",
    "tip_adapter": "clip_tip_adapter_result_v1",
    "tip_adapter_f": "clip_tip_adapter_f_result_v1",
    "lp_plus_plus": "clip_lp_plus_plus_result_v1",
    "lp_plus_plus_fair": "clip_lp_plus_plus_fair_result_v1",
    "zs_lp": "clip_zs_lp_result_v1",
    "clap": "clip_clap_result_v1",
    "rla": "clip_rla_result_v1",
}
ADAPTATION_METHOD_NAMES = {
    "linear_probe": "OpenAI CLIP ViT-B/32 linear probe",
    "tip_adapter": "OpenAI CLIP ViT-B/32 Tip-Adapter",
    "tip_adapter_f": "OpenAI CLIP ViT-B/32 Tip-Adapter-F",
    "lp_plus_plus": "OpenAI CLIP ViT-B/32 LP++",
    "lp_plus_plus_fair": "OpenAI CLIP ViT-B/32 LP++ (fair protocol)",
    "zs_lp": "OpenAI CLIP ViT-B/32 ZS-LP (same-protocol port)",
    "clap": "OpenAI CLIP ViT-B/32 CLAP (same-protocol port)",
    "rla": "OpenAI CLIP ViT-B/32 RLA (same-protocol port)",
}


def _support_augmentation_metadata_for_protocol(
    protocol_name: str,
) -> dict[str, Any]:
    normalized = str(protocol_name).strip().lower()
    dataset = "fgvc_aircraft" if normalized.startswith("fgvc_aircraft_") else "wm811k"
    return tip_adapter_augmentation_metadata(dataset)


def _reset_cuda_peak_memory(device: str) -> None:
    try:
        import torch
    except ImportError:
        return
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(resolved)


def _peak_cuda_memory_mb(device: str) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(resolved) / (1024.0**2))


def _validate_support_inputs(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    *,
    num_classes: int,
    shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = np.asarray(clean_features, dtype=np.float32)
    augmented = np.asarray(augmented_features, dtype=np.float32)
    support_labels = np.asarray(labels, dtype=np.int64)
    expected = num_classes * shot
    if clean.ndim != 2 or clean.shape[0] != expected:
        raise ValueError("clean support features have the wrong shape")
    if augmented.ndim != 3 or augmented.shape[1:] != clean.shape:
        raise ValueError("augmented support features have the wrong shape")
    if support_labels.shape != (expected,):
        raise ValueError("support labels have the wrong shape")
    if not np.isfinite(clean).all() or not np.isfinite(augmented).all():
        raise ValueError("support features contain non-finite values")
    expected_labels = np.repeat(np.arange(num_classes), shot)
    if sorted(support_labels.tolist()) != sorted(expected_labels.tolist()):
        raise ValueError("support labels must contain exactly K samples per class")
    return l2_normalize(clean), l2_normalize(augmented.reshape(-1, clean.shape[1])).reshape(
        augmented.shape
    ), support_labels


@dataclass
class FittedLinearProbe:
    model: LogisticRegression
    selection: dict[str, Any]
    trainable_parameter_count: int

    def decision_scores(self, features: Any) -> np.ndarray:
        normalized = l2_normalize(np.asarray(features, dtype=np.float32))
        scores = np.asarray(self.model.decision_function(normalized))
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return scores

    def predict(self, features: Any) -> np.ndarray:
        indices = np.argmax(self.decision_scores(features), axis=1)
        return np.asarray(self.model.classes_[indices], dtype=np.int64)


@dataclass
class FittedTipAdapter:
    text_prototypes: np.ndarray
    cache_keys: np.ndarray
    cache_values: np.ndarray
    selection: dict[str, Any]
    trainable_parameter_count: int = 0
    normalize_cache_keys: bool = True

    def decision_scores(self, features: Any) -> np.ndarray:
        return tip_adapter_logits(
            features,
            self.text_prototypes,
            self.cache_keys,
            self.cache_values,
            beta=float(self.selection["beta"]),
            alpha=float(self.selection["alpha"]),
            normalize_cache_keys=self.normalize_cache_keys,
        )

    def predict(self, features: Any) -> np.ndarray:
        return np.argmax(self.decision_scores(features), axis=1).astype(np.int64)


def _fit_logistic(
    augmented: np.ndarray, labels: np.ndarray, *, c_value: float
) -> LogisticRegression:
    feature_dimension = augmented.shape[2]
    train_features = augmented.reshape(-1, feature_dimension)
    train_labels = np.tile(labels, augmented.shape[0])
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    model.fit(train_features, train_labels)
    return model


def fit_linear_probe_support_only(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    *,
    num_classes: int,
    shot: int,
    c_grid: tuple[float, ...] = LINEAR_C_GRID,
    support_group_ids: Sequence[str] | None = None,
) -> FittedLinearProbe:
    """Select and refit a deterministic CLIP logistic-regression probe."""

    clean, augmented, support_labels = _validate_support_inputs(
        clean_features,
        augmented_features,
        labels,
        num_classes=num_classes,
        shot=shot,
    )
    if shot == 1:
        selected_c = 1.0
        selection: dict[str, Any] = {
            "strategy": "one_shot_frozen_default",
            "reason": ONE_SHOT_REASON,
            "C": selected_c,
            "validation_macro_f1": None,
            "candidates": [],
        }
    else:
        group_ids: np.ndarray | None = None
        if support_group_ids is None:
            folds = build_stratified_support_folds(
                support_labels, num_classes=num_classes
            )
        else:
            group_ids = np.asarray(support_group_ids, dtype=str)
            folds = build_group_stratified_support_folds(
                support_labels,
                group_ids,
                num_classes=num_classes,
            )
        class_names = tuple(str(index) for index in range(num_classes))
        candidates = []
        best_key = None
        selected_c = None
        selected_f1 = None
        all_indices = np.arange(clean.shape[0])
        for c_value in tuple(float(value) for value in c_grid):
            true_parts = []
            prediction_parts = []
            for fold in folds:
                train_mask = np.ones(clean.shape[0], dtype=bool)
                train_mask[fold] = False
                train_indices = all_indices[train_mask]
                model = _fit_logistic(
                    augmented[:, train_indices, :],
                    support_labels[train_indices],
                    c_value=c_value,
                )
                true_parts.append(support_labels[fold])
                prediction_parts.append(model.predict(clean[fold]))
            y_true = np.concatenate(true_parts)
            y_pred = np.concatenate(prediction_parts)
            macro_f1 = float(
                evaluate_predictions(y_true, y_pred, class_names)["macro_f1"]
            )
            candidates.append({"C": c_value, "macro_f1": macro_f1})
            key = (-macro_f1, c_value)
            if best_key is None or key < best_key:
                best_key = key
                selected_c = c_value
                selected_f1 = macro_f1
        assert selected_c is not None and selected_f1 is not None
        selection = {
            "strategy": "stratified_support_cross_validation",
            "reason": None,
            "C": selected_c,
            "validation_macro_f1": selected_f1,
            "validation_count": int(clean.shape[0]),
            "folds": [fold.tolist() for fold in folds],
            "candidates": candidates,
        }
        if group_ids is not None:
            fold_group_ids = [group_ids[fold].tolist() for fold in folds]
            selection.update(
                {
                    "oof_group_aware": True,
                    "oof_group_ids": group_ids.tolist(),
                    "oof_fold_group_ids": fold_group_ids,
                    "oof_fold_assignment_sha256": hashlib.sha256(
                        json.dumps(
                            {
                                "group_ids": group_ids.tolist(),
                                "fold_group_ids": fold_group_ids,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
    final_model = _fit_logistic(augmented, support_labels, c_value=selected_c)
    trainable_parameters = clean.shape[1] * num_classes + num_classes
    return FittedLinearProbe(final_model, selection, trainable_parameters)


def fit_tip_adapter_support_only(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    num_classes: int,
    shot: int,
    beta_grid: tuple[float, ...] = TIP_BETA_GRID,
    alpha_grid: tuple[float, ...] = TIP_ALPHA_GRID,
) -> FittedTipAdapter:
    """Build Tip-Adapter with hyperparameters selected from support only."""

    clean, augmented, support_labels = _validate_support_inputs(
        clean_features,
        augmented_features,
        labels,
        num_classes=num_classes,
        shot=shot,
    )
    text = l2_normalize(np.asarray(text_prototypes, dtype=np.float32))
    cache_keys, cache_values = build_cache_model(
        augmented, support_labels, num_classes=num_classes
    )
    if shot == 1:
        selection = {
            "strategy": "one_shot_frozen_default",
            "reason": ONE_SHOT_REASON,
            "beta": TIP_DEFAULT_BETA,
            "alpha": TIP_DEFAULT_ALPHA,
            "validation_macro_f1": None,
            "candidates": [],
        }
    else:
        folds = build_stratified_support_folds(
            support_labels, num_classes=num_classes
        )
        selected = select_tip_hyperparameters(
            clean,
            cache_keys,
            support_labels,
            text,
            folds,
            num_classes=num_classes,
            beta_grid=beta_grid,
            alpha_grid=alpha_grid,
        )
        selection = {
            "strategy": "stratified_support_cross_validation",
            "reason": None,
            "beta": selected["beta"],
            "alpha": selected["alpha"],
            "validation_macro_f1": selected["macro_f1"],
            "validation_count": selected["validation_count"],
            "folds": [fold.tolist() for fold in folds],
            "candidates": selected["candidates"],
        }
    return FittedTipAdapter(text, cache_keys, cache_values, selection)


def _train_tip_f_keys(
    augmented: np.ndarray,
    labels: np.ndarray,
    text: np.ndarray,
    *,
    num_classes: int,
    epochs: int,
    device: str,
    beta: float = TIP_DEFAULT_BETA,
    alpha: float = TIP_DEFAULT_ALPHA,
) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    import torch

    torch.manual_seed(0)
    resolved = torch.device(device)
    cache_keys, cache_values = build_cache_model(
        augmented, labels, num_classes=num_classes
    )
    adapter = torch.nn.Linear(cache_keys.shape[1], cache_keys.shape[0], bias=False)
    adapter.weight = torch.nn.Parameter(
        torch.as_tensor(cache_keys, dtype=torch.float32)
    )
    adapter = adapter.to(resolved)
    values = torch.as_tensor(cache_values, dtype=torch.float32, device=resolved)
    text_tensor = torch.as_tensor(text, dtype=torch.float32, device=resolved)
    label_tensor = torch.as_tensor(labels, dtype=torch.long, device=resolved)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    losses = []
    key_history = []
    for epoch in range(epochs):
        features = torch.as_tensor(
            augmented[epoch % augmented.shape[0]],
            dtype=torch.float32,
            device=resolved,
        )
        affinity = adapter(features)
        cache_logits = torch.exp(-float(beta) * (1.0 - affinity)) @ values
        clip_logits = 100.0 * features @ text_tensor.T
        logits = clip_logits + float(alpha) * cache_logits
        loss = torch.nn.functional.cross_entropy(logits, label_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        key_history.append(
            adapter.weight.detach().cpu().numpy().astype(np.float32, copy=True)
        )
    learned_keys = adapter.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    return learned_keys, losses, key_history


def fit_tip_adapter_f_support_only(
    clean_features: Any,
    augmented_features: Any,
    labels: Any,
    text_prototypes: Any,
    *,
    num_classes: int,
    shot: int,
    device: str,
    max_epochs: int = TIP_F_MAX_EPOCHS,
    beta_grid: tuple[float, ...] = TIP_BETA_GRID,
    alpha_grid: tuple[float, ...] = TIP_ALPHA_GRID,
) -> FittedTipAdapter:
    """Fit Tip-Adapter-F with support-fold model selection only."""

    clean, augmented, support_labels = _validate_support_inputs(
        clean_features,
        augmented_features,
        labels,
        num_classes=num_classes,
        shot=shot,
    )
    text = l2_normalize(np.asarray(text_prototypes, dtype=np.float32))
    if shot == 1:
        selected_epoch = int(max_epochs)
        selected_beta = TIP_DEFAULT_BETA
        selected_alpha = TIP_DEFAULT_ALPHA
        selection: dict[str, Any] = {
            "strategy": "one_shot_frozen_default",
            "reason": ONE_SHOT_REASON,
            "epoch": selected_epoch,
            "beta": selected_beta,
            "alpha": selected_alpha,
            "validation_macro_f1": None,
            "candidates": [],
        }
    else:
        folds = build_stratified_support_folds(
            support_labels, num_classes=num_classes
        )
        all_indices = np.arange(clean.shape[0])
        fold_histories = []
        fold_values = []
        for fold in folds:
            train_mask = np.ones(clean.shape[0], dtype=bool)
            train_mask[fold] = False
            train_indices = all_indices[train_mask]
            _, _, history = _train_tip_f_keys(
                augmented[:, train_indices, :],
                support_labels[train_indices],
                text,
                num_classes=num_classes,
                epochs=max_epochs,
                device=device,
            )
            fold_histories.append(history)
            _, values = build_cache_model(
                augmented[:, train_indices, :],
                support_labels[train_indices],
                num_classes=num_classes,
            )
            fold_values.append(values)

        class_names = tuple(str(index) for index in range(num_classes))
        epoch_candidates = []
        best_epoch_key = None
        selected_epoch = None
        selected_epoch_f1 = None
        for epoch_index in range(max_epochs):
            true_parts = []
            prediction_parts = []
            for fold_index, fold in enumerate(folds):
                logits = tip_adapter_logits(
                    clean[fold],
                    text,
                    fold_histories[fold_index][epoch_index],
                    fold_values[fold_index],
                    beta=TIP_DEFAULT_BETA,
                    alpha=TIP_DEFAULT_ALPHA,
                    normalize_cache_keys=False,
                )
                true_parts.append(support_labels[fold])
                prediction_parts.append(np.argmax(logits, axis=1))
            macro_f1 = float(
                evaluate_predictions(
                    np.concatenate(true_parts),
                    np.concatenate(prediction_parts),
                    class_names,
                )["macro_f1"]
            )
            epoch = epoch_index + 1
            epoch_candidates.append({"epoch": epoch, "macro_f1": macro_f1})
            key = (-macro_f1, epoch)
            if best_epoch_key is None or key < best_epoch_key:
                best_epoch_key = key
                selected_epoch = epoch
                selected_epoch_f1 = macro_f1
        assert selected_epoch is not None and selected_epoch_f1 is not None

        hp_candidates = []
        best_hp_key = None
        selected_beta = None
        selected_alpha = None
        selected_hp_f1 = None
        for beta in tuple(float(value) for value in beta_grid):
            for alpha in tuple(float(value) for value in alpha_grid):
                true_parts = []
                prediction_parts = []
                for fold_index, fold in enumerate(folds):
                    logits = tip_adapter_logits(
                        clean[fold],
                        text,
                        fold_histories[fold_index][selected_epoch - 1],
                        fold_values[fold_index],
                        beta=beta,
                        alpha=alpha,
                        normalize_cache_keys=False,
                    )
                    true_parts.append(support_labels[fold])
                    prediction_parts.append(np.argmax(logits, axis=1))
                macro_f1 = float(
                    evaluate_predictions(
                        np.concatenate(true_parts),
                        np.concatenate(prediction_parts),
                        class_names,
                    )["macro_f1"]
                )
                hp_candidates.append(
                    {"beta": beta, "alpha": alpha, "macro_f1": macro_f1}
                )
                key = (
                    -macro_f1,
                    (beta - TIP_DEFAULT_BETA) ** 2
                    + (alpha - TIP_DEFAULT_ALPHA) ** 2,
                    beta,
                    alpha,
                )
                if best_hp_key is None or key < best_hp_key:
                    best_hp_key = key
                    selected_beta = beta
                    selected_alpha = alpha
                    selected_hp_f1 = macro_f1
        assert selected_beta is not None and selected_alpha is not None
        assert selected_hp_f1 is not None
        selection = {
            "strategy": "stratified_support_cross_validation",
            "reason": None,
            "epoch": selected_epoch,
            "beta": selected_beta,
            "alpha": selected_alpha,
            "validation_macro_f1": selected_hp_f1,
            "epoch_selection_macro_f1": selected_epoch_f1,
            "validation_count": int(clean.shape[0]),
            "folds": [fold.tolist() for fold in folds],
            "epoch_candidates": epoch_candidates,
            "candidates": hp_candidates,
        }

    learned_keys, losses, _ = _train_tip_f_keys(
        augmented,
        support_labels,
        text,
        num_classes=num_classes,
        epochs=selected_epoch,
        device=device,
    )
    _, cache_values = build_cache_model(
        augmented, support_labels, num_classes=num_classes
    )
    selection["training_losses"] = losses
    return FittedTipAdapter(
        text,
        learned_keys,
        cache_values,
        selection,
        trainable_parameter_count=int(learned_keys.size),
        normalize_cache_keys=False,
    )


def _deduplicated_records(
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


def _prediction_rows(
    records: Sequence[ImageRecord],
    class_names: Sequence[str],
    predictions: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for row_index, record in enumerate(records):
        predicted_index = int(predictions[row_index])
        selected_score = float(scores[row_index, predicted_index])
        rows.append(
            {
                "original_index": record.original_index,
                "image_path": record.relative_path,
                "true_index": record.label_index,
                "true_label": record.label,
                "predicted_index": predicted_index,
                "predicted_label": class_names[predicted_index],
                # Kept for compatibility with the established artifact schema.
                "cosine_score": selected_score,
                "method_score": selected_score,
                "correct": bool(predicted_index == record.label_index),
            }
        )
    return rows


def _fit_mode(
    mode: str,
    clean_support: np.ndarray,
    augmented_support: np.ndarray,
    support_labels: np.ndarray,
    text_prototypes: np.ndarray | None,
    *,
    num_classes: int,
    shot: int,
    device: str,
    tip_f_max_epochs: int,
    lp_plus_plus_epochs: int,
    clap_epochs: int,
    rla_epochs: int,
    task_seed: int,
    support_group_ids: Sequence[str] | None = None,
) -> FittedLinearProbe | FittedTipAdapter | FittedLPPlusPlus | FittedCLAP | FittedRLA:
    if mode == "linear_probe":
        return fit_linear_probe_support_only(
            clean_support,
            augmented_support,
            support_labels,
            num_classes=num_classes,
            shot=shot,
            support_group_ids=support_group_ids,
        )
    if text_prototypes is None:
        raise ValueError("text-aware adaptation modes require text prototypes")
    if mode == "tip_adapter":
        return fit_tip_adapter_support_only(
            clean_support,
            augmented_support,
            support_labels,
            text_prototypes,
            num_classes=num_classes,
            shot=shot,
        )
    if mode == "tip_adapter_f":
        return fit_tip_adapter_f_support_only(
            clean_support,
            augmented_support,
            support_labels,
            text_prototypes,
            num_classes=num_classes,
            shot=shot,
            device=device,
            max_epochs=tip_f_max_epochs,
        )
    if mode == "lp_plus_plus":
        return fit_lp_plus_plus_support_only(
            clean_support,
            augmented_support,
            support_labels,
            text_prototypes,
            num_classes=num_classes,
            shot=shot,
            device=device,
            epochs=lp_plus_plus_epochs,
        )
    if mode == "lp_plus_plus_fair":
        return fit_fair_lp_plus_plus_support_only(
            clean_support,
            augmented_support,
            support_labels,
            text_prototypes,
            num_classes=num_classes,
            shot=shot,
            device=device,
            epochs=lp_plus_plus_epochs,
        )
    if mode in ("zs_lp", "clap"):
        return fit_clap_support_only(
            clean_support,
            augmented_support,
            support_labels,
            text_prototypes,
            num_classes=num_classes,
            shot=shot,
            device=device,
            mode=mode,
            seed=task_seed,
            epochs=clap_epochs,
        )
    if mode == "rla":
        return fit_rla_support_only(
            clean_support,
            augmented_support,
            support_labels,
            text_prototypes,
            num_classes=num_classes,
            shot=shot,
            device=device,
            seed=task_seed,
            epochs=rla_epochs,
        )
    raise ValueError(f"unsupported adaptation mode: {mode}")


def _clap_task_seed(task: FewShotTask) -> int:
    payload = (
        f"{task.protocol_name}|{task.n_way}|{task.shot}|{task.seed}|clap_zslp_v1"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def run_clip_adaptation_tasks(
    tasks: Sequence[FewShotTask],
    *,
    encoder: Any,
    modes: Sequence[str],
    output_roots: dict[str, str | Path],
    batch_size: int,
    limit_query: int | None,
    aggregate: bool,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    show_progress: bool = True,
    support_augment_views: int = 10,
    tip_f_max_epochs: int = TIP_F_MAX_EPOCHS,
    lp_plus_plus_epochs: int = LP_PLUS_PLUS_EPOCHS,
    clap_epochs: int = CLAP_EPOCHS,
    rla_epochs: int = RLA_EPOCHS,
) -> dict[str, list[dict[str, Any]]]:
    """Share CLIP encoding and evaluate isolated adaptation result roots."""

    mode_tuple = tuple(str(mode) for mode in modes)
    if not mode_tuple or len(set(mode_tuple)) != len(mode_tuple):
        raise ValueError("modes must be non-empty and unique")
    unknown = sorted(set(mode_tuple) - set(ADAPTATION_MODES))
    if unknown:
        raise ValueError(f"unsupported adaptation modes: {unknown}")
    if set(output_roots) != set(mode_tuple):
        raise ValueError("output_roots must exactly match requested modes")
    if aggregate and limit_query is not None:
        raise ValueError("debug query limits cannot be formally aggregated")
    validate_task_grid(
        tasks, require_formal_grid=aggregate, formal_seeds=expected_seeds
    )
    _reset_cuda_peak_memory(encoder.device)

    query_by_task: dict[tuple[int, int], tuple[ImageRecord, ...]] = {}
    for task in tasks:
        query_by_task[(task.shot, task.seed)] = (
            select_debug_query_subset(
                task.query, limit=limit_query, num_classes=task.n_way
            )
            if limit_query is not None
            else task.query
        )
    unique_records = _deduplicated_records(tasks, query_by_task)
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
    runtime_metadata = encoder.runtime_metadata(int(first_feature.shape[0]))

    text_prototypes = None
    prompts: tuple[str, ...] = ()
    text_seconds = 0.0
    if any(mode != "linear_probe" for mode in mode_tuple):
        text_started = time.perf_counter()
        text_prototypes, prompts = encoder.encode_class_text_prototypes(
            tasks[0].class_names
        )
        text_seconds = time.perf_counter() - text_started

    results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in mode_tuple}
    for task in tasks:
        clean_support = np.stack(
            [features_by_path[record.relative_path] for record in task.support]
        )
        support_labels = np.asarray(
            [record.label_index for record in task.support], dtype=np.int64
        )
        support_group_ids = (
            [str(record.content_hash) for record in task.support]
            if task.protocol_name == "wm811k_dedup_8way_v2"
            else None
        )
        query_records = query_by_task[(task.shot, task.seed)]
        query_features = np.stack(
            [features_by_path[record.relative_path] for record in query_records]
        )
        augmentation_seed = 20260816 + task.shot * 100 + task.seed
        augmentation_started = time.perf_counter()
        augmented_support = encoder.encode_augmented_images(
            task.support,
            batch_size=batch_size,
            view_count=support_augment_views,
            seed=augmentation_seed,
            show_progress=show_progress,
        )
        augmentation_seconds = time.perf_counter() - augmentation_started

        for mode in mode_tuple:
            fit_started = time.perf_counter()
            fitted = _fit_mode(
                mode,
                clean_support,
                augmented_support,
                support_labels,
                text_prototypes,
                num_classes=task.n_way,
                shot=task.shot,
                device=encoder.device,
                tip_f_max_epochs=tip_f_max_epochs,
                lp_plus_plus_epochs=lp_plus_plus_epochs,
                clap_epochs=clap_epochs,
                rla_epochs=rla_epochs,
                task_seed=_clap_task_seed(task),
                support_group_ids=support_group_ids,
            )
            fit_seconds = time.perf_counter() - fit_started
            evaluation_started = time.perf_counter()
            scores = fitted.decision_scores(query_features)
            predictions = np.argmax(scores, axis=1).astype(np.int64)
            # Query labels enter for the first time here, after selection/fitting.
            y_true = np.asarray(
                [record.label_index for record in query_records], dtype=np.int64
            )
            metrics = evaluate_predictions(y_true, predictions, task.class_names)
            evaluation_seconds = time.perf_counter() - evaluation_started
            debug_run = limit_query is not None
            result: dict[str, Any] = {
                "schema_version": ADAPTATION_SCHEMAS[mode],
                "method": ADAPTATION_METHOD_NAMES[mode],
                "mode": mode,
                **runtime_metadata,
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
                "query_count": len(query_records),
                "full_query_count": len(task.query),
                "debug_run": debug_run,
                "paper_eligible": bool(aggregate and not debug_run),
                "formal_grid_validated": bool(aggregate),
                "query_limit": limit_query,
                "query_labels_used_for_selection": False,
                "selection_source": "Training support only",
                "selection": fitted.selection,
                "fold_assignment_sha256": fitted.selection.get(
                    "oof_fold_assignment_sha256"
                ),
                "support_augmentation": {
                    **_support_augmentation_metadata_for_protocol(task.protocol_name),
                    "view_count": support_augment_views,
                    "seed": augmentation_seed,
                },
                "text_prompts": list(prompts) if mode != "linear_probe" else [],
                "clip_logit_scale": (
                    1.0
                    if mode in ("lp_plus_plus", "lp_plus_plus_fair")
                    else 100.0
                    if mode != "linear_probe"
                    else None
                ),
                "trainable_parameter_count": fitted.trainable_parameter_count,
                "peak_cuda_memory_mb": _peak_cuda_memory_mb(encoder.device),
                "prediction_score": (
                    "logistic_regression_decision_function"
                    if mode == "linear_probe"
                    else "normalized_clap_linear_logit"
                    if mode in ("zs_lp", "clap")
                    else "lp_plus_plus_visual_text_logit"
                    if mode in ("lp_plus_plus", "lp_plus_plus_fair")
                    else "normalized_rla_visual_projection_logit"
                    if mode == "rla"
                    else "clip_plus_cache_logit"
                ),
                "legacy_prediction_field_note": (
                    "cosine_score stores the selected method score for compatibility; "
                    "use method_score for adaptation baselines"
                ),
                "timing_seconds": {
                    "shared_clean_image_feature_extraction": feature_seconds,
                    "shared_text_encoding": text_seconds,
                    "support_augmentation_encoding": augmentation_seconds,
                    "support_only_fit_and_selection": fit_seconds,
                    "test_evaluation": evaluation_seconds,
                },
                "metrics": metrics,
            }
            if mode in ("zs_lp", "clap"):
                result["class_confidence_coefficients"] = (
                    fitted.class_coefficients.tolist()
                )
            stem = f"{task.n_way}way_{task.shot}shot_seed{task.seed}"
            if debug_run:
                stem += f"_debug_q{len(query_records)}"
            write_run_artifacts(
                output_roots[mode],
                result,
                _prediction_rows(
                    query_records, task.class_names, predictions, scores
                ),
                stem=stem,
                extra_prediction_fields=("method_score",),
            )
            results[mode].append(result)

    if aggregate:
        for mode in mode_tuple:
            write_fewshot_summaries(
                output_roots[mode], results[mode], expected_seeds=expected_seeds
            )
    return results
