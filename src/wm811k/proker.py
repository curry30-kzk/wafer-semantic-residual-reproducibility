"""Exact ProKeR kernel residual, ported from the pinned official implementation.

Reference: ybendou/ProKeR f2ecb3d2a0d7e0993f977e550286b34f06015edb,
trainers/proker.py (Apache-2.0; original source archived under docs/proker).
The fit API has no query labels. Validation selection is explicitly separate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator
import numpy as np
import torch

from .clip_adaptation import build_group_stratified_support_folds
from .metrics import evaluate_predictions
from .prototype import l2_normalize


def _features(value: Any, name: str) -> np.ndarray:
    return l2_normalize(np.asarray(value)).astype(np.float32)


def _inputs(rows, labels, text, device):
    s = _features(rows, 'support')
    t = _features(text, 'text')
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(s) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('labels must be integer class indices matching support')
    if t.shape[1] != s.shape[1] or set(y.tolist()) != set(range(len(t))):
        raise ValueError('text dimensions or complete support class set do not match')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    st = torch.as_tensor(s, device=device)
    tt = torch.as_tensor(t, device=device)
    yt = torch.as_tensor(y.astype(np.int64), device=device)
    rhs = torch.nn.functional.one_hot(yt, len(t)).float() - st @ tt.T
    return s, t, st, tt, rhs


def _positive(value, name):
    if not np.isfinite(value) or float(value) <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def _solve(kernel, rhs, lmbda):
    # solve(K/lambda + I, rhs) == solve(K + lambda I, lambda * rhs).
    matrix = kernel.clone()
    matrix.diagonal().add_(lmbda)
    factor = torch.linalg.cholesky(matrix)
    return torch.cholesky_solve(lmbda * rhs, factor)


@dataclass(frozen=True)
class FittedProKeR:
    support_rows: np.ndarray
    text: np.ndarray
    coefficients: np.ndarray
    beta: float
    lmbda: float
    device: str

    def decision_scores(self, features) -> np.ndarray:
        x = _features(features, 'query')
        if x.shape[1] != self.support_rows.shape[1]:
            raise ValueError('query feature dimension differs')
        s = torch.as_tensor(self.support_rows, device=self.device)
        t = torch.as_tensor(self.text, device=self.device)
        a = torch.as_tensor(self.coefficients, device=self.device)
        parts = []
        with torch.inference_mode():
            for start in range(0, len(x), 512):
                xt = torch.as_tensor(x[start:start+512], device=self.device)
                scores = xt @ t.T + torch.exp(-self.beta * (1 - xt @ s.T)) @ a
                parts.append(scores.cpu().numpy())
        return np.concatenate(parts)

    def predict(self, features) -> np.ndarray:
        return self.decision_scores(features).argmax(axis=1)


def fit_proker(support_rows, labels, text_prototypes, *, beta, lmbda, device='cpu') -> FittedProKeR:
    beta = _positive(beta, 'beta')
    lmbda = _positive(lmbda, 'lambda')
    with torch.inference_mode():
        s, t, st, _, rhs = _inputs(support_rows, labels, text_prototypes, device)
        kernel = torch.exp(-beta * (1 - st @ st.T))
        coeff = _solve(kernel, rhs, lmbda).cpu().numpy().copy()
    if not np.isfinite(coeff).all():
        raise ValueError('nonfinite kernel coefficients')
    return FittedProKeR(s, t, coeff, beta, lmbda, str(device))


def expand_support_views(augmented_features, labels):
    """Keep every view as a row and retain the original-image group identity."""
    v = np.asarray(augmented_features)
    y = np.asarray(labels)
    if v.ndim != 3 or min(v.shape) <= 0 or y.ndim != 1 or len(y) != v.shape[1]:
        raise ValueError('views must be [views, original_images, dimension]')
    rows = _features(v.reshape(-1, v.shape[-1]), 'views')
    return rows, np.tile(y, v.shape[0]), np.tile(np.arange(len(y)), v.shape[0])


def candidate_scores(support_rows, labels, text_prototypes, validation_features,
                     beta_grid, lambda_grid, *, device='cpu') -> Iterator[tuple[float, float, np.ndarray]]:
    """Reuse exact kernel matrices across lambda candidates, without labels."""
    x = _features(validation_features, 'validation')
    with torch.inference_mode():
        _, _, st, tt, rhs = _inputs(support_rows, labels, text_prototypes, device)
        if x.shape[1] != st.shape[1]:
            raise ValueError('validation dimension differs')
        xt = torch.as_tensor(x, device=device)
        prior = xt @ tt.T
        for beta_value in beta_grid:
            beta = _positive(beta_value, 'beta')
            kernel = torch.exp(-beta * (1 - st @ st.T))
            query_kernel = torch.exp(-beta * (1 - xt @ st.T))
            for lambda_value in lambda_grid:
                lmbda = _positive(lambda_value, 'lambda')
                a = _solve(kernel, rhs, lmbda)
                scores = prior + query_kernel @ a
                if not torch.isfinite(scores).all():
                    raise ValueError('nonfinite candidate scores')
                yield beta, lmbda, scores.cpu().numpy().copy()


def choose_candidate(candidates):
    if not candidates:
        raise ValueError('empty candidate grid')
    if any(not np.isfinite(r['macro_f1']) for r in candidates):
        raise ValueError('nonfinite selection metric')
    return min(candidates, key=lambda r: (-r['macro_f1'], r['beta'], r['lambda']))


def select_validation_parameters(augmented_features, support_labels, text_prototypes,
                                 validation_features, validation_labels, beta_grid,
                                 lambda_grid, *, device='cpu'):
    rows, labels, _ = expand_support_views(augmented_features, support_labels)
    names = [str(i) for i in range(len(text_prototypes))]
    candidates = []
    for beta, lmbda, scores in candidate_scores(rows, labels, text_prototypes, validation_features,
                                               beta_grid, lambda_grid, device=device):
        m = evaluate_predictions(validation_labels, scores.argmax(axis=1), names)
        candidates.append({'beta': beta, 'lambda': lmbda, 'macro_f1': m['macro_f1'],
                           'accuracy': m['accuracy']})
    return {'selected': choose_candidate(candidates), 'candidates': candidates,
            'query_labels_used_for_selection': False, 'selection_source': 'Validation'}


def select_support_parameters(clean_features, augmented_features, support_labels,
                              text_prototypes, beta_grid, lambda_grid, *, group_ids,
                              device='cpu', fold_observer=None):
    y = np.asarray(support_labels)
    names = [str(i) for i in range(len(text_prototypes))]
    folds = build_group_stratified_support_folds(y, group_ids, num_classes=len(names), fold_count=5)
    predictions = {(float(b), float(l)): np.full(len(y), -1, dtype=int)
                   for b in beta_grid for l in lambda_grid}
    fold_log = []
    for heldout in folds:
        train = np.setdiff1d(np.arange(len(y)), heldout)
        rows, labels, local_groups = expand_support_views(np.asarray(augmented_features)[:, train], y[train])
        training_groups = train[local_groups]
        if fold_observer:
            fold_observer(train.tolist(), heldout.tolist(), training_groups.tolist())
        fold_log.append({'train_original_indices': train.tolist(), 'heldout_original_indices': heldout.tolist()})
        for b, l, scores in candidate_scores(rows, labels, text_prototypes, np.asarray(clean_features)[heldout],
                                              beta_grid, lambda_grid, device=device):
            predictions[(b, l)][heldout] = scores.argmax(axis=1)
    candidates = []
    for (b, l), pred in predictions.items():
        m = evaluate_predictions(y, pred, names)
        candidates.append({'beta': b, 'lambda': l, 'macro_f1': m['macro_f1'],
                           'accuracy': m['accuracy'], 'oof_predictions': pred.tolist()})
    return {'selected': choose_candidate(candidates), 'candidates': candidates, 'folds': fold_log,
            'selection_source': 'Training support only, grouped fivefold',
            'query_labels_used_for_selection': False}
