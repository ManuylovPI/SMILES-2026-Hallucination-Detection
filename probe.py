"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

HIDDEN_DIM = 3584
GEOM_DIM = 25
META_DIM = 14

LOGREG_C = 0.01
GB_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                 random_state=42)
XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss', verbosity=0,
    use_label_encoder=False,
)


class HallucinationProbe(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._lr_pipe: Pipeline | None = None
        self._xgb: XGBClassifier | None = None
        self._fallback_used = False
        self._threshold: float = 0.5
        self._centroid_truthful: np.ndarray | None = None
        self._centroid_hallucinated: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "HallucinationProbe uses sklearn classifiers, "
            "not torch forward. Use .predict_proba() instead."
        )

    def _sanitize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
        X = np.clip(X, -1e4, 1e4)
        return X

    def _compute_centroid_distances(self, X: np.ndarray) -> np.ndarray:

        if self._centroid_truthful is None or self._centroid_hallucinated is None:
            return np.zeros((X.shape[0], 4), dtype=np.float32)

        layer24 = X[:, 3*896:4*896]

        ct = self._centroid_truthful
        ch = self._centroid_hallucinated

        dist_features = np.zeros((X.shape[0], 4), dtype=np.float32)
        eps = 1e-9
        for i in range(X.shape[0]):
            vec = layer24[i]
            dist_features[i, 0] = np.linalg.norm(vec - ct)
            dist_features[i, 1] = np.linalg.norm(vec - ch)
            dist_features[i, 2] = np.dot(vec, ct) / (
                np.linalg.norm(vec) * np.linalg.norm(ct) + eps)
            dist_features[i, 3] = np.dot(vec, ch) / (
                np.linalg.norm(vec) * np.linalg.norm(ch) + eps)

        dist_features = np.nan_to_num(dist_features, nan=0.0,
                                       posinf=1e4, neginf=-1e4)
        return np.clip(dist_features, -1e4, 1e4)

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        dist = self._compute_centroid_distances(X)
        return np.concatenate([X, dist], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X = self._sanitize(X)
        y = np.asarray(y).astype(np.int64)

        layer24 = X[:, 3*896:4*896]
        if (y == 0).any():
            self._centroid_truthful = layer24[y == 0].mean(axis=0)
        else:
            self._centroid_truthful = layer24.mean(axis=0)

        if (y == 1).any():
            self._centroid_hallucinated = layer24[y == 1].mean(axis=0)
        else:
            self._centroid_hallucinated = layer24.mean(axis=0)

        X_aug = self._augment_features(X)

        self._lr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(
                C=LOGREG_C, max_iter=2000,
                class_weight='balanced', random_state=42)),
        ])
        self._lr_pipe.fit(X_aug, y)

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        if _HAS_XGB:
            xgb_params = {**XGB_PARAMS, 'scale_pos_weight': scale_pos_weight}
            xgb_params.pop('use_label_encoder', None)
            self._xgb = XGBClassifier(**xgb_params)
            self._xgb.fit(X_aug, y)
            self._fallback_used = False
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self._xgb = GradientBoostingClassifier(**GB_PARAMS)
            self._xgb.fit(X_aug, y)
            self._fallback_used = True

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        X = self._sanitize(X)
        X_aug = self._augment_features(X)

        proba_lr = self._lr_pipe.predict_proba(X_aug)[:, 1]
        proba_xgb = self._xgb.predict_proba(X_aug)[:, 1]

        proba_pos = (proba_lr + proba_xgb) / 2.0

        return np.stack([1.0 - proba_pos, proba_pos], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba_pos = self.predict_proba(X)[:, 1]
        return (proba_pos >= self._threshold).astype(int)

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        y_val = np.asarray(y_val).astype(int)

        candidates = np.unique(np.concatenate([
            probs,
            np.linspace(0.0, 1.0, 101),
        ]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self
