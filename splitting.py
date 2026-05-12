"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:

    n_folds = 5
    y = np.asarray(y).astype(int)
    indices = np.arange(len(y))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, y)):

        val_relative = val_size / (1.0 - 0.20)
        n_val = max(1, int(round(len(train_val_idx) * val_relative)))

        rng = np.random.default_rng(random_state + fold_idx)

        labels_pool = y[train_val_idx]
        val_indices_in_pool = []
        for cls in np.unique(labels_pool):
            cls_mask = (labels_pool == cls)
            cls_pool = train_val_idx[cls_mask]
            n_cls_val = max(1, int(round(n_val * cls_mask.mean())))
            n_cls_val = min(n_cls_val, len(cls_pool) - 1)
            chosen = rng.choice(cls_pool, size=n_cls_val, replace=False)
            val_indices_in_pool.append(chosen)

        val_idx = np.concatenate(val_indices_in_pool)
        train_idx = np.array([i for i in train_val_idx if i not in set(val_idx)])

        splits.append((train_idx, val_idx, test_idx))

    return splits
