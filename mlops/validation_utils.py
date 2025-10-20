"""Utility helpers for data splitting and leakage-guarded validation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def rolling_origin_splits(
    X: Any,
    y: Any,
    *,
    n_splits: int = 5,
    train_size: int = 500,
    test_size: int = 100,
) -> List[Dict[str, Any]]:
    """Return walk-forward train/test splits without mutating the source arrays.

    The returned dictionaries contain deep copies of the feature and label
    segments so downstream masking or zeroing of future data cannot alter the
    original ``X``/``y`` inputs.
    """

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("features and labels must share the first dimension")

    total = int(X_arr.shape[0])
    results: List[Dict[str, Any]] = []
    start_idx = 0

    for fold in range(n_splits):
        train_end = start_idx + int(train_size)
        test_end = train_end + int(test_size)

        adjusted = False
        if test_end > total:
            adjusted = True
            test_end = total
            train_end = max(int(train_size), test_end - int(test_size))

        train_slice = slice(start_idx, train_end)
        test_slice = slice(train_end, test_end)

        X_train = np.array(X_arr[train_slice], copy=True)
        X_test = np.array(X_arr[test_slice], copy=True)
        y_train = np.array(y_arr[train_slice], copy=True)
        y_test = np.array(y_arr[test_slice], copy=True)

        results.append(
            {
                "fold": fold + 1,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "train_start": train_slice.start,
                "train_end": train_slice.stop,
                "test_start": test_slice.start,
                "test_end": test_slice.stop,
                "adjusted": adjusted,
            }
        )

        start_idx += int(test_size)
        if test_slice.stop >= total:
            break

    return results


__all__ = ["rolling_origin_splits"]
