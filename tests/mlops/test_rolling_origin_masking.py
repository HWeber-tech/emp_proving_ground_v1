from __future__ import annotations

import numpy as np

from mlops.validation_utils import rolling_origin_splits


def test_masking_future_data_does_not_mutate_sources() -> None:
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.arange(20, dtype=float)

    original_features = X.copy()
    original_labels = y.copy()

    splits = rolling_origin_splits(X, y, n_splits=3, train_size=6, test_size=4)
    assert splits, "expected at least one rolling-origin split"

    for split in splits:
        assert not np.shares_memory(split["X_train"], X)
        assert not np.shares_memory(split["X_test"], X)
        assert not np.shares_memory(split["y_train"], y)
        assert not np.shares_memory(split["y_test"], y)

        # Simulate masking future-known columns by zeroing selected entries
        future_mask = split["X_train"] > split["X_train"].mean()
        split["X_train"][future_mask] = 0.0
        split["X_test"][...] = -1.0
        split["y_train"][...] = -1.0
        split["y_test"][...] = -2.0

    np.testing.assert_array_equal(X, original_features)
    np.testing.assert_array_equal(y, original_labels)
