from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data_foundation.pipelines.class_priors import (
    DailyClassPrior,
    assign_daily_pos_weight,
    compute_daily_class_priors,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 10, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 14, tzinfo=timezone.utc),
                datetime(2024, 1, 3, 9, tzinfo=timezone.utc),
            ],
            "label": [1, 1, 0, 0, 1],
        }
    )


def test_compute_daily_class_priors_returns_history_only() -> None:
    priors = compute_daily_class_priors(_frame(), smoothing=0.0, default_weight=1.0)

    assert [isinstance(prior, DailyClassPrior) for prior in priors]
    assert [prior.history_positive for prior in priors] == [0, 2, 2]
    assert [prior.history_negative for prior in priors] == [0, 0, 2]
    # Day two weight should only reflect day-one data (no leakage of same-day labels)
    assert priors[1].pos_weight == pytest.approx(0.0)
    # By day three both classes have equal historical support
    assert priors[2].pos_weight == pytest.approx(1.0)


def test_assign_daily_pos_weight_aligned_with_frame_index() -> None:
    frame = _frame()
    weights = assign_daily_pos_weight(frame, smoothing=0.0, default_weight=1.0)

    assert list(weights.index) == list(frame.index)
    assert weights.iloc[0] == pytest.approx(1.0)
    assert weights.iloc[1] == pytest.approx(1.0)
    assert weights.iloc[2] == pytest.approx(0.0)
    assert weights.iloc[3] == pytest.approx(0.0)
    assert weights.iloc[4] == pytest.approx(1.0)
