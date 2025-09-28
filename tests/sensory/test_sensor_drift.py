from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sensory.monitoring import evaluate_sensor_drift


def _build_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    baseline_a = rng.normal(loc=0.0, scale=1.0, size=60)
    baseline_b = rng.normal(loc=5.0, scale=0.5, size=60)
    # introduce drift in sensor_a during evaluation window
    recent_a = rng.normal(loc=2.5, scale=1.0, size=20)
    recent_b = rng.normal(loc=5.0, scale=0.5, size=20)
    data = {
        "sensor_a": np.concatenate([baseline_a, recent_a]),
        "sensor_b": np.concatenate([baseline_b, recent_b]),
    }
    return pd.DataFrame(data)


def test_evaluate_sensor_drift_flags_drift() -> None:
    frame = _build_frame()
    summary = evaluate_sensor_drift(
        frame,
        baseline_window=60,
        evaluation_window=20,
        min_observations=15,
        z_threshold=3.0,
    )

    assert len(summary.results) == 2
    flagged = {result.sensor: result for result in summary.results}
    assert flagged["sensor_a"].exceeded is True
    assert flagged["sensor_a"].z_score is not None
    assert flagged["sensor_b"].exceeded is False


def test_evaluate_sensor_drift_subset_columns() -> None:
    frame = _build_frame()
    summary = evaluate_sensor_drift(
        frame,
        sensor_columns=["sensor_a"],
        baseline_window=60,
        evaluation_window=20,
        min_observations=15,
        z_threshold=3.0,
    )
    assert [result.sensor for result in summary.results] == ["sensor_a"]


def test_evaluate_sensor_drift_requires_enough_rows() -> None:
    frame = pd.DataFrame({"sensor_a": [1, 2, 3, 4]})
    with pytest.raises(ValueError):
        evaluate_sensor_drift(frame, baseline_window=3, evaluation_window=2)
