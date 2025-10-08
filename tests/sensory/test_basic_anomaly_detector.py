from __future__ import annotations

import math

from src.sensory.anomaly.basic_detector import BasicAnomalyDetector


def test_basic_detector_flags_outlier() -> None:
    detector = BasicAnomalyDetector(window=10, min_samples=4, z_threshold=2.5)
    values = [1.0 + idx * 0.001 for idx in range(9)] + [5.0]

    evaluation = detector.evaluate(values)

    assert evaluation.sample_size == 10
    assert evaluation.is_anomaly is True
    assert evaluation.z_score > 2.5
    assert math.isclose(evaluation.latest, values[-1])


def test_basic_detector_handles_small_sample_without_alert() -> None:
    detector = BasicAnomalyDetector(window=5, min_samples=4, z_threshold=2.0)
    values = [1.0, 1.02]

    evaluation = detector.evaluate(values)

    assert evaluation.sample_size == 2
    assert evaluation.is_anomaly is False
    assert evaluation.z_score != 0.0


def test_basic_detector_ignores_invalid_values() -> None:
    detector = BasicAnomalyDetector(window=5, min_samples=3, z_threshold=2.5)
    sequence = [1.0, None, float("nan"), "1.1", 1.2]

    evaluation = detector.evaluate(sequence)

    assert evaluation.sample_size == 3
    assert evaluation.is_anomaly is False
    assert evaluation.mean > 1.0
