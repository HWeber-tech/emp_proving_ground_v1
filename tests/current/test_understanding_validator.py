import importlib

import pytest

from src.validation.accuracy import UnderstandingValidator


def test_intelligence_validator_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match="understanding_validator"):
        importlib.import_module("src.validation.accuracy.intelligence_validator")


def test_understanding_validator_generates_metrics() -> None:
    validator = UnderstandingValidator()
    metrics = validator.validate_anomaly_detection([0, 1, 1], [0, 1, 1])

    assert metrics.accuracy == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.false_negative_rate == 0.0
