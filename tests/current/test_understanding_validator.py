from src.validation.accuracy import IntelligenceValidator, UnderstandingValidator


def test_understanding_validator_aliases() -> None:
    assert UnderstandingValidator is IntelligenceValidator


def test_understanding_validator_generates_metrics() -> None:
    validator = UnderstandingValidator()
    metrics = validator.validate_anomaly_detection([0, 1, 1], [0, 1, 1])

    assert metrics.accuracy == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.false_negative_rate == 0.0
