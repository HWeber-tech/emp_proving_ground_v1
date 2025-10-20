from datetime import datetime, timezone

from src.core.validation_models import ValidationResult


def test_validation_result_metadata_is_isolated() -> None:
    source_metadata = {"alpha": 1}

    result = ValidationResult(
        test_name="example",
        passed=True,
        value=0.5,
        threshold=0.4,
        unit="percent",
        metadata=source_metadata,
    )

    assert result.metadata is not source_metadata
    source_metadata["alpha"] = 0

    assert result.metadata["alpha"] == 1
    assert source_metadata["alpha"] == 0


def test_validation_result_to_dict_returns_metadata_copy() -> None:
    result = ValidationResult(
        test_name="example",
        passed=True,
        value=1.0,
        threshold=0.5,
        unit="percent",
        metadata={"beta": 2},
    )

    payload = result.to_dict()

    assert payload["metadata"] is not result.metadata

    payload["metadata"]["beta"] = 3

    assert result.metadata["beta"] == 2


def test_validation_result_preserves_explicit_timestamp() -> None:
    timestamp = datetime(2023, 1, 1, 12, 30, tzinfo=timezone.utc)

    result = ValidationResult(
        test_name="example",
        passed=False,
        value=2.0,
        threshold=1.5,
        unit="percent",
        timestamp=timestamp,
    )

    assert result.timestamp is timestamp
    assert result.to_dict()["timestamp"] == timestamp.isoformat()
