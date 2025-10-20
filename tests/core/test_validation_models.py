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
