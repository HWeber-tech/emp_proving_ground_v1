import pytest

from datetime import datetime, timedelta

from src.validation.validation_framework import ValidationFramework


@pytest.mark.asyncio
async def test_validation_framework_component_integration_initializes_integrator():
    framework = ValidationFramework()

    result = await framework.validate_component_integration()

    assert result.passed is True
    assert result.value == pytest.approx(1.0)
    assert not result.metadata.get("missing")
    assert result.metadata.get("initialized") is True


@pytest.mark.asyncio
async def test_validate_data_integrity_passes_with_valid_payload():
    framework = ValidationFramework()

    result = await framework.validate_data_integrity()

    assert result.passed is True
    assert result.value == pytest.approx(1.0)
    assert result.details == "All required fields satisfied"


@pytest.mark.asyncio
async def test_validate_data_integrity_flags_schema_violations(monkeypatch):
    framework = ValidationFramework()

    def _bad_payload(_self: ValidationFramework) -> dict:
        return {
            "symbol": " ",
            "price": -1.0,
            "volume": 42.5,
            "timestamp": datetime.utcnow() + timedelta(days=365),
        }

    monkeypatch.setattr(ValidationFramework, "get_reference_payload", _bad_payload)

    result = await framework.validate_data_integrity()

    assert result.passed is False
    assert result.value == pytest.approx(0.25)
    assert result.metadata["invalid_fields"] == {
        "symbol": "value cannot be empty",
        "price": "value below minimum 0.0",
        "volume": "value must be an integer",
    }
    assert "Data integrity violations" in result.details
