import pytest

from src.validation.validation_framework import ValidationFramework


@pytest.mark.asyncio
async def test_validation_framework_component_integration_initializes_integrator():
    framework = ValidationFramework()

    result = await framework.validate_component_integration()

    assert result.passed is True
    assert result.value == pytest.approx(1.0)
    assert not result.metadata.get("missing")
    assert result.metadata.get("initialized") is True
