import pytest

from src.validation.real_market_validation import RealMarketValidationFramework


@pytest.mark.asyncio
async def test_real_market_validation_reports_missing_adapters():
    framework = RealMarketValidationFramework()

    result = await framework.validate_anomaly_detection_accuracy()

    assert result.passed is False
    missing = result.historical_data.get("missing_adapters")
    assert missing is not None
    assert "market_data_gateway" in missing
    assert "anomaly_detector" in missing
