import pytest

from src.integration.component_integrator_impl import ComponentIntegratorImpl
from src.sensory.lineage_publisher import SensoryLineagePublisher
from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.config.risk.risk_config import RiskConfig


@pytest.mark.asyncio
async def test_initialize_registers_sensory_aliases():
    integrator = ComponentIntegratorImpl()

    assert await integrator.initialize() is True

    sensory_organ = integrator.get_component("sensory_organ")
    lineage_publisher = integrator.get_component("sensory_lineage_publisher")
    why_sensor = integrator.get_component("why_sensor")
    how_sensor = integrator.get_component("how_sensor")
    what_sensor = integrator.get_component("what_sensor")
    when_sensor = integrator.get_component("when_sensor")
    anomaly_sensor = integrator.get_component("anomaly_sensor")

    assert isinstance(sensory_organ, RealSensoryOrgan)
    assert isinstance(lineage_publisher, SensoryLineagePublisher)
    assert why_sensor is not None
    assert how_sensor is not None
    assert what_sensor is not None
    assert when_sensor is not None
    assert anomaly_sensor is not None

    # Legacy aliases should resolve to the same objects for downstream validators
    assert integrator.get_component("how_organ") is how_sensor
    assert integrator.get_component("what_organ") is what_sensor
    assert integrator.get_component("when_organ") is when_sensor
    assert integrator.get_component("anomaly_organ") is anomaly_sensor

    # Data flow validation should reference whichever sensory key is available
    validation = await integrator.validate_integration()
    data_flow = validation["components"].get("data_flow")
    assert data_flow is not None
    assert data_flow["available"] is True
    assert data_flow["test_passed"] is True
    assert data_flow["source"] in {"sensory_organ", "what_sensor", "what_organ"}

    risk_entry = validation["components"].get("risk_configuration")
    assert risk_entry is not None
    assert risk_entry["available"] is True
    assert risk_entry["runbook"].endswith("risk_api_contract.md")
    summary = risk_entry["summary"]
    assert summary["mandatory_stop_loss"] is True
    assert summary["max_risk_per_trade_pct"] > 0

    risk_manager = integrator.get_component("risk_manager")
    assert risk_manager is not None
    config = getattr(risk_manager, "_risk_config", None)
    assert isinstance(config, RiskConfig)

    await integrator.shutdown()
