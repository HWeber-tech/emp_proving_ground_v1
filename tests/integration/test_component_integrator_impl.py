import pytest

from src.integration.component_integrator_impl import ComponentIntegratorImpl


@pytest.mark.asyncio
async def test_initialize_registers_sensory_aliases():
    integrator = ComponentIntegratorImpl()

    assert await integrator.initialize() is True

    what_sensor = integrator.get_component("what_sensor")
    when_sensor = integrator.get_component("when_sensor")
    anomaly_sensor = integrator.get_component("anomaly_sensor")

    assert what_sensor is not None
    assert when_sensor is not None
    assert anomaly_sensor is not None

    # Legacy aliases should resolve to the same objects for downstream validators
    assert integrator.get_component("what_organ") is what_sensor
    assert integrator.get_component("when_organ") is when_sensor
    assert integrator.get_component("anomaly_organ") is anomaly_sensor

    # Data flow validation should reference whichever sensory key is available
    validation = await integrator.validate_integration()
    data_flow = validation["components"].get("data_flow")
    assert data_flow is not None
    assert data_flow["available"] is True
    assert data_flow["test_passed"] is True
    assert data_flow["source"] in {"what_sensor", "what_organ"}

    await integrator.shutdown()
