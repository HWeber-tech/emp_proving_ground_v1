import pytest

from src.integration.component_integrator import ComponentIntegrator
from src.sensory.lineage_publisher import SensoryLineagePublisher
from src.sensory.real_sensory_organ import RealSensoryOrgan


@pytest.mark.asyncio
async def test_component_integrator_initializes_canonical_sensory_stack() -> None:
    integrator = ComponentIntegrator()

    assert await integrator.initialize_components() is True

    sensory_organ = integrator.components.get("sensory_organ")
    lineage = integrator.components.get("sensory_lineage_publisher")
    how_sensor = integrator.components.get("how_sensor")
    anomaly_sensor = integrator.components.get("anomaly_sensor")

    assert isinstance(sensory_organ, RealSensoryOrgan)
    assert isinstance(lineage, SensoryLineagePublisher)
    assert how_sensor is not None
    assert anomaly_sensor is not None
    assert integrator.components.get("how_organ") is how_sensor

    await integrator.shutdown_components()
