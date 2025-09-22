from __future__ import annotations

from datetime import UTC, datetime, timedelta

import aiohttp
import pytest

from src.governance.system_config import ConnectionProtocol, DataBackboneMode
from src.operations.data_backbone import (
    BackboneComponentSnapshot,
    BackboneStatus,
    DataBackboneReadinessSnapshot,
)
from src.runtime.healthcheck import RuntimeHealthServer, evaluate_runtime_health


class _DummyConfig:
    def __init__(
        self, *, connection_protocol: ConnectionProtocol, data_backbone_mode: DataBackboneMode
    ):
        self.connection_protocol = connection_protocol
        self.data_backbone_mode = data_backbone_mode
        self.extras: dict[str, str] = {}


class _DummyEventBus:
    def __init__(self, running: bool) -> None:
        self._running = running

    def is_running(self) -> bool:
        return self._running


class _DummyFixManager:
    def __init__(self, initiator_present: bool) -> None:
        self._initiator_present = initiator_present

    def get_initiator(self, _session: str):
        return object() if self._initiator_present else None


class _DummyBroker:
    def __init__(self, running: bool) -> None:
        self.running = running


class _DummySensory:
    def __init__(self, generated_at: datetime | None) -> None:
        self._generated_at = generated_at

    def status(self):
        if self._generated_at is None:
            telemetry = {"last_decision": None}
        else:
            telemetry = {"last_decision": {"generated_at": self._generated_at.isoformat()}}
        return {"telemetry": telemetry}


class _DummyApp:
    def __init__(
        self,
        *,
        config: _DummyConfig,
        fix_manager=None,
        broker=None,
        sensory=None,
        snapshot: DataBackboneReadinessSnapshot | None = None,
        event_bus: _DummyEventBus | None = None,
    ) -> None:
        self.config = config
        self.fix_connection_manager = fix_manager
        self.broker_interface = broker
        self.sensory_organ = sensory
        self._snapshot = snapshot
        self.event_bus = event_bus or _DummyEventBus(True)

    def get_last_data_backbone_snapshot(self) -> DataBackboneReadinessSnapshot | None:
        return self._snapshot


def _fresh_snapshot(age_seconds: float) -> DataBackboneReadinessSnapshot:
    generated = datetime.now(tz=UTC) - timedelta(seconds=age_seconds)
    component = BackboneComponentSnapshot(
        name="ingest_health",
        status=BackboneStatus.ok,
        summary="ok",
        metadata={"generated_at": generated.isoformat()},
    )
    return DataBackboneReadinessSnapshot(
        status=BackboneStatus.ok,
        generated_at=datetime.now(tz=UTC),
        components=(component,),
        metadata={},
    )


def test_evaluate_runtime_health_flags_fix_failures() -> None:
    cfg = _DummyConfig(
        connection_protocol=ConnectionProtocol.fix,
        data_backbone_mode=DataBackboneMode.bootstrap,
    )
    app = _DummyApp(
        config=cfg,
        fix_manager=_DummyFixManager(initiator_present=False),
        broker=_DummyBroker(running=False),
        event_bus=_DummyEventBus(True),
    )

    snapshot = evaluate_runtime_health(app)
    fix_check = next(check for check in snapshot.checks if check.name == "fix_connectivity")
    assert fix_check.status == "fail"


def test_evaluate_runtime_health_marks_stale_ingest() -> None:
    cfg = _DummyConfig(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
    )
    stale_snapshot = _fresh_snapshot(age_seconds=3600)
    app = _DummyApp(config=cfg, snapshot=stale_snapshot)

    snapshot = evaluate_runtime_health(app)
    market_check = next(check for check in snapshot.checks if check.name == "market_data")
    assert market_check.status == "fail"


def test_evaluate_runtime_health_detects_missing_kafka_publishers() -> None:
    cfg = _DummyConfig(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
    )
    metadata_component = BackboneComponentSnapshot(
        name="kafka_streaming",
        status=BackboneStatus.warn,
        summary="no publishers",
        metadata={"topics": ["telemetry.ingest"], "publishers": []},
    )
    snapshot = DataBackboneReadinessSnapshot(
        status=BackboneStatus.warn,
        generated_at=datetime.now(tz=UTC),
        components=(metadata_component,),
        metadata={},
    )
    app = _DummyApp(config=cfg, snapshot=snapshot)

    result = evaluate_runtime_health(app)
    telemetry_check = next(check for check in result.checks if check.name == "telemetry_exporters")
    assert telemetry_check.status == "fail"


@pytest.mark.asyncio()
async def test_runtime_health_server_serves_snapshot() -> None:
    cfg = _DummyConfig(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
    )
    snapshot = _fresh_snapshot(age_seconds=10)
    app = _DummyApp(config=cfg, snapshot=snapshot)

    server = RuntimeHealthServer(app, host="127.0.0.1", port=0)
    await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server.url) as response:
                payload = await response.json()
        assert payload["status"] in {"ok", "warn", "fail"}
        assert any(check["name"] == "market_data" for check in payload["checks"])
    finally:
        await server.stop()
