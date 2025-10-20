from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Mapping

import aiohttp
import pytest

from src.core._event_bus_impl import EventBusStatistics
from src.governance.system_config import ConnectionProtocol, DataBackboneMode
from src.operations.data_backbone import (
    BackboneComponentSnapshot,
    BackboneStatus,
    DataBackboneReadinessSnapshot,
)
from src.runtime.healthcheck import RuntimeHealthServer, evaluate_runtime_health
from src.security.auth_tokens import create_access_token


AUTH_SECRET = "unit-test-runtime-health-secret"


def _build_token(*roles: str) -> str:
    return create_access_token(
        "unit-test-client",
        secret=AUTH_SECRET,
        roles=roles,
        expires_in=timedelta(minutes=5),
    )



_TEST_CERT = Path(__file__).parent / "certs" / "server.pem"
_TEST_KEY = Path(__file__).parent / "certs" / "server.key"


class _DummyConfig:
    def __init__(
        self, *, connection_protocol: ConnectionProtocol, data_backbone_mode: DataBackboneMode
    ):
        self.connection_protocol = connection_protocol
        self.data_backbone_mode = data_backbone_mode
        self.extras: dict[str, str] = {}


class _DummyEventBus:
    def __init__(self, running: bool, stats: EventBusStatistics | None = None) -> None:
        self._running = running
        if stats is None:
            now = time.time()
            stats = EventBusStatistics(
                running=running,
                loop_running=running,
                queue_size=0,
                queue_capacity=None,
                subscriber_count=0,
                topic_subscribers={},
                published_events=0,
                dropped_events=0,
                handler_errors=0,
                last_event_timestamp=now,
                last_error_timestamp=None,
                started_at=now,
                uptime_seconds=0.0,
            )
        self._stats = stats

    def is_running(self) -> bool:
        return self._running

    def get_statistics(self) -> EventBusStatistics:
        return self._stats


class _DummyFixManager:
    def __init__(self, initiator_present: bool) -> None:
        self._initiator_present = initiator_present

    def get_initiator(self, _session: str):
        return object() if self._initiator_present else None


class _DummyBroker:
    def __init__(self, running: bool, metrics: Mapping[str, object] | None = None) -> None:
        self.running = running
        self._metrics = dict(metrics or {})

    def describe_metrics(self) -> Mapping[str, object]:
        return dict(self._metrics)


class _DummyTradingManager:
    def __init__(self, stats: Mapping[str, object]) -> None:
        self._stats = dict(stats)

    def get_execution_stats(self) -> Mapping[str, object]:
        return dict(self._stats)


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
        trading_manager=None,
    ) -> None:
        self.config = config
        self.fix_connection_manager = fix_manager
        self.broker_interface = broker
        self.sensory_organ = sensory
        self._snapshot = snapshot
        self.event_bus = event_bus or _DummyEventBus(True)
        self.trading_manager = trading_manager

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

    server = RuntimeHealthServer(app, host="127.0.0.1", port=0, auth_secret=AUTH_SECRET)
    await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            token = _build_token("runtime.health:read")
            async with session.get(
                server.url, headers={"Authorization": f"Bearer {token}"}
            ) as response:
                payload = await response.json()
        assert payload["status"] in {"ok", "warn", "fail"}
        assert any(check["name"] == "market_data" for check in payload["checks"])
    finally:
        await server.stop()


@pytest.mark.asyncio()
async def test_runtime_health_server_serves_metrics() -> None:
    cfg = _DummyConfig(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
    )
    now = time.time()
    bus_stats = EventBusStatistics(
        running=True,
        loop_running=True,
        queue_size=3,
        queue_capacity=10,
        subscriber_count=5,
        topic_subscribers={},
        published_events=12,
        dropped_events=4,
        handler_errors=6,
        last_event_timestamp=now - 1.5,
        last_error_timestamp=None,
        started_at=now - 60,
        uptime_seconds=60.0,
    )
    broker_metrics = {
        "p50_latency_s": 0.12,
        "p90_latency_s": 0.25,
        "p99_latency_s": 0.75,
    }
    tm_stats = {
        "guardrail_force": {"force_paper": True, "expires_at": datetime.now(tz=UTC).isoformat()},
    }
    app = _DummyApp(
        config=cfg,
        snapshot=_fresh_snapshot(age_seconds=5),
        broker=_DummyBroker(running=True, metrics=broker_metrics),
        event_bus=_DummyEventBus(True, stats=bus_stats),
        trading_manager=_DummyTradingManager(tm_stats),
    )

    server = RuntimeHealthServer(app, host="127.0.0.1", port=0, auth_secret=AUTH_SECRET)
    await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            token = _build_token("runtime.health:read", "runtime.metrics:read")
            async with session.get(
                server.metrics_url, headers={"Authorization": f"Bearer {token}"}
            ) as response:
                body = await response.text()
        assert response.status == 200
        assert "event_lag_ms" in body
        assert "p50_infer_ms" in body
        assert "risk_halted 1" in body
        assert "runtime_latency_p99_seconds 0.75" in body
        assert "event_handler_exception_rate_per_minute 6" in body
        assert "runtime_exception_rate_per_minute 6" in body
        assert "runtime_exceptions_total 6" in body
        assert "runtime_health_status 0" in body
        assert 'runtime_health_check_status{check="market_data"} 0' in body
    finally:
        await server.stop()


@pytest.mark.asyncio()
async def test_runtime_health_server_requires_roles() -> None:
    cfg = _DummyConfig(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
    )
    app = _DummyApp(config=cfg, snapshot=_fresh_snapshot(age_seconds=10))

    server = RuntimeHealthServer(app, host="127.0.0.1", port=0, auth_secret=AUTH_SECRET)
    await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server.url) as unauthorized:
                assert unauthorized.status == 401
                await unauthorized.text()

            token = _build_token("runtime.health:read")
            async with session.get(
                server.metrics_url, headers={"Authorization": f"Bearer {token}"}
            ) as forbidden:
                assert forbidden.status == 403
                await forbidden.text()

            metrics_token = _build_token("runtime.health:read", "runtime.metrics:read")
            async with session.get(
                server.metrics_url, headers={"Authorization": f"Bearer {metrics_token}"}
            ) as ok_response:
                assert ok_response.status == 200
                await ok_response.text()
    finally:
        await server.stop()

