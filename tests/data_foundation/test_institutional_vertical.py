"""Security hardening tests for institutional ingest connectivity probes."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest


pytestmark = pytest.mark.guardrail

from src.data_foundation.ingest.configuration import InstitutionalIngestConfig
from src.data_foundation.ingest.institutional_vertical import (
    ConnectivityProbeError,
    InstitutionalIngestServices,
    ManagedConnectorSnapshot,
)
from src.data_foundation.ingest.scheduler import TimescaleIngestScheduler
from src.data_foundation.ingest.timescale_pipeline import TimescaleBackbonePlan
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.streaming.kafka_stream import KafkaConnectionSettings
from src.runtime.task_supervisor import TaskSupervisor


def _make_services() -> InstitutionalIngestServices:
    config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(),
        timescale_settings=TimescaleConnectionSettings(url="sqlite:///tmp/test.db"),
        kafka_settings=KafkaConnectionSettings(bootstrap_servers=""),
    )
    scheduler = Mock(spec=TimescaleIngestScheduler)
    supervisor = Mock(spec=TaskSupervisor)
    return InstitutionalIngestServices(
        config=config,
        scheduler=scheduler,
        task_supervisor=supervisor,
    )


def test_timescale_probe_expected_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    services = _make_services()
    caplog.set_level(logging.WARNING)

    def raise_ose(_: TimescaleConnectionSettings) -> None:
        raise OSError("engine boom")

    monkeypatch.setattr(TimescaleConnectionSettings, "create_engine", raise_ose)

    probe = services._default_connectivity_probes()["timescale"]
    with pytest.raises(ConnectivityProbeError):
        probe()
    assert "timescale connectivity probe failed" in caplog.text


def test_redis_probe_expected_failure(caplog: pytest.LogCaptureFixture) -> None:
    services = _make_services()
    caplog.set_level(logging.WARNING)

    class FailingRedisCache:
        def __init__(self) -> None:
            self.raw_client = self

        def ping(self) -> bool:
            raise TimeoutError("redis ping timeout")

    services.redis_cache = FailingRedisCache()

    probe = services._default_connectivity_probes()["redis"]
    with pytest.raises(ConnectivityProbeError):
        probe()
    assert "redis connectivity probe failed" in caplog.text


@pytest.mark.asyncio
async def test_connectivity_report_marks_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    services = _make_services()

    snapshot = ManagedConnectorSnapshot(
        name="timescale",
        configured=True,
        supervised=True,
        metadata={},
    )
    def manifest(_: InstitutionalIngestServices) -> tuple[ManagedConnectorSnapshot, ...]:
        return (snapshot,)

    monkeypatch.setattr(InstitutionalIngestServices, "managed_manifest", manifest)

    def failing_probe() -> bool:
        raise ConnectivityProbeError("expected failure")

    result = await services.connectivity_report(probes={"timescale": failing_probe})
    assert result[0].healthy is False


@pytest.mark.asyncio
async def test_connectivity_report_propagates_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    services = _make_services()

    snapshot = ManagedConnectorSnapshot(
        name="timescale",
        configured=True,
        supervised=True,
        metadata={},
    )
    def manifest(_: InstitutionalIngestServices) -> tuple[ManagedConnectorSnapshot, ...]:
        return (snapshot,)

    monkeypatch.setattr(InstitutionalIngestServices, "managed_manifest", manifest)

    def unexpected_probe() -> bool:
        raise RuntimeError("probe exploded")

    with pytest.raises(RuntimeError):
        await services.connectivity_report(probes={"timescale": unexpected_probe})
