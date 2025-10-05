"""Tests for configuration audit telemetry."""

import logging
from datetime import UTC, datetime

import pytest

from src.governance.system_config import (
    ConnectionProtocol,
    DataBackboneMode,
    EmpEnvironment,
    EmpTier,
    RunMode,
    SystemConfig,
)
from src.operations.configuration_audit import (
    ConfigurationAuditStatus,
    evaluate_configuration_audit,
    format_configuration_audit_markdown,
    publish_configuration_audit_snapshot,
)
from src.operations.event_bus_failover import EventPublishError


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish_from_sync(self, event) -> int:
        self.events.append(event)
        return 1

    def is_running(self) -> bool:
        return True


def test_configuration_audit_initial_snapshot_marks_initial() -> None:
    config = SystemConfig()

    snapshot = evaluate_configuration_audit(config)

    assert snapshot.status is ConfigurationAuditStatus.passed
    assert snapshot.metadata.get("initial_snapshot") is True
    assert snapshot.changes == ()
    assert snapshot.metadata["severity_counts"] == {"pass": 0, "warn": 0, "fail": 0}
    assert snapshot.metadata["highest_severity_fields"] == []

    markdown = format_configuration_audit_markdown(snapshot)
    assert "Initial configuration snapshot" in markdown
    assert "- Pass: 0" in markdown
    assert "- Warn: 0" in markdown
    assert "- Fail: 0" in markdown


def test_configuration_audit_detects_high_risk_changes() -> None:
    baseline = evaluate_configuration_audit(SystemConfig())

    modified = SystemConfig().with_updated(
        run_mode=RunMode.live,
        confirm_live=True,
        tier=EmpTier.tier_1,
        environment=EmpEnvironment.production,
        connection_protocol=ConnectionProtocol.fix,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "KAFKA_BROKERS": "broker:9092",
            "CUSTOM_FLAG": "enabled",
        },
    )

    snapshot = evaluate_configuration_audit(
        modified,
        previous=baseline.current_config,
        metadata={"timestamp": datetime.now(tz=UTC).isoformat()},
    )

    assert snapshot.status is ConfigurationAuditStatus.fail
    fields = {change.field: change for change in snapshot.changes}
    assert fields["run_mode"].severity is ConfigurationAuditStatus.fail
    assert fields["confirm_live"].severity is ConfigurationAuditStatus.fail
    assert fields["tier"].severity is ConfigurationAuditStatus.warn
    assert fields["connection_protocol"].severity is ConfigurationAuditStatus.warn
    assert fields["extras.KAFKA_BROKERS"].severity is not ConfigurationAuditStatus.passed
    assert snapshot.metadata["severity_counts"] == {"pass": 1, "warn": 5, "fail": 2}
    assert snapshot.metadata["highest_severity_fields"] == ["run_mode", "confirm_live"]

    markdown = snapshot.to_markdown()
    assert "KAFKA_BROKERS" in markdown
    assert "- Fail: 2" in markdown
    assert "- Warn: 5" in markdown
    extras_summary = snapshot.metadata.get("extras_summary")
    assert extras_summary["added"] == ["CUSTOM_FLAG", "KAFKA_BROKERS"]


def test_publish_configuration_audit_snapshot_prefers_runtime_bus(monkeypatch) -> None:
    baseline = evaluate_configuration_audit(SystemConfig())
    config = SystemConfig().with_updated(extras={"KAFKA_BROKERS": "broker:9092"})
    snapshot = evaluate_configuration_audit(
        config,
        previous=baseline.current_config,
    )

    bus = _StubEventBus()
    publish_configuration_audit_snapshot(bus, snapshot)

    assert bus.events, "expected configuration audit event"
    event = bus.events[0]
    assert event.type == "telemetry.runtime.configuration"
    assert event.payload["snapshot_id"] == snapshot.snapshot_id


def test_publish_configuration_audit_snapshot_falls_back_to_global_bus(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    baseline = evaluate_configuration_audit(SystemConfig())
    config = SystemConfig().with_updated(extras={"KAFKA_BROKERS": "broker:9092"})
    snapshot = evaluate_configuration_audit(
        config,
        previous=baseline.current_config,
    )

    class _FailingEventBus(_StubEventBus):
        def publish_from_sync(self, event) -> int:  # type: ignore[override]
            raise RuntimeError("offline")

    class _TopicBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, object, str | None]] = []

        def publish_sync(
            self, topic: str, payload: object, *, source: str | None = None
        ) -> int:
            self.published.append((topic, payload, source))
            return 1

    captured_topic_bus = _TopicBus()

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus",
        lambda: captured_topic_bus,
    )

    bus = _FailingEventBus()

    with caplog.at_level(logging.WARNING):
        publish_configuration_audit_snapshot(bus, snapshot)

    assert captured_topic_bus.published
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "falling back to global bus" in messages


def test_publish_configuration_audit_snapshot_raises_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = evaluate_configuration_audit(SystemConfig())
    config = SystemConfig().with_updated(extras={"KAFKA_BROKERS": "broker:9092"})
    snapshot = evaluate_configuration_audit(
        config,
        previous=baseline.current_config,
    )

    class _UnexpectedBus(_StubEventBus):
        def publish_from_sync(self, event) -> int:  # type: ignore[override]
            raise ValueError("boom")

    bus = _UnexpectedBus()

    with pytest.raises(EventPublishError) as excinfo:
        publish_configuration_audit_snapshot(bus, snapshot)

    assert excinfo.value.stage == "runtime"
    assert excinfo.value.event_type == "telemetry.runtime.configuration"
