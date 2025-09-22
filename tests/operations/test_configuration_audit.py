"""Tests for configuration audit telemetry."""

from datetime import UTC, datetime

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

    markdown = format_configuration_audit_markdown(snapshot)
    assert "Initial configuration snapshot" in markdown


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

    markdown = snapshot.to_markdown()
    assert "KAFKA_BROKERS" in markdown
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
