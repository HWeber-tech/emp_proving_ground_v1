import json
from datetime import datetime, timezone

import pytest

from src.operations.event_bus_failover import EventPublishError
from src.operations.system_validation import (
    SystemValidationCheck,
    SystemValidationSnapshot,
    SystemValidationStatus,
    derive_system_validation_alerts,
    evaluate_system_validation,
    format_system_validation_markdown,
    load_system_validation_snapshot,
    publish_system_validation_snapshot,
)


class StubBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish_from_sync(self, event: object) -> None:
        self.events.append(event)

    def is_running(self) -> bool:
        return True


class RaisingRuntimeBus(StubBus):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    def publish_from_sync(self, event: object) -> None:  # type: ignore[override]
        raise self._exc


class StubTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, object, str | None]] = []

    def publish_sync(self, topic: str, payload: object, *, source: str | None = None) -> None:
        self.events.append((topic, payload, source))


def test_evaluate_system_validation_full_pass() -> None:
    report = {
        "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
        "validator": "System Completeness",
        "total_checks": 2,
        "results": {"core.interfaces": True, "integration": True},
        "summary": {"status": "PASS", "message": "All checks passed"},
    }

    snapshot = evaluate_system_validation(report, metadata={"window": "daily"})

    assert snapshot.status is SystemValidationStatus.passed
    assert snapshot.passed_checks == 2
    assert snapshot.failed_checks == 0
    assert snapshot.success_rate == pytest.approx(1.0)
    markdown = format_system_validation_markdown(snapshot)
    assert "system validation" in markdown.lower()
    assert "all checks passed" in markdown.lower()


def test_evaluate_system_validation_partial_warn() -> None:
    report = {
        "timestamp": datetime(2025, 1, 2, tzinfo=timezone.utc).isoformat(),
        "validator": "System Completeness",
        "total_checks": 3,
        "results": {
            "core.exceptions": True,
            "population_manager": False,
            "system_integration": True,
        },
        "summary": {"status": "PARTIAL", "message": "population manager pending"},
    }

    snapshot = evaluate_system_validation(report)

    assert snapshot.status is SystemValidationStatus.warn
    assert snapshot.passed_checks == 2
    assert snapshot.failed_checks == 1
    assert snapshot.success_rate == pytest.approx(2 / 3)
    assert any(not check.passed for check in snapshot.checks)
    assert snapshot.metadata.get("failing_check_names") == ("population_manager",)
    failing_checks = snapshot.metadata.get("failing_checks")
    assert failing_checks and failing_checks[0]["name"] == "population_manager"
    assert failing_checks[0]["message"] is None


def test_markdown_includes_failing_checks() -> None:
    report = {
        "timestamp": datetime(2025, 1, 2, tzinfo=timezone.utc).isoformat(),
        "total_checks": 2,
        "results": {
            "core.exceptions": True,
            "population_manager": False,
        },
    }

    snapshot = evaluate_system_validation(report)
    markdown = format_system_validation_markdown(snapshot)

    assert "failing checks" in markdown.lower()
    assert "population_manager" in markdown


def test_publish_system_validation_snapshot_emits_event() -> None:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_checks": 1,
        "results": {"core": True},
    }
    snapshot = evaluate_system_validation(report)

    bus = StubBus()
    publish_system_validation_snapshot(bus, snapshot, source="test")

    assert bus.events, "expected system validation telemetry event"
    event = bus.events[-1]
    assert getattr(event, "type", "") == "telemetry.operational.system_validation"
    payload = getattr(event, "payload", {})
    assert payload.get("status") == SystemValidationStatus.passed.value


def test_publish_system_validation_snapshot_falls_back_to_global_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "total_checks": 1, "results": {"core": True}}
    snapshot = evaluate_system_validation(report)

    runtime_bus = RaisingRuntimeBus(RuntimeError("loop stopped"))
    global_bus = StubTopicBus()

    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", lambda: global_bus)

    publish_system_validation_snapshot(runtime_bus, snapshot, source="test")

    assert not runtime_bus.events
    assert global_bus.events
    topic, payload, source = global_bus.events[-1]
    assert topic == "telemetry.operational.system_validation"
    assert payload.get("status") == SystemValidationStatus.passed.value
    assert source == "test"


def test_publish_system_validation_snapshot_raises_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "total_checks": 1, "results": {"core": True}}
    snapshot = evaluate_system_validation(report)

    runtime_bus = RaisingRuntimeBus(ValueError("boom"))
    global_bus = StubTopicBus()

    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", lambda: global_bus)

    with pytest.raises(EventPublishError) as exc_info:
        publish_system_validation_snapshot(runtime_bus, snapshot, source="test")

    assert not runtime_bus.events
    assert not global_bus.events
    assert exc_info.value.stage == "runtime"


def test_load_system_validation_snapshot_reads_file(tmp_path) -> None:
    report_path = tmp_path / "system_validation.json"
    report = {
        "timestamp": "2025-01-03T00:00:00+00:00",
        "validator": "System Completeness",
        "total_checks": 2,
        "results": {"core": True, "integration": False},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    snapshot = load_system_validation_snapshot(report_path, metadata={"run": "ci"})

    assert snapshot is not None
    assert snapshot.metadata.get("report_path") == str(report_path)
    assert snapshot.metadata.get("run") == "ci"
    assert snapshot.failed_checks == 1


def test_derive_system_validation_alerts_include_failing_checks() -> None:
    snapshot = SystemValidationSnapshot(
        status=SystemValidationStatus.fail,
        generated_at=datetime(2025, 2, 1, tzinfo=timezone.utc),
        total_checks=3,
        passed_checks=1,
        failed_checks=2,
        success_rate=1 / 3,
        checks=(
            SystemValidationCheck(name="database", passed=False, message="timeout"),
            SystemValidationCheck(name="event_bus", passed=True),
            SystemValidationCheck(name="ingest", passed=False),
        ),
        metadata={},
    )

    events = derive_system_validation_alerts(snapshot)

    categories = [event.category for event in events]
    assert "system.validation.status" in categories
    assert categories.count("system.validation.check") == 2
    failing_messages = [event.message for event in events if "validation check failed" in event.message.lower()]
    assert len(failing_messages) == 2
    for event in events:
        assert event.context.get("snapshot")


def test_derive_system_validation_alerts_threshold_blocks_warn() -> None:
    snapshot = SystemValidationSnapshot(
        status=SystemValidationStatus.warn,
        generated_at=datetime(2025, 2, 1, tzinfo=timezone.utc),
        total_checks=2,
        passed_checks=1,
        failed_checks=1,
        success_rate=0.5,
        checks=(
            SystemValidationCheck(name="database", passed=False, message="timeout"),
        ),
        metadata={},
    )

    events = derive_system_validation_alerts(snapshot, threshold=SystemValidationStatus.fail)
    assert events == []
