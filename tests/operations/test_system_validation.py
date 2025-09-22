import json
from datetime import datetime, timezone

import pytest

from src.operations.system_validation import (
    SystemValidationStatus,
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
