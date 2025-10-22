"""Regression coverage for :mod:`src.governance.audit_logger`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.governance.audit_logger import AuditLogger


def _write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def test_audit_history_filters_skip_entries_with_invalid_timestamp(tmp_path) -> None:
    log_path = Path(tmp_path) / "audit.jsonl"
    _write_log(
        log_path,
        [
            '{"timestamp": "2024-01-01T00:00:00", "event_type": "system_event"}',
            '{"timestamp": "not-a-timestamp", "event_type": "system_event"}',
            '{"timestamp": "2024-01-03T00:00:00", "event_type": "system_event"}',
        ],
    )

    logger = AuditLogger(log_file=str(log_path))

    history = logger.get_audit_history(start_time=datetime(2024, 1, 2))

    assert [entry["timestamp"] for entry in history] == ["2024-01-03T00:00:00"]


def test_audit_history_preserves_entries_without_filters(tmp_path) -> None:
    log_path = Path(tmp_path) / "audit.jsonl"
    _write_log(
        log_path,
        [
            '{"timestamp": "2024-01-01T00:00:00", "event_type": "system_event"}',
            '{"timestamp": "invalid", "event_type": "system_event"}',
        ],
    )

    logger = AuditLogger(log_file=str(log_path))

    history = logger.get_audit_history()

    assert len(history) == 2


def test_audit_statistics_ignores_corrupt_entries(tmp_path) -> None:
    log_path = Path(tmp_path) / "audit.jsonl"
    _write_log(
        log_path,
        [
            '{"timestamp": "2024-01-01T00:00:00", "event_type": "decision", "strategy_id": "alpha"}',
            '"not-a-dict"',
            '{"timestamp": "invalid", "event_type": "decision"}',
        ],
    )

    logger = AuditLogger(log_file=str(log_path))

    stats = logger.get_audit_statistics()

    assert stats["total_entries"] == 1
    assert stats["event_types"] == {"decision": 1}
    assert stats["strategies"] == {"alpha": 1}
    assert stats["date_range"] == {
        "start": "2024-01-01T00:00:00",
        "end": "2024-01-01T00:00:00",
        "duration_days": 0,
    }


def test_log_entries_include_integrity_chain(tmp_path) -> None:
    log_path = Path(tmp_path) / "audit.jsonl"
    logger = AuditLogger(log_file=str(log_path))

    logger.log_decision("approve", "strategy-1", "genome-1")
    logger.log_system_event("heartbeat", "governance", "info", "ok")

    lines = [line for line in log_path.read_text().splitlines() if line]
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["integrity"]["previous_hash"] is None
    assert second["integrity"]["previous_hash"] == first["integrity"]["hash"]


def test_verify_integrity_reports_tampering(tmp_path) -> None:
    log_path = Path(tmp_path) / "audit.jsonl"
    logger = AuditLogger(log_file=str(log_path))

    logger.log_system_event("heartbeat", "governance", "info", "ok")

    result = logger.verify_integrity()
    assert result["valid"] is True
    assert result["checked_entries"] == 1

    # Tamper with the stored entry and expect verification failure.
    original = log_path.read_text().splitlines()[0]
    payload = json.loads(original)
    payload["message"] = "tampered"
    log_path.write_text(json.dumps(payload) + "\n")

    tampered_result = logger.verify_integrity()
    assert tampered_result["valid"] is False
    assert tampered_result["violations"], "Expected violations after tampering"
    reasons = {violation.reason for violation in tampered_result["violations"]}
    assert "hash mismatch" in reasons
