"""Regression coverage for :mod:`src.governance.audit_logger`."""

from __future__ import annotations

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
