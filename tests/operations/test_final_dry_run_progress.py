from __future__ import annotations

from datetime import timedelta

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run_progress import (
    format_progress_snapshot,
    parse_progress_snapshot,
)


def test_parse_progress_snapshot_basic() -> None:
    payload = {
        "status": "running",
        "phase": "running",
        "now": "2024-03-01T12:01:00Z",
        "started_at": "2024-03-01T12:00:00Z",
        "elapsed_seconds": 60,
        "target_duration_seconds": 3600,
        "required_duration_seconds": 7200,
        "total_lines": 120,
        "line_counts": {"stdout": 100, "stderr": 20},
        "level_counts": {"info": 110, "warning": 10},
        "first_line_at": "2024-03-01T12:00:05Z",
        "last_line_at": "2024-03-01T12:00:55Z",
        "minimum_uptime_ratio": 0.98,
        "require_diary_evidence": True,
        "require_performance_evidence": False,
        "config_metadata": {"run_id": "demo"},
        "incidents": [
            {
                "severity": "warn",
                "occurred_at": "2024-03-01T12:00:45Z",
                "message": "Decision diary stale",
                "metadata": {"path": "diary.jsonl"},
            }
        ],
        "summary": {"status": "warn"},
        "sign_off": {"status": "fail"},
    }

    snapshot = parse_progress_snapshot(payload)

    assert snapshot.status == "running"
    assert snapshot.status_severity is None
    assert snapshot.phase == "running"
    assert snapshot.elapsed == timedelta(seconds=60)
    assert snapshot.target_duration == timedelta(hours=1)
    assert snapshot.required_duration == timedelta(hours=2)
    assert snapshot.countdown == timedelta(seconds=3540)
    assert snapshot.total_lines == 120
    assert snapshot.line_counts == {"stderr": 20, "stdout": 100}
    assert snapshot.level_counts == {"info": 110, "warning": 10}
    assert snapshot.minimum_uptime_ratio == 0.98
    assert snapshot.require_diary_evidence is True
    assert snapshot.require_performance_evidence is False
    assert snapshot.config_metadata == {"run_id": "demo"}
    assert snapshot.summary_status is DryRunStatus.warn
    assert snapshot.sign_off_status is DryRunStatus.fail
    assert len(snapshot.incidents) == 1
    incident = snapshot.incidents[0]
    assert incident.severity is DryRunStatus.warn
    assert incident.message == "Decision diary stale"
    assert incident.metadata == {"path": "diary.jsonl"}


def test_format_progress_snapshot_includes_sections() -> None:
    payload = {
        "status": "pass",
        "phase": "finishing",
        "now": "2024-03-02T12:00:00Z",
        "started_at": "2024-02-28T12:00:00Z",
        "elapsed_seconds": 72 * 3600,
        "target_duration_seconds": 72 * 3600,
        "required_duration_seconds": 72 * 3600,
        "total_lines": 4096,
        "line_counts": {"stdout": 3900, "stderr": 196},
        "level_counts": {"info": 4080, "warning": 16},
        "first_line_at": "2024-02-28T12:00:05Z",
        "last_line_at": "2024-03-02T11:59:58Z",
        "minimum_uptime_ratio": 0.99,
        "require_diary_evidence": True,
        "require_performance_evidence": True,
        "config_metadata": {"objective": "uat"},
        "incidents": [
            {
                "severity": "warn",
                "occurred_at": "2024-02-28T18:00:00Z",
                "message": "Temporary latency spike",
                "metadata": {"scope": "ingest"},
            },
            {
                "severity": "fail",
                "occurred_at": "2024-02-28T20:00:00Z",
                "message": "Evidence stale",
                "metadata": {"artifact": "performance"},
            },
        ],
        "summary": {"status": "pass"},
        "sign_off": {"status": "pass"},
    }

    snapshot = parse_progress_snapshot(payload)

    text = format_progress_snapshot(snapshot, max_incidents=2)

    assert "Status: PASS" in text
    assert "Elapsed:" in text
    assert "3d" in text
    assert "100.0%" in text
    assert "Required minimum" in text
    assert "Streams: stderr=196, stdout=3900" in text
    assert "Levels: info=4080, warning=16" in text
    assert "Required evidence: diary, performance" in text
    assert "Config metadata: objective=uat" in text
    assert "Summary status: PASS" in text
    assert "Sign-off status: PASS" in text
    assert "Incidents:" in text
    assert "[FAIL]" in text
    assert "[WARN]" in text
