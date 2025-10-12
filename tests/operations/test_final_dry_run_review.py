from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys

import datetime as _datetime

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = timezone.utc  # type: ignore[attr-defined]
UTC = _datetime.UTC

_MODULE_ROOT = Path(__file__).resolve().parents[2] / "src" / "operations"

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "src.operations.dry_run_audit", _MODULE_ROOT / "dry_run_audit.py"
)
assert _AUDIT_SPEC and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
sys.modules["src.operations.dry_run_audit"] = _AUDIT_MODULE
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)

_PACKET_SPEC = importlib.util.spec_from_file_location(
    "src.operations.dry_run_packet", _MODULE_ROOT / "dry_run_packet.py"
)
assert _PACKET_SPEC and _PACKET_SPEC.loader is not None
_PACKET_MODULE = importlib.util.module_from_spec(_PACKET_SPEC)
sys.modules["src.operations.dry_run_packet"] = _PACKET_MODULE
_PACKET_SPEC.loader.exec_module(_PACKET_MODULE)

_REVIEW_SPEC = importlib.util.spec_from_file_location(
    "src.operations.final_dry_run_review", _MODULE_ROOT / "final_dry_run_review.py"
)
assert _REVIEW_SPEC and _REVIEW_SPEC.loader is not None
_REVIEW_MODULE = importlib.util.module_from_spec(_REVIEW_SPEC)
sys.modules["src.operations.final_dry_run_review"] = _REVIEW_MODULE
_REVIEW_SPEC.loader.exec_module(_REVIEW_MODULE)

DryRunDiaryIssue = _AUDIT_MODULE.DryRunDiaryIssue
DryRunDiarySummary = _AUDIT_MODULE.DryRunDiarySummary
DryRunIncident = _AUDIT_MODULE.DryRunIncident
DryRunLogSummary = _AUDIT_MODULE.DryRunLogSummary
DryRunPerformanceSummary = _AUDIT_MODULE.DryRunPerformanceSummary
DryRunSignOffFinding = _AUDIT_MODULE.DryRunSignOffFinding
DryRunSignOffReport = _AUDIT_MODULE.DryRunSignOffReport
DryRunStatus = _AUDIT_MODULE.DryRunStatus
DryRunSummary = _AUDIT_MODULE.DryRunSummary
StructuredLogRecord = _AUDIT_MODULE.StructuredLogRecord
build_review = _REVIEW_MODULE.build_review

from src.understanding.decision_diary import DecisionDiaryEntry


def _record(ts: datetime, level: str, event: str, message: str) -> StructuredLogRecord:
    return StructuredLogRecord(
        timestamp=ts,
        level=level,
        event=event,
        message=message,
        payload={},
    )


def test_build_review_collects_incidents() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=3, minutes=5)
    records = (
        _record(start, "info", "startup", "startup"),
        _record(start + timedelta(hours=1), "warning", "latency", "latency"),
        _record(start + timedelta(hours=2), "error", "handler", "failure"),
        _record(end, "info", "shutdown", "shutdown"),
    )
    gap_incident = DryRunIncident(
        severity=DryRunStatus.warn,
        occurred_at=start + timedelta(hours=12),
        summary="6h gap detected",
        metadata={"gap_seconds": 6 * 3600},
    )
    content_incident = DryRunIncident(
        severity=DryRunStatus.warn,
        occurred_at=start + timedelta(hours=30),
        summary="Anomaly cluster",
        metadata={"event": "anomaly_cluster"},
    )
    log_summary = DryRunLogSummary(
        records=records,
        ignored_lines=0,
        level_counts={"info": 2, "warning": 1, "error": 1},
        event_counts={"startup": 1, "latency": 1, "handler": 1, "shutdown": 1},
        gap_incidents=(gap_incident,),
        content_incidents=(content_incident,),
        uptime_ratio=0.992,
    )

    diary_entry = DecisionDiaryEntry(
        entry_id="d-1",
        recorded_at=start + timedelta(days=1),
        policy_id="risk",
        decision={"tactic_id": "risk_beta"},
        regime_state={"regime": "calm"},
        outcomes={"status": "ok"},
    )
    diary_issue = DryRunDiaryIssue(
        entry_id="d-1",
        policy_id="risk",
        recorded_at=diary_entry.recorded_at,
        severity=DryRunStatus.warn,
        reason="Missing Sharpe annotation",
        metadata={},
    )
    diary_summary = DryRunDiarySummary(
        entries=(diary_entry,),
        issues=(diary_issue,),
        policy_counts={"risk": 1},
    )

    performance_summary = DryRunPerformanceSummary(
        generated_at=end,
        period_start=start,
        total_trades=18,
        roi=-0.01,
        win_rate=0.45,
        sharpe_ratio=0.6,
    )

    summary = DryRunSummary(
        generated_at=end,
        log_summary=log_summary,
        diary_summary=diary_summary,
        performance_summary=performance_summary,
        metadata={"run_id": "demo"},
    )

    sign_off_report = DryRunSignOffReport(
        evaluated_at=end,
        findings=(
            DryRunSignOffFinding(
                severity=DryRunStatus.fail,
                message="Dry run duration below required minimum.",
                metadata={"required_seconds": 72 * 3600, "actual_seconds": 60 * 3600},
            ),
        ),
        criteria={"minimum_duration_seconds": 72 * 3600},
    )

    review = build_review(
        summary,
        sign_off_report,
        run_label="Q1 Final Dry Run",
        attendees=["Alice", " Bob "],
        notes=("Follow up on latency alert",),
    )

    assert review.status is DryRunStatus.fail
    assert review.run_label == "Q1 Final Dry Run"
    assert review.attendees == ("Alice", "Bob")
    assert any(item.category == "logs" and item.severity is DryRunStatus.fail for item in review.action_items)
    assert any(item.category == "diary" for item in review.action_items)
    assert any(item.category == "performance" for item in review.action_items)
    assert any(item.category == "sign_off" for item in review.action_items)
    assert "Observed duration" in review.highlights


def test_review_to_markdown_includes_sections() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=3)
    records = (
        _record(start, "info", "startup", "startup"),
        _record(end, "info", "shutdown", "shutdown"),
    )
    log_summary = DryRunLogSummary(
        records=records,
        ignored_lines=0,
        level_counts={"info": 2},
        event_counts={"startup": 1, "shutdown": 1},
        gap_incidents=tuple(),
        content_incidents=tuple(),
        uptime_ratio=0.999,
    )
    summary = DryRunSummary(
        generated_at=end,
        log_summary=log_summary,
    )

    review = build_review(summary, run_label="Stability Sweep", attendees=["Ops"])
    markdown = review.to_markdown()

    assert "Final Dry Run Review" in markdown
    assert "Highlights" in markdown
    assert "Evidence Status" in markdown
    assert "Action Items" in markdown
    assert "Dry run summary" in markdown
    assert "Dry run sign-off" not in markdown
    assert "Stability Sweep" in markdown

    review_no_appendix = build_review(summary)
    markdown_trimmed = review_no_appendix.to_markdown(
        include_summary=False,
        include_sign_off=False,
    )
    assert "## Appendices" not in markdown_trimmed
