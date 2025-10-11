from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "operations" / "dry_run_audit.py"
_SPEC = importlib.util.spec_from_file_location("dry_run_audit_for_tests", _MODULE_PATH)
assert _SPEC and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["dry_run_audit_for_tests"] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DryRunDiaryIssue = _MODULE.DryRunDiaryIssue
DryRunDiarySummary = _MODULE.DryRunDiarySummary
DryRunLogSummary = _MODULE.DryRunLogSummary
DryRunPerformanceSummary = _MODULE.DryRunPerformanceSummary
DryRunSignOffReport = _MODULE.DryRunSignOffReport
DryRunStatus = _MODULE.DryRunStatus
DryRunSummary = _MODULE.DryRunSummary
LogParseResult = _MODULE.LogParseResult
StructuredLogRecord = _MODULE.StructuredLogRecord
analyse_structured_logs = _MODULE.analyse_structured_logs
assess_sign_off_readiness = _MODULE.assess_sign_off_readiness
evaluate_dry_run = _MODULE.evaluate_dry_run
humanise_timedelta = _MODULE.humanise_timedelta
load_structured_logs = _MODULE.load_structured_logs
parse_structured_log_line = _MODULE.parse_structured_log_line
summarise_diary_entries = _MODULE.summarise_diary_entries
from src.understanding.decision_diary import DecisionDiaryEntry


def _record(ts: str, level: str, event: str | None, message: str | None, **payload: object) -> StructuredLogRecord:
    return StructuredLogRecord(
        timestamp=datetime.fromisoformat(ts).replace(tzinfo=UTC),
        level=level,
        event=event,
        message=message,
        payload=payload,
    )


def test_parse_structured_log_line_handles_iso_timestamp() -> None:
    line = json.dumps({
        "timestamp": "2024-01-01T00:00:00Z",
        "level": "INFO",
        "event": "bootstrap.start",
        "message": "starting",
        "component": "alpha",
    })
    record = parse_structured_log_line(line)
    assert record is not None
    assert record.timestamp == datetime(2024, 1, 1, tzinfo=UTC)
    assert record.level == "info"
    assert record.event == "bootstrap.start"
    assert record.payload == {"component": "alpha"}


def test_analyse_structured_logs_flags_errors() -> None:
    records = (
        _record("2024-01-01T00:00:00", "info", "start", "start"),
        _record("2024-01-01T01:00:00", "warning", "latency", "latency high"),
        _record("2024-01-01T02:00:00", "error", "crash", "failure"),
    )
    summary = analyse_structured_logs(LogParseResult(records=records, ignored_lines=1))
    assert summary.started_at == datetime(2024, 1, 1, tzinfo=UTC)
    assert summary.ended_at == datetime(2024, 1, 1, 2, tzinfo=UTC)
    assert summary.duration == timedelta(hours=2)
    assert summary.status is DryRunStatus.fail
    assert len(summary.errors) == 1
    assert len(summary.warnings) == 1
    assert summary.gap_incidents == tuple()
    assert summary.content_incidents == tuple()
    assert summary.uptime_ratio == 1.0
    assert summary.as_dict()["ignored_lines"] == 1
    assert "Errors" in summary.to_markdown()


def test_analyse_structured_logs_flags_traceback_even_without_error_level() -> None:
    records = (
        _record(
            "2024-01-01T00:00:00",
            "info",
            "engine.tick",
            "Traceback (most recent call last): division by zero",
        ),
    )
    summary = analyse_structured_logs(LogParseResult(records=records, ignored_lines=0))
    assert summary.status is DryRunStatus.fail
    assert not summary.gap_incidents
    assert summary.content_incidents
    incident = summary.content_incidents[0]
    assert "traceback" in incident.summary.lower()
    rollup = DryRunSummary(
        generated_at=datetime(2024, 1, 2, tzinfo=UTC),
        log_summary=summary,
    ).to_markdown()
    assert "Log anomalies" in rollup


def test_analyse_structured_logs_enforces_minimum_duration() -> None:
    records = (
        _record("2024-01-01T00:00:00", "info", "start", "start"),
        _record("2024-01-01T00:30:00", "info", "heartbeat", "tick"),
    )
    summary = analyse_structured_logs(
        LogParseResult(records=records, ignored_lines=0),
        minimum_duration=timedelta(hours=2),
    )
    assert summary.status is DryRunStatus.fail
    assert any(
        incident.metadata.get("required_min_duration_seconds")
        for incident in summary.gap_incidents
    )


def test_analyse_structured_logs_enforces_uptime_ratio() -> None:
    records = (
        _record("2024-01-01T00:00:00", "info", "start", "start"),
        _record("2024-01-01T06:00:00", "info", "heartbeat", "tick"),
    )
    summary = analyse_structured_logs(
        LogParseResult(records=records, ignored_lines=0),
        warn_gap=timedelta(minutes=30),
        fail_gap=timedelta(hours=12),
        minimum_uptime_ratio=0.5,
    )
    assert summary.status is DryRunStatus.fail
    assert any(
        "required_minimum_uptime_ratio" in incident.metadata
        for incident in summary.gap_incidents
    )


def test_summarise_diary_entries_detects_incidents() -> None:
    base = {
        "decision": {"status": "ok"},
        "regime_state": {"regime": "bull"},
        "outcomes": {"status": "ok"},
    }
    entry_ok = DecisionDiaryEntry(
        entry_id="ok",
        recorded_at=datetime(2024, 1, 1, tzinfo=UTC),
        policy_id="alpha",
        **base,
    )
    entry_warn = DecisionDiaryEntry(
        entry_id="warn",
        recorded_at=datetime(2024, 1, 2, tzinfo=UTC),
        policy_id="alpha",
        notes=("warning: degraded throughput",),
        **base,
    )
    entry_fail = DecisionDiaryEntry(
        entry_id="fail",
        recorded_at=datetime(2024, 1, 3, tzinfo=UTC),
        policy_id="beta",
        outcomes={"status": "failed"},
        decision=base["decision"],
        regime_state=base["regime_state"],
    )
    summary = summarise_diary_entries((entry_ok, entry_warn, entry_fail))
    assert summary.total_entries == 3
    assert summary.status is DryRunStatus.fail
    assert len(summary.issues) == 2
    assert any(issue.severity is DryRunStatus.fail for issue in summary.issues)
    assert summary.first_recorded_at == entry_ok.recorded_at
    assert summary.last_recorded_at == entry_fail.recorded_at
    assert isinstance(summary.issues[0], DryRunDiaryIssue)


def test_dry_run_summary_combines_components() -> None:
    log_summary = DryRunLogSummary(
        records=(
            _record("2024-01-01T00:00:00", "info", "start", "start"),
        ),
        ignored_lines=0,
        level_counts={"info": 1},
        event_counts={"start": 1},
        gap_incidents=tuple(),
        content_incidents=tuple(),
        uptime_ratio=1.0,
    )
    diary_summary = DryRunDiarySummary(
        entries=tuple(),
        issues=tuple(),
        policy_counts={},
    )
    perf_summary = DryRunPerformanceSummary(
        generated_at=datetime(2024, 1, 2, tzinfo=UTC),
        period_start=datetime(2024, 1, 1, tzinfo=UTC),
        total_trades=10,
        roi=0.05,
        win_rate=0.6,
    )
    summary = DryRunSummary(
        generated_at=datetime(2024, 1, 2, tzinfo=UTC),
        log_summary=log_summary,
        diary_summary=diary_summary,
        performance_summary=perf_summary,
        metadata={"run_id": "demo"},
    )
    assert summary.status is DryRunStatus.pass_
    payload = summary.as_dict()
    assert payload["status"] == "pass"
    markdown = summary.to_markdown()
    assert "Final dry run summary" in markdown
    assert "run_id" in markdown


def test_sign_off_report_passes_with_all_criteria() -> None:
    records = (
        _record("2024-01-01T00:00:00", "info", "start", "start"),
        _record("2024-01-04T00:00:00", "info", "end", "done"),
    )
    log_summary = DryRunLogSummary(
        records=records,
        ignored_lines=0,
        level_counts={"info": 2},
        event_counts={"start": 1, "end": 1},
        gap_incidents=tuple(),
        content_incidents=tuple(),
        uptime_ratio=0.995,
    )
    diary_summary = DryRunDiarySummary(entries=tuple(), issues=tuple(), policy_counts={})
    perf_summary = DryRunPerformanceSummary(
        generated_at=datetime(2024, 1, 4, tzinfo=UTC),
        period_start=datetime(2024, 1, 1, tzinfo=UTC),
        total_trades=12,
        roi=0.02,
        win_rate=0.55,
    )
    summary = DryRunSummary(
        generated_at=datetime(2024, 1, 4, tzinfo=UTC),
        log_summary=log_summary,
        diary_summary=diary_summary,
        performance_summary=perf_summary,
    )
    report = assess_sign_off_readiness(
        summary,
        minimum_duration=timedelta(hours=72),
        minimum_uptime_ratio=0.98,
        require_diary=True,
        require_performance=True,
    )
    assert isinstance(report, DryRunSignOffReport)
    assert report.status is DryRunStatus.pass_
    assert report.findings == tuple()
    payload = report.as_dict()
    assert payload["status"] == "pass"
    assert payload["criteria"]["minimum_duration_seconds"] == 72 * 3600


def test_sign_off_report_flags_missing_evidence() -> None:
    records = (
        _record("2024-01-01T00:00:00", "info", "start", "start"),
        _record("2024-01-01T12:00:00", "info", "heartbeat", "tick"),
    )
    log_summary = DryRunLogSummary(
        records=records,
        ignored_lines=0,
        level_counts={"info": 2},
        event_counts={"start": 1, "heartbeat": 1},
        gap_incidents=tuple(),
        uptime_ratio=0.5,
    )
    summary = DryRunSummary(
        generated_at=datetime(2024, 1, 1, 12, tzinfo=UTC),
        log_summary=log_summary,
    )
    report = assess_sign_off_readiness(
        summary,
        minimum_duration=timedelta(hours=72),
        minimum_uptime_ratio=0.9,
        require_diary=True,
        require_performance=True,
    )
    assert report.status is DryRunStatus.fail
    assert any("duration" in finding.message for finding in report.findings)
    assert any("Performance telemetry" in finding.message for finding in report.findings)
    markdown = report.to_markdown()
    assert "FINDINGS" in markdown.upper()


def test_sign_off_report_warns_when_allowed() -> None:
    records = (
        _record("2024-01-01T00:00:00", "warning", "latency", "slow"),
        _record("2024-01-04T00:00:00", "info", "end", "done"),
    )
    log_summary = DryRunLogSummary(
        records=records,
        ignored_lines=0,
        level_counts={"warning": 1, "info": 1},
        event_counts={"latency": 1, "end": 1},
        gap_incidents=tuple(),
        uptime_ratio=1.0,
    )
    summary = DryRunSummary(
        generated_at=datetime(2024, 1, 4, tzinfo=UTC),
        log_summary=log_summary,
    )
    report = assess_sign_off_readiness(
        summary,
        minimum_duration=timedelta(hours=1),
        allow_warnings=True,
    )
    assert report.status is DryRunStatus.warn
    assert any(
        finding.severity is DryRunStatus.warn and "warnings" in finding.message
        for finding in report.findings
    )


def test_humanise_timedelta_formatting() -> None:
    assert humanise_timedelta(timedelta(seconds=5)) == "0m 5s"
    assert humanise_timedelta(timedelta(hours=26, minutes=5, seconds=7)) == "1d 2h 5m 7s"


def test_evaluate_dry_run_end_to_end(tmp_path: Path) -> None:
    log_path = tmp_path / "logs.jsonl"
    log_payloads = [
        {"timestamp": "2024-01-01T00:00:00Z", "level": "INFO", "event": "start"},
        {"timestamp": "2024-01-01T00:10:00Z", "level": "INFO", "event": "heartbeat"},
    ]
    log_path.write_text("\n".join(json.dumps(entry) for entry in log_payloads), encoding="utf-8")

    diary_path = tmp_path / "diary.json"
    diary_payload = {
        "generated_at": "2024-01-01T00:00:00Z",
        "entries": [
            {
                "entry_id": "dd-1",
                "recorded_at": "2024-01-01T00:05:00Z",
                "policy_id": "alpha",
                "decision": {"status": "ok"},
                "regime_state": {"regime": "bull"},
                "outcomes": {"status": "ok"},
                "notes": [],
                "metadata": {},
                "probes": [],
            }
        ],
    }
    diary_path.write_text(json.dumps(diary_payload), encoding="utf-8")

    performance_path = tmp_path / "performance.json"
    performance_payload = {
        "generated_at": "2024-01-01T23:00:00Z",
        "period_start": "2024-01-01T00:00:00Z",
        "aggregates": {"trades": 5, "roi": 0.1, "win_rate": 0.6},
        "metadata": {"window": "demo"},
    }
    performance_path.write_text(json.dumps(performance_payload), encoding="utf-8")

    summary = evaluate_dry_run(
        log_paths=[log_path],
        diary_path=diary_path,
        performance_path=performance_path,
    )
    assert summary.status is DryRunStatus.pass_
    assert summary.log_summary is not None
    assert summary.log_summary.event_counts["start"] == 1
    assert summary.diary_summary is not None
    assert summary.diary_summary.total_entries == 1
    assert summary.performance_summary is not None
    assert summary.performance_summary.total_trades == 5


def test_evaluate_dry_run_applies_minimum_duration(tmp_path: Path) -> None:
    log_path = tmp_path / "logs.jsonl"
    log_payloads = [
        {"timestamp": "2024-01-01T00:00:00Z", "level": "INFO", "event": "start"},
        {"timestamp": "2024-01-01T00:10:00Z", "level": "INFO", "event": "heartbeat"},
    ]
    log_path.write_text("\n".join(json.dumps(entry) for entry in log_payloads), encoding="utf-8")

    summary = evaluate_dry_run(
        log_paths=[log_path],
        minimum_run_duration=timedelta(hours=1),
        minimum_uptime_ratio=0.9,
    )
    assert summary.status is DryRunStatus.fail
    assert summary.log_summary is not None
    assert any(
        incident.metadata.get("required_min_duration_seconds")
        for incident in summary.log_summary.gap_incidents
    )

def test_load_structured_logs_counts_invalid_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "logs.jsonl"
    log_path.write_text("{}\ninvalid", encoding="utf-8")
    result = load_structured_logs([log_path])
    assert result.ignored_lines == 2
    assert not result.records


def test_analyse_structured_logs_detects_log_gaps() -> None:
    records = (
        _record("2024-01-01T00:00:00", "info", "start", "start"),
        _record("2024-01-01T03:30:00", "info", "heartbeat", "still alive"),
        _record("2024-01-01T04:00:00", "info", "heartbeat", "still alive"),
    )
    summary = analyse_structured_logs(
        LogParseResult(records=records, ignored_lines=0),
        warn_gap=timedelta(hours=1),
        fail_gap=timedelta(hours=4),
    )
    assert summary.gap_incidents
    gap = summary.gap_incidents[0]
    assert gap.severity is DryRunStatus.warn
    assert "3h" in gap.summary
    assert summary.uptime_ratio is not None
    assert summary.uptime_ratio < 1.0


def test_analyse_structured_logs_gap_failures_when_threshold_exceeded() -> None:
    records = (
        _record("2024-01-01T00:00:00", "info", "start", "start"),
        _record("2024-01-02T10:00:00", "info", "heartbeat", "ok"),
    )
    summary = analyse_structured_logs(
        LogParseResult(records=records, ignored_lines=0),
        warn_gap=timedelta(hours=1),
        fail_gap=timedelta(hours=12),
    )
    assert summary.status is DryRunStatus.fail
    assert summary.gap_incidents and summary.gap_incidents[0].severity is DryRunStatus.fail
    assert summary.uptime_ratio is not None
    assert "Log gaps" in DryRunSummary(
        generated_at=datetime(2024, 1, 3, tzinfo=UTC),
        log_summary=summary,
    ).to_markdown()
