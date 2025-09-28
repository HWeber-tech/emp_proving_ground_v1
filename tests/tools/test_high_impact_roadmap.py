"""Tests for the high-impact roadmap evaluation CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.roadmap import high_impact


def _status_map() -> dict[str, high_impact.StreamStatus]:
    return {status.stream: status for status in high_impact.evaluate_streams()}


def test_streams_marked_ready() -> None:
    statuses = _status_map()

    assert set(statuses) == {
        "Stream A – Institutional data backbone",
        "Stream B – Sensory cortex & evolution uplift",
        "Stream C – Execution, risk, compliance, ops readiness",
    }

    for status in statuses.values():
        assert status.status == "Ready"
        assert status.missing == ()
        assert status.evidence, "expected evidence for ready streams"


def test_stream_a_includes_resilience_requirements() -> None:
    status = _status_map()["Stream A – Institutional data backbone"]

    assert "operations.backup.evaluate_backup_readiness" in status.evidence
    assert "operations.spark_stress.execute_spark_stress_drill" in status.evidence


def test_stream_b_lists_all_sensory_organs() -> None:
    status = _status_map()["Stream B – Sensory cortex & evolution uplift"]

    assert {
        "sensory.how.how_sensor.HowSensor",
        "sensory.anomaly.anomaly_sensor.AnomalySensor",
        "sensory.when.gamma_exposure.GammaExposureAnalyzer",
        "sensory.why.why_sensor.WhySensor",
        "sensory.what.what_sensor.WhatSensor",
    }.issubset(status.evidence)


def test_stream_c_covers_execution_lifecycle_artifacts() -> None:
    status = _status_map()[
        "Stream C – Execution, risk, compliance, ops readiness"
    ]

    expected_entries = {
        "operations.event_bus_health.evaluate_event_bus_health",
        "operations.failover_drill.execute_failover_drill",
        "trading.order_management.lifecycle_processor.OrderLifecycleProcessor",
        "trading.order_management.position_tracker.PositionTracker",
        "trading.order_management.event_journal.OrderEventJournal",
        "trading.order_management.reconciliation.replay_order_events",
        "scripts/order_lifecycle_dry_run.py",
        "scripts/reconcile_positions.py",
        "docs/runbooks/execution_lifecycle.md",
    }

    for entry in expected_entries:
        assert entry in status.evidence, f"missing {entry}"


def test_markdown_formatter_outputs_table() -> None:
    statuses = high_impact.evaluate_streams()
    markdown = high_impact.format_markdown(statuses)

    assert markdown.splitlines()[0] == "| Stream | Status | Summary | Next checkpoint |"
    assert "Stream A – Institutional data backbone" in markdown


def test_portfolio_summary_formatter_lists_counts() -> None:
    statuses = high_impact.evaluate_streams()
    summary = high_impact.format_portfolio_summary(statuses)

    assert summary.startswith("# High-impact roadmap summary")
    assert "Total streams" in summary
    assert "Ready" in summary
    assert "Stream A – Institutional data backbone" in summary
    assert "All streams are Ready" in summary


def test_attention_formatter_notes_missing_items() -> None:
    status = high_impact.StreamStatus(
        stream="Stream Ω",
        status="Attention needed",
        summary="Incomplete",
        next_checkpoint="Ship everything",
        evidence=("module.present",),
        missing=("module.missing", "docs.todo"),
    )

    report = high_impact.format_attention([status])

    assert report.startswith("# High-impact roadmap attention")
    assert "Stream Ω" in report
    assert "module.missing" in report
    assert "module.present" in report


def test_summarise_portfolio_counts_ready_streams() -> None:
    statuses = high_impact.evaluate_streams()

    portfolio = high_impact.summarise_portfolio(statuses)

    assert portfolio.total_streams == len(statuses)
    assert portfolio.ready == len(statuses)
    assert portfolio.attention_needed == 0
    assert all(stream.status == "Ready" for stream in portfolio.streams)
    assert portfolio.all_ready
    assert portfolio.attention_streams() == ()
    assert {
        status.stream for status in portfolio.ready_streams()
    } == {status.stream for status in statuses}
    assert portfolio.missing_requirements() == {}

    payload = portfolio.as_dict()
    assert payload["ready"] == portfolio.ready
    assert len(payload["streams"]) == portfolio.total_streams
    assert payload["ready_streams"]
    assert payload["attention_streams"] == []
    assert payload["missing_requirements"] == {}


def test_json_format_includes_evidence() -> None:
    statuses = high_impact.evaluate_streams()
    payload = json.loads(high_impact.format_json(statuses))

    assert isinstance(payload, list)
    assert payload[0]["evidence"], "expected evidence list in JSON output"


def test_cli_supports_json_format(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = high_impact.main(["--format", "json"])
    assert exit_code == 0
    out, err = capsys.readouterr()
    assert not err
    decoded = json.loads(out)
    assert decoded[0]["status"] == "Ready"


def test_cli_attention_format_handles_ready_streams(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = high_impact.main(["--format", "attention"])

    assert exit_code == 0

    out, err = capsys.readouterr()
    assert not err
    assert "All streams are Ready" in out


def test_evaluate_streams_can_filter() -> None:
    [status] = high_impact.evaluate_streams(
        ["Stream A – Institutional data backbone"]
    )

    assert status.stream == "Stream A – Institutional data backbone"


def test_evaluate_streams_preserves_requested_order() -> None:
    statuses = high_impact.evaluate_streams(
        [
            "Stream C – Execution, risk, compliance, ops readiness",
            "Stream A – Institutional data backbone",
        ]
    )

    assert [status.stream for status in statuses] == [
        "Stream C – Execution, risk, compliance, ops readiness",
        "Stream A – Institutional data backbone",
    ]


def test_cli_rejects_unknown_stream(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        high_impact.main(["--stream", "unknown stream"])

    assert excinfo.value.code == 2
    out, err = capsys.readouterr()
    assert not out
    assert "Unknown stream" in err


def test_detail_formatter_includes_evidence() -> None:
    statuses = high_impact.evaluate_streams()
    report = high_impact.format_detail(statuses)

    assert "# High-impact roadmap status" in report
    assert "**Evidence:**" in report
    assert "Stream A" in report


def test_portfolio_summary_mentions_missing_requirements() -> None:
    statuses = [
        high_impact.StreamStatus(
            stream="Stream Ω",
            status="Attention needed",
            summary="Incomplete",
            next_checkpoint="Ship everything",
            evidence=("module.present",),
            missing=("module.missing", "docs.todo"),
        ),
        high_impact.StreamStatus(
            stream="Stream Α",
            status="Ready",
            summary="All good",
            next_checkpoint="Next",
            evidence=("module.present",),
            missing=(),
        ),
    ]

    portfolio = high_impact.summarise_portfolio(statuses)

    assert not portfolio.all_ready
    assert [status.stream for status in portfolio.attention_streams()] == [
        "Stream Ω"
    ]
    assert portfolio.missing_requirements() == {
        "Stream Ω": ("module.missing", "docs.todo"),
    }

    summary = high_impact.format_portfolio_summary(statuses)
    assert "Streams needing attention" in summary
    assert "Stream Ω" in summary
    assert "module.missing" in summary


def test_cli_writes_output_file(tmp_path: Path) -> None:
    destination = tmp_path / "report.md"
    exit_code = high_impact.main([
        "--format",
        "detail",
        "--output",
        str(destination),
    ])

    assert exit_code == 0
    assert destination.exists()
    content = destination.read_text(encoding="utf-8")
    assert "High-impact roadmap status" in content


def test_cli_refresh_docs_accepts_custom_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary = tmp_path / "summary.md"
    detail = tmp_path / "detail.md"
    summary.write_text(
        (
            "Lead-in\n\n"
            "<!-- HIGH_IMPACT_SUMMARY:START -->\n"
            "outdated table\n"
            "<!-- HIGH_IMPACT_SUMMARY:END -->\n"
        ),
        encoding="utf-8",
    )
    detail.write_text("outdated\n", encoding="utf-8")

    exit_code = high_impact.main(
        [
            "--refresh-docs",
            "--summary-path",
            str(summary),
            "--detail-path",
            str(detail),
        ]
    )

    assert exit_code == 0
    out, err = capsys.readouterr()
    assert not err
    assert "Stream A – Institutional data backbone" in out

    updated_summary = summary.read_text(encoding="utf-8")
    assert "Stream A – Institutional data backbone" in updated_summary
    assert updated_summary.startswith("Lead-in")

    updated_detail = detail.read_text(encoding="utf-8")
    assert updated_detail.startswith("# High-impact roadmap status")
    assert updated_detail.endswith("\n")


def test_refresh_docs_updates_summary_and_detail(tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    detail = tmp_path / "detail.md"
    summary.write_text(
        (
            "Header\n\n"
            "<!-- HIGH_IMPACT_SUMMARY:START -->\n"
            "old table\n"
            "<!-- HIGH_IMPACT_SUMMARY:END -->\n\n"
            "Footer\n"
        ),
        encoding="utf-8",
    )
    detail.write_text("outdated\n", encoding="utf-8")

    statuses = high_impact.evaluate_streams()
    high_impact.refresh_docs(statuses, summary_path=summary, detail_path=detail)

    updated_summary = summary.read_text(encoding="utf-8")
    assert "Header" in updated_summary
    assert "Footer" in updated_summary
    assert "| Stream A – Institutional data backbone |" in updated_summary

    updated_detail = detail.read_text(encoding="utf-8")
    assert updated_detail.startswith("# High-impact roadmap status")
    assert updated_detail.endswith("\n")


def test_refresh_docs_requires_markers(tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    detail = tmp_path / "detail.md"
    summary.write_text("missing markers\n", encoding="utf-8")
    detail.write_text("outdated\n", encoding="utf-8")

    statuses = high_impact.evaluate_streams()

    with pytest.raises(ValueError):
        high_impact.refresh_docs(statuses, summary_path=summary, detail_path=detail)
