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
    assert "data_foundation.ingest.anomaly_detection.detect_feed_anomalies" in status.evidence
    assert "data_foundation.streaming.latency_benchmark.StreamingLatencyBenchmark" in status.evidence


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
        "operations.feed_health.evaluate_feed_health",
        "operations.failover_drill.execute_failover_drill",
        "trading.order_management.lifecycle_processor.OrderLifecycleProcessor",
        "trading.order_management.position_tracker.PositionTracker",
        "trading.order_management.event_journal.OrderEventJournal",
        "trading.order_management.reconciliation.replay_order_events",
        "trading.execution.market_regime.classify_market_regime",
        "data_foundation.monitoring.feed_anomaly.analyse_feed",
        "scripts/order_lifecycle_dry_run.py",
        "scripts/reconcile_positions.py",
        "docs/runbooks/execution_lifecycle.md",
    }

    for entry in expected_entries:
        assert entry in status.evidence, f"missing {entry}"


def test_stream_c_includes_risk_analytics_suite() -> None:
    status = _status_map()[
        "Stream C – Execution, risk, compliance, ops readiness"
    ]

    required = {
        "risk.analytics.var.compute_parametric_var",
        "risk.analytics.expected_shortfall.compute_historical_expected_shortfall",
        "risk.analytics.volatility_target.determine_target_allocation",
        "risk.analytics.volatility_regime.classify_volatility_regime",
        "scripts/generate_risk_report.py",
    }

    assert required.issubset(set(status.evidence))


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

    assert isinstance(payload, dict)
    assert payload["portfolio"]["ready"] == len(statuses)
    assert payload["streams"][0]["evidence"], "expected evidence list in JSON output"


def test_detail_json_format_includes_streams() -> None:
    statuses = high_impact.evaluate_streams()
    payload = json.loads(high_impact.format_detail_json(statuses))

    assert payload["portfolio"]["total_streams"] == len(statuses)
    assert payload["streams"], "expected streams list in detail JSON output"
    assert payload["streams"][0]["evidence"], "expected evidence list in detail JSON output"
    assert any(item["deferred_summary"] for item in payload["streams"])  # ensures deferred notes surface


def test_attention_json_format_includes_missing_requirements() -> None:
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

    payload = json.loads(high_impact.format_attention_json(statuses))

    assert payload["portfolio"]["attention_needed"] == 1
    assert payload["portfolio"]["all_ready"] is False
    assert payload["streams"][0]["stream"] == "Stream Ω"
    assert payload["streams"][0]["missing"] == ["module.missing", "docs.todo"]
    assert payload["missing_requirements"] == {
        "Stream Ω": ["module.missing", "docs.todo"],
    }


def test_portfolio_json_format_includes_counts() -> None:
    statuses = high_impact.evaluate_streams()
    payload = json.loads(high_impact.format_portfolio_json(statuses))

    assert payload["total_streams"] == len(statuses)
    assert payload["ready"] == len(statuses)
    assert payload["attention_needed"] == 0


def test_cli_supports_json_format(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = high_impact.main(["--format", "json"])
    assert exit_code == 0
    out, err = capsys.readouterr()
    assert not err
    decoded = json.loads(out)
    assert decoded["streams"][0]["status"] == "Ready"
    assert decoded["portfolio"]["attention_needed"] == 0


def test_cli_attention_format_handles_ready_streams(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = high_impact.main(["--format", "attention"])

    assert exit_code == 0

    out, err = capsys.readouterr()
    assert not err
    assert "All streams are Ready" in out


def test_cli_exit_code_reflects_attention(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _fake_streams(_: list[str] | None = None) -> list[high_impact.StreamStatus]:
        return [
            high_impact.StreamStatus(
                stream="Stream Ω – Test",
                status="Attention needed",
                summary="Incomplete",
                next_checkpoint="Ship everything",
                evidence=("module.present",),
                missing=("module.missing",),
            )
        ]

    monkeypatch.setattr(high_impact, "evaluate_streams", _fake_streams)

    exit_code = high_impact.main([])

    assert exit_code == 1

    out, err = capsys.readouterr()
    assert not err
    assert "Stream Ω – Test" in out


def test_cli_supports_portfolio_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = high_impact.main(["--format", "portfolio-json"])

    assert exit_code == 0

    out, err = capsys.readouterr()
    assert not err
    payload = json.loads(out)
    assert payload["ready"] >= 0


def test_cli_supports_attention_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = high_impact.main(["--format", "attention-json"])

    assert exit_code == 0

    out, err = capsys.readouterr()
    assert not err
    payload = json.loads(out)
    assert "portfolio" in payload
    assert "streams" in payload


def test_cli_supports_detail_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = high_impact.main(["--format", "detail-json"])

    assert exit_code == 0

    out, err = capsys.readouterr()
    assert not err
    payload = json.loads(out)
    assert payload["streams"], "expected streams list in CLI detail JSON output"


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
    assert "**Deferred (Phase 3 backlog):**" in report


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


def test_progress_formatter_includes_percentages() -> None:
    statuses = high_impact.evaluate_streams()
    report = high_impact.format_progress(statuses)

    assert report.startswith("# High-impact roadmap progress")
    assert "Ready streams" in report
    assert "Attention needed" in report
    for status in statuses:
        assert status.stream in report
        assert status.next_checkpoint in report


def test_progress_formatter_lists_missing_requirements() -> None:
    statuses = [
        high_impact.StreamStatus(
            stream="Stream Ω",
            status="Attention needed",
            summary="Incomplete",
            next_checkpoint="Ship everything",
            evidence=("module.present",),
            missing=("module.missing", "docs.todo"),
        ),
    ]

    report = high_impact.format_progress(statuses)

    assert "Attention focus" in report
    assert "module.missing" in report
    assert "docs.todo" in report


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


def test_cli_refresh_docs_accepts_custom_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = tmp_path / "summary.md"
    detail = tmp_path / "detail.md"
    attention = tmp_path / "attention.md"
    portfolio_json = tmp_path / "portfolio.json"
    attention_json = tmp_path / "attention.json"
    summary.write_text(
        (
            "Lead-in\n\n"
            "<!-- HIGH_IMPACT_PORTFOLIO:START -->\n"
            "outdated summary\n"
            "<!-- HIGH_IMPACT_PORTFOLIO:END -->\n\n"
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
            "--attention-path",
            str(attention),
            "--portfolio-json-path",
            str(portfolio_json),
            "--attention-json-path",
            str(attention_json),
        ]
    )

    assert exit_code == 0
    out, err = capsys.readouterr()
    assert not err
    assert "Stream A – Institutional data backbone" in out

    updated_summary = summary.read_text(encoding="utf-8")
    assert "# High-impact roadmap summary" in updated_summary
    assert "Stream A – Institutional data backbone" in updated_summary
    assert updated_summary.startswith("Lead-in")

    updated_detail = detail.read_text(encoding="utf-8")
    assert updated_detail.startswith("# High-impact roadmap status")
    assert updated_detail.endswith("\n")

    updated_attention = attention.read_text(encoding="utf-8")
    assert updated_attention.startswith("# High-impact roadmap attention")
    assert updated_attention.endswith("\n")

    portfolio_payload = json.loads(portfolio_json.read_text(encoding="utf-8"))
    assert portfolio_payload["ready"] == portfolio_payload["total_streams"]

    attention_payload = json.loads(attention_json.read_text(encoding="utf-8"))
    assert attention_payload["portfolio"]["attention_needed"] == 0


def test_cli_rejects_stream_filter_when_refreshing_docs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        high_impact.main(
            [
                "--refresh-docs",
                "--stream",
                "Stream A – Institutional data backbone",
            ]
        )

    assert excinfo.value.code == 2
    out, err = capsys.readouterr()
    assert not out
    assert "cannot be combined" in err


def test_refresh_docs_updates_summary_and_detail(tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    detail = tmp_path / "detail.md"
    attention = tmp_path / "attention.md"
    portfolio_json = tmp_path / "portfolio.json"
    attention_json = tmp_path / "attention.json"
    summary.write_text(
        (
            "Header\n\n"
            "<!-- HIGH_IMPACT_PORTFOLIO:START -->\n"
            "outdated summary\n"
            "<!-- HIGH_IMPACT_PORTFOLIO:END -->\n\n"
            "<!-- HIGH_IMPACT_SUMMARY:START -->\n"
            "old table\n"
            "<!-- HIGH_IMPACT_SUMMARY:END -->\n\n"
            "Footer\n"
        ),
        encoding="utf-8",
    )
    detail.write_text("outdated\n", encoding="utf-8")

    statuses = high_impact.evaluate_streams()
    high_impact.refresh_docs(
        statuses,
        summary_path=summary,
        detail_path=detail,
        attention_path=attention,
        portfolio_json_path=portfolio_json,
        attention_json_path=attention_json,
    )

    updated_summary = summary.read_text(encoding="utf-8")
    assert "Header" in updated_summary
    assert "Footer" in updated_summary
    assert "# High-impact roadmap summary" in updated_summary
    assert "| Stream A – Institutional data backbone |" in updated_summary

    updated_detail = detail.read_text(encoding="utf-8")
    assert updated_detail.startswith("# High-impact roadmap status")
    assert updated_detail.endswith("\n")

    updated_attention = attention.read_text(encoding="utf-8")
    assert updated_attention.startswith("# High-impact roadmap attention")
    assert updated_attention.endswith("\n")

    portfolio_payload = json.loads(portfolio_json.read_text(encoding="utf-8"))
    assert portfolio_payload["ready_streams"]

    attention_payload = json.loads(attention_json.read_text(encoding="utf-8"))
    assert attention_payload["streams"] == []


def test_cli_supports_progress_format(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = high_impact.main(["--format", "progress"])

    assert exit_code == 0

    out, err = capsys.readouterr()
    assert not err
    assert "High-impact roadmap progress" in out


def test_refresh_docs_requires_markers(tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    detail = tmp_path / "detail.md"
    summary.write_text("missing markers\n", encoding="utf-8")
    detail.write_text("outdated\n", encoding="utf-8")

    statuses = high_impact.evaluate_streams()

    with pytest.raises(ValueError):
        high_impact.refresh_docs(statuses, summary_path=summary, detail_path=detail)

    summary.write_text(
        (
            "<!-- HIGH_IMPACT_SUMMARY:START -->\n"
            "table\n"
            "<!-- HIGH_IMPACT_SUMMARY:END -->\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        high_impact.refresh_docs(statuses, summary_path=summary, detail_path=detail)
