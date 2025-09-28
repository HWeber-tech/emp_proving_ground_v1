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


def test_markdown_formatter_outputs_table() -> None:
    statuses = high_impact.evaluate_streams()
    markdown = high_impact.format_markdown(statuses)

    assert markdown.splitlines()[0] == "| Stream | Status | Summary | Next checkpoint |"
    assert "Stream A – Institutional data backbone" in markdown


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


def test_detail_formatter_includes_evidence() -> None:
    statuses = high_impact.evaluate_streams()
    report = high_impact.format_detail(statuses)

    assert "# High-impact roadmap status" in report
    assert "**Evidence:**" in report
    assert "Stream A" in report


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
