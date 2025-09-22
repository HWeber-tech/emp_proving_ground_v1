from __future__ import annotations

import importlib
import sys
from pathlib import Path

import tools.roadmap.snapshot as roadmap_snapshot


def _status_map() -> dict[str, roadmap_snapshot.InitiativeStatus]:
    return {status.initiative: status for status in roadmap_snapshot.evaluate_portfolio_snapshot()}


def test_data_backbone_marked_ready() -> None:
    statuses = _status_map()
    backbone = statuses["Institutional data backbone"]
    assert backbone.status == "Ready"
    assert not backbone.missing
    assert any(
        evidence.startswith("data_foundation.ingest.timescale_pipeline")
        for evidence in backbone.evidence
    )


def test_execution_and_compliance_marked_ready() -> None:
    statuses = _status_map()
    ops = statuses["Execution, risk, compliance, ops readiness"]
    assert ops.status == "Ready"
    assert "runtime.fix_pilot.FixIntegrationPilot" in ops.evidence


def test_markdown_formatter_outputs_table(capsys) -> None:
    markdown = roadmap_snapshot.format_markdown(roadmap_snapshot.evaluate_portfolio_snapshot())
    assert "| Initiative |" in markdown
    assert "Institutional data backbone" in markdown

    roadmap_snapshot.main(["--format", "markdown"])
    captured = capsys.readouterr().out
    assert "Institutional data backbone" in captured
    assert "Ready" in captured


def test_json_format_includes_evidence(capsys) -> None:
    roadmap_snapshot.main(["--format", "json"])
    captured = capsys.readouterr().out
    assert "evidence" in captured
    assert "Institutional data backbone" in captured


def test_cli_recovers_when_src_not_on_sys_path(monkeypatch) -> None:
    src_path = Path(__file__).resolve().parents[2] / "src"
    filtered_path = [
        entry for entry in sys.path if Path(entry or ".").resolve() != src_path.resolve()
    ]
    monkeypatch.setattr(sys, "path", filtered_path, raising=False)

    module = importlib.reload(roadmap_snapshot)
    statuses = {status.initiative: status for status in module.evaluate_portfolio_snapshot()}
    assert statuses["Institutional data backbone"].status == "Ready"
    assert statuses["Sensory cortex & evolution uplift"].status == "Ready"
