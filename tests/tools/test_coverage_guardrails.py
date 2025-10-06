from pathlib import Path

import pytest

from tools.telemetry.coverage_guardrails import (
    evaluate_guardrails,
    main,
    render_report,
)


def _write_report(tmp_path: Path, coverage: dict[str, list[int]]) -> Path:
    lines = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<coverage>",
        "  <packages>",
        "    <package>",
        "      <classes>",
    ]
    for filename, hits in coverage.items():
        lines.append(f'        <class filename="{filename}">')
        lines.append("          <lines>")
        for idx, hit in enumerate(hits, start=1):
            lines.append(f'            <line number="{idx}" hits="{hit}"/>')
        lines.append("          </lines>")
        lines.append("        </class>")
    lines.extend([
        "      </classes>",
        "    </package>",
        "  </packages>",
        "</coverage>",
    ])
    path = tmp_path / "coverage.xml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture()
def sample_report(tmp_path: Path) -> Path:
    coverage = {
        "src/data_foundation/ingest/production_slice.py": [1, 1, 1, 0],
        "src/data_foundation/ingest/timescale_pipeline.py": [1, 1, 1, 1, 0],
        "src/data_foundation/ingest/scheduler.py": [1, 1, 1, 1],
        "src/trading/risk/risk_policy.py": [1, 1, 0, 1, 1],
        "src/trading/risk/policy_telemetry.py": [1, 1, 1, 0],
        "src/data_foundation/ingest/observability.py": [1, 1, 1, 1, 1],
        "src/operations/observability_dashboard.py": [1, 1, 1, 1, 1],
    }
    return _write_report(tmp_path, coverage)


def test_evaluate_guardrails_passes_when_threshold_met(sample_report: Path) -> None:
    report = evaluate_guardrails(sample_report, minimum_percent=60.0)
    assert not report.has_failures
    assert {target.label for target in report.targets} == {
        "ingest_production_slice",
        "timescale_pipeline",
        "ingest_scheduler",
        "risk_policy",
        "risk_policy_telemetry",
        "ingest_observability",
        "observability_dashboard",
    }
    for target in report.targets:
        assert target.percent >= 60.0
        assert not target.missing
    markdown = render_report(report)
    assert "ok" in markdown


def test_evaluate_guardrails_marks_missing_targets(tmp_path: Path) -> None:
    coverage = {
        "src/data_foundation/ingest/production_slice.py": [1, 0, 1],
        "src/trading/risk/risk_policy.py": [1, 0, 0, 1],
    }
    report_path = _write_report(tmp_path, coverage)
    report = evaluate_guardrails(report_path, minimum_percent=80.0)

    assert report.has_failures
    assert set(report.failing) == {
        "timescale_pipeline",
        "ingest_scheduler",
        "ingest_observability",
        "observability_dashboard",
        "risk_policy_telemetry",
    }
    missing = {target.label for target in report.targets if target.missing}
    assert missing == {
        "timescale_pipeline",
        "ingest_scheduler",
        "ingest_observability",
        "observability_dashboard",
        "risk_policy_telemetry",
    }


def test_cli_returns_failure_exit_code_on_low_coverage(sample_report: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([
        "--report",
        str(sample_report),
        "--min-percent",
        "95",
        "--json",
    ])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "failing" in captured.out
