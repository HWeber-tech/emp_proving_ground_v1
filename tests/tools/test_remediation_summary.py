from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.telemetry.ci_metrics import save_metrics
from tools.telemetry.remediation_summary import main, render_remediation_summary


def _write_metrics(path: Path, entries: list[dict[str, object]]) -> None:
    payload = {
        "coverage_trend": [],
        "formatter_trend": [],
        "coverage_domain_trend": [],
        "remediation_trend": entries,
    }
    save_metrics(path, payload)


def test_render_remediation_summary_formats_markdown(tmp_path: Path) -> None:
    metrics_path = tmp_path / "ci_metrics.json"
    _write_metrics(
        metrics_path,
        [
            {
                "label": "2024-05-01",
                "statuses": {"coverage": "76%", "observability": "green"},
                "source": "ci_run_123",
                "note": "Baseline snapshot",
            },
            {
                "label": "2024-05-10",
                "statuses": {"coverage": "78%", "observability": "amber"},
                "source": "ci_run_150",
                "note": "Regression coverage uplift",
            },
        ],
    )

    summary = render_remediation_summary(metrics_path)

    assert summary.startswith("# Remediation progress\n\n| Label |")
    assert "2024-05-10" in summary
    assert "coverage" in summary
    assert "Regression coverage uplift" in summary
    assert "## Latest status overview" in summary
    assert "**coverage**: 78%" in summary


def test_render_remediation_summary_honours_limit(tmp_path: Path) -> None:
    metrics_path = tmp_path / "ci_metrics.json"
    _write_metrics(
        metrics_path,
        [
            {"label": "snapshot-a", "statuses": {"coverage": "70%"}},
            {"label": "snapshot-b", "statuses": {"coverage": "71%"}},
            {"label": "snapshot-c", "statuses": {"coverage": "72%"}},
        ],
    )

    summary = render_remediation_summary(metrics_path, limit=1)

    assert "snapshot-c" in summary
    assert "snapshot-b" not in summary
    assert "snapshot-a" not in summary


def test_render_remediation_summary_handles_empty_metrics(tmp_path: Path) -> None:
    metrics_path = tmp_path / "ci_metrics.json"
    metrics_path.write_text(json.dumps({}))

    summary = render_remediation_summary(metrics_path)

    assert "No remediation snapshots" in summary


def test_cli_writes_output_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    metrics_path = tmp_path / "ci_metrics.json"
    _write_metrics(
        metrics_path,
        [
            {"label": "snapshot", "statuses": {"coverage": "75%"}},
        ],
    )
    output_path = tmp_path / "summary.md"

    exit_code = main(
        [
            "--metrics-path",
            str(metrics_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text().startswith("# Remediation progress")
    captured = capsys.readouterr()
    assert captured.out == ""
