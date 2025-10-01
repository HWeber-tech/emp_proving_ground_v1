import json
from pathlib import Path

import pytest

from tools.telemetry.render_ci_metrics import main, render_markdown


def _sample_metrics() -> dict[str, object]:
    return {
        "coverage_trend": [
            {"label": "2024-07-01", "coverage_percent": 76.25, "source": "coverage.xml"},
            {"label": "2024-07-02", "coverage_percent": 77.0, "source": "coverage.xml"},
        ],
        "coverage_domain_trend": [
            {
                "label": "2024-07-02",
                "threshold": 80.0,
                "lagging_domains": ["core"],
                "domains": [
                    {"name": "core", "coverage_percent": 78.5},
                    {"name": "operations", "coverage_percent": 82.3},
                ],
            }
        ],
        "formatter_trend": [
            {"label": "2024-07-02", "mode": "global", "total_entries": 0},
        ],
        "remediation_trend": [
            {
                "label": "Weekly snapshot",
                "statuses": {"ingest": "green", "risk": "amber"},
                "source": "docs/status/ci_health.md",
                "note": "Expanded regression coverage",
            }
        ],
    }


def test_render_markdown_renders_all_sections() -> None:
    markdown = render_markdown(_sample_metrics(), limit=3)

    assert "# CI metrics snapshot" in markdown
    assert "## Coverage trend" in markdown
    assert "| 2024-07-02 | 77.00% | coverage.xml |" in markdown
    assert "Lagging" in markdown
    assert "Formatter adoption" in markdown
    assert "Remediation progress" in markdown
    assert "- ingest: green" in markdown
    assert "- risk: amber" in markdown
    assert "- Source: docs/status/ci_health.md" in markdown
    assert "- Note: Expanded regression coverage" in markdown


def test_main_writes_output_file(tmp_path: Path) -> None:
    metrics_path = tmp_path / "ci_metrics.json"
    metrics_path.write_text(json.dumps(_sample_metrics()))
    output_path = tmp_path / "snapshot.md"

    exit_code = main(
        [
            "--metrics",
            str(metrics_path),
            "--output",
            str(output_path),
            "--limit",
            "2",
        ]
    )

    assert exit_code == 0
    content = output_path.read_text()
    assert content.startswith("# CI metrics snapshot")
    assert "2024-07-02" in content


def test_main_prints_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    metrics_path = tmp_path / "ci_metrics.json"
    metrics_path.write_text(json.dumps(_sample_metrics()))

    exit_code = main(["--metrics", str(metrics_path), "--limit", "1"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "# CI metrics snapshot" in captured.out
    assert "2024-07-02" in captured.out


def test_main_requires_positive_limit(tmp_path: Path) -> None:
    metrics_path = tmp_path / "ci_metrics.json"
    metrics_path.write_text(json.dumps(_sample_metrics()))

    with pytest.raises(SystemExit) as exc_info:
        main(["--metrics", str(metrics_path), "--limit", "0"])

    assert exc_info.value.code == 2
