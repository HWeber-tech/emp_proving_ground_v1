import json
from pathlib import Path

import pytest

from tools.telemetry.ci_digest import (
    main,
    render_dashboard_summary,
    render_weekly_digest,
    summarise_coverage,
    summarise_coverage_domains,
    summarise_remediation,
)


SAMPLE_METRICS = {
    "coverage_trend": [
        {
            "label": "2025-09-29T12:00:00+00:00",
            "coverage_percent": 75.1,
            "source": "coverage-reports/pytest-2025-09-29.xml",
        },
        {
            "label": "2025-10-06T11:00:00+00:00",
            "coverage_percent": 76.4,
            "source": "coverage-reports/pytest-2025-10-06.xml",
        },
    ],
    "coverage_domain_trend": [
        {
            "label": "2025-09-29T12:00:00+00:00",
            "source": "coverage-reports/pytest-2025-09-29.xml",
            "lagging_count": 3,
            "lagging_domains": ["operations", "trading", "evolution"],
            "worst_domain": {"name": "operations", "coverage_percent": 74.5},
        },
        {
            "label": "2025-10-06T11:00:00+00:00",
            "source": "coverage-reports/pytest-2025-10-06.xml",
            "lagging_count": 2,
            "lagging_domains": ["trading", "evolution"],
            "worst_domain": {"name": "evolution", "coverage_percent": 77.0},
        },
    ],
    "formatter_trend": [
        {
            "label": "2025-09-01T09:00:00+00:00",
            "mode": "global",
            "total_entries": 0,
            "directory_count": 0,
            "file_count": 0,
        }
    ],
    "remediation_trend": [
        {
            "label": "2025-09-29",
            "statuses": {
                "overall_coverage": "75.1",
                "lagging_count": "3",
                "coverage_threshold": "80",
                "worst_domain": "operations",
            },
            "note": "Lagging domains: operations (74.5%), trading (79.2%), evolution (77.1%)",
            "source": "coverage-reports/pytest-2025-09-29.xml",
        },
        {
            "label": "2025-10-06",
            "statuses": {
                "overall_coverage": "76.4",
                "lagging_count": "2",
                "coverage_threshold": "80",
                "worst_domain": "evolution",
            },
            "note": "Lagging domains: trading (79.4%), evolution (77.0%)",
            "source": "coverage-reports/pytest-2025-10-06.xml",
        },
    ],
}


@pytest.fixture()
def metrics_path(tmp_path: Path) -> Path:
    path = tmp_path / "ci_metrics.json"
    path.write_text(json.dumps(SAMPLE_METRICS))
    return path


def test_summarise_coverage_returns_latest(metrics_path: Path) -> None:
    summary = summarise_coverage(json.loads(metrics_path.read_text()))

    assert summary.label == "2025-10-06T11:00:00+00:00"
    assert summary.previous_label == "2025-09-29T12:00:00+00:00"
    assert summary.value == pytest.approx(76.4)
    assert summary.delta == pytest.approx(1.3)
    assert summary.source == "coverage-reports/pytest-2025-10-06.xml"


def test_summarise_coverage_domains_reports_deltas(metrics_path: Path) -> None:
    summary = summarise_coverage_domains(json.loads(metrics_path.read_text()))

    assert summary.lagging_count == 2
    assert summary.lagging_delta == -1
    assert summary.lagging_domains == ("trading", "evolution")
    assert summary.worst_domain == "evolution"
    assert summary.worst_domain_percent == pytest.approx(77.0)


def test_summarise_remediation_extracts_status_deltas(metrics_path: Path) -> None:
    summary = summarise_remediation(json.loads(metrics_path.read_text()))

    assert summary.entries
    coverage_entry = next((entry for entry in summary.entries if entry[0] == "overall_coverage"), None)
    assert coverage_entry is not None
    assert coverage_entry[1] == "76.4"
    assert coverage_entry[2] == pytest.approx(1.3)
    lagging_entry = next((entry for entry in summary.entries if entry[0] == "lagging_count"), None)
    assert lagging_entry is not None
    assert lagging_entry[1] == "2"
    assert lagging_entry[2] == pytest.approx(-1.0)
    assert summary.note == "Lagging domains: trading (79.4%), evolution (77.0%)"
    assert summary.source == "coverage-reports/pytest-2025-10-06.xml"


def test_render_dashboard_summary_includes_key_details(metrics_path: Path) -> None:
    summary = render_dashboard_summary(metrics_path)

    assert "76.40%" in summary
    assert "Δ +1.30" in summary
    assert "Lagging domains: 2" in summary or "2 lagging" in summary
    assert "coverage-reports/pytest-2025-10-06.xml" in summary


def test_render_weekly_digest_mentions_remediation(metrics_path: Path) -> None:
    digest = render_weekly_digest(metrics_path)

    assert "## 2025-10-06" in digest
    assert "Coverage: 76.40%" in digest
    assert "Lagging domains: 2" in digest
    assert "overall_coverage: 76.4" in digest
    assert "Δ +1.3" in digest
    assert "Evidence: coverage-reports/pytest-2025-10-06.xml" in digest


def test_main_supports_output_file(metrics_path: Path) -> None:
    output = metrics_path.parent / "digest.md"

    exit_code = main(["--metrics", str(metrics_path), "--mode", "dashboard", "--output", str(output)])

    assert exit_code == 0
    assert output.exists()
    content = output.read_text()
    assert "Coverage" in content
