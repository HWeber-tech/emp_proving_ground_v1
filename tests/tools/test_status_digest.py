import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tools.telemetry import ci_metrics
from tools.telemetry.status_digest import (
    main as status_digest_main,
    render_ci_dashboard_table,
    render_weekly_status_summary,
)


METRICS_SAMPLE = {
    "coverage_trend": [
        {
            "label": "2024-06-01T10:00:00+00:00",
            "coverage_percent": 74.0,
            "source": "cov1.xml",
        },
        {
            "label": "2024-06-08T10:00:00+00:00",
            "coverage_percent": 76.5,
            "source": "cov2.xml",
        },
    ],
    "coverage_domain_trend": [
        {
            "label": "2024-06-08T10:00:00+00:00",
            "threshold": 80.0,
            "lagging_count": 1,
            "lagging_domains": ["core"],
            "domains": [
                {"name": "core", "coverage_percent": 75.0},
                {"name": "trading", "coverage_percent": 82.0},
            ],
            "worst_domain": {"name": "core", "coverage_percent": 75.0},
            "source": "cov2.xml",
        }
    ],
    "formatter_trend": [
        {
            "label": "2024-06-08T10:00:00+00:00",
            "mode": "allowlist",
            "total_entries": 2,
            "directory_count": 1,
            "file_count": 1,
        }
    ],
    "remediation_trend": [
        {
            "label": "2024-06-08T10:00:00+00:00",
            "statuses": {"lagging_count": "1", "overall_status": "warn"},
            "note": "Lagging domain core",
            "source": "cov2.xml",
        }
    ],
    "alert_response_trend": [
        {
            "label": "ci-alert-2024-06-08",
            "incident_id": "ci-alert-2024-06-08",
            "drill": True,
            "opened_at": "2024-06-08T09:55:00+00:00",
            "acknowledged_at": "2024-06-08T10:00:00+00:00",
            "resolved_at": "2024-06-08T10:12:00+00:00",
            "generated_at": "2024-06-08T10:12:00+00:00",
            "source": "drills/ci-alert-2024-06-08.json",
            "note": "Ack via slack in 0:05:00; Resolve via github in 0:17:00",
            "statuses": {
                "mtta_seconds": "300",
                "mtta_minutes": "5.0",
                "mtta_readable": "0:05:00",
                "mttr_seconds": "1020",
                "mttr_minutes": "17.0",
                "mttr_readable": "0:17:00",
                "ack_channel": "slack",
                "ack_actor": "oncall-analyst",
                "resolve_channel": "github",
                "resolve_actor": "maintainer",
            },
        }
    ],
}

DASHBOARD_SAMPLE = {
    "generated_at": "2024-06-08T11:30:00+00:00",
    "status": "warn",
    "panels": [
        {"name": "Operational readiness", "status": "fail"},
        {"name": "System health", "status": "warn"},
        {"name": "Latency", "status": "ok"},
    ],
}


@pytest.fixture()
def metrics_file(tmp_path: Path) -> Path:
    metrics_path = tmp_path / "ci_metrics.json"
    metrics_path.write_text(json.dumps(METRICS_SAMPLE, indent=2))
    return metrics_path


@pytest.fixture()
def dashboard_file(tmp_path: Path) -> Path:
    path = tmp_path / "dashboard.json"
    path.write_text(json.dumps(DASHBOARD_SAMPLE, indent=2))
    return path


def _parse_table(markdown: str) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    lines = [line for line in markdown.strip().splitlines() if line.startswith("|")]
    for line in lines[2:]:
        parts = [segment.strip() for segment in line.strip().split("|")[1:-1]]
        if len(parts) == 3:
            rows[parts[0]] = (parts[1], parts[2])
    return rows


def test_render_ci_dashboard_table_compiles_rows(
    metrics_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ci_metrics,
        "_now",
        lambda: datetime(2024, 6, 8, 12, tzinfo=UTC),
    )

    table = render_ci_dashboard_table(
        metrics_file,
        dashboard=DASHBOARD_SAMPLE,
        freshness_hours=24.0,
    )

    rows = _parse_table(table)

    coverage_value, coverage_notes = rows["Coverage"]
    assert "76.50%" in coverage_value
    assert "change +2.50pp" in coverage_value
    assert "cov2.xml" in coverage_notes

    domain_value, domain_notes = rows["Coverage domains"]
    assert domain_value.startswith("1 lagging")
    assert "core 75.00%" in domain_notes
    assert "threshold 80.0%" in domain_value

    formatter_value, formatter_notes = rows["Formatter"]
    assert formatter_value.startswith("mode allowlist")
    assert "Allowlist entries: 2" in formatter_notes

    remediation_value, remediation_notes = rows["Remediation"]
    assert "overall_status=warn" in remediation_value
    assert "Lagging domain core" in remediation_notes

    freshness_value, freshness_notes = rows["Telemetry freshness"]
    assert freshness_value == "All telemetry fresh"
    assert "coverage trend" in freshness_notes

    dashboard_value, dashboard_notes = rows["Observability dashboard"]
    assert dashboard_value.startswith("WARN (ok=1, warn=1, fail=1)")
    assert "Failing: Operational readiness" in dashboard_notes
    assert "Warnings: System health" in dashboard_notes
    assert "Healthy: Latency" in dashboard_notes

    alert_value, alert_notes = rows["Alert response"]
    assert alert_value.startswith("MTTA 5.00m / MTTR 17.00m")
    assert "ci-alert-2024-06-08" in alert_value
    assert "ack via slack" in alert_value
    assert "resolve via github" in alert_value
    assert "Opened 2024-06-08T09:55:00+00:00" in alert_notes
    assert "Acknowledged 2024-06-08T10:00:00+00:00 by oncall-analyst" in alert_notes
    assert "Resolved 2024-06-08T10:12:00+00:00 by maintainer" in alert_notes
    assert "MTTA 0:05:00" in alert_notes
    assert "MTTR 0:17:00" in alert_notes
    assert "Source: drills/ci-alert-2024-06-08.json" in alert_notes


def test_render_weekly_status_summary_contains_sections(
    metrics_file: Path, dashboard_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ci_metrics,
        "_now",
        lambda: datetime(2024, 6, 8, 12, tzinfo=UTC),
    )

    summary = render_weekly_status_summary(
        metrics_file,
        dashboard=DASHBOARD_SAMPLE,
        freshness_hours=168.0,
    )

    assert "# Weekly CI telemetry" in summary
    assert "## Coverage" in summary
    assert "Latest: 76.50%" in summary
    assert "Change vs previous" in summary
    assert "Lagging domains" in summary
    assert "core 75.00%" in summary
    assert "Mode: allowlist" in summary
    assert "Allowlist coverage: 2 entries" in summary
    assert "Statuses:" in summary
    assert "overall_status: warn" in summary
    assert "Lagging domain core" in summary
    assert "## Alert response" in summary
    assert "Label: ci-alert-2024-06-08 (drill)" in summary
    assert "Opened: 2024-06-08T09:55:00+00:00" in summary
    assert "Acknowledged: 2024-06-08T10:00:00+00:00 (via slack, by oncall-analyst)" in summary
    assert "Resolved: 2024-06-08T10:12:00+00:00 (via github, by maintainer)" in summary
    assert "MTTA: 5.00 minutes (0:05:00)" in summary
    assert "MTTR: 17.00 minutes (0:17:00)" in summary
    assert "Evidence: drills/ci-alert-2024-06-08.json" in summary
    assert "Evaluated at" in summary
    assert "Status: WARN" in summary
    assert "Failing panels" in summary


def test_cli_renders_to_file_and_stdout(
    metrics_file: Path,
    dashboard_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        ci_metrics,
        "_now",
        lambda: datetime(2024, 6, 8, 12, tzinfo=UTC),
    )

    exit_code = status_digest_main(
        [
            "--metrics",
            str(metrics_file),
            "--dashboard",
            str(dashboard_file),
            "--mode",
            "ci-dashboard",
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Coverage" in stdout
    assert "Observability dashboard" in stdout
    assert "Alert response" in stdout

    output_path = tmp_path / "weekly.md"
    exit_code = status_digest_main(
        [
            "--metrics",
            str(metrics_file),
            "--dashboard",
            str(dashboard_file),
            "--mode",
            "weekly-status",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    content = output_path.read_text()
    assert content.startswith("# Weekly CI telemetry")
    assert "Status: WARN" in content
    assert "## Alert response" in content


def test_alert_response_ignores_unknown_channels(metrics_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = json.loads(metrics_file.read_text())
    entry = data["alert_response_trend"][0]
    statuses = entry["statuses"]
    statuses["ack_channel"] = "unknown"
    statuses["resolve_channel"] = "UNKNOWN"
    statuses["ack_actor"] = "unknown"
    statuses["resolve_actor"] = "Unknown"
    statuses.pop("mtta_minutes", None)
    statuses.pop("mttr_minutes", None)
    statuses.pop("mtta_readable", None)
    statuses.pop("mttr_readable", None)
    metrics_file.write_text(json.dumps(data))

    monkeypatch.setattr(
        ci_metrics,
        "_now",
        lambda: datetime(2024, 6, 8, 12, tzinfo=UTC),
    )

    table = render_ci_dashboard_table(
        metrics_file,
        dashboard=DASHBOARD_SAMPLE,
        freshness_hours=24.0,
    )
    rows = _parse_table(table)
    alert_value, alert_notes = rows["Alert response"]
    assert alert_value.startswith("MTTA 5.00m / MTTR 17.00m")
    assert "ack via" not in alert_value
    assert "resolve via" not in alert_value
    assert "oncall-analyst" not in alert_notes
    assert "maintainer" not in alert_notes

    summary = render_weekly_status_summary(
        metrics_file,
        dashboard=DASHBOARD_SAMPLE,
        freshness_hours=168.0,
    )
    assert "- MTTA: 5.00 minutes (0:05:00)" in summary
    assert "- MTTR: 17.00 minutes (0:17:00)" in summary
    assert "Acknowledged: 2024-06-08T10:00:00+00:00 (" not in summary
    assert "Resolved: 2024-06-08T10:12:00+00:00 (" not in summary
