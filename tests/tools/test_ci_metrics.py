import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tools.telemetry.ci_metrics import (
    load_metrics,
    parse_coverage_percentage,
    record_coverage,
    record_coverage_domains,
    record_dashboard_remediation,
    record_formatter,
    record_remediation,
    summarise_dashboard_payload,
    summarise_trend_staleness,
)
from tools.telemetry.update_ci_metrics import main as update_ci_metrics


COVERAGE_XML_WITH_RATE = """<?xml version='1.0'?>
<coverage branch-rate='0.5' line-rate='0.8125'>
</coverage>
"""

COVERAGE_XML_WITH_LINES = """<?xml version='1.0'?>
<coverage>
  <packages>
    <package>
      <classes>
        <class>
          <lines>
            <line number='1' hits='0'/>
            <line number='2' hits='3'/>
            <line number='3' hits='1'/>
            <line number='4' hits='0'/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""

ALLOWLIST_SAMPLE = """# comment
src/core/
src/module.py
"""

COVERAGE_WITH_FILENAMES = """<?xml version='1.0'?>
<coverage>
  <packages>
    <package>
      <classes>
        <class filename='src/core/example.py'>
          <lines>
            <line number='1' hits='1'/>
            <line number='2' hits='0'/>
          </lines>
        </class>
        <class filename='src/trading/alpha.py'>
          <lines>
            <line number='1' hits='1'/>
            <line number='2' hits='1'/>
            <line number='3' hits='0'/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""

DASHBOARD_PAYLOAD = {
    "generated_at": "2024-05-20T12:00:00+00:00",
    "status": "warn",
    "panels": [
        {"name": "Operational readiness", "status": "fail"},
        {"name": "System health", "status": "warn"},
        {"name": "Latency & throughput", "status": "ok"},
    ],
}


@pytest.mark.parametrize(
    "content,expected",
    [
        (COVERAGE_XML_WITH_RATE, 81.25),
        (COVERAGE_XML_WITH_LINES, 50.0),
    ],
)
def test_parse_coverage_percentage_handles_formats(
    tmp_path: Path, content: str, expected: float
) -> None:
    coverage_path = tmp_path / "coverage.xml"
    coverage_path.write_text(content)

    percent = parse_coverage_percentage(coverage_path)

    assert pytest.approx(percent, rel=1e-6) == expected


def test_record_coverage_appends_entry(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    coverage_path = tmp_path / "coverage.xml"
    coverage_path.write_text(COVERAGE_XML_WITH_RATE)

    record_coverage(metrics_path, coverage_path, label="run-1")

    stored = json.loads(metrics_path.read_text())
    assert stored["coverage_trend"] == [
        {
            "label": "run-1",
            "coverage_percent": 81.25,
            "source": str(coverage_path),
        }
    ]
    assert stored["formatter_trend"] == []
    assert stored["coverage_domain_trend"] == []
    assert stored["remediation_trend"] == []


def test_record_formatter_counts_allowlist_entries(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text(ALLOWLIST_SAMPLE)

    record_formatter(metrics_path, allowlist, mode="allowlist", label="stage-4")

    stored = json.loads(metrics_path.read_text())
    assert stored["formatter_trend"] == [
        {
            "label": "stage-4",
            "mode": "allowlist",
            "total_entries": 2,
            "directory_count": 1,
            "file_count": 1,
        }
    ]
    assert stored["coverage_trend"] == []
    assert stored["coverage_domain_trend"] == []
    assert stored["remediation_trend"] == []


def test_record_formatter_global_mode_records_zero_counts(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"

    record_formatter(metrics_path, None, mode="global", label="repo-wide")

    stored = json.loads(metrics_path.read_text())
    assert stored["formatter_trend"] == [
        {
            "label": "repo-wide",
            "mode": "global",
            "total_entries": 0,
            "directory_count": 0,
            "file_count": 0,
        }
    ]
    assert stored["coverage_trend"] == []
    assert stored["coverage_domain_trend"] == []
    assert stored["remediation_trend"] == []


def test_record_coverage_domains_appends_entry(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    coverage_path = tmp_path / "coverage.xml"
    coverage_path.write_text(COVERAGE_WITH_FILENAMES)

    record_coverage_domains(
        metrics_path,
        coverage_path,
        label="domains-1",
        threshold=90.0,
    )

    stored = json.loads(metrics_path.read_text())
    assert stored["coverage_domain_trend"]
    entry = stored["coverage_domain_trend"][0]
    assert entry["label"] == "domains-1"
    assert entry["source"] == str(coverage_path)
    assert entry["threshold"] == 90.0
    assert entry["lagging_domains"] == ["core", "trading"]
    assert entry["lagging_count"] == 2
    assert entry["worst_domain"]["name"] == "core"
    assert entry["totals"]["files"] == 2
    assert isinstance(entry["generated_at"], str)
    domain_names = [domain["name"] for domain in entry["domains"]]
    assert domain_names == ["core", "trading"]


def test_record_coverage_domains_can_record_remediation(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    coverage_path = tmp_path / "coverage.xml"
    coverage_path.write_text(COVERAGE_WITH_FILENAMES)

    record_coverage_domains(
        metrics_path,
        coverage_path,
        label="domains-with-remediation",
        threshold=60.0,
        record_remediation_entry=True,
        remediation_note="Custom note",
    )

    stored = json.loads(metrics_path.read_text())
    remediation_entry = stored["remediation_trend"][0]
    assert remediation_entry["label"] == "domains-with-remediation"
    assert remediation_entry["source"] == str(coverage_path)
    assert remediation_entry["note"] == "Custom note"
    statuses = remediation_entry["statuses"]
    assert statuses["lagging_count"] == "1"
    assert statuses["overall_coverage"] == "60.0"
    assert statuses["coverage_threshold"] == "60.0"
    assert statuses["worst_domain"] == "core"
    assert statuses["worst_domain_coverage"] == "50.0"


def test_record_coverage_domains_generates_default_note(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    coverage_path = tmp_path / "coverage.xml"
    coverage_path.write_text(COVERAGE_WITH_FILENAMES)

    record_coverage_domains(
        metrics_path,
        coverage_path,
        label="domains-auto-note",
        threshold=65.0,
        record_remediation_entry=True,
    )

    stored = json.loads(metrics_path.read_text())
    remediation_entry = stored["remediation_trend"][0]
    assert remediation_entry["label"] == "domains-auto-note"
    assert remediation_entry["note"] == "Lagging domains: core (50.00%)"


def test_summarise_dashboard_payload_extracts_counts() -> None:
    summary = summarise_dashboard_payload(DASHBOARD_PAYLOAD)

    assert summary["overall_status"] == "warn"
    assert summary["panel_counts"] == {"ok": 1, "warn": 1, "fail": 1}
    assert summary["failing_panels"] == ("Operational readiness",)
    assert summary["warning_panels"] == ("System health",)
    assert summary["healthy_panels"] == ("Latency & throughput",)


def test_record_dashboard_remediation_appends_entry(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"

    record_dashboard_remediation(
        metrics_path,
        summary=DASHBOARD_PAYLOAD,
        label="dashboard-scan",
        source="observability_dashboard.json",
    )

    stored = json.loads(metrics_path.read_text())
    entry = stored["remediation_trend"][0]
    assert entry["label"] == "dashboard-scan"
    assert entry["source"] == "observability_dashboard.json"
    assert entry["statuses"] == {
        "overall_status": "warn",
        "panels_fail": "1",
        "panels_ok": "1",
        "panels_warn": "1",
    }
    assert entry["note"] == "Failing panels: Operational readiness; Warning panels: System health"


def test_record_remediation_appends_entry(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"

    record_remediation(
        metrics_path,
        statuses={"quality": "improved", "observability": "documented"},
        label="snapshot-1",
        source="docs/status/ci_health.md",
        note="Expanded operational metrics coverage",
    )

    stored = json.loads(metrics_path.read_text())
    assert stored["remediation_trend"] == [
        {
            "label": "snapshot-1",
            "note": "Expanded operational metrics coverage",
            "source": "docs/status/ci_health.md",
            "statuses": {
                "observability": "documented",
                "quality": "improved",
            },
        }
    ]
    assert stored["coverage_trend"] == []
    assert stored["formatter_trend"] == []
    assert stored["coverage_domain_trend"] == []


def test_update_ci_metrics_cli_updates_both(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    coverage_path = tmp_path / "coverage.xml"
    allowlist = tmp_path / "allowlist.txt"
    coverage_path.write_text(COVERAGE_WITH_FILENAMES)
    allowlist.write_text(ALLOWLIST_SAMPLE)

    exit_code = update_ci_metrics(
        [
            "--metrics",
            str(metrics_path),
            "--coverage-report",
            str(coverage_path),
            "--coverage-label",
            "coverage-sample",
            "--formatter-mode",
            "allowlist",
            "--allowlist",
            str(allowlist),
            "--formatter-label",
            "formatter-sample",
        ]
    )

    assert exit_code == 0

    stored = json.loads(metrics_path.read_text())
    assert stored["coverage_trend"][-1]["label"] == "coverage-sample"
    assert stored["formatter_trend"][-1]["label"] == "formatter-sample"
    assert stored["formatter_trend"][-1]["total_entries"] == 2
    coverage_entry = stored["coverage_domain_trend"][-1]
    assert coverage_entry["label"] == "coverage-sample"
    assert coverage_entry["domains"]
    assert coverage_entry["lagging_count"] == len(coverage_entry["lagging_domains"])
    assert coverage_entry["worst_domain"]["name"] in coverage_entry["lagging_domains"]
    assert stored["remediation_trend"] == []


def test_update_ci_metrics_cli_records_coverage_remediation(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    coverage_path = tmp_path / "coverage.xml"
    coverage_path.write_text(COVERAGE_WITH_FILENAMES)

    exit_code = update_ci_metrics(
        [
            "--metrics",
            str(metrics_path),
            "--coverage-report",
            str(coverage_path),
            "--coverage-label",
            "coverage-remediation",
            "--coverage-remediation",
            "--coverage-remediation-note",
            "Coverage remediation snapshot",
        ]
    )

    assert exit_code == 0
    stored = json.loads(metrics_path.read_text())
    remediation_entry = stored["remediation_trend"][0]
    assert remediation_entry["label"] == "coverage-remediation"
    assert remediation_entry["note"] == "Coverage remediation snapshot"
    statuses = remediation_entry["statuses"]
    assert statuses["lagging_count"] == str(len(stored["coverage_domain_trend"][0]["lagging_domains"]))
    assert "overall_coverage" in statuses
    assert remediation_entry["source"] == str(coverage_path)


def test_update_ci_metrics_cli_handles_global_mode(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"

    exit_code = update_ci_metrics(
        [
            "--metrics",
            str(metrics_path),
            "--formatter-mode",
            "global",
            "--formatter-label",
            "global-rollout",
        ]
    )

    assert exit_code == 0

    stored = json.loads(metrics_path.read_text())
    assert stored["formatter_trend"][-1] == {
        "label": "global-rollout",
        "mode": "global",
        "total_entries": 0,
        "directory_count": 0,
        "file_count": 0,
    }
    assert stored["coverage_domain_trend"] == []
    assert stored["remediation_trend"] == []


def test_update_ci_metrics_cli_can_skip_domain_breakdown(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    coverage_path = tmp_path / "coverage.xml"
    coverage_path.write_text(COVERAGE_WITH_FILENAMES)

    exit_code = update_ci_metrics(
        [
            "--metrics",
            str(metrics_path),
            "--coverage-report",
            str(coverage_path),
            "--no-domain-breakdown",
        ]
    )

    assert exit_code == 0

    stored = json.loads(metrics_path.read_text())
    assert stored["coverage_trend"]
    assert stored["coverage_domain_trend"] == []
    assert stored["remediation_trend"] == []


def test_summarise_trend_staleness_flags_outdated_trends(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = {
        "coverage_trend": [
            {"label": "2024-06-04T15:30:00+00:00"},
            {"label": "2024-06-02T12:00:00+00:00"},
        ],
        "formatter_trend": [
            {
                "label": "manual",
                "generated_at": "2024-06-05T09:00:00+00:00",
            }
        ],
        "coverage_domain_trend": [
            {"generated_at": "not-a-timestamp"},
        ],
        "remediation_trend": [],
    }

    frozen_now = datetime(2024, 6, 5, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr("tools.telemetry.ci_metrics._now", lambda: frozen_now)

    summary = summarise_trend_staleness(metrics, max_age_hours=24.0)

    assert summary["evaluated_at"] == "2024-06-05T12:00:00+00:00"
    assert summary["threshold_hours"] == 24.0

    coverage = summary["trends"]["coverage_trend"]
    assert coverage["entry_count"] == 2
    assert coverage["last_timestamp"] == "2024-06-04T15:30:00+00:00"
    assert coverage["is_stale"] is False
    assert coverage["age_hours"] == pytest.approx(20.5, rel=1e-6)

    formatter = summary["trends"]["formatter_trend"]
    assert formatter["entry_count"] == 1
    assert formatter["last_timestamp"] == "2024-06-05T09:00:00+00:00"
    assert formatter["is_stale"] is False
    assert formatter["age_hours"] == pytest.approx(3.0, rel=1e-6)

    domain = summary["trends"]["coverage_domain_trend"]
    assert domain["entry_count"] == 1
    assert domain["last_timestamp"] is None
    assert domain["is_stale"] is True
    assert domain["age_hours"] is None

    remediation = summary["trends"]["remediation_trend"]
    assert remediation["entry_count"] == 0
    assert remediation["is_stale"] is True
    assert remediation["age_hours"] is None


def test_update_ci_metrics_records_dashboard_payload(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    dashboard_path = tmp_path / "dashboard.json"
    dashboard_path.write_text(json.dumps(DASHBOARD_PAYLOAD))

    exit_code = update_ci_metrics(
        [
            "--metrics",
            str(metrics_path),
            "--dashboard-json",
            str(dashboard_path),
            "--dashboard-label",
            "ops-scan",
        ]
    )

    assert exit_code == 0
    stored = json.loads(metrics_path.read_text())
    entry = stored["remediation_trend"][0]
    assert entry["label"] == "ops-scan"
    assert entry["source"] == str(dashboard_path)
    assert entry["statuses"]["panels_fail"] == "1"


def test_update_ci_metrics_cli_records_remediation(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"

    exit_code = update_ci_metrics(
        [
            "--metrics",
            str(metrics_path),
            "--remediation-status",
            "quality=regression-added",
            "--remediation-status",
            "observability=docs-refreshed",
            "--remediation-label",
            "snapshot-2",
            "--remediation-source",
            "docs/context/alignment_briefs/quality_observability.md",
            "--remediation-note",
            "Documented regression coverage expansion",
        ]
    )

    assert exit_code == 0

    stored = json.loads(metrics_path.read_text())
    assert stored["coverage_trend"] == []
    assert stored["formatter_trend"] == []
    assert stored["coverage_domain_trend"] == []
    entry = stored["remediation_trend"][-1]
    assert entry["label"] == "snapshot-2"
    assert entry["source"] == "docs/context/alignment_briefs/quality_observability.md"
    assert entry["note"] == "Documented regression coverage expansion"
    assert entry["statuses"] == {
        "observability": "docs-refreshed",
        "quality": "regression-added",
    }


def test_load_metrics_returns_defaults_when_missing(tmp_path: Path) -> None:
    metrics_path = tmp_path / "missing.json"

    metrics = load_metrics(metrics_path)

    assert metrics == {
        "coverage_trend": [],
        "formatter_trend": [],
        "coverage_domain_trend": [],
        "remediation_trend": [],
    }
