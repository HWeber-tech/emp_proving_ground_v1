import json
from pathlib import Path

import pytest

from tools.telemetry.ci_metrics import (
    load_metrics,
    parse_coverage_percentage,
    record_coverage,
    record_formatter,
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


def test_update_ci_metrics_cli_updates_both(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    coverage_path = tmp_path / "coverage.xml"
    allowlist = tmp_path / "allowlist.txt"
    coverage_path.write_text(COVERAGE_XML_WITH_LINES)
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


def test_load_metrics_returns_defaults_when_missing(tmp_path: Path) -> None:
    metrics_path = tmp_path / "missing.json"

    metrics = load_metrics(metrics_path)

    assert metrics == {"coverage_trend": [], "formatter_trend": []}
