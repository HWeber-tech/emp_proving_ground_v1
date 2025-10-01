from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.telemetry.coverage_matrix import (
    build_coverage_matrix,
    main as coverage_matrix_main,
    render_markdown,
)


SAMPLE_COVERAGE = """<?xml version='1.0'?>
<coverage>
  <packages>
    <package>
      <classes>
        <class filename='src/sensory/foo.py'>
          <lines>
            <line number='1' hits='1'/>
            <line number='2' hits='0'/>
            <line number='3' hits='1'/>
          </lines>
        </class>
        <class filename='src/risk/bar.py'>
          <lines>
            <line number='1' hits='0'/>
            <line number='2' hits='0'/>
            <line number='3' hits='1'/>
          </lines>
        </class>
        <class filename='src/intelligence/brain.py'>
          <lines>
            <line number='1' hits='1'/>
            <line number='2' hits='1'/>
            <line number='3' hits='1'/>
          </lines>
        </class>
        <class filename='src/unknown/baz.py'>
          <lines>
            <line number='1' hits='0'/>
          </lines>
        </class>
        <class filename='../src/runtime/../runtime/builder.py'>
          <lines>
            <line number='1' hits='1'/>
            <line number='2' hits='1'/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""


@pytest.fixture()
def coverage_report(tmp_path: Path) -> Path:
    coverage_path = tmp_path / "coverage.xml"
    coverage_path.write_text(SAMPLE_COVERAGE)
    return coverage_path


def test_build_coverage_matrix_groups_by_domain(coverage_report: Path) -> None:
    matrix = build_coverage_matrix(coverage_report)

    assert matrix.totals.files == 5
    assert matrix.totals.covered == 8
    assert matrix.totals.missed == 4
    assert pytest.approx(matrix.totals.percent, rel=1e-6) == 66.67

    domain_names = [domain.name for domain in matrix.domains]
    assert domain_names[:3] == ["other", "risk", "sensory"]
    assert set(domain_names[3:]) == {"runtime", "intelligence"}

    sensory = next(domain for domain in matrix.domains if domain.name == "sensory")
    assert sensory.files == 1
    assert sensory.covered == 2
    assert sensory.missed == 1
    assert pytest.approx(sensory.percent, rel=1e-6) == 66.67

    runtime = next(domain for domain in matrix.domains if domain.name == "runtime")
    assert runtime.files == 1
    assert runtime.covered == 2
    assert runtime.missed == 0
    assert runtime.percent == 100.0

    intelligence = next(
        domain for domain in matrix.domains if domain.name == "intelligence"
    )
    assert intelligence.files == 1
    assert intelligence.covered == 3
    assert intelligence.missed == 0
    assert intelligence.percent == 100.0


def test_render_markdown_highlights_laggards(coverage_report: Path) -> None:
    matrix = build_coverage_matrix(coverage_report)
    markdown = render_markdown(matrix, threshold=60.0)

    assert "| Domain |" in markdown
    assert "sensory" in markdown
    assert "runtime" in markdown
    assert "intelligence" in markdown
    assert "Domains below the 60.00% threshold" in markdown
    assert "other (0.00%)" in markdown
    assert "risk (33.33%)" in markdown


def test_cli_writes_json_payload(tmp_path: Path, coverage_report: Path) -> None:
    output_path = tmp_path / "matrix.json"
    exit_code = coverage_matrix_main(
        [
            "--coverage-report",
            str(coverage_report),
            "--format",
            "json",
            "--threshold",
            "70",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["threshold"] == 70.0
    assert payload["laggards"] == ["other", "risk", "sensory"]
    totals = payload["totals"]
    assert totals["files"] == 5
    assert totals["coverage_percent"] == pytest.approx(66.67)
