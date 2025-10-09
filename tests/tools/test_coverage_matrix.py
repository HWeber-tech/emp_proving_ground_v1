from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.telemetry.coverage_matrix import (
    build_coverage_matrix,
    identify_laggards,
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
        <class filename='src/understanding/brain.py'>
          <lines>
            <line number='1' hits='1'/>
            <line number='2' hits='1'/>
            <line number='3' hits='1'/>
          </lines>
        </class>
        <class filename='src/intelligence/legacy.py'>
          <lines>
            <line number='1' hits='1'/>
            <line number='2' hits='0'/>
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

    assert matrix.totals.files == 6
    assert matrix.totals.covered == 9
    assert matrix.totals.missed == 5
    assert pytest.approx(matrix.totals.percent, rel=1e-6) == 64.29

    domain_names = [domain.name for domain in matrix.domains]
    assert domain_names[:3] == ["other", "risk", "sensory"]
    assert domain_names[3:] == ["understanding", "runtime"]

    sensory = next(domain for domain in matrix.domains if domain.name == "sensory")
    assert sensory.files == 1
    assert sensory.covered == 2
    assert sensory.missed == 1
    assert pytest.approx(sensory.percent, rel=1e-6) == 66.67

    understanding = next(
        domain for domain in matrix.domains if domain.name == "understanding"
    )
    assert understanding.files == 2
    assert understanding.covered == 4
    assert understanding.missed == 1
    assert pytest.approx(understanding.percent, rel=1e-6) == 80.0

    runtime = next(domain for domain in matrix.domains if domain.name == "runtime")
    assert runtime.files == 1
    assert runtime.covered == 2
    assert runtime.missed == 0
    assert runtime.percent == 100.0


def test_render_markdown_highlights_laggards(coverage_report: Path) -> None:
    matrix = build_coverage_matrix(coverage_report)
    markdown = render_markdown(matrix, threshold=60.0)

    assert "| Domain |" in markdown
    assert "sensory" in markdown
    assert "runtime" in markdown
    assert "understanding" in markdown
    assert "Domains below the 60.00% threshold" in markdown
    assert "other (0.00%)" in markdown
    assert "risk (33.33%)" in markdown


def test_identify_laggards_filters_domains(coverage_report: Path) -> None:
    matrix = build_coverage_matrix(coverage_report)

    laggards = identify_laggards(matrix, threshold=70.0)

    assert [domain.name for domain in laggards] == ["other", "risk", "sensory"]


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
    assert payload["lagging_count"] == 3
    assert payload["worst_domain"]["name"] == "other"
    totals = payload["totals"]
    assert totals["files"] == 6
    assert totals["coverage_percent"] == pytest.approx(64.29)
    assert "source_files" in payload
    assert "src/sensory/foo.py" in payload["source_files"]
    assert "src/intelligence/legacy.py" in payload["source_files"]


def test_cli_fails_when_laggards_present_and_flag_enabled(
    coverage_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = coverage_matrix_main(
        [
            "--coverage-report",
            str(coverage_report),
            "--threshold",
            "70",
            "--fail-below-threshold",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Domains below the 70.00% threshold" in captured.out


def test_cli_succeeds_when_flag_enabled_but_no_laggards(
    coverage_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = coverage_matrix_main(
        [
            "--coverage-report",
            str(coverage_report),
            "--threshold",
            "0",
            "--fail-below-threshold",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "All tracked domains meet the 0.00% coverage threshold." in captured.out


def test_cli_succeeds_when_required_files_present(
    coverage_report: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = coverage_matrix_main(
        [
            "--coverage-report",
            str(coverage_report),
            "--require-file",
            "src/sensory/foo.py",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "All required files present in coverage" in captured.out
    assert "src/sensory/foo.py" in captured.out


def test_cli_fails_when_required_file_missing(
    coverage_report: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = coverage_matrix_main(
        [
            "--coverage-report",
            str(coverage_report),
            "--require-file",
            "src/data_foundation/ingest/scheduler.py",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Missing required coverage for" in captured.out
    assert "src/data_foundation/ingest/scheduler.py" in captured.out
    assert "Required files missing from coverage" in captured.err
