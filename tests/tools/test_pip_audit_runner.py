from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import subprocess

import pytest

from tools.security import pip_audit_runner as audit


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture()
def sample_report() -> list[dict[str, Any]]:
    return [
        {
            "name": "example",
            "version": "1.0.0",
            "vulns": [
                {
                    "id": "CVE-0001",
                    "severity": "HIGH",
                    "fix_versions": ["1.2.0"],
                    "description": "Example vulnerability",
                    "aliases": ["GHSA-0001"],
                }
            ],
        }
    ]


def test_run_audit_returns_result_without_findings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: _FakeCompletedProcess(0, stdout="[]"),
    )
    requirement = tmp_path / "requirements.txt"
    requirement.write_text("pytest==8.0.0\n", encoding="utf-8")

    result = audit.run_audit([requirement])

    assert result.ok
    assert result.findings == ()
    assert result.ignored == ()


def test_run_audit_filters_ignored_ids(
    monkeypatch: pytest.MonkeyPatch,
    sample_report: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    payload = json.dumps(sample_report)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: _FakeCompletedProcess(1, stdout=payload),
    )
    requirement = tmp_path / "req.txt"
    requirement.write_text("flask==3.0\n", encoding="utf-8")

    result = audit.run_audit([requirement], ignore_ids={"CVE-0001"})

    assert result.ok
    assert result.findings == ()
    assert len(result.ignored) == 1
    assert result.ignored[0].vulnerability_id == "CVE-0001"


def test_run_audit_surfaces_findings(
    monkeypatch: pytest.MonkeyPatch,
    sample_report: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    payload = json.dumps(sample_report)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: _FakeCompletedProcess(1, stdout=payload),
    )
    requirement = tmp_path / "req.txt"
    requirement.write_text("flask==3.0\n", encoding="utf-8")

    result = audit.run_audit([requirement])

    assert not result.ok
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.vulnerability_id == "CVE-0001"
    assert finding.fix_versions == ("1.2.0",)


def test_run_audit_raises_on_execution_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: _FakeCompletedProcess(2, stdout="", stderr="boom"),
    )
    requirement = tmp_path / "req.txt"
    requirement.write_text("flask==3.0\n", encoding="utf-8")

    with pytest.raises(audit.PipAuditExecutionError) as exc:
        audit.run_audit([requirement])

    assert exc.value.returncode == 2


def test_cli_writes_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        [
            {
                "name": "demo",
                "version": "1.0",
                "vulns": [],
            }
        ]
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: _FakeCompletedProcess(0, stdout=payload),
    )
    requirement = tmp_path / "requirements.txt"
    requirement.write_text("demo==1.0\n", encoding="utf-8")
    output = tmp_path / "report.md"

    exit_code = audit.main(
        ["--requirement", str(requirement), "--output", str(output), "--format", "markdown"]
    )

    assert exit_code == 0
    content = output.read_text(encoding="utf-8")
    assert "Dependency vulnerability scan" in content


def test_cli_nonzero_on_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        [
            {
                "name": "demo",
                "version": "1.0",
                "vulns": [{"id": "CVE-1234", "fix_versions": []}],
            }
        ]
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: _FakeCompletedProcess(1, stdout=payload),
    )
    requirement = tmp_path / "requirements.txt"
    requirement.write_text("demo==1.0\n", encoding="utf-8")

    exit_code = audit.main(["--requirement", str(requirement)])

    assert exit_code == 1
