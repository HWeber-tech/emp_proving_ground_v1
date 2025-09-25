from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_forbidden_integrations import (
    DEFAULT_PATTERN,
    ROOT,
    Match,
    format_matches,
    normalise_allowlist,
    scan_paths,
)


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def create_file(base: Path, relative: str, contents: str) -> Path:
    target = base / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents, encoding="utf-8")
    return target


def test_detects_forbidden_reference(sandbox: Path) -> None:
    suspect = create_file(sandbox, "module/app.py", "from fastapi import FastAPI\n")

    matches = scan_paths(
        [str(sandbox)],
        root=sandbox,
        pattern=DEFAULT_PATTERN,
        allowlist=set(),
    )

    assert matches == [Match(path=suspect.resolve(), line_number=1, line="from fastapi import FastAPI")]


def test_allowlist_excludes_entries(sandbox: Path) -> None:
    allowed = create_file(sandbox, "scripts/helper.py", "import uvicorn\n")

    allowlist = normalise_allowlist([allowed], sandbox)

    matches = scan_paths(
        [str(sandbox)],
        root=sandbox,
        pattern=DEFAULT_PATTERN,
        allowlist=allowlist,
    )

    assert matches == []


def test_cli_reports_matches(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    create_file(workspace, "src/sample.py", "import uvicorn\n")

    script = ROOT / "scripts" / "check_forbidden_integrations.py"
    result = subprocess.run(
        [sys.executable, str(script), str(workspace / "src")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Forbidden integration references detected" in result.stdout
    assert "src/sample.py:1:import uvicorn" in result.stdout


def test_format_matches_includes_relative_paths(sandbox: Path) -> None:
    suspect = create_file(sandbox, "docs/readme.md", "CTRADER_OPEN_API\n")
    matches = [Match(path=suspect.resolve(), line_number=1, line="CTRADER_OPEN_API")]

    message = format_matches(matches, root=sandbox)

    assert "docs/readme.md:1:CTRADER_OPEN_API" in message
