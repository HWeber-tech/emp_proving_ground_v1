from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.deployment.rebuild_degraded_nodes as cli


def _capture_runs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], str]]:
    calls: list[tuple[list[str], str]] = []

    def fake_run(cmd: list[str], *, check: bool, cwd: str) -> subprocess.CompletedProcess[str]:
        calls.append((cmd, cwd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls


def test_main_rebuilds_resources_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    terraform_dir = tmp_path / "infra"
    terraform_dir.mkdir()

    degraded_file = tmp_path / "degraded.json"
    degraded_file.write_text(json.dumps(["hcloud_server.single_box"]), encoding="utf-8")

    calls = _capture_runs(monkeypatch)

    exit_code = cli.main([
        "--terraform-dir",
        str(terraform_dir),
        "--degraded-file",
        str(degraded_file),
    ])

    assert exit_code == 0
    assert calls == [
        (["terraform", "init", "-input=false"], str(terraform_dir)),
        (
            [
                "terraform",
                "apply",
                "-replace=hcloud_server.single_box",
                "-input=false",
                "-auto-approve",
            ],
            str(terraform_dir),
        ),
    ]


def test_main_accepts_cli_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    terraform_dir = tmp_path / "infra"
    terraform_dir.mkdir()

    calls = _capture_runs(monkeypatch)

    exit_code = cli.main([
        "--terraform-dir",
        str(terraform_dir),
        "--resource",
        "hcloud_server.single_box",
    ])

    assert exit_code == 0
    apply_cmd = calls[1][0]
    assert "-replace=hcloud_server.single_box" in apply_cmd


def test_main_returns_zero_when_nothing_to_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    terraform_dir = tmp_path / "infra"
    terraform_dir.mkdir()

    calls = _capture_runs(monkeypatch)

    exit_code = cli.main([
        "--terraform-dir",
        str(terraform_dir),
    ])

    assert exit_code == 0
    assert calls == []


def test_main_surfaces_invalid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    terraform_dir = tmp_path / "infra"
    terraform_dir.mkdir()

    degraded_file = tmp_path / "degraded.json"
    degraded_file.write_text(json.dumps({"resources": ""}), encoding="utf-8")

    calls = _capture_runs(monkeypatch)

    exit_code = cli.main([
        "--terraform-dir",
        str(terraform_dir),
        "--degraded-file",
        str(degraded_file),
    ])

    assert exit_code == 1
    assert calls == []


def test_main_propagates_subprocess_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    terraform_dir = tmp_path / "infra"
    terraform_dir.mkdir()

    degraded_file = tmp_path / "degraded.json"
    degraded_file.write_text(json.dumps(["hcloud_server.single_box"]), encoding="utf-8")

    def fake_run(cmd: list[str], *, check: bool, cwd: str) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main([
        "--terraform-dir",
        str(terraform_dir),
        "--degraded-file",
        str(degraded_file),
    ])

    assert exit_code == 2
