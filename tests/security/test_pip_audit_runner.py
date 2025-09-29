from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import MutableMapping, Sequence

import pytest

from src.security import pip_audit_runner


def _summary(
    *,
    actionable: list[MutableMapping[str, object]] | None = None,
    suppressed: list[MutableMapping[str, object]] | None = None,
    expired: tuple[str, ...] = (),
    unused: tuple[str, ...] = (),
) -> pip_audit_runner.AuditSummary:
    return pip_audit_runner.AuditSummary(
        requirements=("requirements/base.txt",),
        actionable=actionable or [],
        suppressed=suppressed or [],
        expired_allowlist=expired,
        unused_allowlist=unused,
    )


def test_load_allowlist_parses_entries(tmp_path: Path) -> None:
    path = tmp_path / "allowlist.yaml"
    path.write_text(
        """
ignored_vulnerabilities:
  - id: GHSA-aaaa-bbbb-cccc
    reason: Waiting for upstream release
    expires: 2030-01-01
  - id: GHSA-dddd-eeee-ffff
    reason: Legacy dependency awaiting removal
"""
    )

    entries = pip_audit_runner.load_allowlist(path)

    assert set(entries) == {"GHSA-aaaa-bbbb-cccc", "GHSA-dddd-eeee-ffff"}
    first = entries["GHSA-aaaa-bbbb-cccc"]
    assert first.reason == "Waiting for upstream release"
    assert first.expires == date(2030, 1, 1)
    assert not first.is_expired(date(2029, 12, 31))

    second = entries["GHSA-dddd-eeee-ffff"]
    assert second.expires is None
    assert second.is_expired(date(2024, 1, 1)) is False


def test_summarise_audit_filters_allowlist(tmp_path: Path) -> None:
    allowlist = {
        "GHSA-allowed": pip_audit_runner.AllowlistEntry(
            vuln_id="GHSA-allowed",
            reason="Covered by vendor SLA",
            expires=None,
        ),
        "GHSA-expired": pip_audit_runner.AllowlistEntry(
            vuln_id="GHSA-expired",
            reason="Awaiting patch",
            expires=date(2024, 1, 1),
        ),
    }

    payload = [
        {
            "name": "package-a",
            "version": "1.0.0",
            "vulns": [
                {"id": "GHSA-allowed", "fix_versions": ["1.0.1"]},
                {"id": "GHSA-missing", "fix_versions": []},
            ],
        },
        {
            "name": "package-b",
            "version": "2.0.0",
            "vulns": [
                {"id": "GHSA-expired", "fix_versions": ["2.0.1"]},
            ],
        },
    ]

    summary = pip_audit_runner.summarise_audit(
        payload,
        allowlist,
        requirements=[Path("requirements/base.txt")],
        today=date(2025, 1, 1),
    )

    assert summary.actionable_count == 2
    assert summary.suppressed_count == 1
    assert summary.expired_allowlist == ("GHSA-expired",)
    assert summary.unused_allowlist == ()

    actionable_ids = {
        vuln["id"]
        for package in summary.actionable
        for vuln in package["vulns"]
    }
    assert actionable_ids == {"GHSA-missing", "GHSA-expired"}

    suppressed_ids = {
        vuln["id"]
        for package in summary.suppressed
        for vuln in package["vulns"]
    }
    assert suppressed_ids == {"GHSA-allowed"}


def test_run_audit_invokes_pip_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requirement = tmp_path / "requirements.txt"
    requirement.write_text("")

    captured: dict[str, object] = {}

    def _fake_invoke(
        requirements: Sequence[Path], *, pip_audit_bin: str
    ) -> list[MutableMapping[str, object]]:
        captured["requirements"] = tuple(requirements)
        captured["binary"] = pip_audit_bin
        return [{"name": "pkg", "version": "1.0", "vulns": []}]

    monkeypatch.setattr(pip_audit_runner, "invoke_pip_audit", _fake_invoke)

    report_path = tmp_path / "report.json"
    summary = pip_audit_runner.run_audit(
        [requirement],
        allowlist_path=None,
        report_path=report_path,
        pip_audit_bin="pip-audit",
    )

    assert captured["requirements"] == (requirement,)
    assert captured["binary"] == "pip-audit"
    assert report_path.exists()
    assert summary.actionable_count == 0


def test_main_exit_codes(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pip_audit_runner, "run_audit", lambda *_, **__: _summary())

    exit_code = pip_audit_runner.main([
        "--requirement",
        "requirements/base.txt",
        "--ignore-expired-allowlist",
    ])

    assert exit_code == 0
    out, err = capsys.readouterr()
    assert "Dependency vulnerability scan" in out
    assert not err


def test_main_handles_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = _summary(
        actionable=[{"name": "pkg", "version": "1", "vulns": [{"id": "GHSA-1"}]}]
    )
    monkeypatch.setattr(pip_audit_runner, "run_audit", lambda *_, **__: summary)

    exit_code = pip_audit_runner.main(["--requirement", "requirements/base.txt"])

    assert exit_code == 1


def test_main_fails_on_expired_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = _summary(expired=("GHSA-expired",))
    monkeypatch.setattr(pip_audit_runner, "run_audit", lambda *_, **__: summary)

    exit_code = pip_audit_runner.main(["--requirement", "requirements/base.txt"])
    assert exit_code == 1

    exit_code = pip_audit_runner.main(
        [
            "--requirement",
            "requirements/base.txt",
            "--ignore-expired-allowlist",
        ]
    )
    assert exit_code == 0
