from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.governance import run_cadence as cli


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_compliance() -> dict[str, object]:
    return {
        "status": "ok",
        "components": [
            {
                "name": "kyc_aml",
                "status": "ok",
                "summary": "Healthy",
            }
        ],
    }


def _sample_regulatory() -> dict[str, object]:
    return {
        "status": "ok",
        "coverage_ratio": 1.0,
        "signals": [],
        "required_domains": ["kyc_aml"],
        "missing_domains": [],
    }


def _sample_audit() -> dict[str, object]:
    return {
        "metadata": {"configured": True, "dialect": "sqlite"},
        "compliance": {"stats": {"total_records": 1}},
        "kyc": {"stats": {"total_cases": 1}},
    }


def test_cli_skips_when_not_due(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    compliance_path = tmp_path / "compliance.json"
    regulatory_path = tmp_path / "regulatory.json"
    audit_path = tmp_path / "audit.json"
    report_path = tmp_path / "governance.json"

    _write_json(compliance_path, _sample_compliance())
    _write_json(regulatory_path, _sample_regulatory())
    _write_json(audit_path, _sample_audit())
    _write_json(
        report_path,
        {
            "latest": {
                "generated_at": "2024-01-01T00:00:00+00:00",
                "status": "ok",
            },
            "history": [],
        },
    )

    exit_code = cli.main(
        [
            "--compliance",
            str(compliance_path),
            "--regulatory",
            str(regulatory_path),
            "--audit",
            str(audit_path),
            "--report-path",
            str(report_path),
            "--interval",
            "24h",
            "--generated-at",
            "2024-01-01T12:00:00+00:00",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "not due" in captured.out.lower()

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["latest"]["generated_at"] == "2024-01-01T00:00:00+00:00"


def test_cli_generates_report_with_force(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    compliance_path = tmp_path / "compliance.json"
    regulatory_path = tmp_path / "regulatory.json"
    audit_path = tmp_path / "audit.json"
    report_path = tmp_path / "reports" / "governance.json"

    _write_json(compliance_path, _sample_compliance())
    _write_json(regulatory_path, _sample_regulatory())
    _write_json(audit_path, _sample_audit())

    exit_code = cli.main(
        [
            "--compliance",
            str(compliance_path),
            "--regulatory",
            str(regulatory_path),
            "--audit",
            str(audit_path),
            "--report-path",
            str(report_path),
            "--interval",
            "24h",
            "--generated-at",
            "2024-01-03T00:00:00+00:00",
            "--force",
            "--metadata",
            "owner=ops",
            "--strategy-id",
            "alpha",
            "--emit-markdown",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "| Section | Status | Summary |" in captured.out

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    latest = persisted["latest"]

    assert latest["metadata"]["owner"] == "ops"
    assert latest["metadata"]["cadence_runner"] == "tools.governance.run_cadence"
    assert latest["metadata"]["cadence_forced"] is True
    assert latest["metadata"]["strategy_id"] == "alpha"
    assert latest["status"] in {"ok", "warn", "fail"}
    assert len(latest["sections"]) == 3

