from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tools.telemetry import export_governance_report as cli


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_cli_generates_report_and_persists(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    compliance_path = tmp_path / "compliance.json"
    regulatory_path = tmp_path / "regulatory.json"
    audit_path = tmp_path / "audit.json"
    output_path = tmp_path / "report.json"
    persist_path = tmp_path / "history.json"

    _write_json(
        compliance_path,
        {
            "status": "warn",
            "components": [
                {
                    "name": "kyc_aml",
                    "status": "warn",
                    "summary": "Backlog present",
                    "metadata": {"open_cases": 2},
                }
            ],
        },
    )

    _write_json(
        regulatory_path,
        {
            "status": "ok",
            "coverage_ratio": 0.75,
            "signals": [
                {
                    "name": "kyc_aml",
                    "status": "ok",
                    "summary": "Healthy",
                    "observed_at": datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
                }
            ],
            "required_domains": ["kyc_aml"],
            "missing_domains": [],
        },
    )

    _write_json(
        audit_path,
        {
            "metadata": {"configured": True, "dialect": "sqlite"},
            "compliance": {"stats": {"total_records": 3}},
            "kyc": {"stats": {"total_cases": 2}},
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
            "--output",
            str(output_path),
            "--persist",
            str(persist_path),
            "--history-limit",
            "2",
            "--metadata",
            "reviewer=ops",
            "--emit-markdown",
        ]
    )

    assert exit_code == 0

    stdout = capsys.readouterr().out
    assert "Section" in stdout, "Markdown table should be emitted"

    persisted = json.loads(persist_path.read_text(encoding="utf-8"))
    assert persisted["latest"]["metadata"]["reviewer"] == "ops"
    assert len(persisted["history"]) == 1

    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_payload["status"] in {"ok", "warn", "fail"}
    assert output_payload["metadata"]["reviewer"] == "ops"


def test_cli_collects_audit_when_not_provided(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    collected: dict[str, object] = {}

    def _fake_collect(config: object, *, strategy_id: str | None) -> dict[str, object]:
        collected["config"] = config
        collected["strategy_id"] = strategy_id
        return {
            "metadata": {"configured": False, "dialect": "sqlite"},
            "compliance": {"stats": {"total_records": 0}},
            "kyc": {"stats": {"total_cases": 0}},
        }

    monkeypatch.setattr(cli, "collect_audit_evidence", _fake_collect)
    monkeypatch.setattr(cli.SystemConfig, "from_env", classmethod(lambda cls: cls()))

    compliance_path = tmp_path / "compliance.json"
    _write_json(
        compliance_path,
        {
            "status": "ok",
            "components": [
                {
                    "name": "kyc_aml",
                    "status": "ok",
                    "summary": "Healthy",
                }
            ],
        },
    )

    regulatory_path = tmp_path / "regulatory.json"
    _write_json(
        regulatory_path,
        {
            "status": "warn",
            "coverage_ratio": 0.5,
            "signals": [],
            "required_domains": ["kyc_aml"],
            "missing_domains": ["kyc_aml"],
        },
    )

    generated_at = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()

    exit_code = cli.main(
        [
            "--compliance",
            str(compliance_path),
            "--regulatory",
            str(regulatory_path),
            "--generated-at",
            generated_at,
            "--strategy-id",
            "alpha",
        ]
    )

    assert exit_code == 0
    assert collected["strategy_id"] == "alpha"
    assert "config" in collected

