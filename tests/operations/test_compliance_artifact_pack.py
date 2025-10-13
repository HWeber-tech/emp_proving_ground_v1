"""Coverage for the compliance artifact pack builder."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.operations.compliance_artifact_pack import build_compliance_artifact_pack
from src.operations.compliance_readiness import evaluate_compliance_readiness
from src.operations.regulatory_telemetry import (
    RegulatoryTelemetrySignal,
    RegulatoryTelemetryStatus,
    evaluate_regulatory_telemetry,
)


def _write_audit_log(path: Path) -> None:
    entries = [
        {
            "timestamp": datetime(2024, 5, 1, 10, tzinfo=UTC).isoformat(),
            "event_type": "system_event",
            "component": "ingest",
            "severity": "info",
            "message": "ingest pipeline started",
        },
        {
            "timestamp": datetime(2024, 5, 1, 11, tzinfo=UTC).isoformat(),
            "event_type": "governance_decision",
            "decision_type": "approve_strategy",
            "strategy_id": "alpha-1",
            "genome_id": "genome-42",
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def test_build_pack_exports_audit_log_and_metadata(tmp_path: Path) -> None:
    audit_log_path = tmp_path / "audit_log.jsonl"
    _write_audit_log(audit_log_path)

    compliance_snapshot = evaluate_compliance_readiness(
        trade_summary={
            "last_snapshot": {"status": "pass", "checks": []},
            "policy": {"policy_name": "default"},
        },
        kyc_summary={"last_snapshot": {"status": "pass"}},
        workflow_summary={
            "status": "completed",
            "workflows": [
                {
                    "status": "completed",
                    "tasks": [{"status": "completed"}],
                }
            ],
        },
        metadata={"note": "daily compliance check"},
    )

    regulatory_snapshot = evaluate_regulatory_telemetry(
        signals=[
            RegulatoryTelemetrySignal(
                name="trade_compliance",
                status=RegulatoryTelemetryStatus.ok,
                summary="policy checks passing",
                observed_at=datetime(2024, 5, 1, 12, tzinfo=UTC),
                metadata={},
            )
        ],
        metadata={"window": "hourly"},
        stale_after=timedelta(hours=2),
    )

    output_dir = tmp_path / "pack"
    archive_path = tmp_path / "pack.tar.gz"

    result = build_compliance_artifact_pack(
        audit_log_path=audit_log_path,
        output_dir=output_dir,
        compliance_snapshot=compliance_snapshot,
        regulatory_snapshot=regulatory_snapshot,
        metadata={"run_id": "COMPLIANCE-20240501"},
        archive_path=archive_path,
    )

    assert result.audit_log and result.audit_log.exists()
    assert result.compliance_readiness_json and result.compliance_readiness_json.exists()
    assert result.compliance_readiness_markdown and result.compliance_readiness_markdown.exists()
    assert result.regulatory_snapshot_json and result.regulatory_snapshot_json.exists()
    assert result.archive_path == archive_path
    assert archive_path.exists()

    exported_log = result.audit_log.read_text(encoding="utf-8")
    original_log = audit_log_path.read_text(encoding="utf-8")
    assert exported_log == original_log

    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["audit_log"]["line_count"] == 2
    assert manifest["audit_log"]["statistics"]["total_entries"] == 2
    assert "audit_log.jsonl" in manifest["files"]
    assert "compliance_readiness.json" in manifest["files"]
    assert "compliance_readiness.md" in manifest["files"]
    assert "regulatory_telemetry.json" in manifest["files"]
    assert "manifest.json" in manifest["files"]
    assert archive_path.name in manifest["files"]
    assert manifest["metadata"]["run_id"] == "COMPLIANCE-20240501"


def test_build_pack_handles_missing_audit_log(tmp_path: Path) -> None:
    audit_log_path = tmp_path / "missing.jsonl"
    output_dir = tmp_path / "pack"

    result = build_compliance_artifact_pack(
        audit_log_path=audit_log_path,
        output_dir=output_dir,
    )

    assert result.audit_log is None
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["audit_log"]["exported"] is None
    assert manifest["audit_log"]["statistics"] == {}
    assert manifest["files"] == ["manifest.json"]
