from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.governance.system_config import SystemConfig
from src.operations.compliance_readiness import (
    ComplianceReadinessComponent,
    ComplianceReadinessSnapshot,
    ComplianceReadinessStatus,
)
from src.operations.governance_reporting import (
    GovernanceReportStatus,
    collect_audit_evidence,
    generate_governance_report,
    persist_governance_report,
    should_generate_report,
)
from src.operations.regulatory_telemetry import (
    RegulatoryTelemetrySignal,
    RegulatoryTelemetrySnapshot,
    RegulatoryTelemetryStatus,
)


class _StubJournal:
    def __init__(self, *, stats: dict[str, object] | None = None, error: Exception | None = None) -> None:
        self._stats = stats or {}
        self._error = error
        self.closed = False

    def summarise(self, **_: object) -> dict[str, object]:
        if self._error is not None:
            raise self._error
        return dict(self._stats)

    def close(self) -> None:
        self.closed = True


def test_generate_governance_report_composes_sections() -> None:
    compliance_snapshot = ComplianceReadinessSnapshot(
        status=ComplianceReadinessStatus.warn,
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
        components=(
            ComplianceReadinessComponent(
                name="kyc_aml",
                status=ComplianceReadinessStatus.warn,
                summary="KYC monitor degraded",
                metadata={"open_cases": 2},
            ),
        ),
        metadata={},
    )

    regulatory_snapshot = RegulatoryTelemetrySnapshot(
        generated_at=datetime(2024, 1, 1, 12, tzinfo=UTC),
        status=RegulatoryTelemetryStatus.ok,
        coverage_ratio=1.0,
        signals=(
            RegulatoryTelemetrySignal(
                name="kyc_aml",
                status=RegulatoryTelemetryStatus.ok,
                summary="KYC telemetry healthy",
                observed_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
        ),
        required_domains=("kyc_aml",),
        missing_domains=(),
        metadata={"coverage_percent": 100.0},
    )

    audit_evidence = {
        "metadata": {"configured": True, "dialect": "sqlite"},
        "compliance": {"stats": {"total_records": 5, "last_recorded_at": "2024-01-01T00:00:00Z"}},
        "kyc": {"stats": {"total_cases": 3, "last_recorded_at": "2024-01-01T12:00:00Z"}},
    }

    report = generate_governance_report(
        compliance_readiness=compliance_snapshot,
        regulatory_snapshot=regulatory_snapshot,
        audit_evidence=audit_evidence,
        generated_at=datetime(2024, 1, 2, tzinfo=UTC),
        metadata={"cadence": "daily"},
    )

    assert report.status is GovernanceReportStatus.warn
    names = [section.name for section in report.sections]
    assert names == ["kyc_aml", "regulatory_telemetry", "audit_storage"]
    kyc_section = report.sections[0]
    assert kyc_section.status is GovernanceReportStatus.warn
    assert "open_cases" in kyc_section.metadata
    audit_section = report.sections[2]
    assert "compliance" in audit_section.metadata
    markdown = report.to_markdown()
    assert "Section" in markdown and "KYC monitor degraded" in markdown


def test_collect_audit_evidence_records_errors() -> None:
    config = SystemConfig()

    compliance_stats = {"total_records": 0}

    evidence = collect_audit_evidence(
        config,
        journal_factories={
            "compliance": lambda _: _StubJournal(stats=compliance_stats),
            "kyc": lambda _: _StubJournal(error=RuntimeError("journal offline")),
        },
    )

    assert evidence["metadata"]["dialect"] == "sqlite"
    assert evidence["compliance"]["stats"] == compliance_stats
    assert "error" in evidence["kyc"]
    assert "errors" in evidence["metadata"]


def test_should_generate_report_respects_interval() -> None:
    last = datetime(2024, 1, 1, tzinfo=UTC)
    interval = timedelta(hours=6)

    assert not should_generate_report(last, interval, reference=last + timedelta(hours=5, minutes=59))
    assert should_generate_report(last, interval, reference=last + timedelta(hours=6))
    assert should_generate_report(None, interval)


def test_persist_governance_report_trims_history(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "governance.json"

    def _report(status: GovernanceReportStatus, ts: int):
        return generate_governance_report(
            compliance_readiness=None,
            regulatory_snapshot=None,
            audit_evidence=None,
            generated_at=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=ts),
            metadata={"status": status.value},
        )

    persist_governance_report(_report(GovernanceReportStatus.ok, 0), path, history_limit=2)
    persist_governance_report(_report(GovernanceReportStatus.warn, 1), path, history_limit=2)
    persist_governance_report(_report(GovernanceReportStatus.fail, 2), path, history_limit=2)

    data = path.read_text(encoding="utf-8")
    payload = json.loads(data)

    assert payload["latest"]["metadata"]["status"] == GovernanceReportStatus.fail.value
    assert len(payload["history"]) == 2
    statuses = [entry["metadata"]["status"] for entry in payload["history"]]
    assert statuses == [GovernanceReportStatus.warn.value, GovernanceReportStatus.fail.value]

