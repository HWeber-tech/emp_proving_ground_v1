from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.governance.system_config import SystemConfig
from src.operations.compliance_readiness import (
    ComplianceReadinessComponent,
    ComplianceReadinessSnapshot,
    ComplianceReadinessStatus,
)
from src.operations.governance_reporting import (
    AuditJournalError,
    GovernanceReportStatus,
    build_governance_report_from_config,
    collect_audit_evidence,
    generate_governance_report,
    load_governance_context_from_config,
    persist_governance_report,
    publish_governance_report,
    should_generate_report,
)
from src.operations.event_bus_failover import EventPublishError
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


class _RuntimeOnlyBus:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.published: list[object] = []

    def is_running(self) -> bool:
        return True

    def publish_from_sync(self, event: object) -> None:
        if self._error is not None:
            raise self._error
        self.published.append(event)


class _StubTopicBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object, str]] = []

    def publish_sync(self, event_type: str, payload: object, *, source: str) -> None:
        self.published.append((event_type, payload, source))


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
        strategy_id="alpha",
        journal_factories={
            "compliance": lambda _: _StubJournal(stats=compliance_stats),
            "kyc": lambda _: _StubJournal(
                error=AuditJournalError("journal offline")
            ),
        },
    )

    assert evidence["metadata"]["dialect"] == "sqlite"
    assert evidence["metadata"]["strategy_id"] == "alpha"
    assert "collected_at" in evidence["metadata"]
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


def test_audit_section_marks_stale_journals() -> None:
    stale_audit = {
        "metadata": {"configured": True, "dialect": "sqlite"},
        "compliance": {
            "stats": {
                "total_records": 5,
                "recent_records": 0,
                "recent_window_seconds": 86_400,
                "last_recorded_at": "2023-01-01T00:00:00+00:00",
            }
        },
        "kyc": {
            "stats": {
                "total_cases": 2,
                "recent_cases": 0,
                "recent_window_seconds": 86_400,
                "last_recorded_at": "2023-01-01T00:00:00+00:00",
            }
        },
    }

    report = generate_governance_report(
        compliance_readiness=None,
        regulatory_snapshot=None,
        audit_evidence=stale_audit,
    )

    audit_section = next(section for section in report.sections if section.name == "audit_storage")
    assert audit_section.status is GovernanceReportStatus.warn
    assert "stale" in audit_section.summary.lower()


def test_publish_governance_report_falls_back_to_global_bus() -> None:
    runtime_bus = _RuntimeOnlyBus(error=RuntimeError("runtime down"))
    topic_bus = _StubTopicBus()

    report = generate_governance_report(
        compliance_readiness=None,
        regulatory_snapshot=None,
        audit_evidence=None,
    )

    publish_governance_report(
        runtime_bus,
        report,
        global_bus_factory=lambda: topic_bus,
    )

    assert len(topic_bus.published) == 1


def test_build_governance_report_from_config_uses_context(tmp_path: Path) -> None:
    compliance_path = tmp_path / "compliance.json"
    compliance_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "generated_at": "2024-03-01T00:00:00+00:00",
                "components": [
                    {
                        "name": "kyc_aml",
                        "status": "ok",
                        "summary": "KYC readiness stable",
                        "metadata": {"open_cases": 0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    regulatory_path = tmp_path / "regulatory.json"
    regulatory_path.write_text(
        json.dumps(
            {
                "generated_at": "2024-03-01T12:00:00+00:00",
                "status": "ok",
                "coverage_ratio": 1.0,
                "required_domains": ["kyc_aml"],
                "missing_domains": [],
                "signals": [
                    {
                        "name": "kyc_aml",
                        "status": "ok",
                        "summary": "Telemetry green",
                        "observed_at": "2024-03-01T12:00:00+00:00",
                        "metadata": {"checks": 4},
                    }
                ],
                "metadata": {"notes": "baseline"},
            }
        ),
        encoding="utf-8",
    )

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "metadata": {"configured": True, "dialect": "sqlite"},
                "compliance": {"stats": {"total_records": 5}},
                "kyc": {"stats": {"total_cases": 1}},
            }
        ),
        encoding="utf-8",
    )

    config = SystemConfig(
        extras={
            "GOVERNANCE_CONTEXT_DIR": str(tmp_path),
            "GOVERNANCE_COMPLIANCE_CONTEXT": "compliance.json",
            "GOVERNANCE_REGULATORY_CONTEXT": "regulatory.json",
            "GOVERNANCE_AUDIT_CONTEXT": "audit.json",
        }
    )

    sources = load_governance_context_from_config(config)
    assert sources.compliance is not None
    assert sources.compliance_path == compliance_path
    assert sources.regulatory_path == regulatory_path
    assert sources.audit_path == audit_path

    generated_at = datetime(2024, 3, 2, 0, 0, tzinfo=UTC)
    report = build_governance_report_from_config(
        config,
        generated_at=generated_at,
        metadata={"cadence": "weekly"},
    )

    assert report.status is GovernanceReportStatus.ok
    assert report.generated_at == generated_at
    assert report.metadata["cadence"] == "weekly"
    assert report.metadata["source"] == "governance_context"
    context_sources = report.metadata["context_sources"]
    assert context_sources["compliance"].endswith("compliance.json")
    assert context_sources["regulatory"].endswith("regulatory.json")
    assert context_sources["audit"].endswith("audit.json")

    kyc_section = next(section for section in report.sections if section.name == "kyc_aml")
    assert kyc_section.status is GovernanceReportStatus.ok
    audit_section = next(section for section in report.sections if section.name == "audit_storage")
    assert "records=5" in audit_section.summary


def test_publish_governance_report_raises_on_unexpected_runtime_error() -> None:
    runtime_bus = _RuntimeOnlyBus(error=ValueError("bad runtime"))
    topic_bus = _StubTopicBus()

    report = generate_governance_report(
        compliance_readiness=None,
        regulatory_snapshot=None,
        audit_evidence=None,
    )

    with pytest.raises(EventPublishError):
        publish_governance_report(
            runtime_bus,
            report,
            global_bus_factory=lambda: topic_bus,
        )

    assert not topic_bus.published
