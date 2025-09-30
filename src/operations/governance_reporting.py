"""Governance reporting cadence for KYC/AML and regulatory telemetry.

This module bundles the compliance surfaces highlighted in the roadmap into a
repeatable reporting cadence.  It fuses the latest KYC/AML readiness snapshot,
regulatory telemetry coverage, and Timescale audit evidence into a single
report that can be published to the event bus and persisted for audit trails.

The implementation intentionally mirrors existing operations utilities so it
stays lightweight enough for CI while still providing deterministic outputs for
governance reviews.  Consumers can use :func:`should_generate_report` to decide
when the cadence needs to emit a new artefact, call
``generate_governance_report`` to assemble the payload, publish it via
``publish_governance_report``, and persist the JSON bundle with
``persist_governance_report``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus, get_global_bus
from src.governance.system_config import SystemConfig
from src.operations.compliance_readiness import (
    ComplianceReadinessComponent,
    ComplianceReadinessSnapshot,
    ComplianceReadinessStatus,
)
from src.operations.regulatory_telemetry import (
    RegulatoryTelemetrySnapshot,
    RegulatoryTelemetryStatus,
)

from src.data_foundation.persist.timescale import (
    TimescaleComplianceJournal,
    TimescaleConnectionSettings,
    TimescaleKycJournal,
)

__all__ = [
    "GovernanceReportStatus",
    "GovernanceReportSection",
    "GovernanceReport",
    "should_generate_report",
    "collect_audit_evidence",
    "generate_governance_report",
    "publish_governance_report",
    "persist_governance_report",
]


class GovernanceReportStatus(StrEnum):
    """Normalised status labels used for the governance report."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[GovernanceReportStatus, int] = {
    GovernanceReportStatus.ok: 0,
    GovernanceReportStatus.warn: 1,
    GovernanceReportStatus.fail: 2,
}


def _escalate(
    current: GovernanceReportStatus, candidate: GovernanceReportStatus
) -> GovernanceReportStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _normalise_status(value: object | None) -> GovernanceReportStatus:
    if isinstance(value, GovernanceReportStatus):
        return value
    if isinstance(value, ComplianceReadinessStatus):
        return GovernanceReportStatus(value.value)
    if isinstance(value, RegulatoryTelemetryStatus):
        return GovernanceReportStatus(value.value)
    label = str(value or "").strip().lower()
    if label in {"ok", "pass", "green"}:
        return GovernanceReportStatus.ok
    if label in {"warn", "warning", "amber"}:
        return GovernanceReportStatus.warn
    if label in {"fail", "failed", "error", "red"}:
        return GovernanceReportStatus.fail
    return GovernanceReportStatus.warn


@dataclass(slots=True, frozen=True)
class GovernanceReportSection:
    """Structured section included in the governance report."""

    name: str
    status: GovernanceReportStatus
    summary: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True, frozen=True)
class GovernanceReport:
    """Aggregated governance artefact ready for publication and storage."""

    status: GovernanceReportStatus
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    sections: tuple[GovernanceReportSection, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "sections": [section.as_dict() for section in self.sections],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        if not self.sections:
            return "| Section | Status | Summary |\n| --- | --- | --- |\n"

        rows = ["| Section | Status | Summary |", "| --- | --- | --- |"]
        for section in self.sections:
            rows.append(
                f"| {section.name} | {section.status.value.upper()} | {section.summary} |"
            )
        if self.metadata:
            rows.append("")
            rows.append("Metadata:")
            for key, value in sorted(self.metadata.items()):
                rows.append(f"- **{key}**: {value}")
        return "\n".join(rows)


def should_generate_report(
    last_generated_at: datetime | None,
    interval: timedelta,
    *,
    reference: datetime | None = None,
) -> bool:
    """Return ``True`` when the cadence interval has elapsed."""

    now = reference or datetime.now(tz=UTC)
    if last_generated_at is None:
        return True
    if last_generated_at.tzinfo is None:
        last_generated_at = last_generated_at.replace(tzinfo=UTC)
    delta = now - last_generated_at
    return delta >= interval


def _coerce_mapping(value: object | None) -> MutableMapping[str, object]:
    if isinstance(value, MutableMapping):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _component_lookup(
    snapshot: ComplianceReadinessSnapshot | Mapping[str, object] | None,
    *,
    name: str,
) -> ComplianceReadinessComponent | Mapping[str, object] | None:
    if snapshot is None:
        return None
    if isinstance(snapshot, ComplianceReadinessSnapshot):
        for component in snapshot.components:
            if component.name == name:
                return component
        return None
    mapping = _coerce_mapping(snapshot)
    components = mapping.get("components")
    if isinstance(components, Sequence):
        for payload in components:
            candidate = _coerce_mapping(payload)
            if str(candidate.get("name")) == name:
                return candidate
    return None


def _section_from_compliance(
    snapshot: ComplianceReadinessSnapshot | Mapping[str, object] | None,
) -> GovernanceReportSection:
    component = _component_lookup(snapshot, name="kyc_aml")
    if component is None:
        return GovernanceReportSection(
            name="kyc_aml",
            status=GovernanceReportStatus.warn,
            summary="kyc snapshot missing",
            metadata={"reason": "kyc_snapshot_missing"},
        )

    if isinstance(component, ComplianceReadinessComponent):
        status = _normalise_status(component.status)
        summary = component.summary
        metadata = component.metadata
    else:
        status = _normalise_status(component.get("status"))
        summary = str(component.get("summary") or "kyc snapshot available")
        metadata = _coerce_mapping(component.get("metadata"))

    return GovernanceReportSection(
        name="kyc_aml",
        status=status,
        summary=summary,
        metadata=dict(metadata),
    )


def _section_from_regulatory(
    snapshot: RegulatoryTelemetrySnapshot | Mapping[str, object] | None,
) -> GovernanceReportSection:
    if snapshot is None:
        return GovernanceReportSection(
            name="regulatory_telemetry",
            status=GovernanceReportStatus.warn,
            summary="regulatory telemetry missing",
            metadata={"reason": "regulatory_snapshot_missing"},
        )

    if isinstance(snapshot, RegulatoryTelemetrySnapshot):
        status = _normalise_status(snapshot.status)
        coverage_percent = round(snapshot.coverage_ratio * 100.0, 2)
        summary = f"coverage={coverage_percent}% across {len(snapshot.signals)} signals"
        metadata = {
            "required_domains": list(snapshot.required_domains),
            "missing_domains": list(snapshot.missing_domains),
        }
        if snapshot.metadata:
            metadata.update(snapshot.metadata)
    else:
        mapping = _coerce_mapping(snapshot)
        status = _normalise_status(mapping.get("status"))
        coverage = mapping.get("coverage_ratio")
        try:
            coverage_percent = round(float(coverage) * 100.0, 2)
        except (TypeError, ValueError):
            coverage_percent = 0.0
        summary = mapping.get("summary") or (
            f"coverage={coverage_percent}% across {len(mapping.get('signals') or [])} signals"
        )
        metadata = {
            "required_domains": list(mapping.get("required_domains") or []),
            "missing_domains": list(mapping.get("missing_domains") or []),
        }
        extra_meta = mapping.get("metadata")
        if isinstance(extra_meta, Mapping):
            metadata.update(extra_meta)

    return GovernanceReportSection(
        name="regulatory_telemetry",
        status=status,
        summary=str(summary),
        metadata=metadata,
    )


def _coerce_int(value: object | None) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _section_from_audit_evidence(evidence: Mapping[str, object] | None) -> GovernanceReportSection:
    if evidence is None:
        return GovernanceReportSection(
            name="audit_storage",
            status=GovernanceReportStatus.warn,
            summary="audit evidence unavailable",
            metadata={"reason": "audit_evidence_missing"},
        )

    mapping = _coerce_mapping(evidence)
    metadata = _coerce_mapping(mapping.get("metadata"))
    compliance = _coerce_mapping(mapping.get("compliance"))
    kyc = _coerce_mapping(mapping.get("kyc"))

    summary_parts: list[str] = []
    status = GovernanceReportStatus.ok

    configured = bool(metadata.get("configured", False))
    if not configured:
        status = _escalate(status, GovernanceReportStatus.warn)
        summary_parts.append("Timescale fallback in use")

    errors: list[str] = []
    if "error" in compliance:
        errors.append(f"compliance journal: {compliance['error']}")
    if "error" in kyc:
        errors.append(f"kyc journal: {kyc['error']}")
    if errors:
        status = GovernanceReportStatus.fail
        summary_parts.extend(errors)

    def _summarise_block(block: Mapping[str, object], *, key: str) -> None:
        nonlocal status
        stats = block.get("stats")
        if not isinstance(stats, Mapping):
            return
        total = _coerce_int(
            stats.get("total_records") or stats.get("total_cases") or stats.get("total_entries")
        )
        if total <= 0:
            status = _escalate(status, GovernanceReportStatus.warn)
            summary_parts.append(f"{key} journal empty")
        else:
            summary_parts.append(f"{key} journal records={total}")

    if not errors:
        _summarise_block(compliance, key="compliance")
        _summarise_block(kyc, key="kyc")

    if not summary_parts:
        summary_parts.append("audit posture healthy")

    combined_metadata: dict[str, object] = {}
    if metadata:
        combined_metadata.update(metadata)
    if compliance:
        combined_metadata["compliance"] = compliance
    if kyc:
        combined_metadata["kyc"] = kyc

    return GovernanceReportSection(
        name="audit_storage",
        status=status,
        summary="; ".join(summary_parts),
        metadata=combined_metadata,
    )


def collect_audit_evidence(
    config: SystemConfig,
    *,
    strategy_id: str | None = None,
    journal_factories: Mapping[
        str, Callable[[TimescaleConnectionSettings], object]
    ] | None = None,
) -> dict[str, object]:
    """Collect Timescale audit evidence for compliance governance reports.

    ``journal_factories`` allows dependency injection during testing.  Factories
    should return an object exposing ``summarise`` and ``close`` methods.
    """

    settings = TimescaleConnectionSettings.from_mapping(config.extras)
    metadata: dict[str, object] = {
        "configured": settings.configured,
        "dialect": "postgresql" if settings.is_postgres() else "sqlite",
    }

    factories = journal_factories or {
        "compliance": lambda s: TimescaleComplianceJournal(s.create_engine()),
        "kyc": lambda s: TimescaleKycJournal(s.create_engine()),
    }

    results: dict[str, object] = {}
    errors: list[str] = []

    for key, factory in factories.items():
        try:
            journal = factory(settings)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{key}: {exc}")
            results[key] = {"error": str(exc)}
            continue

        try:
            if key == "compliance":
                stats = journal.summarise(strategy_id=strategy_id)
            elif key == "kyc":
                stats = journal.summarise(strategy_id=strategy_id)
            else:
                stats = journal.summarise(strategy_id=strategy_id)
            results[key] = {"stats": stats}
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{key}: {exc}")
            results[key] = {"error": str(exc)}
        finally:
            try:
                journal.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    if errors:
        metadata["errors"] = errors

    results["metadata"] = metadata
    return results


def generate_governance_report(
    *,
    compliance_readiness: ComplianceReadinessSnapshot | Mapping[str, object] | None,
    regulatory_snapshot: RegulatoryTelemetrySnapshot | Mapping[str, object] | None,
    audit_evidence: Mapping[str, object] | None,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
    generated_at: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> GovernanceReport:
    """Assemble a governance report from the supplied telemetry surfaces."""

    generated = generated_at or datetime.now(tz=UTC)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=UTC)

    period_end = period_end or generated
    if period_end.tzinfo is None:
        period_end = period_end.replace(tzinfo=UTC)

    period_start = period_start or (period_end - timedelta(days=1))
    if period_start.tzinfo is None:
        period_start = period_start.replace(tzinfo=UTC)

    sections = (
        _section_from_compliance(compliance_readiness),
        _section_from_regulatory(regulatory_snapshot),
        _section_from_audit_evidence(audit_evidence),
    )

    overall = GovernanceReportStatus.ok
    for section in sections:
        overall = _escalate(overall, section.status)

    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("sections", [section.name for section in sections])

    return GovernanceReport(
        status=overall,
        generated_at=generated,
        period_start=period_start,
        period_end=period_end,
        sections=sections,
        metadata=payload_metadata,
    )


def publish_governance_report(
    event_bus: EventBus,
    report: GovernanceReport,
    *,
    channel: str = "telemetry.compliance.governance",
) -> None:
    """Publish the governance report to the provided event bus."""

    event = Event(
        type=channel,
        payload=report.as_dict(),
        source="governance_reporting",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync):
        try:
            if event_bus.is_running():
                publish_from_sync(event)
                return
        except Exception:  # pragma: no cover - best-effort logging path
            pass

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - best-effort logging path
        pass


def persist_governance_report(
    report: GovernanceReport,
    path: Path,
    *,
    history_limit: int = 12,
) -> None:
    """Persist the report to disk, trimming history to the configured limit."""

    payload = report.as_dict()
    history: list[Mapping[str, object]] = []

    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            existing = {}
        if isinstance(existing, Mapping):
            raw_history = existing.get("history")
            if isinstance(raw_history, list):
                history = [
                    entry
                    for entry in raw_history
                    if isinstance(entry, Mapping)
                ]
    history.append(payload)

    if history_limit > 0 and len(history) > history_limit:
        history = history[-history_limit:]

    output = {
        "latest": payload,
        "history": history,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

