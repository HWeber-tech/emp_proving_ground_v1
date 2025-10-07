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
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

from sqlalchemy.exc import SQLAlchemyError

from src.core.event_bus import Event, EventBus, TopicBus
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
from src.operations.event_bus_failover import publish_event_with_failover

__all__ = [
    "GovernanceReportStatus",
    "GovernanceReportSection",
    "GovernanceReport",
    "GovernanceContextSources",
    "should_generate_report",
    "collect_audit_evidence",
    "generate_governance_report",
    "publish_governance_report",
    "persist_governance_report",
    "load_governance_context_from_config",
    "build_governance_report_from_config",
]


logger = logging.getLogger(__name__)


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


class AuditJournalError(RuntimeError):
    """Raised when a Timescale governance journal cannot be accessed."""


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


@dataclass(slots=True, frozen=True)
class GovernanceContextSources:
    """Resolved governance telemetry context sources derived from configuration."""

    compliance: Mapping[str, object] | None
    compliance_path: Path | None
    regulatory: Mapping[str, object] | None
    regulatory_path: Path | None
    audit: Mapping[str, object] | None
    audit_path: Path | None


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


def _parse_timestamp(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


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

        recent_key = None
        for candidate in ("recent_records", "recent_cases", "recent_entries"):
            if candidate in stats:
                recent_key = candidate
                break
        if recent_key is not None:
            recent_total = _coerce_int(stats.get(recent_key))
            if recent_total <= 0:
                status = _escalate(status, GovernanceReportStatus.warn)
                last_recorded_raw = stats.get("last_recorded_at")
                last_recorded = _parse_timestamp(last_recorded_raw)
                stale_message = f"{key} journal stale"
                if last_recorded is not None:
                    age = datetime.now(tz=UTC) - last_recorded
                    hours = max(age.total_seconds() / 3600.0, 0.0)
                    stale_message += f" (last {hours:.1f}h ago)"
                summary_parts.append(stale_message)

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
    metadata["collected_at"] = datetime.now(tz=UTC).isoformat()
    if strategy_id is not None:
        metadata["strategy_id"] = strategy_id

    factories = journal_factories or {
        "compliance": lambda s: TimescaleComplianceJournal(s.create_engine()),
        "kyc": lambda s: TimescaleKycJournal(s.create_engine()),
    }

    results: dict[str, object] = {}
    errors: list[str] = []

    for key, factory in factories.items():
        journal: object | None = None
        try:
            journal = factory(settings)
        except (AuditJournalError, OSError, SQLAlchemyError) as exc:
            logger.error("Failed to initialise %s audit journal", key, exc_info=exc)
            errors.append(f"{key}: {exc}")
            results[key] = {"error": str(exc)}
            continue

        try:
            summarise = getattr(journal, "summarise", None)
            if callable(summarise):
                stats = summarise(strategy_id=strategy_id)
            else:
                raise AuditJournalError(f"journal {key!r} does not expose summarise()")
        except (AuditJournalError, OSError, SQLAlchemyError) as exc:
            logger.error("Failed to summarise %s audit journal", key, exc_info=exc)
            errors.append(f"{key}: {exc}")
            results[key] = {"error": str(exc)}
        else:
            results[key] = {"stats": stats}
        finally:
            close = getattr(journal, "close", None)
            if callable(close):
                try:
                    close()
                except (AuditJournalError, OSError) as exc:
                    logger.warning("Error closing %s audit journal", key, exc_info=exc)

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
    global_bus_factory: Callable[[], TopicBus] | None = None,
) -> None:
    """Publish the governance report to the provided event bus."""

    event = Event(
        type=channel,
        payload=report.as_dict(),
        source="governance_reporting",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=(
            "Runtime bus rejected governance report; falling back to global bus"
        ),
        runtime_unexpected_message=(
            "Unexpected error publishing governance report via runtime bus"
        ),
        runtime_none_message=(
            "Runtime publish_from_sync returned None for governance report"
        ),
        global_not_running_message=(
            "Global topic bus is not running; governance report not published"
        ),
        global_unexpected_message=(
            "Unexpected error publishing governance report via global bus"
        ),
        global_bus_factory=global_bus_factory,
    )


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


def _context_base_directory(
    extras: Mapping[str, str] | None,
    base_path: Path | None,
) -> Path:
    root_hint = None if extras is None else extras.get("GOVERNANCE_CONTEXT_DIR")
    if root_hint:
        root = Path(root_hint)
        if not root.is_absolute():
            anchor = base_path or Path.cwd()
            return anchor / root
        return root
    return base_path or Path.cwd()


def _load_context_payload(path: Path) -> Mapping[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Governance context file missing: %s", path)
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load governance context from %s", path, exc_info=exc)
        return None
    if isinstance(data, Mapping):
        return {str(key): value for key, value in data.items()}
    logger.warning("Governance context file %s did not contain a JSON object", path)
    return None


def load_governance_context_from_config(
    config: SystemConfig,
    *,
    base_path: Path | None = None,
) -> GovernanceContextSources:
    """Resolve governance telemetry context payloads referenced by SystemConfig extras."""

    extras = config.extras or {}
    root = _context_base_directory(extras, base_path)

    def _resolve(key: str) -> tuple[Mapping[str, object] | None, Path | None]:
        location = extras.get(key)
        if not location:
            return None, None
        candidate = Path(location)
        if not candidate.is_absolute():
            candidate = root / candidate
        payload = _load_context_payload(candidate)
        return payload, candidate

    compliance_payload, compliance_path = _resolve("GOVERNANCE_COMPLIANCE_CONTEXT")
    regulatory_payload, regulatory_path = _resolve("GOVERNANCE_REGULATORY_CONTEXT")
    audit_payload, audit_path = _resolve("GOVERNANCE_AUDIT_CONTEXT")

    return GovernanceContextSources(
        compliance=compliance_payload,
        compliance_path=compliance_path,
        regulatory=regulatory_payload,
        regulatory_path=regulatory_path,
        audit=audit_payload,
        audit_path=audit_path,
    )


def build_governance_report_from_config(
    config: SystemConfig,
    *,
    event_bus: EventBus | None = None,
    base_path: Path | None = None,
    generated_at: datetime | None = None,
    strategy_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
    compliance_provider: Callable[[], ComplianceReadinessSnapshot | Mapping[str, object] | None]
    | None = None,
    regulatory_provider: Callable[[], RegulatoryTelemetrySnapshot | Mapping[str, object] | None]
    | None = None,
    audit_collector: Callable[[SystemConfig, str | None], Mapping[str, object]] | None = None,
) -> GovernanceReport:
    """Compose and optionally publish a governance report using config-driven context."""

    context_sources = load_governance_context_from_config(config, base_path=base_path)

    compliance_payload = (
        compliance_provider() if compliance_provider is not None else context_sources.compliance
    )
    regulatory_payload = (
        regulatory_provider() if regulatory_provider is not None else context_sources.regulatory
    )

    if audit_collector is not None:
        audit_payload = audit_collector(config, strategy_id)
    elif context_sources.audit is not None:
        audit_payload = context_sources.audit
    else:
        audit_payload = collect_audit_evidence(config, strategy_id=strategy_id)

    metadata_payload: dict[str, object] = dict(metadata or {})
    context_metadata: dict[str, str] = {}
    if context_sources.compliance_path is not None:
        context_metadata["compliance"] = str(context_sources.compliance_path)
    if context_sources.regulatory_path is not None:
        context_metadata["regulatory"] = str(context_sources.regulatory_path)
    if context_sources.audit_path is not None:
        context_metadata["audit"] = str(context_sources.audit_path)
    if context_metadata:
        existing = metadata_payload.get("context_sources")
        merged: dict[str, str] = {}
        if isinstance(existing, Mapping):
            merged.update({str(key): str(value) for key, value in existing.items()})
        merged.update(context_metadata)
        metadata_payload["context_sources"] = merged
    metadata_payload.setdefault("source", "governance_context")

    report = generate_governance_report(
        compliance_readiness=compliance_payload,
        regulatory_snapshot=regulatory_payload,
        audit_evidence=audit_payload,
        generated_at=generated_at,
        metadata=metadata_payload,
    )

    if event_bus is not None:
        publish_governance_report(event_bus, report)

    return report
