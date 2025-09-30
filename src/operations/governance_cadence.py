"""Governance reporting cadence orchestrator.

This module coordinates the governance reporting workflow described in the
roadmap.  It wraps the lower-level helpers from
``src.operations.governance_reporting`` with interval gating, persistence, and
event-bus publication so professional runtimes can trigger the cadence from a
single entrypoint.  The runner keeps track of the last generated report,
hydrates Timescale audit evidence, and emits the fused governance artefact when
the configured interval has elapsed.

The implementation keeps side-effects injectable to preserve unit-test
ergonomics.  Providers can be supplied for compliance readiness, regulatory
telemetry, system configuration, and audit evidence collection.  Publishing and
persistence functions are also injectable, mirroring the defensive patterns
used by other operational modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, MutableMapping

from src.core.event_bus import EventBus
from src.governance.system_config import SystemConfig
from src.operations.compliance_readiness import ComplianceReadinessSnapshot
from src.operations.governance_reporting import (
    GovernanceReport,
    collect_audit_evidence,
    generate_governance_report,
    persist_governance_report,
    publish_governance_report,
    should_generate_report,
)
from src.operations.regulatory_telemetry import RegulatoryTelemetrySnapshot

GovernanceSnapshot = (
    ComplianceReadinessSnapshot | Mapping[str, object] | None
)
RegulatorySnapshot = (
    RegulatoryTelemetrySnapshot | Mapping[str, object] | None
)


def _parse_timestamp(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _load_last_generated(path: Path) -> datetime | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, MutableMapping):
        return None
    latest = payload.get("latest")
    if isinstance(latest, Mapping):
        return _parse_timestamp(latest.get("generated_at"))
    return None


def _collect_audit(config: SystemConfig, strategy_id: str | None) -> Mapping[str, object]:
    return collect_audit_evidence(config, strategy_id=strategy_id)


def _persist_report(
    report: GovernanceReport, path: Path, *, history_limit: int
) -> None:
    persist_governance_report(report, path, history_limit=history_limit)


def _metadata_with_defaults(
    metadata: Mapping[str, object] | None, *, interval: timedelta, strategy_id: str | None
) -> Mapping[str, object]:
    payload = dict(metadata or {})
    payload.setdefault("cadence_interval_seconds", int(interval.total_seconds()))
    if strategy_id:
        payload.setdefault("strategy_id", strategy_id)
    return payload


@dataclass(slots=True)
class GovernanceCadenceRunner:
    """Run the governance reporting cadence when due."""

    event_bus: EventBus
    config_provider: Callable[[], SystemConfig]
    compliance_provider: Callable[[], GovernanceSnapshot]
    regulatory_provider: Callable[[], RegulatorySnapshot]
    report_path: Path
    interval: timedelta
    history_limit: int = 12
    strategy_id_provider: Callable[[], str | None] | None = None
    metadata_provider: Callable[[], Mapping[str, object] | None] | None = None
    audit_collector: Callable[[SystemConfig, str | None], Mapping[str, object]] = _collect_audit
    publisher: Callable[[EventBus, GovernanceReport], None] = publish_governance_report
    persister: Callable[[GovernanceReport, Path, int], None] | None = None
    _last_generated_at: datetime | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._last_generated_at = _load_last_generated(self.report_path)
        if self.persister is None:
            self.persister = lambda report, path, limit: _persist_report(
                report, path, history_limit=limit
            )

    @property
    def last_generated_at(self) -> datetime | None:
        return self._last_generated_at

    def run(self, *, reference: datetime | None = None) -> GovernanceReport | None:
        """Execute the cadence if the interval has elapsed."""

        if not should_generate_report(self._last_generated_at, self.interval, reference=reference):
            return None

        generated_at = reference
        if generated_at is not None and generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)

        config = self.config_provider()
        strategy_id = self.strategy_id_provider() if self.strategy_id_provider else None
        compliance = self.compliance_provider()
        regulatory = self.regulatory_provider()
        audit_evidence = self.audit_collector(config, strategy_id)
        metadata = _metadata_with_defaults(
            self.metadata_provider() if self.metadata_provider else None,
            interval=self.interval,
            strategy_id=strategy_id,
        )

        report = generate_governance_report(
            compliance_readiness=compliance,
            regulatory_snapshot=regulatory,
            audit_evidence=audit_evidence,
            generated_at=generated_at,
            metadata=metadata,
        )

        self.publisher(self.event_bus, report)
        assert self.persister is not None  # for type checkers
        self.persister(report, self.report_path, self.history_limit)
        self._last_generated_at = report.generated_at
        return report

