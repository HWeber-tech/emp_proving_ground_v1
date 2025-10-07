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
import inspect
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Optional, TypeVar, cast

from src.core.event_bus import EventBus
from src.governance.system_config import SystemConfig
from src.operations.compliance_readiness import ComplianceReadinessSnapshot
from src.operations.governance_reporting import (
    GovernanceContextSources,
    GovernanceReport,
    collect_audit_evidence,
    generate_governance_report,
    load_governance_context_from_config,
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


__all__ = [
    "GovernanceCadenceRunner",
    "build_governance_cadence_runner_from_config",
]


class _CadenceContextResolver:
    """Resolve governance context packs alongside SystemConfig snapshots."""

    def __init__(
        self,
        config_provider: Callable[[], SystemConfig],
        *,
        base_path: Path | None,
    ) -> None:
        self._config_provider = config_provider
        self._base_path = base_path
        self._context: GovernanceContextSources | None = None

    def config(self) -> SystemConfig:
        config = self._config_provider()
        self._context = load_governance_context_from_config(
            config,
            base_path=self._base_path,
        )
        return config

    def _ensure_context(self) -> GovernanceContextSources:
        if self._context is None:
            # Loading the config refreshes the cached context.
            self.config()
        assert self._context is not None
        return self._context

    def compliance(self) -> GovernanceSnapshot:
        return self._ensure_context().compliance

    def regulatory(self) -> RegulatorySnapshot:
        return self._ensure_context().regulatory

    def audit(self) -> Mapping[str, object] | None:
        return self._ensure_context().audit

    def context_paths(self) -> Mapping[str, Path | None]:
        if self._context is None:
            return {}
        return {
            "compliance": self._context.compliance_path,
            "regulatory": self._context.regulatory_path,
            "audit": self._context.audit_path,
        }


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

    def run(
        self,
        *,
        reference: datetime | None = None,
        force: bool = False,
    ) -> GovernanceReport | None:
        """Execute the cadence if the interval has elapsed or ``force`` is set."""

        if not force and not should_generate_report(
            self._last_generated_at, self.interval, reference=reference
        ):
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


def build_governance_cadence_runner_from_config(
    *,
    event_bus: EventBus,
    report_path: Path,
    interval: timedelta,
    config: SystemConfig | None = None,
    config_provider: Callable[[], SystemConfig] | None = None,
    base_path: Path | None = None,
    history_limit: int = 12,
    compliance_provider: Callable[[], GovernanceSnapshot] | None = None,
    regulatory_provider: Callable[[], RegulatorySnapshot] | None = None,
    audit_collector: Callable[[SystemConfig, str | None], Mapping[str, object] | None]
    | None = None,
    metadata: Mapping[str, object] | None = None,
    metadata_provider: Callable[[], Mapping[str, object] | None] | None = None,
    strategy_id: str | None = None,
    strategy_id_provider: Callable[[], str | None] | None = None,
    publisher: Callable[[EventBus, GovernanceReport], None] | None = None,
) -> GovernanceCadenceRunner:
    """Build a cadence runner wired to governance context packs.

    Parameters mirror :class:`GovernanceCadenceRunner` with convenience hooks for
    configuration-driven context resolution.  The caller may supply either a
    ``config`` instance or a ``config_provider`` callable; when neither is
    provided the environment-backed :meth:`SystemConfig.from_env` loader is
    used.  Compliance, regulatory, and audit telemetry fall back to the JSON
    context packs referenced by ``SystemConfig.extras``.
    """

    if config is not None and config_provider is not None:
        raise ValueError("Provide either config or config_provider, not both")

    if config_provider is None:
        if config is not None:
            config_provider_fn: Callable[[], SystemConfig] = lambda: config
        else:
            config_provider_fn = lambda: SystemConfig.from_env()
    else:
        config_provider_fn = config_provider

    resolver = _CadenceContextResolver(
        config_provider_fn,
        base_path=base_path,
    )

    config_callable = resolver.config

    compliance_callable = compliance_provider or resolver.compliance
    regulatory_callable = regulatory_provider or resolver.regulatory

    def _audit_collector(
        config_obj: SystemConfig, strategy_id_value: str | None
    ) -> Mapping[str, object] | None:
        if audit_collector is not None:
            return audit_collector(config_obj, strategy_id_value)
        context_payload = resolver.audit()
        if context_payload is not None:
            return context_payload
        return collect_audit_evidence(config_obj, strategy_id=strategy_id_value)

    base_metadata = dict(metadata or {})

    def _metadata_callable() -> Mapping[str, object] | None:
        merged: dict[str, object] = dict(base_metadata)

        if metadata_provider is not None:
            extra = metadata_provider()
            if isinstance(extra, Mapping):
                for key, value in extra.items():
                    merged[str(key)] = value

        context_paths = {
            key: path
            for key, path in resolver.context_paths().items()
            if path is not None
        }
        if context_paths:
            merged.setdefault("source", "governance_context")
            existing = merged.get("context_sources")
            context_metadata: dict[str, str] = {}
            if isinstance(existing, Mapping):
                context_metadata.update(
                    {str(key): str(value) for key, value in existing.items()}
                )
            context_metadata.update({key: str(path) for key, path in context_paths.items()})
            merged["context_sources"] = context_metadata

        return merged or None

    if strategy_id is not None and strategy_id_provider is not None:
        raise ValueError(
            "Provide either strategy_id or strategy_id_provider, not both"
        )

    resolved_strategy_provider = strategy_id_provider
    if resolved_strategy_provider is None and strategy_id is not None:
        resolved_strategy_provider = lambda: strategy_id

    publisher_callable = publisher or publish_governance_report

    return GovernanceCadenceRunner(
        event_bus=event_bus,
        config_provider=config_callable,
        compliance_provider=compliance_callable,
        regulatory_provider=regulatory_callable,
        report_path=report_path,
        interval=interval,
        history_limit=history_limit,
        strategy_id_provider=resolved_strategy_provider,
        metadata_provider=_metadata_callable,
        audit_collector=_audit_collector,
        publisher=publisher_callable,
    )
