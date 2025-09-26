"""Configuration change telemetry aligned with the modernization roadmap.

This module evaluates differences between successive ``SystemConfig`` snapshots,
grades the risk of each change, and produces reusable telemetry that can be
persisted to Timescale, published on the runtime event bus, and surfaced in
operator summaries.  It gives the roadmap's "configuration audit" milestone a
concrete implementation so institutional runs record who changed what before
high-impact toggles (tier, backbone, run mode, credentials) go live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Callable, Mapping, MutableMapping
from uuid import uuid4

from src.core.event_bus import Event, EventBus, get_global_bus
from src.governance.system_config import SystemConfig

logger = logging.getLogger(__name__)


class ConfigurationAuditStatus(StrEnum):
    """Severity levels exposed by configuration audit telemetry."""

    passed = "pass"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: dict[ConfigurationAuditStatus, int] = {
    ConfigurationAuditStatus.passed: 0,
    ConfigurationAuditStatus.warn: 1,
    ConfigurationAuditStatus.fail: 2,
}


@dataclass(frozen=True)
class ConfigurationChange:
    """A single configuration field transition."""

    field: str
    previous: object | None
    current: object | None
    severity: ConfigurationAuditStatus
    note: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "field": self.field,
            "severity": self.severity.value,
        }
        if self.previous is not None:
            payload["previous"] = self.previous
        if self.current is not None:
            payload["current"] = self.current
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass(frozen=True)
class ConfigurationAuditSnapshot:
    """Audit snapshot describing configuration changes and associated metadata."""

    snapshot_id: str
    status: ConfigurationAuditStatus
    applied_at: datetime
    changes: tuple[ConfigurationChange, ...] = field(default_factory=tuple)
    current_config: Mapping[str, object] = field(default_factory=dict)
    previous_config: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "status": self.status.value,
            "applied_at": self.applied_at.astimezone(UTC).isoformat(),
            "changes": [change.as_dict() for change in self.changes],
            "current_config": dict(self.current_config),
            "previous_config": dict(self.previous_config) if self.previous_config else None,
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        heading = f"# Configuration audit ({self.status.value.upper()})"
        lines: list[str] = [heading]
        if not self.changes:
            if self.metadata.get("initial_snapshot"):
                lines.append(
                    "- Initial configuration snapshot recorded; no baseline for comparison yet."
                )
            else:
                lines.append("- No configuration changes detected.")
        else:
            for change in self.changes:
                previous = _format_value(change.previous)
                current = _format_value(change.current)
                note = f" — {change.note}" if change.note else ""
                lines.append(
                    f"- **{change.field}**: {previous} → {current}"
                    f" ({change.severity.value.upper()}){note}"
                )
        extras_summary = self.metadata.get("extras_summary")
        if isinstance(extras_summary, Mapping) and any(extras_summary.values()):
            added = extras_summary.get("added") or []
            removed = extras_summary.get("removed") or []
            updated = extras_summary.get("updated") or []
            lines.append("\n## Extras summary")
            lines.append(f"- Total extras: {extras_summary.get('total', 0)}")
            if added:
                lines.append(f"- Added keys: {', '.join(added)}")
            if removed:
                lines.append(f"- Removed keys: {', '.join(removed)}")
            if updated:
                lines.append(f"- Updated keys: {', '.join(updated)}")
        return "\n".join(lines)


def evaluate_configuration_audit(
    current: SystemConfig | Mapping[str, object],
    *,
    previous: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    applied_at: datetime | None = None,
) -> ConfigurationAuditSnapshot:
    """Evaluate configuration differences and return a telemetry snapshot."""

    baseline_template = SystemConfig().to_dict()
    current_config = _normalise_config(current, baseline=baseline_template)
    previous_config = (
        _normalise_config(previous, baseline=baseline_template) if previous else None
    )

    moment = applied_at or datetime.now(tz=UTC)
    status = ConfigurationAuditStatus.passed
    changes: list[ConfigurationChange] = []

    def register_change(
        field: str,
        prev: object | None,
        curr: object | None,
        severity: ConfigurationAuditStatus,
        note: str | None = None,
    ) -> None:
        nonlocal status
        change = ConfigurationChange(
            field=field, previous=prev, current=curr, severity=severity, note=note
        )
        changes.append(change)
        if _STATUS_ORDER[severity] > _STATUS_ORDER[status]:
            status = severity

    tracked_fields = (
        "run_mode",
        "environment",
        "tier",
        "confirm_live",
        "connection_protocol",
        "data_backbone_mode",
    )
    if previous_config:
        for field in tracked_fields:
            prev_value = previous_config.get(field)
            curr_value = current_config.get(field)
            if prev_value == curr_value:
                continue
            severity = _field_change_severity(field, prev_value, curr_value)
            note = None
            if field == "confirm_live" and bool(curr_value):
                note = "Live trading confirmations enabled"
            register_change(field, prev_value, curr_value, severity, note)

    extras_summary = _diff_extras(register_change, previous_config, current_config)

    metadata_payload: dict[str, object] = {
        "tier": current_config.get("tier"),
        "run_mode": current_config.get("run_mode"),
        "environment": current_config.get("environment"),
        "connection_protocol": current_config.get("connection_protocol"),
        "data_backbone_mode": current_config.get("data_backbone_mode"),
        "extras_summary": extras_summary,
        "changed_fields": [change.field for change in changes],
    }
    if metadata:
        metadata_payload.update(dict(metadata))

    if not previous_config:
        metadata_payload["initial_snapshot"] = True

    snapshot = ConfigurationAuditSnapshot(
        snapshot_id=str(uuid4()),
        status=status,
        applied_at=moment,
        changes=tuple(changes),
        current_config=current_config,
        previous_config=previous_config,
        metadata=metadata_payload,
    )
    return snapshot


def publish_configuration_audit_snapshot(
    event_bus: EventBus,
    snapshot: ConfigurationAuditSnapshot,
) -> None:
    """Publish the configuration audit snapshot onto the runtime event bus."""

    payload = snapshot.as_dict()
    payload["markdown"] = snapshot.to_markdown()
    event = Event(
        type="telemetry.runtime.configuration",
        payload=payload,
        source="operations.configuration_audit",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and getattr(event_bus, "is_running", lambda: False)():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to publish configuration audit via runtime bus", exc_info=True)

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("Configuration audit telemetry publish skipped", exc_info=True)


def format_configuration_audit_markdown(snapshot: ConfigurationAuditSnapshot) -> str:
    """Helper mirroring other operations modules for summary rendering."""

    return snapshot.to_markdown()


def _normalise_config(
    config: SystemConfig | Mapping[str, object] | None,
    *,
    baseline: MutableMapping[str, object],
) -> dict[str, object]:
    """Convert ``SystemConfig`` or mapping inputs into a comparable dict."""

    baseline_copy: dict[str, object] = {
        key: value for key, value in baseline.items() if key != "extras"
    }
    baseline_extras = _extract_extras(baseline)

    if isinstance(config, SystemConfig):
        payload = config.to_dict()
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        payload = {}

    result = {
        "run_mode": str(payload.get("run_mode") or baseline_copy.get("run_mode") or "mock"),
        "environment": str(
            payload.get("environment") or baseline_copy.get("environment") or "demo"
        ),
        "tier": str(payload.get("tier") or baseline_copy.get("tier") or "tier_0"),
        "confirm_live": _coerce_bool(
            payload.get("confirm_live", baseline_copy.get("confirm_live"))
        ),
        "connection_protocol": str(
            payload.get("connection_protocol")
            or baseline_copy.get("connection_protocol")
            or "bootstrap"
        ),
        "data_backbone_mode": str(
            payload.get("data_backbone_mode")
            or baseline_copy.get("data_backbone_mode")
            or "bootstrap"
        ),
    }

    extras_raw = payload.get("extras", baseline_extras)
    extras = _coerce_extras_mapping(extras_raw, default=baseline_extras)

    result["extras"] = extras
    return result


def _coerce_bool(value: object, default: bool | None = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _field_change_severity(
    field: str,
    previous: object | None,
    current: object | None,
) -> ConfigurationAuditStatus:
    if field == "run_mode" and str(current).lower() == "live":
        return ConfigurationAuditStatus.fail
    if field == "confirm_live" and bool(current):
        return ConfigurationAuditStatus.fail
    if field == "tier":
        current_value = str(current).lower()
        if current_value in {"tier_2", "tier2"}:
            return ConfigurationAuditStatus.fail
        if current_value in {"tier_1", "tier1"}:
            return ConfigurationAuditStatus.warn
    if field == "connection_protocol" and str(current).lower() == "fix":
        return ConfigurationAuditStatus.warn
    if field == "data_backbone_mode" and str(current).lower() == "institutional":
        return ConfigurationAuditStatus.warn
    if field == "environment" and str(current).lower() == "production":
        return ConfigurationAuditStatus.warn
    return ConfigurationAuditStatus.passed


def _diff_extras(
    register: Callable[
        [str, object | None, object | None, ConfigurationAuditStatus, str | None], None
    ],
    previous_config: Mapping[str, object] | None,
    current_config: Mapping[str, object],
) -> dict[str, object]:
    prev_extras = _extract_extras(previous_config)
    curr_extras = _extract_extras(current_config)

    added = sorted(set(curr_extras) - set(prev_extras))
    removed = sorted(set(prev_extras) - set(curr_extras))
    updated = sorted(
        key
        for key in set(curr_extras).intersection(prev_extras)
        if curr_extras[key] != prev_extras[key]
    )

    for key in added:
        severity = _extras_change_severity(key, curr_extras[key])
        register(f"extras.{key}", None, curr_extras[key], severity, "added key")
    for key in removed:
        severity = _extras_change_severity(key, None)
        register(f"extras.{key}", prev_extras[key], None, severity, "removed key")
    for key in updated:
        severity = _extras_change_severity(key, curr_extras[key])
        register(
            f"extras.{key}",
            prev_extras[key],
            curr_extras[key],
            severity,
            "updated value",
        )

    return {
        "total": len(curr_extras),
        "added": added,
        "removed": removed,
        "updated": updated,
    }


def _extras_change_severity(key: str, value: object | None) -> ConfigurationAuditStatus:
    normalized = key.upper()
    high_risk_tokens = ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "WEBHOOK", "CREDENTIAL")
    if any(token in normalized for token in high_risk_tokens):
        return ConfigurationAuditStatus.fail
    warn_prefixes = (
        "KAFKA_",
        "REDIS_",
        "FIX_",
        "BROKER_",
        "TIMESCALE",
        "INGEST_",
        "SECURITY_",
        "COMPLIANCE_",
        "SCHEDULER_",
        "BACKUP_",
        "EVENT_BUS_",
        "SPARK_",
    )
    if normalized.startswith(warn_prefixes):
        return ConfigurationAuditStatus.warn
    return ConfigurationAuditStatus.passed


def _format_value(value: object | None) -> str:
    if value is None:
        return "`<unset>`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    text = str(value)
    if not text:
        return "``"
    return f"`{text}`"


def _coerce_extras_mapping(
    extras_raw: object,
    *,
    default: Mapping[str, str] | None = None,
) -> dict[str, str]:
    extras: dict[str, str] = {}
    if isinstance(extras_raw, Mapping):
        for key, value in extras_raw.items():
            key_text = str(key)
            extras[key_text] = "" if value is None else str(value)
        return extras
    if default is not None:
        return dict(default)
    return extras


def _extract_extras(source: Mapping[str, object] | None) -> dict[str, str]:
    if source is None:
        return {}
    return _coerce_extras_mapping(source.get("extras"), default={})


__all__ = [
    "ConfigurationAuditStatus",
    "ConfigurationChange",
    "ConfigurationAuditSnapshot",
    "evaluate_configuration_audit",
    "publish_configuration_audit_snapshot",
    "format_configuration_audit_markdown",
]
