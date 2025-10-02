"""Regulatory telemetry aggregation utilities.

This module transforms raw compliance and surveillance telemetry feeds into a
normalised snapshot that can be published to the wider runtime.  The
implementation mirrors the high-impact roadmap requirement to provide
regulatory coverage telemetry alongside the existing execution and compliance
surfaces.  It purposely avoids heavyweight dependencies so it can operate
inside the current CI footprint while still surfacing actionable metadata for
dashboards and runbooks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus
from src.operations.event_bus_failover import publish_event_with_failover

logger = logging.getLogger(__name__)

__all__ = [
    "RegulatoryTelemetryStatus",
    "RegulatoryTelemetrySignal",
    "RegulatoryTelemetrySnapshot",
    "evaluate_regulatory_telemetry",
    "publish_regulatory_telemetry",
]


class RegulatoryTelemetryStatus(StrEnum):
    """Severity levels exposed by regulatory telemetry."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[RegulatoryTelemetryStatus, int] = {
    RegulatoryTelemetryStatus.ok: 0,
    RegulatoryTelemetryStatus.warn: 1,
    RegulatoryTelemetryStatus.fail: 2,
}


DEFAULT_REQUIRED_DOMAINS: tuple[str, ...] = (
    "trade_compliance",
    "kyc_aml",
    "trade_reporting",
    "surveillance",
)


@dataclass(slots=True, frozen=True)
class RegulatoryTelemetrySignal:
    """Normalised regulatory telemetry observation."""

    name: str
    status: RegulatoryTelemetryStatus
    summary: str
    observed_at: datetime | None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
        }
        if self.observed_at is not None:
            payload["observed_at"] = self.observed_at.isoformat()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True, frozen=True)
class RegulatoryTelemetrySnapshot:
    """Aggregated telemetry coverage for regulatory controls."""

    generated_at: datetime
    status: RegulatoryTelemetryStatus
    coverage_ratio: float
    signals: tuple[RegulatoryTelemetrySignal, ...]
    required_domains: tuple[str, ...]
    missing_domains: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "coverage_ratio": self.coverage_ratio,
            "required_domains": list(self.required_domains),
            "missing_domains": list(self.missing_domains),
            "signals": [signal.as_dict() for signal in self.signals],
            "metadata": dict(self.metadata),
        }


def _escalate(
    current: RegulatoryTelemetryStatus, candidate: RegulatoryTelemetryStatus
) -> RegulatoryTelemetryStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _coerce_mapping(value: object | None) -> MutableMapping[str, object]:
    if isinstance(value, MutableMapping):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_sequence(value: object | None) -> Sequence[object]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        return value
    return ()


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
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _parse_datetime(value: object | None) -> datetime | None:
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


def _normalise_status(value: object | None) -> RegulatoryTelemetryStatus:
    if isinstance(value, RegulatoryTelemetryStatus):
        return value
    label = str(value or "").strip().lower()
    if label in {"ok", "pass", "green"}:
        return RegulatoryTelemetryStatus.ok
    if label in {"warn", "warning", "amber"}:
        return RegulatoryTelemetryStatus.warn
    if label in {"fail", "failed", "red", "error"}:
        return RegulatoryTelemetryStatus.fail
    return RegulatoryTelemetryStatus.warn


def _normalise_name(value: object | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace(" ", "_").lower()


def _derive_summary(payload: Mapping[str, object]) -> str:
    for key in ("summary", "description", "message"):
        value = payload.get(key)
        if value:
            text = str(value).strip()
            if text:
                return text
    return "no summary provided"


def _merge_metadata(*values: Mapping[str, object]) -> MutableMapping[str, object]:
    merged: MutableMapping[str, object] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key, val in value.items():
            merged[str(key)] = val
    return merged


def _finalise_signal(
    name: str,
    *,
    status: RegulatoryTelemetryStatus,
    summary: str,
    observed_at: datetime | None,
    generated_at: datetime,
    stale_after: timedelta,
    metadata: MutableMapping[str, object],
) -> RegulatoryTelemetrySignal:
    effective_status = status
    stale = False
    if observed_at is None:
        stale = True
    else:
        if generated_at - observed_at > stale_after:
            stale = True

    breaches = _coerce_int(metadata.get("breaches"))
    violations = _coerce_int(metadata.get("violations"))

    if breaches > 0 or violations > 0:
        effective_status = RegulatoryTelemetryStatus.fail
    elif stale and effective_status == RegulatoryTelemetryStatus.ok:
        effective_status = RegulatoryTelemetryStatus.warn

    if stale:
        metadata.setdefault("stale", True)

    return RegulatoryTelemetrySignal(
        name=name,
        status=effective_status,
        summary=summary,
        observed_at=observed_at,
        metadata=dict(metadata),
    )


def evaluate_regulatory_telemetry(
    *,
    signals: Iterable[Mapping[str, object] | RegulatoryTelemetrySignal] | None = None,
    required_domains: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
    stale_after: timedelta | None = None,
) -> RegulatoryTelemetrySnapshot:
    """Fuse regulatory telemetry feeds into a normalised snapshot."""

    generated_at = datetime.now(tz=UTC)
    required = tuple(required_domains or DEFAULT_REQUIRED_DOMAINS)
    stale_window = stale_after or timedelta(minutes=60)

    processed: dict[str, RegulatoryTelemetrySignal] = {}
    observed_domains: set[str] = set()
    overall = RegulatoryTelemetryStatus.ok

    for payload in signals or ():
        if isinstance(payload, RegulatoryTelemetrySignal):
            signal = payload
        else:
            mapping = _coerce_mapping(payload)
            name = _normalise_name(
                mapping.get("name")
                or mapping.get("domain")
                or mapping.get("surface")
            )
            if not name:
                continue
            status = _normalise_status(mapping.get("status"))
            observed_at = _parse_datetime(
                mapping.get("observed_at") or mapping.get("timestamp")
            )
            summary = _derive_summary(mapping)
            metadata_payload = _merge_metadata(
                mapping.get("metadata", {}),
                {
                    "violations": mapping.get("violations"),
                    "breaches": mapping.get("breaches"),
                },
            )
            signal = _finalise_signal(
                name,
                status=status,
                summary=summary,
                observed_at=observed_at,
                generated_at=generated_at,
                stale_after=stale_window,
                metadata=metadata_payload,
            )

        key = _normalise_name(signal.name)
        if not key:
            continue

        observed_domains.add(key)
        existing = processed.get(key)
        if existing is None or _STATUS_ORDER[signal.status] >= _STATUS_ORDER[
            existing.status
        ]:
            processed[key] = signal
        overall = _escalate(overall, signal.status)

    for domain in required:
        key = _normalise_name(domain)
        if key in processed:
            continue
        placeholder = RegulatoryTelemetrySignal(
            name=domain,
            status=RegulatoryTelemetryStatus.fail,
            summary="telemetry missing",
            observed_at=None,
            metadata={"reason": "telemetry_missing"},
        )
        processed[key] = placeholder
        overall = _escalate(overall, placeholder.status)

    signals_tuple = tuple(processed.values())
    required_lower = tuple(_normalise_name(domain) for domain in required)
    coverage = 1.0
    missing = tuple(
        domain
        for domain, key in zip(required, required_lower)
        if key and key not in observed_domains
    )
    if required_lower:
        covered = sum(1 for key in required_lower if key in observed_domains)
        coverage = covered / len(required_lower)

    snapshot_metadata = dict(metadata or {})
    snapshot_metadata.setdefault("signals", [signal.name for signal in signals_tuple])
    snapshot_metadata["coverage_percent"] = round(coverage * 100.0, 2)

    return RegulatoryTelemetrySnapshot(
        generated_at=generated_at,
        status=overall,
        coverage_ratio=coverage,
        signals=signals_tuple,
        required_domains=required,
        missing_domains=missing,
        metadata=snapshot_metadata,
    )


def publish_regulatory_telemetry(
    event_bus: EventBus,
    snapshot: RegulatoryTelemetrySnapshot,
    *,
    channel: str = "telemetry.compliance.regulatory",
) -> None:
    """Publish the snapshot to the provided event bus."""

    event = Event(
        type=channel,
        payload=snapshot.as_dict(),
        source="regulatory_telemetry",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=(
            "Primary event bus publish_from_sync failed; falling back to global bus"
        ),
        runtime_unexpected_message=(
            "Unexpected error publishing regulatory telemetry via runtime event bus"
        ),
        runtime_none_message=(
            "Primary event bus publish_from_sync returned None; falling back to global bus"
        ),
        global_not_running_message=(
            "Global event bus not running while publishing regulatory telemetry"
        ),
        global_unexpected_message=(
            "Unexpected error publishing regulatory telemetry via global bus"
        ),
    )
