"""Sensory drift telemetry helpers.

This module analyses recent sensory audit entries emitted by the
professional runtime and converts them into reusable telemetry snapshots.
The resulting payload captures dimension-level drift, severity, and
markdown summaries suitable for operator dashboards and event-bus feeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from statistics import fmean
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus, get_global_bus


class DriftSeverity(str, Enum):
    """Severity levels exposed by the sensory drift snapshot."""

    normal = "normal"
    warn = "warn"
    alert = "alert"


_SEVERITY_ORDER: dict[DriftSeverity, int] = {
    DriftSeverity.normal: 0,
    DriftSeverity.warn: 1,
    DriftSeverity.alert: 2,
}


def _max_severity(current: DriftSeverity, candidate: DriftSeverity) -> DriftSeverity:
    if _SEVERITY_ORDER[candidate] > _SEVERITY_ORDER[current]:
        return candidate
    return current


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _sanitize_sequence(value: Iterable[Any]) -> list[Mapping[str, Any]]:
    cleaned: list[Mapping[str, Any]] = []
    for entry in value:
        if isinstance(entry, Mapping):
            cleaned.append(dict(entry))
    return cleaned


@dataclass(frozen=True)
class SensoryDimensionDrift:
    """Drift statistics for a single sensory dimension."""

    name: str
    current_signal: float
    baseline_signal: float | None
    delta: float | None
    current_confidence: float | None
    baseline_confidence: float | None
    confidence_delta: float | None
    severity: DriftSeverity
    samples: int

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "current_signal": self.current_signal,
            "severity": self.severity.value,
            "samples": self.samples,
        }
        if self.baseline_signal is not None:
            payload["baseline_signal"] = self.baseline_signal
        if self.delta is not None:
            payload["delta"] = self.delta
        if self.current_confidence is not None:
            payload["current_confidence"] = self.current_confidence
        if self.baseline_confidence is not None:
            payload["baseline_confidence"] = self.baseline_confidence
        if self.confidence_delta is not None:
            payload["confidence_delta"] = self.confidence_delta
        return payload


@dataclass(frozen=True)
class SensoryDriftSnapshot:
    """Aggregate drift telemetry across all sensory dimensions."""

    generated_at: datetime
    status: DriftSeverity
    dimensions: Mapping[str, SensoryDimensionDrift]
    sample_window: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "sample_window": self.sample_window,
            "metadata": dict(self.metadata),
            "dimensions": {
                name: dimension.as_dict() for name, dimension in self.dimensions.items()
            },
        }

    def to_markdown(self) -> str:
        if not self.dimensions:
            return "No sensory audit data available."

        header = "| Dimension | Signal Δ | Confidence Δ | Status | Samples |"
        separator = "| --- | --- | --- | --- | --- |"
        rows: list[str] = [header, separator]

        for name in sorted(self.dimensions):
            dimension = self.dimensions[name]
            delta = dimension.delta if dimension.delta is not None else 0.0
            confidence_delta = (
                dimension.confidence_delta if dimension.confidence_delta is not None else 0.0
            )
            rows.append(
                f"| {name} | {delta:+.3f} | {confidence_delta:+.3f} | {dimension.severity.value} | {dimension.samples} |"
            )

        return "\n".join(rows)


def evaluate_sensory_drift(
    audit_entries: Sequence[Mapping[str, Any]],
    *,
    lookback: int = 20,
    warn_threshold: float = 0.25,
    alert_threshold: float = 0.5,
    metadata: Mapping[str, Any] | None = None,
) -> SensoryDriftSnapshot:
    """Evaluate sensor drift from recent audit entries."""

    cleaned_entries = _sanitize_sequence(audit_entries)
    generated_at = datetime.utcnow()
    if not cleaned_entries:
        snapshot = SensoryDriftSnapshot(
            generated_at=generated_at,
            status=DriftSeverity.normal,
            dimensions={},
            sample_window=0,
            metadata={"reason": "no_audit_entries"} | (dict(metadata) if metadata else {}),
        )
        return snapshot

    latest = cleaned_entries[0]
    history = cleaned_entries[1 : lookback + 1]

    latest_dimensions = latest.get("dimensions")
    if not isinstance(latest_dimensions, Mapping):
        latest_dimensions = {}

    dimension_payloads: MutableMapping[str, SensoryDimensionDrift] = {}
    aggregate_status = DriftSeverity.normal

    for name, payload in latest_dimensions.items():
        if not isinstance(payload, Mapping):
            continue

        current_signal = _coerce_float(payload.get("signal"))
        if current_signal is None:
            continue

        current_confidence = _coerce_float(payload.get("confidence"))

        baseline_signals: list[float] = []
        baseline_confidences: list[float] = []

        for entry in history:
            entry_dims = entry.get("dimensions")
            if not isinstance(entry_dims, Mapping):
                continue
            historic_payload = entry_dims.get(name)
            if not isinstance(historic_payload, Mapping):
                continue
            baseline_signal = _coerce_float(historic_payload.get("signal"))
            if baseline_signal is not None:
                baseline_signals.append(baseline_signal)
            baseline_confidence = _coerce_float(historic_payload.get("confidence"))
            if baseline_confidence is not None:
                baseline_confidences.append(baseline_confidence)

        baseline_signal_value = fmean(baseline_signals) if baseline_signals else None
        baseline_confidence_value = fmean(baseline_confidences) if baseline_confidences else None

        delta: float | None = None
        severity = DriftSeverity.normal

        if baseline_signal_value is not None:
            delta = current_signal - baseline_signal_value
            absolute_delta = abs(delta)
            if absolute_delta >= alert_threshold:
                severity = DriftSeverity.alert
            elif absolute_delta >= warn_threshold:
                severity = DriftSeverity.warn

        aggregate_status = _max_severity(aggregate_status, severity)

        confidence_delta: float | None = None
        if baseline_confidence_value is not None and current_confidence is not None:
            confidence_delta = current_confidence - baseline_confidence_value

        dimension_payloads[name] = SensoryDimensionDrift(
            name=name,
            current_signal=current_signal,
            baseline_signal=baseline_signal_value,
            delta=delta,
            current_confidence=current_confidence,
            baseline_confidence=baseline_confidence_value,
            confidence_delta=confidence_delta,
            severity=severity,
            samples=1 + len(baseline_signals),
        )

    snapshot_metadata = {"entries": len(cleaned_entries)}
    if metadata:
        snapshot_metadata.update(dict(metadata))

    return SensoryDriftSnapshot(
        generated_at=generated_at,
        status=aggregate_status,
        dimensions=dimension_payloads,
        sample_window=min(len(cleaned_entries), lookback + 1),
        metadata=snapshot_metadata,
    )


def publish_sensory_drift(event_bus: EventBus, snapshot: SensoryDriftSnapshot) -> None:
    """Publish the sensory drift snapshot to the runtime event bus."""

    payload = snapshot.as_dict()
    event = Event(
        type="telemetry.sensory.drift",
        payload=payload,
        source="sensory_drift",
    )

    publish = getattr(event_bus, "publish_from_sync", None)
    if callable(publish) and event_bus.is_running():
        try:
            publish(event)
            return
        except Exception:  # pragma: no cover - defensive publish fallback
            pass

    topic_bus = get_global_bus()
    topic_bus.publish_sync(event.type, event.payload, source=event.source)


__all__ = [
    "DriftSeverity",
    "SensoryDimensionDrift",
    "SensoryDriftSnapshot",
    "evaluate_sensory_drift",
    "publish_sensory_drift",
]
