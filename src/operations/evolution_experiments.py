"""Evolution experiment telemetry helpers.

This module analyses recent paper-trading experiment events captured during
runtime execution and converts them into reusable telemetry snapshots. The
resulting payload highlights execution/rejection ratios, average intent
confidence, ROI posture, and the most common rejection reasons so operators can
monitor how well evolution-driven experiments perform in staging environments.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from statistics import fmean
from typing import Any, Mapping, Sequence

from src.core.event_bus import Event, EventBus
from src.operations.event_bus_failover import publish_event_with_failover
from src.operations.roi import RoiStatus, RoiTelemetrySnapshot


logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Overall status for evolution experiment telemetry."""

    normal = "normal"
    warn = "warn"
    alert = "alert"


_STATUS_ORDER: dict[ExperimentStatus, int] = {
    ExperimentStatus.normal: 0,
    ExperimentStatus.warn: 1,
    ExperimentStatus.alert: 2,
}


def _elevate(current: ExperimentStatus, candidate: ExperimentStatus) -> ExperimentStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ExperimentMetrics:
    """Aggregated metrics derived from recent evolution experiments."""

    total_events: int
    executed: int
    rejected: int
    failed: int
    execution_rate: float
    rejection_rate: float
    failure_rate: float
    avg_confidence: float | None
    avg_notional: float | None
    roi_status: str | None = None
    roi: float | None = None
    net_pnl: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_events": self.total_events,
            "executed": self.executed,
            "rejected": self.rejected,
            "failed": self.failed,
            "execution_rate": self.execution_rate,
            "rejection_rate": self.rejection_rate,
            "failure_rate": self.failure_rate,
        }
        if self.avg_confidence is not None:
            payload["avg_confidence"] = self.avg_confidence
        if self.avg_notional is not None:
            payload["avg_notional"] = self.avg_notional
        if self.roi_status is not None:
            payload["roi_status"] = self.roi_status
        if self.roi is not None:
            payload["roi"] = self.roi
        if self.net_pnl is not None:
            payload["net_pnl"] = self.net_pnl
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class EvolutionExperimentSnapshot:
    """Telemetry snapshot describing live paper-trading experiments."""

    generated_at: datetime
    status: ExperimentStatus
    metrics: ExperimentMetrics
    rejection_reasons: Mapping[str, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "metrics": self.metrics.as_dict(),
            "rejection_reasons": dict(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        rows = [
            "| Metric | Value |",
            "| --- | --- |",
            f"| Events | {self.metrics.total_events} |",
            f"| Executed | {self.metrics.executed} |",
            f"| Rejected | {self.metrics.rejected} |",
            f"| Failed | {self.metrics.failed} |",
            f"| Execution rate | {self.metrics.execution_rate:.2%} |",
            f"| Rejection rate | {self.metrics.rejection_rate:.2%} |",
            f"| Failure rate | {self.metrics.failure_rate:.2%} |",
        ]
        if self.metrics.avg_confidence is not None:
            rows.append(f"| Avg. confidence | {self.metrics.avg_confidence:.3f} |")
        if self.metrics.avg_notional is not None:
            rows.append(f"| Avg. notional | {self.metrics.avg_notional:,.2f} |")
        if self.metrics.roi_status is not None:
            if self.metrics.roi is not None:
                roi_value = f"{self.metrics.roi:.2%}"
            else:
                roi_value = "n/a"
            rows.append(f"| ROI status | {self.metrics.roi_status} ({roi_value}) |")
        if self.metrics.net_pnl is not None:
            rows.append(f"| Net PnL | {self.metrics.net_pnl:,.2f} |")

        if self.rejection_reasons:
            reasons = ", ".join(
                f"{reason}: {count}" for reason, count in sorted(self.rejection_reasons.items())
            )
            rows.append(f"| Rejection reasons | {reasons} |")

        return "\n".join(rows)


def _normalise_events(
    events: Sequence[Mapping[str, Any]], lookback: int
) -> list[Mapping[str, Any]]:
    cleaned: list[Mapping[str, Any]] = []
    for entry in events[:lookback]:
        if isinstance(entry, Mapping):
            cleaned.append(dict(entry))
    return cleaned


def _extract_reason(event: Mapping[str, Any]) -> str | None:
    reason = event.get("reason")
    if reason:
        return str(reason)
    metadata = event.get("metadata")
    if isinstance(metadata, Mapping):
        reason_value = metadata.get("reason") or metadata.get("error")
        if reason_value:
            return str(reason_value)
    decision = event.get("decision")
    if isinstance(decision, Mapping):
        decision_reason = decision.get("reason")
        if decision_reason:
            return str(decision_reason)
    return None


def evaluate_evolution_experiments(
    events: Sequence[Mapping[str, Any]],
    *,
    roi_snapshot: RoiTelemetrySnapshot | Mapping[str, Any] | None = None,
    lookback: int = 50,
    warn_execution_rate: float = 0.4,
    alert_execution_rate: float = 0.2,
    metadata: Mapping[str, Any] | None = None,
) -> EvolutionExperimentSnapshot:
    """Aggregate paper-trading experiment telemetry into a snapshot."""

    lookback = max(1, int(lookback))
    window = _normalise_events(events, lookback)
    generated_at = datetime.now(tz=UTC)

    total = len(window)
    executed = sum(1 for item in window if str(item.get("status", "")).lower() == "executed")
    rejected = sum(1 for item in window if str(item.get("status", "")).lower() == "rejected")
    failed = sum(1 for item in window if str(item.get("status", "")).lower() == "failed")

    execution_rate = executed / total if total else 0.0
    rejection_rate = rejected / total if total else 0.0
    failure_rate = failed / total if total else 0.0

    confidences = [
        value
        for value in (_coerce_float(item.get("confidence")) for item in window)
        if value is not None
    ]
    notionals = [
        value
        for value in (_coerce_float(item.get("notional")) for item in window)
        if value is not None
    ]
    avg_confidence = fmean(confidences) if confidences else None
    avg_notional = fmean(notionals) if notionals else None

    roi_status: str | None = None
    roi_value: float | None = None
    roi_net_pnl: float | None = None
    if isinstance(roi_snapshot, RoiTelemetrySnapshot):
        roi_status = roi_snapshot.status.value
        roi_value = roi_snapshot.roi
        roi_net_pnl = roi_snapshot.net_pnl
    elif isinstance(roi_snapshot, Mapping):
        status_value = roi_snapshot.get("status")
        if status_value is not None:
            roi_status = str(status_value)
        roi_value = _coerce_float(roi_snapshot.get("roi"))
        roi_net_pnl = _coerce_float(roi_snapshot.get("net_pnl"))

    rejection_reasons: Counter[str] = Counter()
    for item in window:
        reason = _extract_reason(item)
        if reason:
            rejection_reasons[reason] += 1

    status = ExperimentStatus.normal
    if total == 0:
        status = ExperimentStatus.alert
    else:
        if execution_rate < alert_execution_rate:
            status = _elevate(status, ExperimentStatus.alert)
        elif execution_rate < warn_execution_rate:
            status = _elevate(status, ExperimentStatus.warn)

        if failure_rate > 0.5:
            status = _elevate(status, ExperimentStatus.alert)
        elif failure_rate > 0.25:
            status = _elevate(status, ExperimentStatus.warn)

        if rejection_rate > 0.6:
            status = _elevate(status, ExperimentStatus.warn)

    if roi_status is not None:
        if roi_status == RoiStatus.at_risk.value:
            status = _elevate(status, ExperimentStatus.warn)
            if roi_value is not None and roi_value < -0.05:
                status = _elevate(status, ExperimentStatus.alert)

    metrics = ExperimentMetrics(
        total_events=total,
        executed=executed,
        rejected=rejected,
        failed=failed,
        execution_rate=execution_rate,
        rejection_rate=rejection_rate,
        failure_rate=failure_rate,
        avg_confidence=avg_confidence,
        avg_notional=avg_notional,
        roi_status=roi_status,
        roi=roi_value,
        net_pnl=roi_net_pnl,
        metadata=dict(metadata or {}),
    )

    snapshot = EvolutionExperimentSnapshot(
        generated_at=generated_at,
        status=status,
        metrics=metrics,
        rejection_reasons=dict(rejection_reasons),
        metadata=dict(metadata or {}),
    )
    return snapshot


def publish_evolution_experiment_snapshot(
    event_bus: EventBus, snapshot: EvolutionExperimentSnapshot
) -> None:
    """Publish the snapshot on the runtime and global event buses."""

    event = Event(
        type="telemetry.evolution.experiments",
        payload=snapshot.as_dict(),
        source="professional_runtime",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=(
            "Primary event bus publish_from_sync failed; falling back to global bus"
        ),
        runtime_unexpected_message=(
            "Unexpected error publishing evolution experiment snapshot via runtime event bus"
        ),
        runtime_none_message=(
            "Primary event bus publish_from_sync returned None; falling back to global bus"
        ),
        global_not_running_message=(
            "Global event bus not running while publishing evolution experiment snapshot"
        ),
        global_unexpected_message=(
            "Unexpected error publishing evolution experiment snapshot via global bus"
        ),
    )


def format_evolution_experiment_markdown(
    snapshot: EvolutionExperimentSnapshot,
) -> str:
    """Return a Markdown rendering for dashboards and runbooks."""

    return snapshot.to_markdown()


__all__ = [
    "ExperimentStatus",
    "ExperimentMetrics",
    "EvolutionExperimentSnapshot",
    "evaluate_evolution_experiments",
    "publish_evolution_experiment_snapshot",
    "format_evolution_experiment_markdown",
]
