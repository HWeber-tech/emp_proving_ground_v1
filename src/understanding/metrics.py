"""Prometheus export helpers for understanding loop telemetry."""

from __future__ import annotations

from typing import Iterable, Mapping

from src.operational import metrics as operational_metrics
from src.operations.observability_diary import ThrottleStateSnapshot
from src.understanding.diagnostics import UnderstandingLoopSnapshot

__all__ = [
    "export_throttle_metrics",
    "export_understanding_throttle_metrics",
]


def _coerce_throttle_state(
    snapshot: ThrottleStateSnapshot | Mapping[str, object]
) -> ThrottleStateSnapshot:
    if isinstance(snapshot, ThrottleStateSnapshot):
        return snapshot

    if hasattr(snapshot, "name") and hasattr(snapshot, "state"):
        multiplier_value = getattr(snapshot, "multiplier", None)
        reason_value = getattr(snapshot, "reason", None)
        metadata_value = getattr(snapshot, "metadata", {})
        try:
            multiplier = float(multiplier_value) if multiplier_value is not None else None
        except (TypeError, ValueError):
            multiplier = None
        reason = str(reason_value) if isinstance(reason_value, str) else None
        metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        return ThrottleStateSnapshot(
            name=str(getattr(snapshot, "name")),
            state=str(getattr(snapshot, "state")),
            active=bool(getattr(snapshot, "active", False)),
            multiplier=multiplier,
            reason=reason,
            metadata=metadata,
        )

    data = dict(snapshot)
    multiplier = data.get("multiplier")
    reason = data.get("reason")
    return ThrottleStateSnapshot(
        name=str(data.get("name", "throttle")),
        state=str(data.get("state", "observing")),
        active=bool(data.get("active", False)),
        multiplier=float(multiplier) if multiplier is not None else None,
        reason=str(reason) if isinstance(reason, str) else None,
        metadata=dict(data.get("metadata", {})),
    )


def export_throttle_metrics(
    throttle_states: Iterable[ThrottleStateSnapshot | Mapping[str, object]],
    *,
    regime: str | None = None,
    decision_id: str | None = None,
) -> None:
    """Emit Prometheus gauges for each throttle state."""

    for throttle in throttle_states:
        snapshot = _coerce_throttle_state(throttle)
        operational_metrics.set_understanding_throttle_state(
            snapshot.name,
            state=snapshot.state,
            active=snapshot.active,
            multiplier=snapshot.multiplier,
            regime=regime,
            decision=decision_id,
        )


def export_understanding_throttle_metrics(snapshot: UnderstandingLoopSnapshot) -> None:
    """Export throttle telemetry from an understanding loop snapshot."""

    regime = snapshot.regime_state.regime
    decision_id = snapshot.decision.tactic_id
    export_throttle_metrics(
        snapshot.capsule.throttle_states,
        regime=regime,
        decision_id=decision_id,
    )
