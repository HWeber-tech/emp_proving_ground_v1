"""DriftSentry gating helpers for selective paper-trade execution.

This module keeps the trading layer decoupled from the raw sensory telemetry
structures exposed by ``src.operations.sensory_drift`` while providing an
explicit decision object the runtime can surface in summaries and telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Mapping, Sequence

from src.governance.policy_ledger import PolicyLedgerStage
from src.operations.sensory_drift import DriftSeverity, SensoryDimensionDrift, SensoryDriftSnapshot

__all__ = [
    "DriftSentryDecision",
    "DriftSentryGate",
]


_SEVERITY_ORDER: Mapping[DriftSeverity, int] = {
    DriftSeverity.normal: 0,
    DriftSeverity.warn: 1,
    DriftSeverity.alert: 2,
}


def _severity_ge(left: DriftSeverity, right: DriftSeverity) -> bool:
    return _SEVERITY_ORDER[left] >= _SEVERITY_ORDER[right]


@dataclass(frozen=True, slots=True)
class DriftSentryDecision:
    """Decision returned by the DriftSentry gate when evaluating a trade."""

    allowed: bool
    severity: DriftSeverity
    evaluated_at: datetime
    reason: str | None = None
    requirements: Mapping[str, Any] = field(default_factory=dict)
    blocked_dimensions: tuple[str, ...] = ()
    snapshot_metadata: Mapping[str, Any] = field(default_factory=dict)
    force_paper: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "allowed": self.allowed,
            "severity": self.severity.value,
            "evaluated_at": self.evaluated_at.isoformat(),
            "blocked_dimensions": list(self.blocked_dimensions),
            "requirements": dict(self.requirements),
            "snapshot_metadata": dict(self.snapshot_metadata),
            "force_paper": self.force_paper,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


class DriftSentryGate:
    """Evaluate sensory drift posture and gate paper-trade execution accordingly."""

    def __init__(
        self,
        *,
        warn_confidence_floor: float = 0.6,
        warn_notional_limit: float | None = 25_000.0,
        block_severity: DriftSeverity = DriftSeverity.alert,
        exempt_strategies: Sequence[str] | None = None,
        threshold_resolver: Callable[[str | None], Mapping[str, Any] | None] | None = None,
    ) -> None:
        if not 0.0 <= warn_confidence_floor <= 1.0:
            raise ValueError("warn_confidence_floor must be between 0 and 1")

        self._warn_confidence_floor = float(warn_confidence_floor)
        self._warn_notional_limit = float(warn_notional_limit) if warn_notional_limit else None
        self._block_severity = block_severity
        self._exempt_strategies = {
            strategy.strip().lower()
            for strategy in (exempt_strategies or tuple())
            if strategy and strategy.strip()
        }
        self._latest_snapshot: SensoryDriftSnapshot | None = None
        self._last_decision: DriftSentryDecision | None = None
        self._threshold_resolver = threshold_resolver

    @property
    def latest_snapshot(self) -> SensoryDriftSnapshot | None:
        """Return the most recent drift snapshot applied to the gate."""

        return self._latest_snapshot

    @property
    def last_decision(self) -> DriftSentryDecision | None:
        """Expose the most recent trade decision (if any)."""

        return self._last_decision

    def update_snapshot(self, snapshot: SensoryDriftSnapshot | None) -> None:
        """Refresh the gate with the latest sensory drift snapshot."""

        self._latest_snapshot = snapshot

    def attach_threshold_resolver(
        self, resolver: Callable[[str | None], Mapping[str, Any] | None] | None
    ) -> None:
        """Attach a resolver that provides adaptive thresholds per strategy."""

        self._threshold_resolver = resolver

    def evaluate_trade(
        self,
        *,
        symbol: str | None,
        strategy_id: str | None,
        confidence: float | None,
        quantity: float | None,
        notional: float | None,
        metadata: Mapping[str, Any] | None = None,
        threshold_overrides: Mapping[str, Any] | None = None,
    ) -> DriftSentryDecision:
        """Evaluate trade intent metadata against the current drift posture."""

        snapshot = self._latest_snapshot
        evaluated_at = datetime.now(tz=UTC)
        severity = snapshot.status if snapshot is not None else DriftSeverity.normal
        warn_confidence_floor, warn_notional_limit, block_severity, applied_stage = (
            self._resolve_thresholds(strategy_id, threshold_overrides)
        )
        blocked_dimensions = self._blocked_dimensions(snapshot)
        snapshot_metadata: Mapping[str, Any]
        if snapshot is not None and isinstance(snapshot.metadata, Mapping):
            snapshot_metadata = dict(snapshot.metadata)
        else:
            snapshot_metadata = {}

        requirements: dict[str, Any] = {
            "confidence_floor": warn_confidence_floor,
        }
        if warn_notional_limit is not None:
            requirements["warn_notional_limit"] = warn_notional_limit
        requirements["block_severity"] = block_severity.value
        if applied_stage:
            requirements["release_stage"] = applied_stage
        if metadata:
            requirements.update({f"context.{key}": value for key, value in metadata.items()})

        finalize = self._finalise_decision

        if snapshot is None:
            return finalize(
                allowed=True,
                severity=severity,
                evaluated_at=evaluated_at,
                requirements=requirements,
                blocked_dimensions=blocked_dimensions,
                snapshot_metadata=snapshot_metadata,
                applied_stage=applied_stage,
                reason="no_snapshot",
            )

        strategy_key = strategy_id.strip().lower() if strategy_id else None
        if strategy_key and strategy_key in self._exempt_strategies:
            return finalize(
                allowed=True,
                severity=severity,
                evaluated_at=evaluated_at,
                requirements=requirements,
                blocked_dimensions=blocked_dimensions,
                snapshot_metadata=snapshot_metadata,
                applied_stage=applied_stage,
                reason="strategy_exempt",
            )

        if _severity_ge(severity, block_severity):
            reason = (
                f"drift severity {severity.value} reached block threshold {block_severity.value}"
            )
            return finalize(
                allowed=False,
                severity=severity,
                evaluated_at=evaluated_at,
                requirements=requirements,
                blocked_dimensions=blocked_dimensions,
                snapshot_metadata=snapshot_metadata,
                applied_stage=applied_stage,
                reason=reason,
            )

        if _severity_ge(severity, DriftSeverity.warn):
            allowed = True
            reason: str | None = None

            if confidence is None or confidence < warn_confidence_floor:
                allowed = False
                if confidence is None:
                    reason = "confidence missing during DriftSentry warn gating"
                else:
                    reason = (
                        f"confidence {confidence:.3f} below DriftSentry floor "
                        f"{warn_confidence_floor:.3f}"
                    )

            if allowed and warn_notional_limit is not None and notional is not None:
                if notional > warn_notional_limit:
                    allowed = False
                    reason = (
                        f"notional {notional:,.2f} exceeds DriftSentry warn limit "
                        f"{warn_notional_limit:,.2f}"
                    )

            return finalize(
                allowed=allowed,
                severity=severity,
                evaluated_at=evaluated_at,
                requirements=requirements,
                blocked_dimensions=blocked_dimensions,
                snapshot_metadata=snapshot_metadata,
                applied_stage=applied_stage,
                reason=reason,
            )

        return finalize(
            allowed=True,
            severity=severity,
            evaluated_at=evaluated_at,
            requirements=requirements,
            blocked_dimensions=blocked_dimensions,
            snapshot_metadata=snapshot_metadata,
            applied_stage=applied_stage,
        )

    @staticmethod
    def _blocked_dimensions(snapshot: SensoryDriftSnapshot | None) -> tuple[str, ...]:
        if snapshot is None:
            return tuple()
        dimensions = []
        for name, dimension in snapshot.dimensions.items():
            if not isinstance(dimension, SensoryDimensionDrift):
                continue
            if _severity_ge(dimension.severity, DriftSeverity.warn):
                dimensions.append(name)
        return tuple(sorted({dim for dim in dimensions}))

    def describe(self) -> Mapping[str, Any]:
        """Return a serialisable view of the current gate posture."""

        snapshot = self._latest_snapshot
        decision = self._last_decision
        return {
            "snapshot": snapshot.as_dict() if snapshot is not None else None,
            "last_decision": decision.as_dict() if decision is not None else None,
            "warn_confidence_floor": self._warn_confidence_floor,
            "warn_notional_limit": self._warn_notional_limit,
            "block_severity": self._block_severity.value,
            "exempt_strategies": sorted(self._exempt_strategies),
            "adaptive_thresholds": bool(self._threshold_resolver),
        }

    def _resolve_thresholds(
        self,
        strategy_id: str | None,
        threshold_overrides: Mapping[str, Any] | None,
    ) -> tuple[float, float | None, DriftSeverity, str | None]:
        warn_confidence = self._warn_confidence_floor
        warn_notional = self._warn_notional_limit
        block_severity = self._block_severity
        applied_stage: str | None = None

        combined: dict[str, Any] = {}
        if self._threshold_resolver is not None:
            try:
                resolved = self._threshold_resolver(strategy_id)
            except Exception:  # pragma: no cover - defensive guard
                resolved = None
            if resolved:
                combined.update(dict(resolved))
        if threshold_overrides:
            combined.update(dict(threshold_overrides))

        if "stage" in combined:
            applied_stage = str(combined["stage"])

        if "warn_confidence_floor" in combined:
            try:
                warn_confidence = float(combined["warn_confidence_floor"])
            except (TypeError, ValueError):
                warn_confidence = self._warn_confidence_floor
        if "warn_notional_limit" in combined:
            try:
                value = combined["warn_notional_limit"]
                warn_notional = float(value) if value is not None else None
            except (TypeError, ValueError):
                warn_notional = self._warn_notional_limit
        if "block_severity" in combined:
            try:
                block_severity = DriftSeverity(str(combined["block_severity"]).lower())
            except Exception:
                block_severity = self._block_severity

        return warn_confidence, warn_notional, block_severity, applied_stage

    def _finalise_decision(
        self,
        *,
        allowed: bool,
        severity: DriftSeverity,
        evaluated_at: datetime,
        requirements: Mapping[str, Any],
        blocked_dimensions: Sequence[str],
        snapshot_metadata: Mapping[str, Any],
        applied_stage: str | None,
        reason: str | None = None,
    ) -> DriftSentryDecision:
        final_allowed = bool(allowed)
        final_reason = reason
        requirements_payload = dict(requirements)
        force_paper = False

        stage_reason = self._stage_gate_reason(applied_stage)
        if stage_reason:
            force_paper = True
            requirements_payload.setdefault("release_stage_gate", stage_reason)
            if final_allowed:
                final_reason = stage_reason
            elif final_reason is None:
                final_reason = stage_reason

        if _severity_ge(severity, DriftSeverity.warn):
            force_paper = True

        decision = DriftSentryDecision(
            allowed=final_allowed,
            severity=severity,
            evaluated_at=evaluated_at,
            reason=final_reason,
            requirements=requirements_payload,
            blocked_dimensions=tuple(blocked_dimensions),
            snapshot_metadata=snapshot_metadata,
            force_paper=force_paper,
        )
        self._last_decision = decision
        return decision

    @staticmethod
    def _stage_gate_reason(stage: str | None) -> str | None:
        if not stage:
            return None
        try:
            resolved_stage = PolicyLedgerStage.from_value(stage)
        except Exception:
            return None
        if resolved_stage is PolicyLedgerStage.EXPERIMENT:
            return "release_stage_experiment_requires_paper_or_better"
        if resolved_stage is PolicyLedgerStage.PAPER:
            return "release_stage_paper_requires_paper_execution"
        return None
