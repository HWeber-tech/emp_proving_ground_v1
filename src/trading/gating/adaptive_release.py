"""Adaptive threshold resolver for release-managed drift gating.

This module extends the policy-led release management workflow by deriving
per-strategy drift gating thresholds based on the current sensory drift
posture.  The resolver consumes the :class:`LedgerReleaseManager` to honour
stage-specific thresholds and then tightens the guardrails when the runtime
reports sustained drift.  The resulting mapping plugs directly into the
:class:`DriftSentryGate`, allowing tactics to graduate from experiments to
paper trading and eventually limited live capital without bypassing
governance approvals captured in the policy ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage
from src.operations.sensory_drift import DriftSeverity, SensoryDimensionDrift, SensoryDriftSnapshot


__all__ = ["AdaptiveReleaseThresholds"]


@dataclass(slots=True)
class _AdaptiveTuning:
    """Configuration switches controlling adaptive adjustments."""

    warn_confidence_delta: float = 0.03
    alert_confidence_delta: float = 0.10
    warn_dimension_delta: float = 0.01
    alert_dimension_delta: float = 0.02
    warn_notional_multiplier: float = 0.75
    alert_notional_multiplier: float = 0.40
    warn_notional_dimension_penalty: float = 0.05
    alert_notional_dimension_penalty: float = 0.15
    minimum_notional_ratio: float = 0.05
    min_confidence_floor: float = 0.5
    max_confidence_floor: float = 0.99
    default_confidence_floor: float = 0.6

    def clamp_confidence(self, value: float) -> float:
        return max(self.min_confidence_floor, min(self.max_confidence_floor, value))


@dataclass(slots=True)
class AdaptiveReleaseThresholds:
    """Resolve adaptive drift thresholds backed by the policy ledger."""

    release_manager: LedgerReleaseManager
    tuning: _AdaptiveTuning = field(default_factory=_AdaptiveTuning)

    _last_thresholds: MutableMapping[str, Mapping[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def resolve(
        self,
        *,
        strategy_id: str | None,
        snapshot: SensoryDriftSnapshot | None,
    ) -> Mapping[str, Any]:
        """Resolve adaptive thresholds for a strategy given the current drift snapshot."""

        base = self._base_thresholds(strategy_id)
        stage = PolicyLedgerStage.from_value(base.get("stage"))
        severity = snapshot.status if snapshot is not None else DriftSeverity.normal

        if snapshot is not None:
            warn_dimensions, alert_dimensions = self._dimension_counts(snapshot)
            base["warn_confidence_floor"] = self._confidence_floor(
                base.get("warn_confidence_floor"),
                stage=stage,
                severity=severity,
                warn_dimensions=warn_dimensions,
                alert_dimensions=alert_dimensions,
            )
            notional_limit = base.get("warn_notional_limit")
            if isinstance(notional_limit, (int, float)):
                base["warn_notional_limit"] = self._notional_limit(
                    float(notional_limit),
                    stage=stage,
                    severity=severity,
                    warn_dimensions=warn_dimensions,
                    alert_dimensions=alert_dimensions,
                )
            base["block_severity"] = self._block_severity(
                base.get("block_severity"),
                stage=stage,
                severity=severity,
            )
            base["adaptive_source"] = severity.value
        else:
            # Normalise values even when no snapshot is available.
            floor_value = self._confidence_floor(base.get("warn_confidence_floor"), stage=stage)
            base["warn_confidence_floor"] = floor_value
            notional_limit = base.get("warn_notional_limit")
            if isinstance(notional_limit, (int, float)) and notional_limit > 0:
                base["warn_notional_limit"] = round(float(notional_limit), 2)
            base["block_severity"] = self._normalize_block_severity(base.get("block_severity"))
            base["adaptive_source"] = DriftSeverity.normal.value

        key = strategy_id or "__default__"
        self._last_thresholds[key] = dict(base)
        return base

    def last_thresholds(self, strategy_id: str | None = None) -> Mapping[str, Any] | None:
        """Return the most recently computed thresholds for the strategy."""

        key = strategy_id or "__default__"
        return self._last_thresholds.get(key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_thresholds(self, strategy_id: str | None) -> MutableMapping[str, Any]:
        try:
            resolved = dict(self.release_manager.resolve_thresholds(strategy_id))
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Failed to resolve base thresholds: {exc}") from exc
        stage_value = resolved.get("stage")
        if stage_value is None:
            stage = self.release_manager.resolve_stage(strategy_id)
            resolved["stage"] = stage.value
        return resolved

    @staticmethod
    def _dimension_counts(snapshot: SensoryDriftSnapshot) -> tuple[int, int]:
        warn_dimensions = 0
        alert_dimensions = 0
        for dimension in snapshot.dimensions.values():
            if not isinstance(dimension, SensoryDimensionDrift):
                continue
            if dimension.severity is DriftSeverity.warn:
                warn_dimensions += 1
            elif dimension.severity is DriftSeverity.alert:
                alert_dimensions += 1
        return warn_dimensions, alert_dimensions

    def _confidence_floor(
        self,
        value: object,
        *,
        stage: PolicyLedgerStage,
        severity: DriftSeverity = DriftSeverity.normal,
        warn_dimensions: int = 0,
        alert_dimensions: int = 0,
    ) -> float:
        base = self._coerce_float(value, self.tuning.default_confidence_floor)
        increase = 0.0
        if severity is DriftSeverity.warn:
            increase += self.tuning.warn_confidence_delta
        elif severity is DriftSeverity.alert:
            increase += self.tuning.alert_confidence_delta
        increase += warn_dimensions * self.tuning.warn_dimension_delta
        increase += alert_dimensions * self.tuning.alert_dimension_delta
        if stage in {PolicyLedgerStage.EXPERIMENT, PolicyLedgerStage.PAPER}:
            increase += 0.01
        return self.tuning.clamp_confidence(base + increase)

    def _notional_limit(
        self,
        base: float,
        *,
        stage: PolicyLedgerStage,
        severity: DriftSeverity,
        warn_dimensions: int,
        alert_dimensions: int,
    ) -> float:
        if base <= 0:
            return 0.0
        ratio = 1.0
        if severity is DriftSeverity.warn:
            ratio *= self.tuning.warn_notional_multiplier
        elif severity is DriftSeverity.alert:
            ratio *= self.tuning.alert_notional_multiplier
        ratio *= max(0.0, 1.0 - warn_dimensions * self.tuning.warn_notional_dimension_penalty)
        ratio *= max(0.0, 1.0 - alert_dimensions * self.tuning.alert_notional_dimension_penalty)
        if stage is PolicyLedgerStage.EXPERIMENT:
            ratio *= 0.85
        ratio = max(self.tuning.minimum_notional_ratio, ratio)
        return round(base * ratio, 2)

    def _block_severity(
        self,
        value: object,
        *,
        stage: PolicyLedgerStage,
        severity: DriftSeverity,
    ) -> str:
        current = self._normalize_block_severity(value)
        if severity is DriftSeverity.alert:
            return DriftSeverity.warn.value
        if severity is DriftSeverity.warn and stage is PolicyLedgerStage.EXPERIMENT:
            return DriftSeverity.warn.value
        return current

    @staticmethod
    def _normalize_block_severity(value: object) -> str:
        try:
            return DriftSeverity(str(value).strip().lower()).value
        except Exception:
            return DriftSeverity.alert.value

    @staticmethod
    def _coerce_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
