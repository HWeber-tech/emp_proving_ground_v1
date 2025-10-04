"""Automated evolution tuning recommendations derived from runtime telemetry."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus
from src.operations.evolution_experiments import (
    ExperimentMetrics,
    ExperimentStatus,
    EvolutionExperimentSnapshot,
)
from src.operations.strategy_performance import (
    StrategyPerformanceMetrics,
    StrategyPerformanceSnapshot,
    StrategyPerformanceStatus,
)
from src.operations.event_bus_failover import publish_event_with_failover


logger = logging.getLogger(__name__)


class EvolutionTuningStatus(str, Enum):
    """Severity levels for evolution tuning recommendations."""

    normal = "normal"
    warn = "warn"
    alert = "alert"


_STATUS_ORDER: Mapping[EvolutionTuningStatus, int] = {
    EvolutionTuningStatus.normal: 0,
    EvolutionTuningStatus.warn: 1,
    EvolutionTuningStatus.alert: 2,
}


def _escalate(
    current: EvolutionTuningStatus, candidate: EvolutionTuningStatus
) -> EvolutionTuningStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class EvolutionTuningRecommendation:
    """Recommended follow-up for a strategy or portfolio dimension."""

    strategy_id: str
    action: str
    rationale: str
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "strategy_id": self.strategy_id,
            "action": self.action,
            "rationale": self.rationale,
            "confidence": float(self.confidence),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class EvolutionTuningSummary:
    """Roll-up statistics for the tuning snapshot."""

    total_recommendations: int
    action_counts: Mapping[str, int]
    execution_rate: float | None = None
    roi: float | None = None
    roi_status: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_recommendations": self.total_recommendations,
            "action_counts": dict(self.action_counts),
        }
        if self.execution_rate is not None:
            payload["execution_rate"] = float(self.execution_rate)
        if self.roi is not None:
            payload["roi"] = float(self.roi)
        if self.roi_status is not None:
            payload["roi_status"] = self.roi_status
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class EvolutionTuningSnapshot:
    """Snapshot describing tuning guidance for the evolution engine."""

    generated_at: datetime
    status: EvolutionTuningStatus
    summary: EvolutionTuningSummary
    recommendations: Sequence[EvolutionTuningRecommendation]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "summary": self.summary.as_dict(),
            "recommendations": [rec.as_dict() for rec in self.recommendations],
            "metadata": dict(self.metadata),
        }


def _coerce_experiment(snapshot: Any) -> EvolutionExperimentSnapshot | None:
    if isinstance(snapshot, EvolutionExperimentSnapshot):
        return snapshot
    if isinstance(snapshot, Mapping):
        metrics = snapshot.get("metrics")
        if isinstance(metrics, Mapping):
            try:
                coerced = EvolutionExperimentSnapshot(
                    generated_at=datetime.fromisoformat(
                        str(snapshot.get("generated_at"))
                    ).astimezone(UTC)
                    if snapshot.get("generated_at")
                    else datetime.now(tz=UTC),
                    status=ExperimentStatus(str(snapshot.get("status", "normal"))),
                    metrics=ExperimentMetrics(
                        total_events=int(metrics.get("total_events", 0)),
                        executed=int(metrics.get("executed", 0)),
                        rejected=int(metrics.get("rejected", 0)),
                        failed=int(metrics.get("failed", 0)),
                        execution_rate=float(metrics.get("execution_rate", 0.0)),
                        rejection_rate=float(metrics.get("rejection_rate", 0.0)),
                        failure_rate=float(metrics.get("failure_rate", 0.0)),
                        avg_confidence=metrics.get("avg_confidence"),
                        avg_notional=metrics.get("avg_notional"),
                        roi_status=metrics.get("roi_status"),
                        roi=metrics.get("roi"),
                        net_pnl=metrics.get("net_pnl"),
                        metadata=dict(metrics.get("metadata", {})),
                    ),
                    rejection_reasons=dict(snapshot.get("rejection_reasons", {})),
                    metadata=dict(snapshot.get("metadata", {})),
                )
            except (TypeError, ValueError):  # pragma: no cover - defensive path
                return None
            return coerced
    return None


def _iter_strategy_metrics(
    snapshot: StrategyPerformanceSnapshot | Mapping[str, Any] | None,
) -> Iterable[StrategyPerformanceMetrics | Mapping[str, Any]]:
    if snapshot is None:
        return []
    if isinstance(snapshot, StrategyPerformanceSnapshot):
        return tuple(snapshot.strategies)
    if isinstance(snapshot, Mapping):
        strategies = snapshot.get("strategies", [])
        if isinstance(strategies, Sequence):
            return tuple(item for item in strategies if isinstance(item, Mapping))
    return []


def _extract_status(value: Any) -> StrategyPerformanceStatus | None:
    if isinstance(value, StrategyPerformanceStatus):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        for status in StrategyPerformanceStatus:
            if status.value == lowered:
                return status
    return None


def _extract_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _summary_metadata(
    experiment: EvolutionExperimentSnapshot | None,
    metadata: Mapping[str, Any] | None,
) -> MutableMapping[str, Any]:
    merged: MutableMapping[str, Any] = {}
    if metadata:
        merged.update(dict(metadata))
    if experiment is not None:
        merged.setdefault("experiment_status", experiment.status.value)
        merged.setdefault("experiment_metadata", dict(experiment.metadata))
    return merged


def evaluate_evolution_tuning(
    experiment_snapshot: EvolutionExperimentSnapshot | Mapping[str, Any] | None,
    performance_snapshot: StrategyPerformanceSnapshot | Mapping[str, Any] | None,
    *,
    warn_execution_rate: float = 0.4,
    alert_execution_rate: float = 0.2,
    failure_alert_threshold: float = 0.25,
    rejection_warn_threshold: float = 0.45,
    roi_warn_threshold: float = 0.0,
    roi_alert_threshold: float = -0.05,
    metadata: Mapping[str, Any] | None = None,
) -> EvolutionTuningSnapshot:
    """Derive evolution tuning recommendations from telemetry snapshots."""

    generated_at = datetime.now(tz=UTC)
    experiment = _coerce_experiment(experiment_snapshot)
    strategies = _iter_strategy_metrics(performance_snapshot)

    recommendations: list[EvolutionTuningRecommendation] = []
    status = EvolutionTuningStatus.normal
    action_counts: Counter[str] = Counter()

    roi_status: str | None = None
    roi_value: float | None = None
    execution_rate: float | None = None

    if experiment is not None:
        execution_rate = experiment.metrics.execution_rate
        roi_status = experiment.metrics.roi_status
        roi_value = experiment.metrics.roi
        if experiment.status is ExperimentStatus.alert:
            status = _escalate(status, EvolutionTuningStatus.alert)
        elif experiment.status is ExperimentStatus.warn:
            status = _escalate(status, EvolutionTuningStatus.warn)

    for entry in strategies:
        if isinstance(entry, StrategyPerformanceMetrics):
            strategy_id = entry.strategy_id
            exec_rate = entry.execution_rate
            reject_rate = entry.rejection_rate
            failure_rate = entry.failure_rate
            avg_confidence = entry.avg_confidence
            strat_status = entry.status
            strat_metadata = entry.metadata
        else:
            strategy_id = str(entry.get("strategy_id", "(unknown)") or "(unknown)")
            exec_rate = float(entry.get("execution_rate", 0.0))
            reject_rate = float(entry.get("rejection_rate", 0.0))
            failure_rate = float(entry.get("failure_rate", 0.0))
            avg_confidence = _extract_float(entry.get("avg_confidence"))
            strat_status = _extract_status(entry.get("status")) or StrategyPerformanceStatus.normal
            strat_metadata = entry.get("metadata", {}) if isinstance(entry, Mapping) else {}

        action: str | None = None
        rationale: str | None = None
        confidence = 0.0
        extra_meta: dict[str, Any] = {
            "execution_rate": exec_rate,
            "rejection_rate": reject_rate,
            "failure_rate": failure_rate,
        }
        if avg_confidence is not None:
            extra_meta["avg_confidence"] = avg_confidence
        if strat_metadata:
            extra_meta["metadata"] = dict(strat_metadata)

        if failure_rate >= failure_alert_threshold or (
            strat_status is StrategyPerformanceStatus.alert
        ):
            action = "disable_strategy"
            rationale = (
                f"Failure rate {failure_rate:.0%} exceeds the {failure_alert_threshold:.0%}"
                " alert threshold"
            )
            confidence = _clamp_confidence(failure_rate / max(failure_alert_threshold, 1e-6))
            status = _escalate(status, EvolutionTuningStatus.alert)
        elif reject_rate >= rejection_warn_threshold:
            action = "review_policy"
            rationale = (
                f"Rejection rate {reject_rate:.0%} exceeds the {rejection_warn_threshold:.0%}"
                " warning threshold"
            )
            confidence = _clamp_confidence(reject_rate)
            status = _escalate(status, EvolutionTuningStatus.warn)
        elif exec_rate <= alert_execution_rate:
            action = "boost_execution_support"
            rationale = (
                f"Execution rate {exec_rate:.0%} is below the {alert_execution_rate:.0%}"
                " alert threshold"
            )
            confidence = _clamp_confidence(alert_execution_rate - exec_rate + 0.1)
            status = _escalate(status, EvolutionTuningStatus.alert)
        elif exec_rate <= warn_execution_rate:
            action = "increase_liquidity_tests"
            rationale = (
                f"Execution rate {exec_rate:.0%} is below the {warn_execution_rate:.0%}"
                " warning threshold"
            )
            confidence = _clamp_confidence(warn_execution_rate - exec_rate + 0.05)
            status = _escalate(status, EvolutionTuningStatus.warn)
        elif roi_value is not None and roi_value <= roi_alert_threshold:
            action = "reduce_allocation"
            rationale = (
                f"Portfolio ROI {roi_value:.2%} breaches the {roi_alert_threshold:.2%}"
                " alert floor despite healthy execution"
            )
            confidence = _clamp_confidence(abs(roi_value - roi_alert_threshold) * 5)
            status = _escalate(status, EvolutionTuningStatus.alert)
        elif roi_value is not None and roi_value <= roi_warn_threshold:
            action = "rebalance_risk"
            rationale = (
                f"Portfolio ROI {roi_value:.2%} is below the target threshold"
                " and may require risk rebalancing"
            )
            confidence = _clamp_confidence(abs(roi_value - roi_warn_threshold) * 3)
            status = _escalate(status, EvolutionTuningStatus.warn)
        elif roi_value is not None and roi_value > roi_warn_threshold and exec_rate >= 0.7:
            action = "scale_successful_strategy"
            rationale = (
                f"Execution rate {exec_rate:.0%} is strong and ROI {roi_value:.2%}"
                " is above target; consider scaling"
            )
            confidence = _clamp_confidence(exec_rate)

        if action and rationale:
            recommendation = EvolutionTuningRecommendation(
                strategy_id=strategy_id,
                action=action,
                rationale=rationale,
                confidence=confidence,
                metadata=extra_meta,
            )
            recommendations.append(recommendation)
            action_counts[action] += 1

    if roi_value is not None:
        if roi_value <= roi_alert_threshold:
            status = _escalate(status, EvolutionTuningStatus.alert)
        elif roi_value <= roi_warn_threshold:
            status = _escalate(status, EvolutionTuningStatus.warn)

    summary = EvolutionTuningSummary(
        total_recommendations=len(recommendations),
        action_counts=action_counts,
        execution_rate=execution_rate,
        roi=roi_value,
        roi_status=roi_status,
        metadata=_summary_metadata(experiment, metadata),
    )

    snapshot_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}

    return EvolutionTuningSnapshot(
        generated_at=generated_at,
        status=status,
        summary=summary,
        recommendations=tuple(recommendations),
        metadata=snapshot_metadata,
    )


def format_evolution_tuning_markdown(snapshot: EvolutionTuningSnapshot) -> str:
    """Render the tuning snapshot into a Markdown table."""

    rows = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Recommendations | {snapshot.summary.total_recommendations} |",
    ]
    if snapshot.summary.execution_rate is not None:
        rows.append(f"| Execution rate | {snapshot.summary.execution_rate:.2%} |")
    if snapshot.summary.roi_status is not None:
        if snapshot.summary.roi is not None:
            roi_value = f"{snapshot.summary.roi:.2%}"
        else:
            roi_value = "n/a"
        rows.append(f"| ROI status | {snapshot.summary.roi_status} ({roi_value}) |")

    if snapshot.summary.action_counts:
        actions = ", ".join(
            f"{action}: {count}" for action, count in sorted(snapshot.summary.action_counts.items())
        )
        rows.append(f"| Action breakdown | {actions} |")

    markdown = "\n".join(rows)

    if snapshot.recommendations:
        bullet_lines = ["", "### Recommendations"]
        for rec in snapshot.recommendations:
            bullet_lines.append(
                f"- **{rec.strategy_id}** → {rec.action}"
                f" (confidence {rec.confidence:.0%}): {rec.rationale}"
            )
        markdown = markdown + "\n" + "\n".join(bullet_lines)

    return markdown


def publish_evolution_tuning_snapshot(
    event_bus: EventBus,
    snapshot: EvolutionTuningSnapshot,
    *,
    source: str = "operations.evolution_tuning",
) -> None:
    """Publish evolution tuning telemetry onto the runtime bus."""

    event = Event(
        type="telemetry.evolution.tuning",
        payload=snapshot.as_dict(),
        source=source,
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message="Runtime event bus unavailable for evolution tuning; falling back to global bus",
        runtime_unexpected_message="Unexpected error publishing evolution tuning snapshot via runtime event bus",
        runtime_none_message="Runtime event bus returned no result for evolution tuning snapshot; falling back to global bus",
        global_not_running_message="Global event bus unavailable while publishing evolution tuning snapshot",
        global_unexpected_message="Unexpected error publishing evolution tuning snapshot via global bus",
    )


__all__ = [
    "EvolutionTuningStatus",
    "EvolutionTuningRecommendation",
    "EvolutionTuningSummary",
    "EvolutionTuningSnapshot",
    "evaluate_evolution_tuning",
    "format_evolution_tuning_markdown",
    "publish_evolution_tuning_snapshot",
]
