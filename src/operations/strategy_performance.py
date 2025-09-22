"""Strategy performance telemetry helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from statistics import fmean
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus, get_global_bus
from src.operations.roi import RoiStatus, RoiTelemetrySnapshot


class StrategyPerformanceStatus(str, Enum):
    """Severity levels for strategy performance telemetry."""

    normal = "normal"
    warn = "warn"
    alert = "alert"


_STATUS_ORDER: Mapping[StrategyPerformanceStatus, int] = {
    StrategyPerformanceStatus.normal: 0,
    StrategyPerformanceStatus.warn: 1,
    StrategyPerformanceStatus.alert: 2,
}


def _escalate(
    current: StrategyPerformanceStatus, candidate: StrategyPerformanceStatus
) -> StrategyPerformanceStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _normalise_strategy_id(value: Any) -> str:
    if value is None:
        return "(unknown)"
    return str(value).strip() or "(unknown)"


def _normalise_status(value: Any) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"executed", "filled", "fill", "filled_order"}:
        return "executed"
    if lowered in {"rejected", "reject", "blocked"}:
        return "rejected"
    if lowered in {"failed", "error", "exception"}:
        return "failed"
    return "other"


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _extract_reason(event: Mapping[str, Any]) -> str | None:
    reason = event.get("reason")
    if reason:
        return str(reason)
    metadata = event.get("metadata")
    if isinstance(metadata, Mapping):
        meta_reason = metadata.get("reason") or metadata.get("error")
        if meta_reason:
            return str(meta_reason)
    decision = event.get("decision")
    if isinstance(decision, Mapping):
        decision_reason = decision.get("reason")
        if decision_reason:
            return str(decision_reason)
    return None


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
class StrategyPerformanceMetrics:
    """Aggregated telemetry for a single strategy."""

    strategy_id: str
    status: StrategyPerformanceStatus
    total_events: int
    executed: int
    rejected: int
    failed: int
    other: int
    execution_rate: float
    rejection_rate: float
    failure_rate: float
    avg_confidence: float | None = None
    avg_notional: float | None = None
    last_event_at: datetime | None = None
    rejection_reasons: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "strategy_id": self.strategy_id,
            "status": self.status.value,
            "total_events": self.total_events,
            "executed": self.executed,
            "rejected": self.rejected,
            "failed": self.failed,
            "other": self.other,
            "execution_rate": self.execution_rate,
            "rejection_rate": self.rejection_rate,
            "failure_rate": self.failure_rate,
        }
        if self.avg_confidence is not None:
            payload["avg_confidence"] = self.avg_confidence
        if self.avg_notional is not None:
            payload["avg_notional"] = self.avg_notional
        if self.last_event_at is not None:
            payload["last_event_at"] = self.last_event_at.astimezone(UTC).isoformat()
        if self.rejection_reasons:
            payload["rejection_reasons"] = dict(self.rejection_reasons)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class StrategyPerformanceTotals:
    """Roll-up metrics across all observed strategies."""

    total_events: int
    executed: int
    rejected: int
    failed: int
    other: int
    execution_rate: float
    rejection_rate: float
    failure_rate: float
    roi_status: str | None = None
    roi: float | None = None
    net_pnl: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_events": self.total_events,
            "executed": self.executed,
            "rejected": self.rejected,
            "failed": self.failed,
            "other": self.other,
            "execution_rate": self.execution_rate,
            "rejection_rate": self.rejection_rate,
            "failure_rate": self.failure_rate,
        }
        if self.roi_status is not None:
            payload["roi_status"] = self.roi_status
        if self.roi is not None:
            payload["roi"] = self.roi
        if self.net_pnl is not None:
            payload["net_pnl"] = self.net_pnl
        return payload


@dataclass(frozen=True)
class StrategyPerformanceSnapshot:
    """Telemetry snapshot describing live strategy performance."""

    generated_at: datetime
    status: StrategyPerformanceStatus
    strategies: Sequence[StrategyPerformanceMetrics]
    totals: StrategyPerformanceTotals
    lookback: int
    top_rejection_reasons: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "strategies": [metric.as_dict() for metric in self.strategies],
            "totals": self.totals.as_dict(),
            "lookback": self.lookback,
            "top_rejection_reasons": dict(self.top_rejection_reasons),
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        lines = [
            f"**Strategy performance** — status: {self.status.value}",
            f"Generated at: {self.generated_at.astimezone(UTC).isoformat()}",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total intents | {self.totals.total_events} |",
            f"| Executed | {self.totals.executed} ({self.totals.execution_rate:.2%}) |",
            f"| Rejected | {self.totals.rejected} ({self.totals.rejection_rate:.2%}) |",
            f"| Failed | {self.totals.failed} ({self.totals.failure_rate:.2%}) |",
        ]
        if self.totals.other:
            lines.append(f"| Other | {self.totals.other} |")
        if self.totals.roi_status is not None:
            roi_display = f"{self.totals.roi:.2%}" if self.totals.roi is not None else "n/a"
            lines.append(f"| ROI status | {self.totals.roi_status} ({roi_display}) |")
        if self.totals.net_pnl is not None:
            lines.append(f"| Net PnL | {self.totals.net_pnl:,.2f} |")

        if self.top_rejection_reasons:
            lines.append("")
            lines.append("**Top rejection reasons**")
            for reason, count in self.top_rejection_reasons.items():
                lines.append(f"- {reason}: {count}")

        for metric in self.strategies:
            lines.append("")
            lines.append(f"### Strategy {metric.strategy_id}")
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            lines.append(f"| Status | {metric.status.value} |")
            lines.append(f"| Total intents | {metric.total_events} |")
            lines.append(f"| Executed | {metric.executed} ({metric.execution_rate:.2%}) |")
            lines.append(f"| Rejected | {metric.rejected} ({metric.rejection_rate:.2%}) |")
            lines.append(f"| Failed | {metric.failed} ({metric.failure_rate:.2%}) |")
            if metric.other:
                lines.append(f"| Other | {metric.other} |")
            if metric.avg_confidence is not None:
                lines.append(f"| Avg confidence | {metric.avg_confidence:.3f} |")
            if metric.avg_notional is not None:
                lines.append(f"| Avg notional | {metric.avg_notional:,.2f} |")
            if metric.last_event_at is not None:
                lines.append(
                    "| Last event | " + metric.last_event_at.astimezone(UTC).isoformat() + " |"
                )
            if metric.rejection_reasons:
                reasons = ", ".join(
                    f"{reason}: {count}" for reason, count in metric.rejection_reasons.items()
                )
                lines.append(f"| Rejection reasons | {reasons} |")
        return "\n".join(lines)


def format_strategy_performance_markdown(
    snapshot: StrategyPerformanceSnapshot,
) -> str:
    """Render the strategy performance snapshot as Markdown."""

    return snapshot.to_markdown()


def _normalise_events(
    events: Sequence[Mapping[str, Any]], lookback: int
) -> list[Mapping[str, Any]]:
    if lookback <= 0:
        return []
    cleaned: list[Mapping[str, Any]] = []
    for entry in events[:lookback]:
        if isinstance(entry, Mapping):
            cleaned.append(dict(entry))
    return cleaned


def _derive_status(
    execution_rate: float,
    failure_rate: float,
    *,
    warn_execution_rate: float,
    alert_execution_rate: float,
    warn_failure_rate: float,
    alert_failure_rate: float,
) -> StrategyPerformanceStatus:
    if execution_rate <= max(alert_execution_rate, 0.0) or failure_rate >= min(
        max(alert_failure_rate, 0.0), 1.0
    ):
        return StrategyPerformanceStatus.alert
    if execution_rate <= max(warn_execution_rate, 0.0) or failure_rate >= min(
        max(warn_failure_rate, 0.0), 1.0
    ):
        return StrategyPerformanceStatus.warn
    return StrategyPerformanceStatus.normal


def evaluate_strategy_performance(
    events: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    roi_snapshot: RoiTelemetrySnapshot | Mapping[str, Any] | None = None,
    lookback: int = 100,
    warn_execution_rate: float = 0.35,
    alert_execution_rate: float = 0.15,
    warn_failure_rate: float = 0.25,
    alert_failure_rate: float = 0.4,
    metadata: Mapping[str, Any] | None = None,
) -> StrategyPerformanceSnapshot:
    """Aggregate strategy performance telemetry into a snapshot."""

    if isinstance(events, Iterable) and not isinstance(events, Sequence):
        events = list(events)

    window = _normalise_events(list(events), lookback)
    generated_at = datetime.now(tz=UTC)

    strategy_data: MutableMapping[str, MutableMapping[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "executed": 0,
            "rejected": 0,
            "failed": 0,
            "other": 0,
            "confidences": [],
            "notionals": [],
            "timestamps": [],
            "reasons": Counter(),
        }
    )

    aggregated_reasons: Counter[str] = Counter()

    for entry in window:
        strategy_id = _normalise_strategy_id(entry.get("strategy_id"))
        record = strategy_data[strategy_id]

        record["total"] += 1
        status = _normalise_status(entry.get("status"))
        record[status] += 1

        confidence = _coerce_float(entry.get("confidence"))
        if confidence is not None:
            record["confidences"].append(confidence)

        notional = _coerce_float(entry.get("notional"))
        if notional is not None:
            record["notionals"].append(notional)

        timestamp = _parse_timestamp(entry.get("timestamp"))
        if timestamp is not None:
            record["timestamps"].append(timestamp)

        reason = _extract_reason(entry)
        if reason:
            record["reasons"][reason] += 1
            aggregated_reasons[reason] += 1

    strategies: list[StrategyPerformanceMetrics] = []
    overall_status = StrategyPerformanceStatus.normal

    total_events = 0
    total_executed = 0
    total_rejected = 0
    total_failed = 0
    total_other = 0

    for strategy_id, record in strategy_data.items():
        total = int(record["total"])
        executed = int(record["executed"])
        rejected = int(record["rejected"])
        failed = int(record["failed"])
        other = int(record["other"])

        total_events += total
        total_executed += executed
        total_rejected += rejected
        total_failed += failed
        total_other += other

        execution_rate = executed / total if total else 0.0
        rejection_rate = rejected / total if total else 0.0
        failure_rate = failed / total if total else 0.0

        status = _derive_status(
            execution_rate,
            failure_rate,
            warn_execution_rate=warn_execution_rate,
            alert_execution_rate=alert_execution_rate,
            warn_failure_rate=warn_failure_rate,
            alert_failure_rate=alert_failure_rate,
        )

        overall_status = _escalate(overall_status, status)

        confidences = record["confidences"]
        avg_confidence = fmean(confidences) if confidences else None

        notionals = record["notionals"]
        avg_notional = fmean(notionals) if notionals else None

        timestamps = sorted(record["timestamps"], reverse=True)
        last_event_at = timestamps[0] if timestamps else None

        rejection_reasons = dict(record["reasons"].most_common(5))

        strategies.append(
            StrategyPerformanceMetrics(
                strategy_id=strategy_id,
                status=status,
                total_events=total,
                executed=executed,
                rejected=rejected,
                failed=failed,
                other=other,
                execution_rate=execution_rate,
                rejection_rate=rejection_rate,
                failure_rate=failure_rate,
                avg_confidence=avg_confidence,
                avg_notional=avg_notional,
                last_event_at=last_event_at,
                rejection_reasons=rejection_reasons,
            )
        )

    strategies.sort(
        key=lambda metric: (
            -_STATUS_ORDER[metric.status],
            -metric.total_events,
            metric.strategy_id,
        )
    )

    totals = StrategyPerformanceTotals(
        total_events=total_events,
        executed=total_executed,
        rejected=total_rejected,
        failed=total_failed,
        other=total_other,
        execution_rate=(total_executed / total_events) if total_events else 0.0,
        rejection_rate=(total_rejected / total_events) if total_events else 0.0,
        failure_rate=(total_failed / total_events) if total_events else 0.0,
    )

    roi_status: str | None = None
    roi_value: float | None = None
    net_pnl: float | None = None
    if roi_snapshot is not None:
        if isinstance(roi_snapshot, RoiTelemetrySnapshot):
            roi_status = roi_snapshot.status.value
            roi_value = float(roi_snapshot.roi)
            net_pnl = float(roi_snapshot.net_pnl)
        elif isinstance(roi_snapshot, Mapping):
            raw_status = roi_snapshot.get("status")
            if isinstance(raw_status, RoiStatus):
                roi_status = raw_status.value
            elif raw_status is not None:
                roi_status = str(raw_status)
            roi_value = _coerce_float(roi_snapshot.get("roi"))
            net_pnl = _coerce_float(roi_snapshot.get("net_pnl"))

    if roi_status is not None or roi_value is not None or net_pnl is not None:
        totals = StrategyPerformanceTotals(
            total_events=totals.total_events,
            executed=totals.executed,
            rejected=totals.rejected,
            failed=totals.failed,
            other=totals.other,
            execution_rate=totals.execution_rate,
            rejection_rate=totals.rejection_rate,
            failure_rate=totals.failure_rate,
            roi_status=roi_status,
            roi=roi_value,
            net_pnl=net_pnl,
        )

    snapshot_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    top_reasons = dict(aggregated_reasons.most_common(5))

    return StrategyPerformanceSnapshot(
        generated_at=generated_at,
        status=overall_status,
        strategies=tuple(strategies),
        totals=totals,
        lookback=min(len(window), max(lookback, 0)),
        top_rejection_reasons=top_reasons,
        metadata=snapshot_metadata,
    )


def publish_strategy_performance_snapshot(
    event_bus: EventBus,
    snapshot: StrategyPerformanceSnapshot,
    *,
    source: str = "operations.strategy_performance",
) -> None:
    """Publish the strategy performance snapshot onto the runtime bus."""

    event = Event(
        type="telemetry.strategy.performance",
        payload=snapshot.as_dict(),
        source=source,
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - diagnostics only
            pass

    try:
        global_bus = get_global_bus()
        global_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - diagnostics only
        return


__all__ = [
    "StrategyPerformanceStatus",
    "StrategyPerformanceMetrics",
    "StrategyPerformanceTotals",
    "StrategyPerformanceSnapshot",
    "evaluate_strategy_performance",
    "format_strategy_performance_markdown",
    "publish_strategy_performance_snapshot",
]
