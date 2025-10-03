"""Execution readiness evaluation and telemetry helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Callable, Mapping

from src.core.event_bus import Event, EventBus, TopicBus
from src.operations.event_bus_failover import publish_event_with_failover

logger = logging.getLogger(__name__)


class ExecutionStatus(StrEnum):
    """Severity levels exposed by execution readiness telemetry."""

    passed = "pass"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[ExecutionStatus, int] = {
    ExecutionStatus.passed: 0,
    ExecutionStatus.warn: 1,
    ExecutionStatus.fail: 2,
}


def _escalate(current: ExecutionStatus, candidate: ExecutionStatus) -> ExecutionStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if value is None:
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _coerce_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        return tuple(part for part in parts if part)
    if isinstance(value, Sequence):
        items: list[str] = []
        for entry in value:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                items.append(text)
        return tuple(items)
    return tuple()


def _coerce_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class ExecutionPolicy:
    """Thresholds describing the expected execution service posture."""

    min_fill_rate: float = 0.9
    max_rejection_rate: float = 0.05
    max_pending_orders: int = 5
    max_avg_latency_ms: float = 500.0
    max_latency_ms: float = 1500.0
    require_connection: bool = True
    require_drop_copy: bool = False
    max_drop_copy_lag_seconds: float = 10.0
    min_active_sessions: int = 0
    warn_on_missing_sessions: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "ExecutionPolicy":
        mapping = mapping or {}
        return cls(
            min_fill_rate=max(
                0.0, min(1.0, _coerce_float(mapping.get("EXECUTION_MIN_FILL_RATE"), 0.9) or 0.9)
            ),
            max_rejection_rate=max(
                0.0,
                min(1.0, _coerce_float(mapping.get("EXECUTION_MAX_REJECTION_RATE"), 0.05) or 0.05),
            ),
            max_pending_orders=max(0, _coerce_int(mapping.get("EXECUTION_MAX_PENDING_ORDERS"), 5)),
            max_avg_latency_ms=float(
                max(0.0, _coerce_float(mapping.get("EXECUTION_MAX_AVG_LATENCY_MS"), 500.0) or 500.0)
            ),
            max_latency_ms=float(
                max(0.0, _coerce_float(mapping.get("EXECUTION_MAX_LATENCY_MS"), 1500.0) or 1500.0)
            ),
            require_connection=_coerce_bool(mapping.get("EXECUTION_REQUIRE_CONNECTION"), True),
            require_drop_copy=_coerce_bool(mapping.get("EXECUTION_REQUIRE_DROP_COPY"), False),
            max_drop_copy_lag_seconds=float(
                max(
                    0.0, _coerce_float(mapping.get("EXECUTION_DROP_COPY_MAX_LAG_SEC"), 10.0) or 10.0
                )
            ),
            min_active_sessions=max(
                0, _coerce_int(mapping.get("EXECUTION_MIN_ACTIVE_SESSIONS"), 0)
            ),
            warn_on_missing_sessions=_coerce_bool(
                mapping.get("EXECUTION_WARN_ON_MISSING_SESSIONS"), True
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "min_fill_rate": self.min_fill_rate,
            "max_rejection_rate": self.max_rejection_rate,
            "max_pending_orders": self.max_pending_orders,
            "max_avg_latency_ms": self.max_avg_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "require_connection": self.require_connection,
            "require_drop_copy": self.require_drop_copy,
            "max_drop_copy_lag_seconds": self.max_drop_copy_lag_seconds,
            "min_active_sessions": self.min_active_sessions,
            "warn_on_missing_sessions": self.warn_on_missing_sessions,
        }


@dataclass(frozen=True)
class ExecutionState:
    """Observed execution telemetry inputs."""

    orders_submitted: int = 0
    orders_executed: int = 0
    orders_failed: int = 0
    pending_orders: int = 0
    avg_latency_ms: float | None = None
    max_latency_ms: float | None = None
    drop_copy_lag_seconds: float | None = None
    drop_copy_active: bool | None = None
    connection_healthy: bool = True
    sessions_active: tuple[str, ...] = field(default_factory=tuple)
    last_error: str | None = None
    last_execution_at: datetime | None = None
    drop_copy_metrics: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_sources(
        cls,
        *,
        mapping: Mapping[str, object] | None = None,
        stats: Mapping[str, object] | None = None,
        drop_copy_metrics: Mapping[str, object] | None = None,
        drop_copy_active: bool | None = None,
        connection_healthy: bool | None = None,
        sessions: Sequence[str] | None = None,
    ) -> "ExecutionState":
        mapping = mapping or {}
        stats = stats or {}
        orders_submitted = _coerce_int(
            stats.get("orders_submitted"), _coerce_int(mapping.get("EXECUTION_ORDERS_SUBMITTED"), 0)
        )
        orders_executed = _coerce_int(
            stats.get("orders_executed"), _coerce_int(mapping.get("EXECUTION_ORDERS_EXECUTED"), 0)
        )
        orders_failed = _coerce_int(
            stats.get("orders_failed"), _coerce_int(mapping.get("EXECUTION_ORDERS_FAILED"), 0)
        )
        pending = _coerce_int(
            stats.get("pending_orders"), _coerce_int(mapping.get("EXECUTION_PENDING_ORDERS"), 0)
        )
        avg_latency = _coerce_float(stats.get("avg_latency_ms"))
        if avg_latency is None:
            avg_latency = _coerce_float(mapping.get("EXECUTION_AVG_LATENCY_MS"))
        max_latency = _coerce_float(stats.get("max_latency_ms"))
        if max_latency is None:
            max_latency = _coerce_float(mapping.get("EXECUTION_MAX_LATENCY_MS"))
        lag_seconds = _coerce_float(stats.get("drop_copy_lag_seconds"))
        if lag_seconds is None:
            lag_seconds = _coerce_float(mapping.get("EXECUTION_DROP_COPY_LAG_SEC"))

        drop_default = _coerce_bool(stats.get("drop_copy_active"), False)
        if drop_copy_active is not None:
            drop_default = drop_copy_active
        drop_copy_active = _coerce_bool(mapping.get("EXECUTION_DROP_COPY_ACTIVE"), drop_default)

        connection_default = _coerce_bool(stats.get("connection_healthy"), True)
        if connection_healthy is not None:
            connection_default = connection_healthy
        connection = _coerce_bool(mapping.get("EXECUTION_CONNECTION_HEALTHY"), connection_default)

        configured_sessions = _coerce_tuple(mapping.get("EXECUTION_ACTIVE_SESSIONS"))
        if configured_sessions:
            sessions_active = configured_sessions
        elif sessions:
            sessions_active = tuple(str(session) for session in sessions)
        else:
            sessions_active = tuple()
        last_error = stats.get("last_error") or mapping.get("EXECUTION_LAST_ERROR")
        if last_error is not None:
            last_error = str(last_error)
        last_execution_at = stats.get("last_execution_at") or mapping.get(
            "EXECUTION_LAST_EXECUTION_AT"
        )
        timestamp = _coerce_datetime(last_execution_at)
        metrics_items: tuple[tuple[str, int], ...]
        if drop_copy_metrics:
            metrics_items = tuple(
                (str(key), _coerce_int(value)) for key, value in drop_copy_metrics.items()
            )
        else:
            metrics_items = tuple()
        metadata: dict[str, object] = {}
        if stats:
            excluded = {
                "orders_submitted",
                "orders_executed",
                "orders_failed",
                "pending_orders",
                "avg_latency_ms",
                "max_latency_ms",
                "drop_copy_lag_seconds",
                "drop_copy_active",
                "connection_healthy",
                "last_error",
                "last_execution_at",
            }
            metadata.update({k: v for k, v in stats.items() if k not in excluded and v is not None})
        extra_meta = mapping.get("EXECUTION_METADATA")
        if isinstance(extra_meta, MappingABC):
            metadata.update(extra_meta)

        return cls(
            orders_submitted=orders_submitted,
            orders_executed=orders_executed,
            orders_failed=orders_failed,
            pending_orders=pending,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            drop_copy_lag_seconds=lag_seconds,
            drop_copy_active=drop_copy_active,
            connection_healthy=connection,
            sessions_active=sessions_active,
            last_error=last_error,
            last_execution_at=timestamp,
            drop_copy_metrics=metrics_items,
            metadata=metadata,
        )

    @property
    def fill_rate(self) -> float | None:
        if self.orders_submitted <= 0:
            return None
        return self.orders_executed / max(self.orders_submitted, 1)

    @property
    def failure_rate(self) -> float | None:
        if self.orders_submitted <= 0:
            return None
        return self.orders_failed / max(self.orders_submitted, 1)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "orders_submitted": self.orders_submitted,
            "orders_executed": self.orders_executed,
            "orders_failed": self.orders_failed,
            "pending_orders": self.pending_orders,
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "drop_copy_lag_seconds": self.drop_copy_lag_seconds,
            "drop_copy_active": self.drop_copy_active,
            "connection_healthy": self.connection_healthy,
            "sessions_active": list(self.sessions_active),
            "fill_rate": self.fill_rate,
            "failure_rate": self.failure_rate,
            "metadata": dict(self.metadata),
        }
        if self.last_error:
            payload["last_error"] = self.last_error
        if self.last_execution_at is not None:
            payload["last_execution_at"] = self.last_execution_at.isoformat()
        if self.drop_copy_metrics:
            payload["drop_copy_metrics"] = {k: v for k, v in self.drop_copy_metrics}
        return payload


@dataclass(frozen=True)
class ExecutionIssue:
    """Structured issue captured during readiness evaluation."""

    code: str
    message: str
    severity: ExecutionStatus
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionReadinessSnapshot:
    """Aggregate execution readiness snapshot."""

    service: str
    generated_at: datetime
    status: ExecutionStatus
    policy: ExecutionPolicy
    state: ExecutionState
    issues: tuple[ExecutionIssue, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def fill_rate(self) -> float | None:
        return self.state.fill_rate

    @property
    def failure_rate(self) -> float | None:
        return self.state.failure_rate

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "service": self.service,
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "fill_rate": self.fill_rate,
            "failure_rate": self.failure_rate,
            "pending_orders": self.state.pending_orders,
            "avg_latency_ms": self.state.avg_latency_ms,
            "max_latency_ms": self.state.max_latency_ms,
            "drop_copy_lag_seconds": self.state.drop_copy_lag_seconds,
            "drop_copy_active": self.state.drop_copy_active,
            "connection_healthy": self.state.connection_healthy,
            "sessions_active": list(self.state.sessions_active),
            "issues": [issue.as_dict() for issue in self.issues],
            "policy": self.policy.as_dict(),
            "state": self.state.as_dict(),
            "metadata": dict(self.metadata),
        }
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"**Execution readiness – {self.service}**",
            f"- Status: {self.status.value}",
            f"- Fill rate: {self.fill_rate:.2%}"
            if self.fill_rate is not None
            else "- Fill rate: n/a",
            f"- Failure rate: {self.failure_rate:.2%}"
            if self.failure_rate is not None
            else "- Failure rate: n/a",
            f"- Pending orders: {self.state.pending_orders}",
            f"- Average latency: {self.state.avg_latency_ms:.1f} ms"
            if self.state.avg_latency_ms is not None
            else "- Average latency: n/a",
            f"- Max latency: {self.state.max_latency_ms:.1f} ms"
            if self.state.max_latency_ms is not None
            else "- Max latency: n/a",
            f"- Drop-copy lag: {self.state.drop_copy_lag_seconds:.1f} s"
            if self.state.drop_copy_lag_seconds is not None
            else "- Drop-copy lag: n/a",
            f"- Drop-copy active: {self.state.drop_copy_active if self.state.drop_copy_active is not None else 'n/a'}",
            f"- Connection healthy: {'yes' if self.state.connection_healthy else 'no'}",
            f"- Active sessions: {', '.join(self.state.sessions_active) if self.state.sessions_active else 'n/a'}",
        ]
        if self.issues:
            lines.append("")
            lines.append("**Issues**")
            for issue in self.issues:
                lines.append(f"- [{issue.severity.value}] {issue.message}")
        return "\n".join(lines)


def format_execution_markdown(snapshot: ExecutionReadinessSnapshot) -> str:
    """Convenience wrapper mirroring other operational telemetry helpers."""

    return snapshot.to_markdown()


def evaluate_execution_readiness(
    policy: ExecutionPolicy,
    state: ExecutionState,
    *,
    metadata: Mapping[str, object] | None = None,
    service: str = "execution",
    now: datetime | None = None,
) -> ExecutionReadinessSnapshot:
    """Evaluate execution posture and return a structured snapshot."""

    moment = now or datetime.now(tz=UTC)
    issues: list[ExecutionIssue] = []
    status = ExecutionStatus.passed

    if state.orders_submitted == 0:
        issues.append(
            ExecutionIssue(
                code="no_orders",
                message="no executions observed during the evaluation window",
                severity=ExecutionStatus.warn,
            )
        )
        status = ExecutionStatus.warn
    else:
        fill_rate = state.fill_rate or 0.0
        if fill_rate < policy.min_fill_rate:
            severity = ExecutionStatus.warn
            if state.orders_submitted >= 5 and fill_rate < policy.min_fill_rate * 0.8:
                severity = ExecutionStatus.fail
            issues.append(
                ExecutionIssue(
                    code="fill_rate_below_target",
                    message=f"fill rate {fill_rate:.2%} below target {policy.min_fill_rate:.0%}",
                    severity=severity,
                    metadata={"fill_rate": fill_rate, "orders": state.orders_submitted},
                )
            )
            status = _escalate(status, severity)

        failure_rate = state.failure_rate or 0.0
        if failure_rate > policy.max_rejection_rate:
            severity = ExecutionStatus.warn
            if state.orders_submitted >= 5 and failure_rate > policy.max_rejection_rate * 1.5:
                severity = ExecutionStatus.fail
            issues.append(
                ExecutionIssue(
                    code="rejection_rate_high",
                    message=f"rejection rate {failure_rate:.2%} above limit {policy.max_rejection_rate:.0%}",
                    severity=severity,
                    metadata={"failure_rate": failure_rate},
                )
            )
            status = _escalate(status, severity)

    if state.pending_orders > policy.max_pending_orders:
        severity = ExecutionStatus.warn
        if state.pending_orders >= policy.max_pending_orders * 2:
            severity = ExecutionStatus.fail
        issues.append(
            ExecutionIssue(
                code="pending_orders_exceeded",
                message=f"{state.pending_orders} pending orders exceed limit {policy.max_pending_orders}",
                severity=severity,
            )
        )
        status = _escalate(status, severity)

    if state.avg_latency_ms is not None and state.avg_latency_ms > policy.max_avg_latency_ms:
        severity = ExecutionStatus.warn
        if state.avg_latency_ms > policy.max_avg_latency_ms * 1.5:
            severity = ExecutionStatus.fail
        issues.append(
            ExecutionIssue(
                code="latency_average",
                message=f"average latency {state.avg_latency_ms:.1f}ms exceeds {policy.max_avg_latency_ms:.1f}ms",
                severity=severity,
            )
        )
        status = _escalate(status, severity)

    if state.max_latency_ms is not None and state.max_latency_ms > policy.max_latency_ms:
        issues.append(
            ExecutionIssue(
                code="latency_peak",
                message=f"peak latency {state.max_latency_ms:.1f}ms exceeds {policy.max_latency_ms:.1f}ms",
                severity=ExecutionStatus.warn,
            )
        )
        status = _escalate(status, ExecutionStatus.warn)

    if policy.require_connection and not state.connection_healthy:
        issues.append(
            ExecutionIssue(
                code="connection_down",
                message="execution connection reported unhealthy",
                severity=ExecutionStatus.fail,
            )
        )
        status = ExecutionStatus.fail

    if policy.require_drop_copy:
        if not state.drop_copy_active:
            issues.append(
                ExecutionIssue(
                    code="drop_copy_inactive",
                    message="required drop-copy bridge inactive",
                    severity=ExecutionStatus.fail,
                )
            )
            status = ExecutionStatus.fail
        elif (
            state.drop_copy_lag_seconds is not None
            and state.drop_copy_lag_seconds > policy.max_drop_copy_lag_seconds
        ):
            issues.append(
                ExecutionIssue(
                    code="drop_copy_lag",
                    message=f"drop-copy lag {state.drop_copy_lag_seconds:.1f}s exceeds {policy.max_drop_copy_lag_seconds:.1f}s",
                    severity=ExecutionStatus.warn,
                )
            )
            status = _escalate(status, ExecutionStatus.warn)

    if state.drop_copy_metrics:
        metrics = {k: v for k, v in state.drop_copy_metrics}
        if metrics.get("dropped", 0) > 0:
            issues.append(
                ExecutionIssue(
                    code="drop_copy_dropped_messages",
                    message=f"drop-copy queues dropped {metrics['dropped']} messages",
                    severity=ExecutionStatus.warn,
                )
            )
            status = _escalate(status, ExecutionStatus.warn)

    if policy.min_active_sessions > 0:
        active_count = len(state.sessions_active)
        if active_count < policy.min_active_sessions:
            severity = ExecutionStatus.warn
            if not policy.warn_on_missing_sessions:
                severity = ExecutionStatus.fail
            issues.append(
                ExecutionIssue(
                    code="missing_sessions",
                    message=f"only {active_count} active sessions (expected {policy.min_active_sessions})",
                    severity=severity,
                )
            )
            status = _escalate(status, severity)

    if state.last_error:
        issues.append(
            ExecutionIssue(
                code="last_error",
                message=f"last execution error: {state.last_error}",
                severity=ExecutionStatus.warn,
            )
        )
        status = _escalate(status, ExecutionStatus.warn)

    snapshot_metadata: dict[str, object] = {}
    if isinstance(metadata, MappingABC):
        snapshot_metadata.update(dict(metadata))
    snapshot_metadata.setdefault("orders_submitted", state.orders_submitted)
    snapshot_metadata.setdefault("orders_executed", state.orders_executed)
    snapshot_metadata.setdefault("orders_failed", state.orders_failed)
    if state.drop_copy_metrics:
        snapshot_metadata.setdefault(
            "drop_copy_metrics", {k: v for k, v in state.drop_copy_metrics}
        )

    return ExecutionReadinessSnapshot(
        service=service,
        generated_at=moment,
        status=status,
        policy=policy,
        state=state,
        issues=tuple(issues),
        metadata=dict(snapshot_metadata),
    )


def publish_execution_snapshot(
    event_bus: EventBus,
    snapshot: ExecutionReadinessSnapshot,
    *,
    source: str = "operations.execution",
    global_bus_factory: Callable[[], TopicBus] | None = None,
) -> None:
    """Publish the execution readiness snapshot onto the runtime/global event buses."""

    event = Event(
        type="telemetry.operational.execution",
        payload=snapshot.as_dict(),
        source=source,
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=(
            "Primary event bus publish_from_sync failed; falling back to global bus "
            "for execution readiness telemetry"
        ),
        runtime_unexpected_message=(
            "Unexpected error publishing execution readiness snapshot via runtime bus"
        ),
        runtime_none_message=(
            "Primary event bus publish_from_sync returned None; falling back to global bus "
            "for execution readiness telemetry"
        ),
        global_not_running_message=(
            "Global event bus not running while publishing execution readiness snapshot"
        ),
        global_unexpected_message=(
            "Unexpected error publishing execution readiness snapshot via global bus"
        ),
        global_bus_factory=global_bus_factory,
    )


__all__ = [
    "ExecutionStatus",
    "ExecutionPolicy",
    "ExecutionState",
    "ExecutionIssue",
    "ExecutionReadinessSnapshot",
    "evaluate_execution_readiness",
    "format_execution_markdown",
    "publish_execution_snapshot",
]
