"""Telemetry helpers for the FIX integration pilot."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus, TopicBus, get_global_bus
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.runtime.fix_pilot import FixPilotState

logger = logging.getLogger(__name__)


class FixPilotStatus(StrEnum):
    """Severity levels reported by FIX pilot telemetry."""

    passed = "pass"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER = {
    FixPilotStatus.passed: 0,
    FixPilotStatus.warn: 1,
    FixPilotStatus.fail: 2,
}


def _coerce_bool(value: object, default: bool) -> bool:
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


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class FixPilotPolicy:
    """Policy thresholds for evaluating FIX pilot readiness."""

    require_sessions: bool = True
    require_sensory: bool = True
    require_broker: bool = True
    max_queue_drops: int = 0
    warn_on_idle_queues: bool = True
    require_compliance: bool = False
    require_risk_stats: bool = False
    require_dropcopy: bool = False
    max_dropcopy_backlog: int = 0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "FixPilotPolicy":
        mapping = mapping or {}
        return cls(
            require_sessions=_coerce_bool(mapping.get("FIX_PILOT_REQUIRE_SESSIONS"), True),
            require_sensory=_coerce_bool(mapping.get("FIX_PILOT_REQUIRE_SENSORY"), True),
            require_broker=_coerce_bool(mapping.get("FIX_PILOT_REQUIRE_BROKER"), True),
            max_queue_drops=max(0, _coerce_int(mapping.get("FIX_PILOT_MAX_QUEUE_DROPS"), 0)),
            warn_on_idle_queues=_coerce_bool(mapping.get("FIX_PILOT_WARN_ON_IDLE_QUEUES"), True),
            require_compliance=_coerce_bool(mapping.get("FIX_PILOT_REQUIRE_COMPLIANCE"), False),
            require_risk_stats=_coerce_bool(mapping.get("FIX_PILOT_REQUIRE_RISK_STATS"), False),
            require_dropcopy=_coerce_bool(mapping.get("FIX_PILOT_REQUIRE_DROPCOPY"), False),
            max_dropcopy_backlog=max(
                0, _coerce_int(mapping.get("FIX_PILOT_MAX_DROPCOPY_BACKLOG"), 0)
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "require_sessions": self.require_sessions,
            "require_sensory": self.require_sensory,
            "require_broker": self.require_broker,
            "max_queue_drops": self.max_queue_drops,
            "warn_on_idle_queues": self.warn_on_idle_queues,
            "require_compliance": self.require_compliance,
            "require_risk_stats": self.require_risk_stats,
            "require_dropcopy": self.require_dropcopy,
            "max_dropcopy_backlog": self.max_dropcopy_backlog,
        }


@dataclass(frozen=True)
class FixPilotComponent:
    """Component-level status reported by the FIX pilot."""

    name: str
    status: FixPilotStatus
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FixPilotSnapshot:
    """Aggregated FIX pilot telemetry snapshot."""

    status: FixPilotStatus
    timestamp: datetime
    components: tuple[FixPilotComponent, ...]
    metadata: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "components": [
                {"name": c.name, "status": c.status.value, "details": dict(c.details)}
                for c in self.components
            ],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        lines = ["| Component | Status | Details |", "| --- | --- | --- |"]
        for component in self.components:
            details = ", ".join(
                f"{key}: {value}" for key, value in sorted(component.details.items())
            )
            lines.append(
                f"| {component.name} | {component.status.value.upper()} | {details or '-'} |"
            )
        return "\n".join(lines)


def _escalate(current: FixPilotStatus, candidate: FixPilotStatus) -> FixPilotStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def evaluate_fix_pilot(
    policy: FixPilotPolicy,
    state: "FixPilotState",
    *,
    metadata: Mapping[str, object] | None = None,
) -> FixPilotSnapshot:
    """Convert runtime state into a structured FIX pilot snapshot."""

    components: list[FixPilotComponent] = []
    status = FixPilotStatus.passed

    connection_status = FixPilotStatus.passed
    connection_details: MutableMapping[str, object] = {
        "sessions_started": state.sessions_started,
    }
    if policy.require_sessions and not state.sessions_started:
        connection_status = FixPilotStatus.fail
        connection_details["issue"] = "sessions not started"
    components.append(
        FixPilotComponent(
            name="sessions", status=connection_status, details=dict(connection_details)
        )
    )
    status = _escalate(status, connection_status)

    sensory_status = FixPilotStatus.passed
    sensory_details: MutableMapping[str, object] = {"running": state.sensory_running}
    if policy.require_sensory and not state.sensory_running:
        sensory_status = FixPilotStatus.fail
        sensory_details["issue"] = "sensory organ stopped"
    components.append(
        FixPilotComponent(name="sensory", status=sensory_status, details=dict(sensory_details))
    )
    status = _escalate(status, sensory_status)

    broker_status = FixPilotStatus.passed
    broker_details: MutableMapping[str, object] = {
        "running": state.broker_running,
        "active_orders": state.active_orders,
    }
    if policy.require_broker and not state.broker_running:
        broker_status = FixPilotStatus.fail
        broker_details["issue"] = "broker offline"
    if state.last_order:
        broker_details["last_order"] = dict(state.last_order)
    components.append(
        FixPilotComponent(name="broker", status=broker_status, details=dict(broker_details))
    )
    status = _escalate(status, broker_status)

    queue_details: MutableMapping[str, object] = {}
    queue_status = FixPilotStatus.passed
    queue_issues: list[str] = []
    for name, metrics in state.queue_metrics.items():
        queue_details[name] = dict(metrics)
        drops = int(metrics.get("dropped", 0))
        delivered = int(metrics.get("delivered", 0))
        if drops > policy.max_queue_drops:
            queue_status = _escalate(queue_status, FixPilotStatus.warn)
            queue_issues.append(f"{name} dropped {drops} messages")
        if policy.warn_on_idle_queues and delivered == 0:
            queue_status = _escalate(queue_status, FixPilotStatus.warn)
            queue_issues.append(f"{name} delivered 0 events")
    if queue_issues:
        queue_details["issues"] = queue_issues
    components.append(
        FixPilotComponent(name="queues", status=queue_status, details=dict(queue_details))
    )
    status = _escalate(status, queue_status)

    compliance_status = FixPilotStatus.passed
    compliance_details: MutableMapping[str, object] = {}
    if state.compliance_summary is not None:
        compliance_details.update(dict(state.compliance_summary))
    elif policy.require_compliance:
        compliance_status = FixPilotStatus.warn
        compliance_details["issue"] = "no compliance monitor"
    components.append(
        FixPilotComponent(
            name="compliance", status=compliance_status, details=dict(compliance_details)
        )
    )
    status = _escalate(status, compliance_status)

    risk_status = FixPilotStatus.passed
    risk_details: MutableMapping[str, object] = {}
    if state.risk_summary is not None:
        risk_details.update(dict(state.risk_summary))
    elif policy.require_risk_stats:
        risk_status = FixPilotStatus.warn
        risk_details["issue"] = "risk stats unavailable"
    components.append(
        FixPilotComponent(name="risk", status=risk_status, details=dict(risk_details))
    )
    status = _escalate(status, risk_status)

    dropcopy_status = FixPilotStatus.passed
    dropcopy_details: MutableMapping[str, object] = {
        "running": state.dropcopy_running,
        "backlog": state.dropcopy_backlog,
    }
    dropcopy_issues: list[str] = []
    if state.last_dropcopy_event is not None:
        dropcopy_details["last_event"] = dict(state.last_dropcopy_event)
    reconciliation = (
        dict(state.dropcopy_reconciliation)
        if isinstance(state.dropcopy_reconciliation, Mapping)
        else {}
    )
    if reconciliation:
        dropcopy_details["reconciliation"] = reconciliation
    if not state.dropcopy_running:
        dropcopy_status = FixPilotStatus.warn
        dropcopy_issues.append("listener stopped")
        if policy.require_dropcopy:
            dropcopy_status = FixPilotStatus.fail
    if state.dropcopy_backlog > policy.max_dropcopy_backlog:
        dropcopy_status = _escalate(dropcopy_status, FixPilotStatus.warn)
        dropcopy_issues.append(
            f"backlog {state.dropcopy_backlog} > {policy.max_dropcopy_backlog}"
        )
    missing_orders = reconciliation.get("orders_without_dropcopy")
    if isinstance(missing_orders, Sequence) and missing_orders:
        severity = FixPilotStatus.fail if policy.require_dropcopy else FixPilotStatus.warn
        dropcopy_status = _escalate(dropcopy_status, severity)
        dropcopy_issues.append(f"missing dropcopy for {len(missing_orders)} orders")
    mismatches = reconciliation.get("status_mismatches")
    if isinstance(mismatches, Sequence) and mismatches:
        dropcopy_status = _escalate(
            dropcopy_status,
            FixPilotStatus.fail if policy.require_dropcopy else FixPilotStatus.warn,
        )
        dropcopy_issues.append(f"status mismatch for {len(mismatches)} orders")
    if dropcopy_issues:
        dropcopy_details["issues"] = dropcopy_issues
    components.append(
        FixPilotComponent(name="dropcopy", status=dropcopy_status, details=dict(dropcopy_details))
    )
    status = _escalate(status, dropcopy_status)

    orders_details: MutableMapping[str, object] = {
        "open_orders": len(state.open_orders),
        "positions_tracked": len(state.positions),
    }
    if state.total_exposure is not None:
        orders_details["total_exposure"] = state.total_exposure
    if state.order_journal_path:
        orders_details["journal"] = state.order_journal_path
    components.append(
        FixPilotComponent(name="orders", status=FixPilotStatus.passed, details=dict(orders_details))
    )
    status = _escalate(status, FixPilotStatus.passed)

    snapshot_metadata: dict[str, object] = {"policy": policy.as_dict()}
    if metadata:
        snapshot_metadata.update(dict(metadata))
    if state.open_orders:
        snapshot_metadata["open_orders"] = [dict(order) for order in state.open_orders]
    if state.positions:
        snapshot_metadata["positions"] = [dict(position) for position in state.positions]
    if state.total_exposure is not None:
        snapshot_metadata["total_exposure"] = state.total_exposure
    if state.order_journal_path:
        snapshot_metadata["order_journal_path"] = state.order_journal_path

    return FixPilotSnapshot(
        status=status,
        timestamp=state.timestamp,
        components=tuple(components),
        metadata=snapshot_metadata,
    )


def format_fix_pilot_markdown(snapshot: FixPilotSnapshot) -> str:
    header = (
        f"**FIX Pilot Status:** {snapshot.status.value.upper()} at {snapshot.timestamp.isoformat()}"
    )
    return "\n\n".join([header, snapshot.to_markdown()])


def publish_fix_pilot_snapshot(
    event_bus: EventBus | TopicBus | None,
    snapshot: FixPilotSnapshot,
    *,
    channel: str = "telemetry.execution.fix_pilot",
) -> None:
    """Publish FIX pilot telemetry on the event bus."""

    bus = event_bus or get_global_bus()
    try:
        payload = snapshot.as_dict()
        if isinstance(bus, TopicBus):
            bus.publish(channel, payload)
        else:
            bus.publish_from_sync(Event(channel, payload))
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("Failed to publish FIX pilot snapshot", exc_info=True)


__all__ = [
    "FixPilotStatus",
    "FixPilotPolicy",
    "FixPilotComponent",
    "FixPilotSnapshot",
    "evaluate_fix_pilot",
    "format_fix_pilot_markdown",
    "publish_fix_pilot_snapshot",
]
