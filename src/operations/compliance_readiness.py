"""Aggregate institutional compliance telemetry into readiness snapshots."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus
from src.core.coercion import coerce_int
from src.operations.event_bus_failover import publish_event_with_failover
from src.compliance.workflow import WorkflowTaskStatus


logger = logging.getLogger(__name__)


class ComplianceReadinessStatus(Enum):
    """Severity levels exposed by the compliance readiness snapshot."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: dict[ComplianceReadinessStatus, int] = {
    ComplianceReadinessStatus.ok: 0,
    ComplianceReadinessStatus.warn: 1,
    ComplianceReadinessStatus.fail: 2,
}


def _combine_status(
    current: ComplianceReadinessStatus, candidate: ComplianceReadinessStatus
) -> ComplianceReadinessStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class ComplianceReadinessComponent:
    """Point-in-time compliance posture for an individual surface."""

    name: str
    status: ComplianceReadinessStatus
    summary: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ComplianceReadinessSnapshot:
    """Aggregated view across trade compliance and KYC telemetry."""

    status: ComplianceReadinessStatus
    generated_at: datetime
    components: tuple[ComplianceReadinessComponent, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "components": [component.as_dict() for component in self.components],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        if not self.components:
            return "| Component | Status | Summary |\n| --- | --- | --- |\n"

        rows = ["| Component | Status | Summary |", "| --- | --- | --- |"]
        for component in self.components:
            rows.append(
                f"| {component.name} | {component.status.value.upper()} | {component.summary} |"
            )
        return "\n".join(rows)


def _coerce_mapping(value: object) -> MutableMapping[str, object]:
    if isinstance(value, MutableMapping):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _extract_trade_component(
    summary: Mapping[str, object] | None,
) -> ComplianceReadinessComponent:
    if not summary:
        return ComplianceReadinessComponent(
            name="trade_compliance",
            status=ComplianceReadinessStatus.warn,
            summary="monitor inactive",
            metadata={"reason": "trade_compliance_monitor_missing"},
        )

    last_snapshot = _coerce_mapping(summary.get("last_snapshot"))
    policy = _coerce_mapping(summary.get("policy"))
    history_value = summary.get("history")
    history: tuple[object, ...] = tuple(_coerce_sequence(history_value)) if history_value else tuple()

    status_label = str(last_snapshot.get("status") or "").lower()
    raw_checks = last_snapshot.get("checks")
    checks = [
        _coerce_mapping(check)
        for check in _coerce_sequence(raw_checks)
        if isinstance(check, Mapping)
    ]
    failed_checks = [check for check in checks if not bool(check.get("passed"))]
    critical_failures = [
        check for check in failed_checks if str(check.get("severity", "")).lower() == "critical"
    ]

    if status_label == "fail" or critical_failures:
        status = ComplianceReadinessStatus.fail
    elif status_label == "warn" or failed_checks:
        status = ComplianceReadinessStatus.warn
    elif status_label == "pass":
        status = ComplianceReadinessStatus.ok
    elif status_label:
        status = ComplianceReadinessStatus.warn
    else:
        status = ComplianceReadinessStatus.warn

    summary_text = (
        f"{status_label.upper() or 'UNKNOWN'} under policy {policy.get('policy_name', 'default')}"
    )

    metadata = {
        "policy": dict(policy),
        "checks_total": len(checks),
        "checks_failed": len(failed_checks),
        "critical_failures": len(critical_failures),
    }

    if history:
        metadata["history_samples"] = min(len(history), 5)

    daily_totals = summary.get("daily_totals")
    if isinstance(daily_totals, Mapping):
        metadata["daily_totals"] = {
            symbol: _coerce_mapping(details)
            for symbol, details in daily_totals.items()
            if isinstance(details, Mapping)
        }

    return ComplianceReadinessComponent(
        name="trade_compliance",
        status=status,
        summary=summary_text,
        metadata=metadata,
    )


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _extract_kyc_component(
    summary: Mapping[str, object] | None,
) -> ComplianceReadinessComponent:
    if not summary:
        return ComplianceReadinessComponent(
            name="kyc_aml",
            status=ComplianceReadinessStatus.warn,
            summary="monitor inactive",
            metadata={"reason": "kyc_monitor_missing"},
        )

    last_snapshot = _coerce_mapping(summary.get("last_snapshot"))
    risk_rating = str(last_snapshot.get("risk_rating") or "").upper()
    status_label = str(last_snapshot.get("status") or "").upper()
    outstanding = list(_coerce_sequence(last_snapshot.get("outstanding_items")))
    watchlist = list(_coerce_sequence(last_snapshot.get("watchlist_hits")))
    alerts = list(_coerce_sequence(last_snapshot.get("alerts")))

    status = ComplianceReadinessStatus.ok
    if status_label in {"ESCALATED", "REJECTED", "BLOCKED", "SUSPENDED"}:
        status = ComplianceReadinessStatus.fail
    elif risk_rating in {"CRITICAL"}:
        status = ComplianceReadinessStatus.fail
    elif watchlist:
        status = ComplianceReadinessStatus.fail
    elif status_label not in {"APPROVED", "CLEARED"}:
        status = ComplianceReadinessStatus.warn
    elif outstanding or alerts:
        status = ComplianceReadinessStatus.warn

    open_cases = coerce_int(summary.get("open_cases"), default=0) or 0
    escalations = coerce_int(summary.get("escalations"), default=0) or 0
    if status is ComplianceReadinessStatus.ok:
        if open_cases or escalations:
            status = ComplianceReadinessStatus.warn

    next_review_due = _parse_datetime(last_snapshot.get("next_review_due"))
    overdue = False
    if next_review_due is not None:
        overdue = next_review_due < datetime.now(tz=UTC)
        if overdue:
            status = _combine_status(status, ComplianceReadinessStatus.warn)

    summary_text = status_label or "NO CASES"
    if risk_rating:
        summary_text = f"{summary_text} (risk {risk_rating})"
    if outstanding:
        summary_text += f" – {len(outstanding)} outstanding"
    if watchlist:
        summary_text += f" – watchlist hits: {len(watchlist)}"

    metadata = {
        "risk_rating": risk_rating,
        "status": status_label,
        "outstanding_items": len(outstanding),
        "watchlist_hits": len(watchlist),
        "alerts": len(alerts),
        "open_cases": open_cases,
        "escalations": escalations,
    }
    if overdue:
        metadata["next_review_overdue"] = True
    if next_review_due is not None:
        metadata["next_review_due"] = next_review_due.isoformat()

    return ComplianceReadinessComponent(
        name="kyc_aml",
        status=status,
        summary=summary_text,
        metadata=metadata,
    )


def _extract_workflow_component(
    summary: Mapping[str, object] | None,
) -> ComplianceReadinessComponent:
    if not summary:
        return ComplianceReadinessComponent(
            name="compliance_workflows",
            status=ComplianceReadinessStatus.warn,
            summary="workflow snapshot missing",
            metadata={"reason": "workflow_snapshot_missing"},
        )

    status_label = str(summary.get("status") or "").lower()
    workflows = [
        _coerce_mapping(workflow)
        for workflow in _coerce_sequence(summary.get("workflows"))
        if isinstance(workflow, Mapping)
    ]

    workflow_counts: dict[str, int] = {
        WorkflowTaskStatus.completed.value: 0,
        WorkflowTaskStatus.in_progress.value: 0,
        WorkflowTaskStatus.todo.value: 0,
        WorkflowTaskStatus.blocked.value: 0,
    }
    total_tasks = 0
    blocked_tasks = 0
    active_tasks = 0

    for workflow in workflows:
        workflow_status = str(workflow.get("status") or "").lower()
        if workflow_status in workflow_counts:
            workflow_counts[workflow_status] += 1

        tasks = [
            _coerce_mapping(task)
            for task in _coerce_sequence(workflow.get("tasks"))
            if isinstance(task, Mapping)
        ]
        for task in tasks:
            status = str(task.get("status") or "").lower()
            total_tasks += 1
            if status == WorkflowTaskStatus.blocked.value:
                blocked_tasks += 1
            if status and status != WorkflowTaskStatus.completed.value:
                active_tasks += 1

    if status_label == WorkflowTaskStatus.blocked.value:
        status = ComplianceReadinessStatus.fail
    elif workflow_counts[WorkflowTaskStatus.blocked.value] or blocked_tasks:
        status = ComplianceReadinessStatus.fail
    elif status_label in {
        WorkflowTaskStatus.todo.value,
        WorkflowTaskStatus.in_progress.value,
    }:
        status = ComplianceReadinessStatus.warn
    elif active_tasks:
        status = ComplianceReadinessStatus.warn
    elif status_label == WorkflowTaskStatus.completed.value:
        status = ComplianceReadinessStatus.ok
    else:
        status = ComplianceReadinessStatus.warn

    summary_text_parts = [status_label.upper() or "UNKNOWN"]
    total_workflows = sum(workflow_counts.values())
    if total_workflows:
        summary_text_parts.append(f"{total_workflows} workflows")
    if blocked_tasks:
        summary_text_parts.append(f"{blocked_tasks} blocked tasks")
    elif active_tasks:
        summary_text_parts.append(f"{active_tasks} active tasks")

    metadata: dict[str, object] = {
        "workflows_total": total_workflows,
        "workflows_completed": workflow_counts[WorkflowTaskStatus.completed.value],
        "workflows_in_progress": workflow_counts[WorkflowTaskStatus.in_progress.value],
        "workflows_todo": workflow_counts[WorkflowTaskStatus.todo.value],
        "workflows_blocked": workflow_counts[WorkflowTaskStatus.blocked.value],
        "tasks_total": total_tasks,
        "tasks_blocked": blocked_tasks,
        "tasks_active": active_tasks,
    }

    snapshot_metadata = summary.get("metadata")
    if isinstance(snapshot_metadata, Mapping):
        metadata["snapshot_metadata"] = dict(snapshot_metadata)

    return ComplianceReadinessComponent(
        name="compliance_workflows",
        status=status,
        summary=" – ".join(summary_text_parts),
        metadata=metadata,
    )


def evaluate_compliance_readiness(
    *,
    trade_summary: Mapping[str, object] | None = None,
    kyc_summary: Mapping[str, object] | None = None,
    workflow_summary: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ComplianceReadinessSnapshot:
    """Fuse trade compliance and KYC telemetry into a readiness snapshot."""

    generated_at = datetime.now(tz=UTC)
    components: list[ComplianceReadinessComponent] = []
    overall = ComplianceReadinessStatus.ok

    if trade_summary is not None or kyc_summary is not None:
        if trade_summary is not None:
            trade_component = _extract_trade_component(trade_summary)
        else:
            trade_component = _extract_trade_component(None)
        components.append(trade_component)
        overall = _combine_status(overall, trade_component.status)

        if kyc_summary is not None:
            kyc_component = _extract_kyc_component(kyc_summary)
        else:
            kyc_component = _extract_kyc_component(None)
        components.append(kyc_component)
        overall = _combine_status(overall, kyc_component.status)

        if workflow_summary is not None:
            workflow_component = _extract_workflow_component(workflow_summary)
            components.append(workflow_component)
            overall = _combine_status(overall, workflow_component.status)

    snapshot_metadata: dict[str, object] = {}
    if isinstance(metadata, Mapping):
        snapshot_metadata.update(dict(metadata))
    snapshot_metadata.setdefault("components", [component.name for component in components])

    return ComplianceReadinessSnapshot(
        status=overall,
        generated_at=generated_at,
        components=tuple(components),
        metadata=snapshot_metadata,
    )


def publish_compliance_readiness(
    event_bus: EventBus,
    snapshot: ComplianceReadinessSnapshot,
    *,
    channel: str = "telemetry.compliance.readiness",
) -> None:
    """Emit the readiness snapshot on the runtime and global event buses."""

    event = Event(
        type=channel,
        payload=snapshot.as_dict(),
        source="compliance_readiness",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=
        "Runtime event bus rejected compliance readiness snapshot; falling back to global bus",
        runtime_unexpected_message=
        "Unexpected error publishing compliance readiness snapshot via runtime bus",
        runtime_none_message=
        "Runtime event bus returned None while publishing compliance readiness snapshot; using global bus",
        global_not_running_message=
        "Global event bus not running while publishing compliance readiness snapshot",
        global_unexpected_message=
        "Unexpected error publishing compliance readiness snapshot via global bus",
    )


__all__ = [
    "ComplianceReadinessComponent",
    "ComplianceReadinessSnapshot",
    "ComplianceReadinessStatus",
    "evaluate_compliance_readiness",
    "publish_compliance_readiness",
]
