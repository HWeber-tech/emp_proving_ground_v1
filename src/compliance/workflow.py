"""Compliance workflow checklists and telemetry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus
from src.core.coercion import coerce_int


logger = logging.getLogger(__name__)


class WorkflowTaskStatus(Enum):
    """Status values tracked for compliance workflow tasks."""

    completed = "completed"
    in_progress = "in_progress"
    todo = "todo"
    blocked = "blocked"


_TASK_ORDER: dict[WorkflowTaskStatus, int] = {
    WorkflowTaskStatus.completed: 0,
    WorkflowTaskStatus.in_progress: 1,
    WorkflowTaskStatus.todo: 2,
    WorkflowTaskStatus.blocked: 3,
}


def _combine_status(
    current: WorkflowTaskStatus, candidate: WorkflowTaskStatus
) -> WorkflowTaskStatus:
    if _TASK_ORDER[candidate] > _TASK_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class ComplianceWorkflowTask:
    """Single actionable item inside a compliance workflow."""

    task_id: str
    title: str
    status: WorkflowTaskStatus
    summary: str
    severity: str = "medium"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status.value,
            "summary": self.summary,
            "severity": self.severity,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ComplianceWorkflowChecklist:
    """Grouped tasks that satisfy a regulatory workflow."""

    name: str
    regulation: str
    status: WorkflowTaskStatus
    tasks: tuple[ComplianceWorkflowTask, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "regulation": self.regulation,
            "status": self.status.value,
            "tasks": [task.as_dict() for task in self.tasks],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        rows = [f"### {self.name} ({self.status.value.upper()})"]
        if not self.tasks:
            rows.append("(no tasks configured)")
            return "\n".join(rows)

        rows.extend(["| Task | Status | Summary |", "| --- | --- | --- |"])
        for task in self.tasks:
            rows.append(f"| {task.title} | {task.status.value.upper()} | {task.summary} |")
        return "\n".join(rows)


@dataclass(frozen=True)
class ComplianceWorkflowSnapshot:
    """Aggregated compliance workflow status."""

    status: WorkflowTaskStatus
    generated_at: datetime
    workflows: tuple[ComplianceWorkflowChecklist, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "workflows": [workflow.as_dict() for workflow in self.workflows],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        if not self.workflows:
            return "| Workflow | Status |\n| --- | --- |\n"

        blocks = [f"## Compliance Workflows ({self.status.value.upper()})"]
        for workflow in self.workflows:
            blocks.append(workflow.to_markdown())
        return "\n\n".join(blocks)


def _coerce_mapping(value: object) -> MutableMapping[str, object]:
    if isinstance(value, MutableMapping):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _extract_trade_details(
    summary: Mapping[str, object] | None,
) -> dict[str, object]:
    details: dict[str, object] = {
        "active": summary is not None,
        "status": "",
        "failed_checks": 0,
        "critical_failures": 0,
        "history_length": 0,
        "daily_totals": {},
        "journal_present": False,
        "policy_name": None,
    }
    if summary is None:
        return details

    mapping = dict(summary)
    policy = mapping.get("policy")
    if isinstance(policy, Mapping):
        details["policy_name"] = policy.get("policy_name")

    last_snapshot = mapping.get("last_snapshot")
    snapshot_map = _coerce_mapping(last_snapshot)
    status_label = str(snapshot_map.get("status") or "").lower()
    details["status"] = status_label

    checks_raw = snapshot_map.get("checks", [])
    if isinstance(checks_raw, Sequence):
        checks = [_coerce_mapping(check) for check in checks_raw if isinstance(check, Mapping)]
        failed_checks = [check for check in checks if not bool(check.get("passed"))]
        critical_failures = [
            check
            for check in failed_checks
            if str(check.get("severity") or "").lower() == "critical"
        ]
        details["failed_checks"] = len(failed_checks)
        details["critical_failures"] = len(critical_failures)

    history = mapping.get("history")
    if isinstance(history, Sequence):
        details["history_length"] = len(history)

    totals = mapping.get("daily_totals")
    if isinstance(totals, Mapping):
        details["daily_totals"] = {
            str(symbol): dict(_coerce_mapping(info))
            for symbol, info in totals.items()
            if isinstance(info, Mapping)
        }

    journal = mapping.get("journal")
    if isinstance(journal, Mapping):
        journal_map = _coerce_mapping(journal)
        if journal_map.get("last_entry") or journal_map.get("recent_entries"):
            details["journal_present"] = True

    return details


def _extract_kyc_details(
    summary: Mapping[str, object] | None,
) -> dict[str, object]:
    details: dict[str, object] = {
        "active": summary is not None,
        "open_cases": 0,
        "escalations": 0,
        "recent_count": 0,
        "last_status": "",
        "risk_rating": "",
        "outstanding_items": 0,
        "watchlist_hits": 0,
        "alerts": 0,
        "journal_present": False,
    }
    if summary is None:
        return details

    mapping = dict(summary)
    details["open_cases"] = coerce_int(mapping.get("open_cases"), default=0)
    details["escalations"] = coerce_int(mapping.get("escalations"), default=0)

    recent = mapping.get("recent")
    if isinstance(recent, Sequence):
        details["recent_count"] = len(recent)

    last_snapshot = mapping.get("last_snapshot")
    snapshot_map = _coerce_mapping(last_snapshot)
    details["last_status"] = str(snapshot_map.get("status") or "").upper()
    details["risk_rating"] = str(snapshot_map.get("risk_rating") or "").upper()

    outstanding = snapshot_map.get("outstanding_items")
    if isinstance(outstanding, Sequence):
        details["outstanding_items"] = len(outstanding)
    watchlist = snapshot_map.get("watchlist_hits")
    if isinstance(watchlist, Sequence):
        details["watchlist_hits"] = len(watchlist)
    alerts = snapshot_map.get("alerts")
    if isinstance(alerts, Sequence):
        details["alerts"] = len(alerts)

    journal = mapping.get("journal")
    if isinstance(journal, Mapping):
        journal_map = _coerce_mapping(journal)
        if journal_map.get("last_entry") or journal_map.get("recent_entries"):
            details["journal_present"] = True

    return details


def _build_mifid_workflow(trade: Mapping[str, object]) -> ComplianceWorkflowChecklist:
    tasks: list[ComplianceWorkflowTask] = []
    overall = WorkflowTaskStatus.completed
    active = bool(trade.get("active"))

    if not active:
        blocked = ComplianceWorkflowTask(
            task_id="mifid-transaction-reporting",
            title="Article 26 transaction reporting",
            status=WorkflowTaskStatus.blocked,
            summary="Trade compliance monitor inactive",
            severity="high",
            metadata={"reason": "trade_monitor_missing"},
        )
        tasks.append(blocked)
        overall = _combine_status(overall, blocked.status)
        recordkeeping = ComplianceWorkflowTask(
            task_id="mifid-recordkeeping",
            title="Article 16 recordkeeping",
            status=WorkflowTaskStatus.blocked,
            summary="Trade compliance monitor inactive",
            severity="high",
            metadata={"reason": "trade_monitor_missing"},
        )
        tasks.append(recordkeeping)
        overall = _combine_status(overall, recordkeeping.status)
        return ComplianceWorkflowChecklist(
            name="MiFID II controls",
            regulation="MiFID II",
            status=overall,
            tasks=tuple(tasks),
            metadata={"monitor_active": False},
        )

    status_label = str(trade.get("status") or "")
    failed_checks = coerce_int(trade.get("failed_checks"), default=0)
    critical_failures = coerce_int(trade.get("critical_failures"), default=0)
    policy_name = trade.get("policy_name")

    if status_label == "fail" or critical_failures:
        txn_status = WorkflowTaskStatus.blocked
        txn_summary = (
            f"Failing snapshot with {critical_failures or failed_checks} critical breaches"
        )
        severity = "critical"
    elif status_label == "warn" or failed_checks:
        txn_status = WorkflowTaskStatus.in_progress
        txn_summary = f"{failed_checks} policy checks require remediation"
        severity = "high"
    elif status_label == "pass":
        txn_status = WorkflowTaskStatus.completed
        txn_summary = f"Policy {policy_name or 'default'} passing latest snapshot"
        severity = "medium"
    else:
        txn_status = WorkflowTaskStatus.todo
        txn_summary = "No recent trade compliance snapshot recorded"
        severity = "medium"

    txn_task = ComplianceWorkflowTask(
        task_id="mifid-transaction-reporting",
        title="Article 26 transaction reporting",
        status=txn_status,
        summary=txn_summary,
        severity=severity,
        metadata={
            "policy_name": policy_name,
            "status": status_label,
            "failed_checks": failed_checks,
            "critical_failures": critical_failures,
        },
    )
    tasks.append(txn_task)
    overall = _combine_status(overall, txn_task.status)

    journal_present = bool(trade.get("journal_present"))
    history_length = coerce_int(trade.get("history_length"), default=0)
    if journal_present:
        rec_status = WorkflowTaskStatus.completed
        rec_summary = "Compliance journal persistence enabled"
    elif history_length:
        rec_status = WorkflowTaskStatus.in_progress
        rec_summary = "Snapshots captured but Timescale journal disabled"
    else:
        rec_status = WorkflowTaskStatus.todo
        rec_summary = "Enable compliance journal retention"

    recordkeeping_task = ComplianceWorkflowTask(
        task_id="mifid-recordkeeping",
        title="Article 16 recordkeeping",
        status=rec_status,
        summary=rec_summary,
        severity="medium",
        metadata={
            "history_samples": history_length,
            "journal_present": journal_present,
        },
    )
    tasks.append(recordkeeping_task)
    overall = _combine_status(overall, recordkeeping_task.status)

    metadata = {
        "monitor_active": True,
        "policy_name": policy_name,
        "status": status_label,
    }
    return ComplianceWorkflowChecklist(
        name="MiFID II controls",
        regulation="MiFID II",
        status=overall,
        tasks=tuple(tasks),
        metadata=metadata,
    )


def _build_dodd_frank_workflow(trade: Mapping[str, object]) -> ComplianceWorkflowChecklist:
    tasks: list[ComplianceWorkflowTask] = []
    overall = WorkflowTaskStatus.completed
    active = bool(trade.get("active"))

    if not active:
        blocked = ComplianceWorkflowTask(
            task_id="doddfrank-large-trader",
            title="Large trader threshold monitoring",
            status=WorkflowTaskStatus.blocked,
            summary="Trade compliance monitor inactive",
            severity="high",
            metadata={"reason": "trade_monitor_missing"},
        )
        tasks.append(blocked)
        overall = _combine_status(overall, blocked.status)
        audit = ComplianceWorkflowTask(
            task_id="doddfrank-audit",
            title="Swap data repository audit trail",
            status=WorkflowTaskStatus.blocked,
            summary="Trade compliance monitor inactive",
            severity="high",
            metadata={"reason": "trade_monitor_missing"},
        )
        tasks.append(audit)
        overall = _combine_status(overall, audit.status)
        return ComplianceWorkflowChecklist(
            name="Dodd-Frank controls",
            regulation="Dodd-Frank",
            status=overall,
            tasks=tuple(tasks),
            metadata={"monitor_active": False},
        )

    status_label = str(trade.get("status") or "")
    failed_checks = coerce_int(trade.get("failed_checks"), default=0)
    totals = trade.get("daily_totals")
    if not isinstance(totals, Mapping):
        totals = {}

    if status_label == "fail" or failed_checks:
        large_status = WorkflowTaskStatus.blocked
        large_summary = "Policy violations block large trader reporting"
        severity = "high"
    elif totals:
        if status_label == "pass":
            large_status = WorkflowTaskStatus.completed
            large_summary = "Daily totals captured for swap data reporting"
        else:
            large_status = WorkflowTaskStatus.in_progress
            large_summary = "Monitoring thresholds with outstanding warnings"
        severity = "medium"
    else:
        large_status = WorkflowTaskStatus.todo
        large_summary = "Enable notional aggregation for large trader reports"
        severity = "medium"

    large_task = ComplianceWorkflowTask(
        task_id="doddfrank-large-trader",
        title="Large trader threshold monitoring",
        status=large_status,
        summary=large_summary,
        severity=severity,
        metadata={
            "status": status_label,
            "failed_checks": failed_checks,
            "symbols_tracked": len(totals),
        },
    )
    tasks.append(large_task)
    overall = _combine_status(overall, large_task.status)

    history_length = coerce_int(trade.get("history_length"), default=0)
    if history_length >= 5:
        audit_status = WorkflowTaskStatus.completed
        audit_summary = "Historical compliance snapshots retained"
    elif history_length:
        audit_status = WorkflowTaskStatus.in_progress
        audit_summary = "Limited snapshot history captured"
    else:
        audit_status = WorkflowTaskStatus.todo
        audit_summary = "Record compliance snapshots for audit trail"

    audit_task = ComplianceWorkflowTask(
        task_id="doddfrank-audit",
        title="Swap data repository audit trail",
        status=audit_status,
        summary=audit_summary,
        severity="medium",
        metadata={"history_samples": history_length},
    )
    tasks.append(audit_task)
    overall = _combine_status(overall, audit_task.status)

    metadata = {
        "monitor_active": True,
        "status": status_label,
        "history_samples": history_length,
        "symbols_tracked": len(totals),
    }
    return ComplianceWorkflowChecklist(
        name="Dodd-Frank controls",
        regulation="Dodd-Frank",
        status=overall,
        tasks=tuple(tasks),
        metadata=metadata,
    )


def _build_kyc_workflow(kyc: Mapping[str, object]) -> ComplianceWorkflowChecklist:
    tasks: list[ComplianceWorkflowTask] = []
    overall = WorkflowTaskStatus.completed
    active = bool(kyc.get("active"))

    if not active:
        blocked = ComplianceWorkflowTask(
            task_id="kyc-due-diligence",
            title="Customer due diligence",
            status=WorkflowTaskStatus.blocked,
            summary="KYC/AML monitor inactive",
            severity="high",
            metadata={"reason": "kyc_monitor_missing"},
        )
        tasks.append(blocked)
        overall = _combine_status(overall, blocked.status)
        watchlist = ComplianceWorkflowTask(
            task_id="kyc-watchlist",
            title="Watchlist screening",
            status=WorkflowTaskStatus.blocked,
            summary="KYC/AML monitor inactive",
            severity="high",
            metadata={"reason": "kyc_monitor_missing"},
        )
        tasks.append(watchlist)
        overall = _combine_status(overall, watchlist.status)
        monitoring = ComplianceWorkflowTask(
            task_id="kyc-ongoing",
            title="Ongoing monitoring",
            status=WorkflowTaskStatus.blocked,
            summary="KYC/AML monitor inactive",
            severity="high",
            metadata={"reason": "kyc_monitor_missing"},
        )
        tasks.append(monitoring)
        overall = _combine_status(overall, monitoring.status)
        return ComplianceWorkflowChecklist(
            name="KYC / AML workflows",
            regulation="KYC/AML",
            status=overall,
            tasks=tuple(tasks),
            metadata={"monitor_active": False},
        )

    open_cases = coerce_int(kyc.get("open_cases"), default=0)
    escalations = coerce_int(kyc.get("escalations"), default=0)
    outstanding = coerce_int(kyc.get("outstanding_items"), default=0)
    watchlist_hits = coerce_int(kyc.get("watchlist_hits"), default=0)
    alerts = coerce_int(kyc.get("alerts"), default=0)
    recent_count = coerce_int(kyc.get("recent_count"), default=0)
    risk_rating = str(kyc.get("risk_rating") or "")

    if escalations:
        dd_status = WorkflowTaskStatus.blocked
        dd_summary = f"{escalations} escalations awaiting resolution"
        severity = "high"
    elif open_cases or outstanding:
        dd_status = WorkflowTaskStatus.in_progress
        dd_summary = f"{open_cases or outstanding} open onboarding tasks"
        severity = "medium"
    else:
        dd_status = WorkflowTaskStatus.completed
        dd_summary = "All customer dossiers cleared"
        severity = "medium"

    due_diligence = ComplianceWorkflowTask(
        task_id="kyc-due-diligence",
        title="Customer due diligence",
        status=dd_status,
        summary=dd_summary,
        severity=severity,
        metadata={
            "open_cases": open_cases,
            "escalations": escalations,
            "outstanding_items": outstanding,
        },
    )
    tasks.append(due_diligence)
    overall = _combine_status(overall, due_diligence.status)

    if watchlist_hits:
        wl_status = WorkflowTaskStatus.blocked
        wl_summary = f"{watchlist_hits} entities flagged on watchlists"
    elif alerts:
        wl_status = WorkflowTaskStatus.in_progress
        wl_summary = f"{alerts} alerts require review"
    else:
        wl_status = WorkflowTaskStatus.completed
        wl_summary = "No watchlist hits detected"

    watchlist_task = ComplianceWorkflowTask(
        task_id="kyc-watchlist",
        title="Watchlist screening",
        status=wl_status,
        summary=wl_summary,
        severity="medium",
        metadata={"watchlist_hits": watchlist_hits, "alerts": alerts},
    )
    tasks.append(watchlist_task)
    overall = _combine_status(overall, watchlist_task.status)

    if recent_count:
        monitoring_status = WorkflowTaskStatus.completed
        monitoring_summary = "Continuous monitoring cadence active"
    else:
        monitoring_status = WorkflowTaskStatus.todo
        monitoring_summary = "Schedule recurring KYC monitoring reviews"

    monitoring_task = ComplianceWorkflowTask(
        task_id="kyc-ongoing",
        title="Ongoing monitoring",
        status=monitoring_status,
        summary=monitoring_summary,
        severity="low",
        metadata={"recent_reviews": recent_count},
    )
    tasks.append(monitoring_task)
    overall = _combine_status(overall, monitoring_task.status)

    metadata = {
        "monitor_active": True,
        "risk_rating": risk_rating,
        "open_cases": open_cases,
        "escalations": escalations,
        "watchlist_hits": watchlist_hits,
    }
    return ComplianceWorkflowChecklist(
        name="KYC / AML workflows",
        regulation="KYC/AML",
        status=overall,
        tasks=tuple(tasks),
        metadata=metadata,
    )


def _build_audit_workflow(
    trade: Mapping[str, object], kyc: Mapping[str, object]
) -> ComplianceWorkflowChecklist:
    tasks: list[ComplianceWorkflowTask] = []
    overall = WorkflowTaskStatus.completed

    trade_active = bool(trade.get("active"))
    kyc_active = bool(kyc.get("active"))

    if trade_active:
        if trade.get("journal_present"):
            trade_status = WorkflowTaskStatus.completed
            trade_summary = "Trade compliance journal retained in Timescale"
        elif coerce_int(trade.get("history_length"), default=0):
            trade_status = WorkflowTaskStatus.in_progress
            trade_summary = "Snapshots captured but journal persistence disabled"
        else:
            trade_status = WorkflowTaskStatus.todo
            trade_summary = "Enable trade compliance journaling"
    else:
        trade_status = WorkflowTaskStatus.blocked
        trade_summary = "Trade compliance monitor inactive"

    trade_task = ComplianceWorkflowTask(
        task_id="audit-trade-journal",
        title="Trade compliance journal retention",
        status=trade_status,
        summary=trade_summary,
        severity="medium" if trade_status is WorkflowTaskStatus.completed else "high",
        metadata={
            "monitor_active": trade_active,
            "journal_present": trade.get("journal_present"),
        },
    )
    tasks.append(trade_task)
    overall = _combine_status(overall, trade_task.status)

    if kyc_active:
        if kyc.get("journal_present"):
            kyc_status = WorkflowTaskStatus.completed
            kyc_summary = "KYC case journal retained in Timescale"
        elif coerce_int(kyc.get("open_cases"), default=0):
            kyc_status = WorkflowTaskStatus.in_progress
            kyc_summary = "Open cases without Timescale journaling"
        else:
            kyc_status = WorkflowTaskStatus.todo
            kyc_summary = "Enable KYC case journaling"
    else:
        kyc_status = WorkflowTaskStatus.blocked
        kyc_summary = "KYC/AML monitor inactive"

    kyc_task = ComplianceWorkflowTask(
        task_id="audit-kyc-journal",
        title="KYC journal retention",
        status=kyc_status,
        summary=kyc_summary,
        severity="medium" if kyc_status is WorkflowTaskStatus.completed else "high",
        metadata={
            "monitor_active": kyc_active,
            "journal_present": kyc.get("journal_present"),
        },
    )
    tasks.append(kyc_task)
    overall = _combine_status(overall, kyc_task.status)

    metadata = {
        "trade_monitor_active": trade_active,
        "kyc_monitor_active": kyc_active,
    }
    return ComplianceWorkflowChecklist(
        name="Audit trail readiness",
        regulation="Multi-regulation",
        status=overall,
        tasks=tuple(tasks),
        metadata=metadata,
    )


def _build_strategy_governance_workflow(
    registry: Mapping[str, object] | None,
) -> ComplianceWorkflowChecklist:
    tasks: list[ComplianceWorkflowTask] = []

    if registry is None:
        status = WorkflowTaskStatus.blocked
        summary = "Strategy registry unavailable"
        metadata: Mapping[str, object] = {}
        tasks.append(
            ComplianceWorkflowTask(
                task_id="catalogue-provenance",
                title="Catalogue provenance recorded",
                status=status,
                summary=summary,
                severity="high",
                metadata={},
            )
        )
        tasks.append(
            ComplianceWorkflowTask(
                task_id="strategy-approvals",
                title="Strategy approvals documented",
                status=status,
                summary="Strategy registry unavailable",
                severity="high",
                metadata={},
            )
        )
        tasks.append(
            ComplianceWorkflowTask(
                task_id="strategy-coverage",
                title="Registry coverage",
                status=status,
                summary="Strategy registry unavailable",
                severity="medium",
                metadata={},
            )
        )
        return ComplianceWorkflowChecklist(
            name="Strategy governance",
            regulation="MiFID II / internal governance",
            status=status,
            tasks=tuple(tasks),
            metadata={},
        )

    seeded = coerce_int(registry.get("catalogue_seeded"), default=0)
    missing = coerce_int(registry.get("catalogue_missing_provenance"), default=0)
    approved = coerce_int(registry.get("approved_count"), default=0)
    total = coerce_int(registry.get("total_strategies"), default=0)
    versions_raw = registry.get("catalogue_versions")
    if isinstance(versions_raw, Sequence) and not isinstance(
        versions_raw, (str, bytes, bytearray)
    ):
        catalogue_versions = [str(version) for version in versions_raw]
    else:
        catalogue_versions = []

    if seeded == 0:
        provenance_status = WorkflowTaskStatus.todo
        provenance_summary = "No catalogue-backed strategies registered"
    elif missing == 0:
        provenance_status = WorkflowTaskStatus.completed
        provenance_summary = "All catalogue-backed strategies record provenance"
    else:
        provenance_status = WorkflowTaskStatus.in_progress
        provenance_summary = f"{missing} catalogue-backed strategies missing provenance metadata"

    tasks.append(
        ComplianceWorkflowTask(
            task_id="catalogue-provenance",
            title="Catalogue provenance recorded",
            status=provenance_status,
            summary=provenance_summary,
            severity="high" if provenance_status is not WorkflowTaskStatus.completed else "medium",
            metadata={
                "catalogue_seeded": seeded,
                "catalogue_missing_provenance": missing,
                "catalogue_versions": catalogue_versions,
            },
        )
    )

    if approved > 0:
        approvals_status = WorkflowTaskStatus.completed
        approvals_summary = f"{approved} strategy approvals recorded"
    elif total > 0:
        approvals_status = WorkflowTaskStatus.in_progress
        approvals_summary = "Strategies evolved but approvals pending"
    else:
        approvals_status = WorkflowTaskStatus.todo
        approvals_summary = "No strategies registered in governance registry"

    tasks.append(
        ComplianceWorkflowTask(
            task_id="strategy-approvals",
            title="Strategy approvals documented",
            status=approvals_status,
            summary=approvals_summary,
            severity="high" if approvals_status is WorkflowTaskStatus.todo else "medium",
            metadata={
                "approved_count": approved,
                "total_strategies": total,
            },
        )
    )

    if total > 0:
        coverage_status = WorkflowTaskStatus.completed
        coverage_summary = f"{total} strategies tracked in registry"
    else:
        coverage_status = WorkflowTaskStatus.todo
        coverage_summary = "Populate the strategy registry"

    tasks.append(
        ComplianceWorkflowTask(
            task_id="strategy-coverage",
            title="Registry coverage",
            status=coverage_status,
            summary=coverage_summary,
            severity="medium",
            metadata={
                "total_strategies": total,
                "seed_source_counts": registry.get("seed_source_counts", {}),
            },
        )
    )

    overall = WorkflowTaskStatus.completed
    for task in tasks:
        overall = _combine_status(overall, task.status)

    return ComplianceWorkflowChecklist(
        name="Strategy governance",
        regulation="MiFID II / internal governance",
        status=overall,
        tasks=tuple(tasks),
        metadata=dict(registry),
    )


def evaluate_compliance_workflows(
    *,
    trade_summary: Mapping[str, object] | None,
    kyc_summary: Mapping[str, object] | None,
    strategy_registry: Mapping[str, object] | None = None,
    policy_workflow_snapshot: ComplianceWorkflowSnapshot | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ComplianceWorkflowSnapshot:
    """Convert compliance monitor telemetry into workflow snapshots."""

    trade_details = _extract_trade_details(trade_summary)
    kyc_details = _extract_kyc_details(kyc_summary)

    workflows = [
        _build_mifid_workflow(trade_details),
        _build_dodd_frank_workflow(trade_details),
        _build_kyc_workflow(kyc_details),
        _build_audit_workflow(trade_details, kyc_details),
        _build_strategy_governance_workflow(strategy_registry),
    ]

    policy_workflow_names: tuple[str, ...] = ()
    if policy_workflow_snapshot is not None:
        workflows.extend(policy_workflow_snapshot.workflows)
        policy_workflow_names = tuple(
            workflow.name for workflow in policy_workflow_snapshot.workflows
        )

    status = WorkflowTaskStatus.completed
    for workflow in workflows:
        status = _combine_status(status, workflow.status)

    snapshot_metadata: dict[str, object] = {
        "trade_monitor_active": trade_details["active"],
        "kyc_monitor_active": kyc_details["active"],
        "trade_status": trade_details["status"],
        "trade_failed_checks": trade_details["failed_checks"],
        "trade_critical_failures": trade_details["critical_failures"],
        "kyc_open_cases": kyc_details["open_cases"],
        "kyc_escalations": kyc_details["escalations"],
    }
    if strategy_registry:
        snapshot_metadata["strategy_registry"] = dict(strategy_registry)
    if policy_workflow_snapshot is not None:
        snapshot_metadata["policy_workflow_status"] = policy_workflow_snapshot.status.value
        snapshot_metadata["policy_workflow_generated_at"] = (
            policy_workflow_snapshot.generated_at.isoformat()
        )
        if policy_workflow_snapshot.metadata:
            snapshot_metadata["policy_workflow_metadata"] = dict(
                policy_workflow_snapshot.metadata
            )
        snapshot_metadata["policy_workflow_names"] = list(policy_workflow_names)
        snapshot_metadata["policy_workflow_count"] = len(policy_workflow_names)
    if metadata:
        snapshot_metadata.update(metadata)

    return ComplianceWorkflowSnapshot(
        status=status,
        generated_at=datetime.now(UTC),
        workflows=tuple(workflows),
        metadata=snapshot_metadata,
    )


def publish_compliance_workflows(
    event_bus: EventBus,
    snapshot: ComplianceWorkflowSnapshot,
    *,
    channel: str = "telemetry.compliance.workflow",
) -> None:
    """Publish the compliance workflow snapshot to the runtime bus."""

    event = Event(
        type=channel,
        payload=snapshot.as_dict(),
        source="compliance_workflows",
    )

    from src.operations.event_bus_failover import publish_event_with_failover

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=
        "Runtime bus rejected compliance workflow publish; falling back to global bus",
        runtime_unexpected_message=
        "Unexpected error publishing compliance workflow snapshot via runtime bus",
        runtime_none_message=
        "Compliance workflow publish returned None; falling back to global bus",
        global_not_running_message=
        "Global bus not running while publishing compliance workflow snapshot",
        global_unexpected_message=
        "Unexpected error publishing compliance workflow snapshot via global bus",
    )


__all__ = [
    "ComplianceWorkflowChecklist",
    "ComplianceWorkflowSnapshot",
    "ComplianceWorkflowTask",
    "WorkflowTaskStatus",
    "evaluate_compliance_workflows",
    "publish_compliance_workflows",
]
