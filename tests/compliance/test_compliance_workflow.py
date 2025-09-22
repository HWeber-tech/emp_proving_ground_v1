from datetime import UTC, datetime

from src.compliance.workflow import (
    ComplianceWorkflowSnapshot,
    WorkflowTaskStatus,
    evaluate_compliance_workflows,
    publish_compliance_workflows,
)
from src.core.event_bus import Event


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._running = True

    def is_running(self) -> bool:  # pragma: no cover - trivial
        return self._running

    def publish_from_sync(self, event: Event) -> int:
        self.events.append(event)
        return 1


def _build_trade_summary(status: str = "pass") -> dict[str, object]:
    history = [
        {
            "status": status,
            "generated_at": datetime(2025, 1, day, tzinfo=UTC).isoformat(),
        }
        for day in range(1, 6)
    ]
    return {
        "policy": {"policy_name": "tier1"},
        "last_snapshot": {
            "status": status,
            "checks": [
                {"rule_id": "single_trade_notional", "passed": status == "pass"},
                {"rule_id": "daily_limit", "passed": True},
            ],
            "generated_at": datetime(2025, 1, 1, tzinfo=UTC).isoformat(),
        },
        "history": history,
        "daily_totals": {"EURUSD": {"notional": 120000.0, "trades": 3}},
        "journal": {"last_entry": {"trade_id": "evt-1"}},
    }


def _build_kyc_summary(status: str = "APPROVED") -> dict[str, object]:
    return {
        "last_snapshot": {
            "case_id": "case-1",
            "status": status,
            "risk_rating": "LOW",
            "outstanding_items": [],
            "watchlist_hits": [],
            "alerts": [],
        },
        "recent": [
            {
                "case_id": "case-1",
                "status": status,
                "generated_at": datetime(2025, 1, 2, tzinfo=UTC).isoformat(),
            }
        ],
        "open_cases": 0,
        "escalations": 0,
        "journal": {"last_entry": {"case_id": "case-1"}},
    }


def test_evaluate_compliance_workflows_builds_completed_snapshot() -> None:
    snapshot = evaluate_compliance_workflows(
        trade_summary=_build_trade_summary(),
        kyc_summary=_build_kyc_summary(),
        strategy_registry={
            "total_strategies": 2,
            "approved_count": 1,
            "active_count": 1,
            "evolved_count": 1,
            "avg_fitness_score": 1.2,
            "max_fitness_score": 1.5,
            "min_fitness_score": 0.9,
            "catalogue_seeded": 2,
            "catalogue_entry_count": 2,
            "catalogue_missing_provenance": 0,
            "seed_source_counts": {"catalogue": 2},
            "catalogue_versions": ["2025.09"],
            "catalogue_names": ["institutional_default"],
            "latest_catalogue_seeded_at": 1_726_000_000.0,
        },
        metadata={"ingest_success": True},
    )

    assert isinstance(snapshot, ComplianceWorkflowSnapshot)
    assert snapshot.status is WorkflowTaskStatus.completed
    assert snapshot.metadata["ingest_success"] is True
    mifid = next(
        workflow for workflow in snapshot.workflows if workflow.name == "MiFID II controls"
    )
    assert all(task.status is WorkflowTaskStatus.completed for task in mifid.tasks)
    governance = next(
        workflow for workflow in snapshot.workflows if workflow.name == "Strategy governance"
    )
    assert governance.status is WorkflowTaskStatus.completed
    provenance_task = next(
        task for task in governance.tasks if task.task_id == "catalogue-provenance"
    )
    assert provenance_task.status is WorkflowTaskStatus.completed
    markdown = snapshot.to_markdown()
    assert "MiFID II controls" in markdown


def test_evaluate_compliance_workflows_handles_missing_monitors() -> None:
    snapshot = evaluate_compliance_workflows(
        trade_summary=None,
        kyc_summary=None,
        strategy_registry=None,
    )

    assert snapshot.status is WorkflowTaskStatus.blocked
    audit = next(
        workflow for workflow in snapshot.workflows if workflow.name == "Audit trail readiness"
    )
    assert any(task.status is WorkflowTaskStatus.blocked for task in audit.tasks)
    governance = next(
        workflow for workflow in snapshot.workflows if workflow.name == "Strategy governance"
    )
    assert governance.status is WorkflowTaskStatus.blocked


def test_publish_compliance_workflows_emits_event() -> None:
    bus = _StubEventBus()
    snapshot = evaluate_compliance_workflows(
        trade_summary=_build_trade_summary("warn"),
        kyc_summary=_build_kyc_summary("ESCALATED"),
        strategy_registry={
            "total_strategies": 0,
            "approved_count": 0,
            "active_count": 0,
            "evolved_count": 0,
            "avg_fitness_score": 0.0,
            "max_fitness_score": 0.0,
            "min_fitness_score": 0.0,
            "catalogue_seeded": 0,
            "catalogue_entry_count": 0,
            "catalogue_missing_provenance": 0,
            "seed_source_counts": {},
            "catalogue_versions": [],
            "catalogue_names": [],
            "latest_catalogue_seeded_at": None,
        },
    )

    publish_compliance_workflows(bus, snapshot)

    assert bus.events
    event = bus.events[0]
    assert event.type == "telemetry.compliance.workflow"
    assert event.payload["status"] == snapshot.status.value
