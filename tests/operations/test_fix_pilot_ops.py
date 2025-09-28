from datetime import UTC, datetime

from src.operations.fix_pilot import (
    FixPilotPolicy,
    FixPilotStatus,
    evaluate_fix_pilot,
    format_fix_pilot_markdown,
)
from src.runtime.fix_pilot import FixPilotState


def test_evaluate_fix_pilot_pass():
    state = FixPilotState(
        sessions_started=True,
        sensory_running=True,
        broker_running=True,
        queue_metrics={"price": {"delivered": 5, "dropped": 0}},
        active_orders=2,
        last_order={"order_id": "ORD-2", "status": "ACK"},
        compliance_summary={"policy": {"name": "default"}},
        risk_summary={"avg_latency_ms": 120.0},
        dropcopy_running=True,
        dropcopy_backlog=0,
        last_dropcopy_event=None,
        dropcopy_reconciliation=None,
        timestamp=datetime.now(tz=UTC),
    )
    policy = FixPilotPolicy()
    snapshot = evaluate_fix_pilot(policy, state, metadata={"ingest_success": True})

    assert snapshot.status is FixPilotStatus.passed
    components = {comp.name: comp for comp in snapshot.components}
    assert "broker" in components
    assert components["dropcopy"].status is FixPilotStatus.passed
    assert "orders" in components
    markdown = format_fix_pilot_markdown(snapshot)
    assert "FIX Pilot Status" in markdown


def test_evaluate_fix_pilot_warn_and_fail():
    state = FixPilotState(
        sessions_started=False,
        sensory_running=False,
        broker_running=True,
        queue_metrics={"trade": {"delivered": 0, "dropped": 3}},
        active_orders=0,
        last_order=None,
        compliance_summary=None,
        risk_summary=None,
        dropcopy_running=False,
        dropcopy_backlog=2,
        last_dropcopy_event=None,
        dropcopy_reconciliation={"status_mismatches": ["ORD-1"]},
        timestamp=datetime.now(tz=UTC),
    )
    policy = FixPilotPolicy(require_compliance=True, max_queue_drops=1)
    snapshot = evaluate_fix_pilot(policy, state)

    components = {component.name: component for component in snapshot.components}
    assert components["sessions"].status is FixPilotStatus.fail
    assert components["sensory"].status is FixPilotStatus.fail
    assert components["queues"].status is FixPilotStatus.warn
    assert components["compliance"].status is FixPilotStatus.warn
    assert components["dropcopy"].status is FixPilotStatus.warn
    assert "orders" in components
