from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.compliance.workflow import WorkflowTaskStatus
from src.governance.review_gates import (
    ReviewCriterionStatus,
    ReviewGateDecision,
    ReviewGateRegistry,
    ReviewVerdict,
)


def _write_definitions(path: Path) -> None:
    path.write_text(
        """
        gates:
          - gate_id: alpha_gate
            title: Alpha Gate
            description: Alpha description
            severity: high
            owners:
              - Owner A
            criteria:
              - id: requirement_a
                description: Requirement A
                mandatory: true
              - id: optional_b
                description: Optional B
                mandatory: false
          - gate_id: beta_gate
            title: Beta Gate
            description: Beta description
            severity: medium
            criteria:
              - id: control_a
                description: Control A
                mandatory: true
        """.strip()
    )


def test_review_gate_registry_merges_state(tmp_path: Path) -> None:
    definitions_path = tmp_path / "defs.yaml"
    state_path = tmp_path / "state.json"
    _write_definitions(definitions_path)

    state_payload = {
        "version": 1,
        "gates": [
            {
                "gate_id": "alpha_gate",
                "verdict": "pass",
                "decided_at": "2025-01-01T00:00:00+00:00",
                "decided_by": ["Owner A"],
                "notes": ["Initial approval"],
                "criteria_status": {
                    "requirement_a": "met",
                },
            }
        ],
    }
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    registry = ReviewGateRegistry.load(definitions_path, state_path=state_path)

    alpha_entry = registry.get("alpha_gate")
    assert alpha_entry is not None
    assert alpha_entry.decision is not None
    assert alpha_entry.decision.verdict is ReviewVerdict.pass_
    assert alpha_entry.status() is WorkflowTaskStatus.completed

    summary = registry.to_summary()
    alpha_summary = next(item for item in summary["gates"] if item["gate_id"] == "alpha_gate")
    assert alpha_summary["status"] == WorkflowTaskStatus.completed.value
    requirement_status = next(
        criterion["status"]
        for criterion in alpha_summary["criteria"]
        if criterion["id"] == "requirement_a"
    )
    assert requirement_status == ReviewCriterionStatus.met.value
    assert summary["patch_proposals"] == []

    decision = ReviewGateDecision(
        gate_id="beta_gate",
        verdict=ReviewVerdict.fail,
        decided_at=datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        decided_by=("Owner B",),
        criteria_status={"control_a": ReviewCriterionStatus.not_met},
    )
    beta_entry = registry.record_decision(decision)
    assert beta_entry.status() is WorkflowTaskStatus.blocked

    summary = registry.to_summary()
    proposals = summary["patch_proposals"]
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["proposal_id"] == "rfc-beta_gate-remediation"
    assert "control_a" in proposal["metadata"]["mandatory_failures"]
    assert any("control_a" in step for step in proposal["proposed_steps"])

    snapshot = registry.to_workflow_snapshot()
    assert snapshot.status is WorkflowTaskStatus.blocked
    assert snapshot.workflows[0].metadata["gate_count"] == 2

    markdown = registry.to_markdown()
    assert "Alpha Gate" in markdown
    assert "Beta Gate" in markdown
