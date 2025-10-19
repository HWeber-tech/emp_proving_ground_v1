from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.reflection.trm.governance import (
    AutoApplyRuleConfig,
    ProposalEvaluation,
    publish_governance_artifacts,
)
from src.reflection.trm.types import RIMWindow


def test_auto_apply_rule_accepts_when_conditions_met() -> None:
    config = AutoApplyRuleConfig(uplift_threshold=0.05)
    evaluation = ProposalEvaluation(
        suggestion_id="rim-0001",
        oos_uplift=0.08,
        risk_hits=0,
        invariant_breaches=0,
        budget_remaining=25.0,
        budget_utilisation=0.4,
    )

    decision = config.evaluate(evaluation)

    assert decision.auto_applied is True
    assert decision.reasons == ()


def test_auto_apply_rule_collects_failure_reasons() -> None:
    config = AutoApplyRuleConfig(
        uplift_threshold=0.05,
        max_risk_hits=0,
        min_budget_remaining=10.0,
        max_budget_utilisation=0.9,
    )
    evaluation = ProposalEvaluation(
        suggestion_id="rim-0002",
        oos_uplift=0.01,
        risk_hits=2,
        invariant_breaches=1,
        budget_remaining=5.0,
        budget_utilisation=0.95,
    )

    decision = config.evaluate(evaluation)

    assert decision.auto_applied is False
    reasons = decision.reasons
    assert any(reason.startswith("uplift_below_threshold") for reason in reasons)
    assert "risk_hits_exceeded:2>0" in reasons
    assert "budget_exhausted" in reasons
    assert any(reason.startswith("budget_over_utilised") for reason in reasons)
    assert "invariants_breached:1" in reasons


def test_auto_apply_rule_requires_invariant_context() -> None:
    config = AutoApplyRuleConfig(uplift_threshold=0.0)
    evaluation = ProposalEvaluation(
        suggestion_id="rim-unknown",
        oos_uplift=0.12,
        risk_hits=0,
        invariant_breaches=None,
        budget_remaining=20.0,
        budget_utilisation=0.25,
    )

    decision = config.evaluate(evaluation)

    assert decision.auto_applied is False
    assert "invariants_unknown" in decision.reasons


def test_auto_apply_rule_requires_budget_context() -> None:
    config = AutoApplyRuleConfig(uplift_threshold=0.0)
    evaluation = ProposalEvaluation(
        suggestion_id="rim-budget",
        oos_uplift=0.12,
        risk_hits=0,
        invariant_breaches=0,
        budget_remaining=25.0,
        budget_utilisation=None,
    )

    decision = config.evaluate(evaluation)

    assert decision.auto_applied is False
    assert "budget_unknown" in decision.reasons


def test_publish_governance_artifacts_marks_auto_applied(tmp_path) -> None:
    now = datetime.now(tz=UTC)
    window = RIMWindow(
        start=now - timedelta(minutes=60),
        end=now,
        minutes=60,
    )

    suggestions = [
        {
            "suggestion_id": "rim-abc",
            "type": "WEIGHT_ADJUST",
            "confidence": 0.91,
            "payload": {"strategy_id": "mean_rev_v1"},
            "rationale": "Positive replay uplift",
        }
    ]

    evaluations = [
        ProposalEvaluation(
            suggestion_id="rim-abc",
            oos_uplift=0.12,
            risk_hits=0,
            invariant_breaches=0,
            budget_remaining=12.0,
            budget_utilisation=0.5,
        )
    ]

    config = AutoApplyRuleConfig(uplift_threshold=0.05)

    queue_path = tmp_path / "queue.jsonl"
    digest_path = tmp_path / "digest.json"
    markdown_path = tmp_path / "digest.md"

    publish_governance_artifacts(
        suggestions,
        run_id="run-001",
        run_timestamp=now,
        window=window,
        input_hash="input-hash",
        model_hash="model-hash",
        config_hash="config-hash",
        queue_path=queue_path,
        digest_path=digest_path,
        markdown_path=markdown_path,
        proposal_evaluations=evaluations,
        auto_apply_config=config,
    )

    queue_lines = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert queue_lines, "expected queue entry"
    governance_payload = queue_lines[0]["governance"]
    assert governance_payload["status"] == "auto_applied"
    auto_apply_payload = governance_payload.get("auto_apply")
    assert auto_apply_payload["applied"] is True
    assert auto_apply_payload["reasons"] == []
    assert auto_apply_payload["evaluation"]["oos_uplift"] == pytest.approx(0.12)
    assert auto_apply_payload["evaluation"]["invariant_breaches"] == 0

    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    auto_apply_summary = digest.get("auto_apply")
    assert auto_apply_summary["auto_applied"] == 1
    assert auto_apply_summary["applied_suggestions"] == ["rim-abc"]

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Auto-apply summary" in markdown
    assert "Auto-applied: 1" in markdown
