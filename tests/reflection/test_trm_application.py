from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Mapping

from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from src.governance.policy_ledger import LedgerReleaseManager
from src.reflection.trm.application import apply_auto_applied_suggestions_to_ledger


def _write_queue_lines(path: Path, suggestions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(entry) for entry in suggestions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_queue_entry(
    strategy_id: str,
    *,
    suggestion_id: str,
    delta: float,
    evaluation_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    run_id = "rim-run-001"
    applied_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    trace_payload = {
        "code_hash": "test-code",
        "config_hash": "cfg-hash",
        "model_hash": "model-hash",
        "batch_input_hash": "batch-hash",
        "target_strategy_ids": [strategy_id],
        "diary_slice": {
            "source_path": "artifacts/diaries/diaries-0001.jsonl",
            "window": {"start": timestamp, "end": timestamp, "minutes": 60},
            "aggregates": {"total_entries": 1},
            "entry_count": 1,
            "strategy_entry_count": 1,
            "strategy_entries": [
                {
                    "timestamp": timestamp,
                    "strategy_id": strategy_id,
                    "instrument": "test-instrument",
                    "pnl": 0.0,
                    "action": "hold",
                    "input_hash": "entry-001",
                    "risk_flags": [],
                    "outcome_labels": [],
                    "raw": {"strategy_id": strategy_id},
                }
            ],
        },
    }
    entry = {
        "suggestion_id": suggestion_id,
        "type": "WEIGHT_ADJUST",
        "input_hash": "batch-hash",
        "model_hash": "model-hash",
        "config_hash": "cfg-hash",
        "payload": {
            "strategy_id": strategy_id,
            "proposed_weight_delta": delta,
            "window_minutes": 1440,
        },
        "confidence": 0.83,
        "audit_ids": ["diary-001"],
        "affected_regimes": ["calm"],
        "evidence": {
            "input_hash": "batch-hash",
            "audit_ids": ["diary-001"],
            "target_strategy_ids": [strategy_id],
            "window": {"start": timestamp, "end": timestamp, "minutes": 60},
            "diary_source": "artifacts/diaries/diaries-0001.jsonl",
        },
        "trace": trace_payload,
        "governance": {
            "queue": "reflection.trm",
            "status": "auto_applied",
            "run_id": run_id,
            "applied_at": applied_at,
            "auto_apply": {
                "applied": True,
                "reasons": [],
                "evaluation": {
                    "suggestion_id": suggestion_id,
                    "oos_uplift": 0.12,
                    "risk_hits": 0,
                    "budget_remaining": 25.0,
                    "budget_utilisation": 0.25,
                },
            },
        },
    }

    if evaluation_overrides:
        evaluation = entry["governance"]["auto_apply"]["evaluation"]
        evaluation.update(dict(evaluation_overrides))

    return entry


def test_apply_auto_applied_suggestions_records_metadata(tmp_path: Path) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    queue_path = tmp_path / "queue" / "reflection_queue.jsonl"

    store = PolicyLedgerStore(ledger_path)
    release_manager = LedgerReleaseManager(store)
    store.upsert(
        policy_id="bootstrap-strategy",
        tactic_id="bootstrap-strategy",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="diary-001",
    )

    suggestion = _build_queue_entry("bootstrap-strategy", suggestion_id="rim-abc", delta=-0.07)
    rejected = _build_queue_entry("bootstrap-strategy", suggestion_id="rim-reject", delta=-0.02)
    rejected_governance = rejected["governance"]
    rejected_governance["status"] = "pending"
    rejected_governance.pop("applied_at", None)
    auto_apply_block = rejected_governance["auto_apply"]
    auto_apply_block["applied"] = False
    auto_apply_block["reasons"] = ["risk_hits_exceeded:2>0"]
    evaluation_block = auto_apply_block.get("evaluation", {})
    evaluation_block["risk_hits"] = 2
    evaluation_block["oos_uplift"] = 0.05
    auto_apply_block["evaluation"] = evaluation_block
    _write_queue_lines(queue_path, [suggestion, rejected])

    applied = apply_auto_applied_suggestions_to_ledger(
        queue_path,
        store,
        release_manager=release_manager,
    )

    assert applied == ("rim-abc",)

    record = store.get("bootstrap-strategy")
    assert record is not None
    assert "rim-auto" in record.approvals
    key = "rim_weight_delta__bootstrap-strategy"
    assert record.threshold_overrides.get(key) == -0.07
    metadata = record.metadata.get("rim_auto_apply") if record.metadata else None
    assert metadata is not None
    assert "rim-abc" in metadata
    payload = metadata["rim-abc"]
    assert payload["type"] == "WEIGHT_ADJUST"
    assert payload["auto_apply"]["applied"] is True
    assert payload["payload"]["proposed_weight_delta"] == -0.07
    assert payload.get("affected_regimes") == ["calm"]
    evidence = payload.get("evidence")
    assert evidence is not None
    assert evidence.get("input_hash") == "batch-hash"
    assert evidence.get("audit_ids") == ["diary-001"]
    assert evidence.get("target_strategy_ids") == ["bootstrap-strategy"]
    window = evidence.get("window")
    assert isinstance(window, dict)
    assert window.get("minutes") == 60
    assert evidence.get("diary_source") == "artifacts/diaries/diaries-0001.jsonl"
    trace = payload.get("trace")
    assert trace is not None
    assert trace.get("code_hash") == "test-code"
    diary_slice = trace.get("diary_slice")
    assert diary_slice is not None and diary_slice.get("strategy_entries")
    assert record.accepted_proposals == ("rim-abc",)

    rejections = record.metadata.get("rim_auto_apply_rejections") if record.metadata else None
    assert rejections is not None
    rejection_payload = rejections.get("rim-reject")
    assert rejection_payload is not None
    assert rejection_payload["auto_apply"]["reasons"] == ["risk_hits_exceeded:2>0"]
    assert record.rejected_proposals == ("rim-reject",)

    # Re-running should be idempotent.
    reapplied = apply_auto_applied_suggestions_to_ledger(
        queue_path,
        store,
        release_manager=release_manager,
    )
    assert reapplied == ()


def test_auto_apply_skips_without_budget_context(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    queue_path = tmp_path / "queue" / "reflection_queue.jsonl"

    store = PolicyLedgerStore(ledger_path)
    release_manager = LedgerReleaseManager(store)
    store.upsert(
        policy_id="bootstrap-strategy",
        tactic_id="bootstrap-strategy",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="diary-001",
    )

    suggestion = _build_queue_entry("bootstrap-strategy", suggestion_id="rim-no-budget", delta=-0.05)
    evaluation = suggestion["governance"]["auto_apply"]["evaluation"]
    evaluation.pop("budget_remaining", None)
    evaluation.pop("budget_utilisation", None)

    _write_queue_lines(queue_path, [suggestion])

    applied = apply_auto_applied_suggestions_to_ledger(
        queue_path,
        store,
        release_manager=release_manager,
    )

    assert applied == ()

    record = store.get("bootstrap-strategy")
    assert record is not None
    assert record.accepted_proposals == ()
    metadata = record.metadata.get("rim_auto_apply") if record.metadata else None
    assert not metadata or "rim-no-budget" not in metadata


def test_auto_apply_skips_when_risk_hits_reported(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    queue_path = tmp_path / "queue" / "reflection_queue.jsonl"

    store = PolicyLedgerStore(ledger_path)
    release_manager = LedgerReleaseManager(store)
    store.upsert(
        policy_id="bootstrap-strategy",
        tactic_id="bootstrap-strategy",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="diary-001",
    )

    suggestion = _build_queue_entry(
        "bootstrap-strategy",
        suggestion_id="rim-risk-hit",
        delta=-0.05,
        evaluation_overrides={"risk_hits": 2},
    )

    _write_queue_lines(queue_path, [suggestion])

    applied = apply_auto_applied_suggestions_to_ledger(
        queue_path,
        store,
        release_manager=release_manager,
    )

    assert applied == ()

    record = store.get("bootstrap-strategy")
    assert record is not None
    assert record.accepted_proposals == ()
    metadata = record.metadata.get("rim_auto_apply") if record.metadata else None
    assert not metadata or "rim-risk-hit" not in metadata
