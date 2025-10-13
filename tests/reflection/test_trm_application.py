from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from src.governance.policy_ledger import LedgerReleaseManager
from src.reflection.trm.application import apply_auto_applied_suggestions_to_ledger


def _write_queue_lines(path: Path, suggestions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(entry) for entry in suggestions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_queue_entry(strategy_id: str, *, suggestion_id: str, delta: float) -> dict[str, object]:
    run_id = "rim-run-001"
    applied_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return {
        "suggestion_id": suggestion_id,
        "type": "WEIGHT_ADJUST",
        "payload": {
            "strategy_id": strategy_id,
            "proposed_weight_delta": delta,
            "window_minutes": 1440,
        },
        "confidence": 0.83,
        "audit_ids": ["diary-001"],
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
                },
            },
        },
    }


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
    pending = {
        "suggestion_id": "rim-pending",
        "type": "WEIGHT_ADJUST",
        "payload": {"strategy_id": "bootstrap-strategy", "proposed_weight_delta": -0.02},
        "confidence": 0.6,
        "governance": {"status": "pending"},
    }
    _write_queue_lines(queue_path, [suggestion, pending])

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

    # Re-running should be idempotent.
    reapplied = apply_auto_applied_suggestions_to_ledger(
        queue_path,
        store,
        release_manager=release_manager,
    )
    assert reapplied == ()
