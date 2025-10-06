from __future__ import annotations

import json
from pathlib import Path

from tools.governance.rebuild_policy import main

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage, PolicyLedgerStore


def test_rebuild_policy_cli_outputs_payload(tmp_path: Path) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(ledger_path)
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id="diary-123",
        policy_delta={"risk_config": {"max_leverage": 6}},
    )

    output_path = tmp_path / "rebuild.json"
    exit_code = main(
        [
            "--ledger",
            str(ledger_path),
            "--output",
            str(output_path),
            "--indent",
            "0",
        ]
    )
    assert exit_code == 0

    payload = json.loads(output_path.read_text())
    assert payload["policy_count"] == 1
    assert "alpha.policy" in payload["policies"]
    policy_payload = payload["policies"]["alpha.policy"]
    assert policy_payload["risk_config"]["max_leverage"] == 6.0
    assert payload["governance_workflow"]["status"] in {"completed", "in_progress"}
