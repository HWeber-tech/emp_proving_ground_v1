from __future__ import annotations

import json
from pathlib import Path

from tools.governance.rebuild_policy_phenotype import main

from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerStage,
    PolicyLedgerStore,
)


def _seed_policy(ledger_path: Path) -> str:
    store = PolicyLedgerStore(ledger_path)
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id="diary-101",
        policy_delta={
            "risk_config": {"max_leverage": 6},
            "router_guardrails": {"max_latency_ms": 180},
        },
    )
    return "alpha.policy"


def test_rebuild_policy_phenotype_cli_outputs_payload(tmp_path: Path) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    policy_id = _seed_policy(ledger_path)

    output_path = tmp_path / "phenotype.json"
    exit_code = main(
        [
            "--ledger",
            str(ledger_path),
            "--policy-id",
            policy_id,
            "--output",
            str(output_path),
            "--indent",
            "0",
        ]
    )
    assert exit_code == 0

    payload = json.loads(output_path.read_text())
    phenotype = payload["phenotype"]
    assert phenotype["policy_id"] == policy_id
    assert phenotype["risk_config"]["max_leverage"] == 6.0
    assert phenotype["router_guardrails"]["max_latency_ms"] == 180
    assert phenotype["policy_hash"]

    hash_output = tmp_path / "phenotype_by_hash.json"
    exit_code = main(
        [
            "--ledger",
            str(ledger_path),
            "--policy-hash",
            phenotype["policy_hash"],
            "--output",
            str(hash_output),
            "--indent",
            "0",
        ]
    )
    assert exit_code == 0
    payload_by_hash = json.loads(hash_output.read_text())
    assert payload_by_hash["phenotype"]["policy_hash"] == phenotype["policy_hash"]


def test_rebuild_policy_phenotype_cli_list(tmp_path: Path, capsys) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    _seed_policy(ledger_path)

    exit_code = main(
        [
            "--ledger",
            str(ledger_path),
            "--list",
            "--indent",
            "0",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr().out.strip()
    listing = json.loads(captured)
    assert listing
    row = listing[0]
    assert row["policy_id"] == "alpha.policy"
    assert row["policy_hash"]

