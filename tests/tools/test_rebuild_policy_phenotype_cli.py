from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from tools.governance.rebuild_policy_phenotype import main

from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.governance.policy_phenotype import build_policy_phenotypes
from src.governance.strategy_rebuilder import rebuild_strategy


def _seed_policy(ledger_path: Path) -> None:
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


def test_rebuild_policy_phenotype_cli_outputs_payload(tmp_path: Path) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    _seed_policy(ledger_path)

    store = PolicyLedgerStore(ledger_path)
    phenotype = build_policy_phenotypes(store)[0]

    output_path = tmp_path / "phenotype.json"
    runtime_output = tmp_path / "runtime_config.json"
    exit_code = main(
        [
            "--ledger",
            str(ledger_path),
            "--policy-id",
            phenotype.policy_id,
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0

    expected_config = rebuild_strategy(phenotype.policy_hash, store=store)
    output_bytes = output_path.read_bytes()
    assert output_bytes == expected_config.json_bytes

    payload = json.loads(output_bytes.decode("utf-8"))
    assert payload["policy_id"] == phenotype.policy_id
    assert payload["policy_hash"] == phenotype.policy_hash
    assert payload["risk_config"]["max_leverage"] == pytest.approx(6.0)
    assert payload["router_guardrails"]["max_latency_ms"] == 180
    assert sha256(output_bytes).hexdigest() == expected_config.digest

    assert payload["runtime_digest"]
    assert payload["runtime_config"]["policy_hash"] == phenotype["policy_hash"]
    assert payload["runtime_config"]["risk_config"]["max_leverage"] == 6.0
    assert payload["runtime_config_path"] == str(runtime_output)

    runtime_bytes = runtime_output.read_bytes()
    assert hashlib.sha256(runtime_bytes).hexdigest() == payload["runtime_digest"]
    runtime_payload = json.loads(runtime_bytes.decode("utf-8"))
    assert runtime_payload == payload["runtime_config"]

    hash_output = tmp_path / "phenotype_by_hash.json"
    exit_code = main(
        [
            "--ledger",
            str(ledger_path),
            "--policy-hash",
            phenotype.policy_hash,
            "--output",
            str(hash_output),
        ]
    )
    assert exit_code == 0
    assert hash_output.read_bytes() == expected_config.json_bytes


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
