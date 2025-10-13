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
    changelog_path = tmp_path / "governance.md"
    exit_code = main(
        [
            "--ledger",
            str(ledger_path),
            "--output",
            str(output_path),
            "--indent",
            "0",
            "--changelog",
            str(changelog_path),
            "--runbook-url",
            "https://runbooks/policy",
        ]
    )
    assert exit_code == 0

    payload = json.loads(output_path.read_text())
    assert payload["policy_count"] == 1
    assert "alpha.policy" in payload["policies"]
    policy_payload = payload["policies"]["alpha.policy"]
    assert policy_payload["risk_config"]["max_leverage"] == 6.0
    assert payload["governance_workflow"]["status"] in {"completed", "in_progress"}
    assert "phenotype_paths" not in payload

    changelog = changelog_path.read_text()
    assert "alpha.policy" in changelog
    assert "https://runbooks/policy" in changelog


def test_rebuild_policy_cli_filters_and_emits_phenotype(tmp_path: Path) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(ledger_path)
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id="diary-123",
        metadata={"policy_hash": "hash-alpha"},
    )
    manager.promote(
        policy_id="beta.policy",
        tactic_id="tactic.beta",
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk", "ops"),
        evidence_id="diary-456",
        metadata={"policy_hash": "hash-beta"},
        policy_delta={"risk_config": {"max_leverage": 9}},
    )

    phenotype_dir = tmp_path / "phenotypes"
    summary_path = tmp_path / "summary.json"
    exit_code = main(
        [
            "--ledger",
            str(ledger_path),
            "--policy",
            "hash-beta",
            "--phenotype-dir",
            str(phenotype_dir),
            "--output",
            str(summary_path),
            "--indent",
            "0",
        ]
    )
    assert exit_code == 0

    payload = json.loads(summary_path.read_text())
    assert payload["policy_count"] == 1
    assert list(payload["policies"].keys()) == ["beta.policy"]
    assert payload["policies"]["beta.policy"]["risk_config"]["max_leverage"] == 9.0
    phenotype_paths = payload.get("phenotype_paths")
    assert phenotype_paths is not None
    assert len(phenotype_paths) == 1

    phenotype_path = Path(phenotype_paths[0])
    assert phenotype_path.exists()
    phenotype_payload = json.loads(phenotype_path.read_text())
    assert phenotype_payload["policy_id"] == "beta.policy"
