from __future__ import annotations

from pathlib import Path

import pytest

from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyDelta,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.governance.policy_phenotype import (
    build_policy_phenotypes,
    select_policy_phenotype,
)


def _bootstrap_policy(tmp_path: Path) -> tuple[PolicyLedgerStore, Path, str]:
    ledger_path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(ledger_path)
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id="diary-1",
        policy_delta=PolicyDelta(
            risk_config={"max_leverage": 8},
            router_guardrails={"max_latency_ms": 250},
        ),
    )
    return store, ledger_path, "alpha.policy"


def test_policy_phenotype_hash_is_deterministic(tmp_path: Path) -> None:
    store, ledger_path, policy_id = _bootstrap_policy(tmp_path)
    phenotypes = build_policy_phenotypes(store)
    assert len(phenotypes) == 1
    phenotype = phenotypes[0]
    assert phenotype.policy_id == policy_id
    assert phenotype.policy_hash

    # Reload and recompute to ensure determinism.
    persisted = PolicyLedgerStore(ledger_path)
    replayed = build_policy_phenotypes(persisted)
    assert replayed[0].policy_hash == phenotype.policy_hash


def test_select_policy_phenotype_by_hash_and_id(tmp_path: Path) -> None:
    store, _, policy_id = _bootstrap_policy(tmp_path)
    phenotypes = build_policy_phenotypes(store)
    phenotype = phenotypes[0]

    by_hash = select_policy_phenotype(phenotypes, policy_hash=phenotype.policy_hash)
    assert by_hash.policy_id == policy_id

    by_id = select_policy_phenotype(phenotypes, policy_id=policy_id)
    assert by_id.policy_hash == phenotype.policy_hash

    with pytest.raises(LookupError):
        select_policy_phenotype(phenotypes, policy_hash="deadbeef")

    with pytest.raises(ValueError):
        select_policy_phenotype(phenotypes)


def test_policy_phenotype_as_dict_structure(tmp_path: Path) -> None:
    store, _, _ = _bootstrap_policy(tmp_path)
    phenotype = build_policy_phenotypes(store)[0]

    payload = phenotype.as_dict()
    assert payload["policy_id"] == "alpha.policy"
    assert payload["policy_hash"] == phenotype.policy_hash
    assert payload["stage"] == PolicyLedgerStage.PAPER.value
    assert payload["risk_config"]["max_leverage"] == 8.0
    assert payload["router_guardrails"]["max_latency_ms"] == 250
    assert payload["approvals"] == ["compliance", "risk"]
    assert payload["updated_at"]
