from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, MutableMapping, cast

import pytest

from src.config.risk.risk_config import RiskConfig
from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyDelta,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.governance.policy_phenotype import build_policy_phenotypes
from src.governance.strategy_rebuilder import StrategyRuntimeConfig, rebuild_strategy


def _seed_policy(
    ledger_path: Path,
    *,
    stage: PolicyLedgerStage = PolicyLedgerStage.PAPER,
    approvals: tuple[str, ...] = ("risk", "compliance"),
) -> tuple[PolicyLedgerStore, str]:
    store = PolicyLedgerStore(ledger_path)
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=stage,
        approvals=approvals,
        evidence_id="diary-alpha-001",
        policy_delta=PolicyDelta(
            risk_config={
                "max_leverage": 7,
                "max_risk_per_trade_pct": 0.04,
            },
            router_guardrails={
                "max_latency_ms": 180,
            },
        ),
        metadata={
            "runtime": {
                "strategy_id": "alpha.policy",
                "deployment": "paper",
            }
        },
    )

    phenotype = build_policy_phenotypes(store)[0]
    return store, phenotype.policy_hash


def test_rebuild_strategy_returns_deterministic_payload(tmp_path: Path) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    store, policy_hash = _seed_policy(ledger_path)

    first = rebuild_strategy(policy_hash, store=store)
    second = rebuild_strategy(policy_hash, store=store)

    assert isinstance(first, StrategyRuntimeConfig)
    assert first.policy_id == "alpha.policy"
    assert first.tactic_id == "tactic.alpha"
    assert first.stage is PolicyLedgerStage.PAPER
    assert set(first.approvals) == {"risk", "compliance"}
    assert first.risk_config["max_leverage"] == pytest.approx(7.0)
    assert first.risk_config["max_risk_per_trade_pct"] == pytest.approx(0.04)
    assert first.router_guardrails["max_latency_ms"] == 180

    assert first.json_bytes == second.json_bytes == first.json().encode("utf-8")
    assert first.digest == second.digest

    payload = json.loads(first.json())
    assert payload["policy_hash"] == policy_hash
    assert payload["stage"] == PolicyLedgerStage.PAPER.value
    assert payload["risk_config"]["max_leverage"] == pytest.approx(7.0)


def test_rebuild_strategy_honours_base_config_and_guardrails(tmp_path: Path) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(ledger_path)
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="beta.policy",
        tactic_id="tactic.beta",
        stage=PolicyLedgerStage.PAPER,
        approvals=("ops", "risk"),
        evidence_id="diary-beta-001",
        policy_delta=PolicyDelta(
            risk_config={"max_leverage": 9},
            router_guardrails={"max_latency_ms": 150},
        ),
    )

    base_config = RiskConfig(
        max_leverage=Decimal("5"),
        max_total_exposure_pct=Decimal("0.40"),
        max_drawdown_pct=Decimal("0.35"),
    )
    default_guardrails = {
        "max_latency_ms": 250,
        "max_orders_per_minute": 120,
    }

    phenotype = build_policy_phenotypes(
        store,
        base_config=base_config,
        default_guardrails=default_guardrails,
    )[0]

    config = rebuild_strategy(
        phenotype.policy_hash,
        store=store,
        base_config=base_config,
        default_guardrails=default_guardrails,
    )

    assert config.policy_id == "beta.policy"
    assert config.risk_config["max_total_exposure_pct"] == pytest.approx(0.40)
    assert config.risk_config["max_leverage"] == pytest.approx(9.0)
    assert config.router_guardrails["max_latency_ms"] == 150
    assert config.router_guardrails["max_orders_per_minute"] == 120


def test_strategy_runtime_config_canonical_output_resists_mutation(tmp_path: Path) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    store, policy_hash = _seed_policy(ledger_path)

    config = rebuild_strategy(policy_hash, store=store)

    canonical_json = config.json()
    canonical_dict = json.loads(canonical_json)

    payload_mutable = cast(MutableMapping[str, Any], config.payload)
    risk_config_mutable = cast(MutableMapping[str, Any], payload_mutable["risk_config"])
    risk_config_mutable["max_leverage"] = 42

    assert payload_mutable["risk_config"]["max_leverage"] == 42
    assert config.json() == canonical_json
    pretty_expected = json.dumps(canonical_dict, sort_keys=True, indent=2, ensure_ascii=False)
    assert config.json(indent=2) == pretty_expected
    assert config.as_dict() == canonical_dict


def test_rebuild_strategy_unknown_hash(tmp_path: Path) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    store, policy_hash = _seed_policy(ledger_path)

    assert policy_hash
    with pytest.raises(LookupError):
        rebuild_strategy("deadbeef", store=store)
