from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from src.governance.promotion import (
    PromotionFeatureFlags,
    promote_manifest_to_registry,
)
from src.governance.promotion_integrity import PromotionGuard
from src.governance.strategy_registry import StrategyRegistry, StrategyStatus
from src.understanding.decision_diary import DecisionDiaryStore


def _write_manifest(path: Path, fitness: float = 4.2) -> Path:
    payload = {
        "experiment": "ma_crossover_ga",
        "generated_at": "2025-09-28T19:01:38Z",
        "seed": 2025,
        "config": {"generations": 18, "population_size": 24},
        "dataset": {"name": "synthetic", "metadata": {"length": 64}},
        "best_genome": {
            "short_window": 8,
            "long_window": 64,
            "risk_fraction": 0.35,
            "use_var_guard": True,
            "use_drawdown_guard": True,
        },
        "best_metrics": {
            "fitness": fitness,
            "sharpe": 3.9,
            "sortino": 5.7,
            "max_drawdown": 0.01,
            "total_return": 0.12,
        },
    }
    manifest_path = path / "manifest.json"
    manifest_path.write_text(json.dumps(payload))
    return manifest_path


def _expected_genome_id() -> str:
    return "ma_crossover_ga::2025::s8::l64::r350"


def _promotion_guard(tmp_path: Path, policy_id: str) -> PromotionGuard:
    ledger_path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id=f"dd-{policy_id}-promotion",
    )
    diary_path = tmp_path / "decision_diary.json"
    diary = DecisionDiaryStore(diary_path, publish_on_record=False)
    regimes = ("balanced", "bullish", "bearish")
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    for index, regime in enumerate(regimes):
        recorded_at = base + timedelta(minutes=index * 7)
        diary.record(
            policy_id=policy_id,
            decision={
                "tactic_id": policy_id,
                "parameters": {},
                "selected_weight": 1.0,
                "guardrails": {},
                "rationale": "promotion-check",
                "experiments_applied": (),
                "reflection_summary": {},
                "weight_breakdown": {},
            },
            regime_state={
                "regime": regime,
                "confidence": 0.75,
                "features": {},
                "timestamp": recorded_at.isoformat(),
            },
            outcomes={"paper_pnl": 0.0},
            metadata={
                "release_stage": PolicyLedgerStage.PAPER.value,
                "release_execution": {"stage": PolicyLedgerStage.PAPER.value, "route": "paper"},
            },
            recorded_at=recorded_at,
        )
    return PromotionGuard(
        ledger_path=ledger_path,
        diary_path=diary_path,
        required_regimes=regimes,
        min_decisions_per_regime=1,
    )


def _registry(tmp_path: Path, policy_id: str | None = None) -> StrategyRegistry:
    guard = _promotion_guard(tmp_path, policy_id or _expected_genome_id())
    return StrategyRegistry(
        db_path=str(tmp_path / "registry.db"),
        promotion_guard=guard,
    )


def test_promotion_skipped_when_flag_disabled(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    registry = _registry(tmp_path)
    flags = PromotionFeatureFlags(register_enabled=False)

    result = promote_manifest_to_registry(manifest, registry, flags=flags)

    assert result.skipped is True
    assert result.registered is False
    summary = registry.get_registry_summary()
    assert summary["total_strategies"] == 0


def test_promotion_registers_genome(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    registry = _registry(tmp_path)
    flags = PromotionFeatureFlags(register_enabled=True, min_fitness=0.0)

    result = promote_manifest_to_registry(manifest, registry, flags=flags)

    assert result.registered is True
    assert result.skipped is False
    assert result.genome_id is not None

    stored = registry.get_strategy(result.genome_id)
    assert stored is not None
    assert stored["status"] == StrategyStatus.EVOLVED.value
    metadata = stored["fitness_report"].get("metadata", {})
    assert metadata.get("experiment") == "ma_crossover_ga"


def test_promotion_auto_approve(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    registry = _registry(tmp_path)
    flags = PromotionFeatureFlags(
        register_enabled=True,
        auto_approve=True,
        target_status=StrategyStatus.APPROVED,
    )

    result = promote_manifest_to_registry(manifest, registry, flags=flags)

    assert result.status_updated is True
    stored = registry.get_strategy(result.genome_id or "")
    assert stored is not None
    assert stored["status"] == StrategyStatus.APPROVED.value


def test_promotion_respects_min_fitness(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, fitness=0.1)
    registry = _registry(tmp_path)
    flags = PromotionFeatureFlags(register_enabled=True, min_fitness=1.0)

    result = promote_manifest_to_registry(manifest, registry, flags=flags)

    assert result.skipped is True
    assert result.reason == "fitness_below_threshold"
    summary = registry.get_registry_summary()
    assert summary["total_strategies"] == 0
