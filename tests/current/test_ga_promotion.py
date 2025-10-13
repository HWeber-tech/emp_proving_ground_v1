from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.evolution.experiments.promotion import promote_ma_crossover_champion
from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from src.governance.promotion_integrity import PromotionGuard
from src.governance.strategy_registry import StrategyRegistry, StrategyStatus
from src.understanding.decision_diary import DecisionDiaryStore


@pytest.fixture()
def sample_manifest(tmp_path: Path) -> Path:
    payload = {
        "experiment": "ma_crossover_ga",
        "seed": 2025,
        "config": {
            "population_size": 12,
            "generations": 5,
            "elite_count": 3,
            "crossover_rate": 0.7,
            "mutation_rate": 0.25,
        },
        "dataset": {
            "name": "unit_test_series",
            "metadata": {"length": 64},
        },
        "best_genome": {
            "short_window": 8,
            "long_window": 55,
            "risk_fraction": 0.24,
            "use_var_guard": True,
            "use_drawdown_guard": True,
        },
        "best_metrics": {
            "fitness": 4.25,
            "sharpe": 2.15,
            "sortino": 2.8,
            "max_drawdown": 0.08,
            "total_return": 0.31,
        },
        "leaderboard": [
            {"generation": 1},
            {"generation": 2},
            {"generation": 3},
        ],
        "replay": {
            "command": "python scripts/generate_evolution_lab.py --seed 2025",
            "seed": 2025,
        },
        "notes": "unit-test manifest",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _resolve_genome_id(manifest_path: Path) -> str:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    genome = payload.get("best_genome", {})
    experiment = str(payload.get("experiment", "ma_crossover_ga"))
    seed = payload.get("seed", "seed")
    short_window = int(genome.get("short_window", 0))
    long_window = int(genome.get("long_window", 0))
    risk_fraction = float(genome.get("risk_fraction", 0.0))
    window_part = f"{short_window}-{long_window}-{risk_fraction:.3f}"
    return "::".join((experiment, str(seed), window_part))


def _promotion_guard(
    tmp_path: Path,
    policy_id: str,
    *,
    stage: PolicyLedgerStage = PolicyLedgerStage.PAPER,
    regimes: tuple[str, ...] = ("balanced", "bullish", "bearish"),
) -> PromotionGuard:
    ledger_path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=stage,
        approvals=("risk", "compliance"),
        evidence_id=f"dd-{policy_id}-promotion",
    )
    diary_path = tmp_path / "decision_diary.json"
    diary = DecisionDiaryStore(diary_path, publish_on_record=False)
    base = datetime(2024, 3, 1, tzinfo=timezone.utc)
    for index, regime in enumerate(regimes):
        recorded_at = base + timedelta(minutes=index * 5)
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
                "confidence": 0.78,
                "features": {},
                "timestamp": recorded_at.isoformat(),
            },
            outcomes={"paper_pnl": 0.0},
            metadata={
                "release_stage": stage.value,
                "release_execution": {
                    "stage": stage.value,
                    "route": "live" if stage is PolicyLedgerStage.LIMITED_LIVE else "paper",
                },
            },
            recorded_at=recorded_at,
        )
    return PromotionGuard(
        ledger_path=ledger_path,
        diary_path=diary_path,
        required_regimes=regimes,
        min_decisions_per_regime=1,
    )


def _build_registry(
    tmp_path: Path,
    policy_id: str,
    *,
    stage: PolicyLedgerStage = PolicyLedgerStage.PAPER,
) -> StrategyRegistry:
    guard = _promotion_guard(tmp_path, policy_id, stage=stage)
    db_path = tmp_path / "registry.db"
    return StrategyRegistry(db_path=str(db_path), promotion_guard=guard)


def test_promote_ga_champion_registers_evolved_by_default(
    sample_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PAPER_TRADE_GA_MA_CROSSOVER", raising=False)
    policy_id = _resolve_genome_id(sample_manifest)
    registry = _build_registry(tmp_path, policy_id)
    try:
        result = promote_ma_crossover_champion(sample_manifest, registry)
        assert result.registered is True
        assert result.target_status is None

        record = registry.get_strategy(result.genome_id or "")
        assert record is not None
        assert record["status"] == StrategyStatus.EVOLVED.value
        report = record["fitness_report"]
        assert isinstance(report, dict)
        metadata = report.get("metadata", {})
        assert metadata["feature_flag"]["name"] == "PAPER_TRADE_GA_MA_CROSSOVER"
        assert metadata["feature_flag"].get("value") is None
        assert metadata["feature_flag"].get("target_status") is None
    finally:
        registry.close()


def test_promote_ga_champion_honours_approved_flag(
    sample_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PAPER_TRADE_GA_MA_CROSSOVER", "approved")
    policy_id = _resolve_genome_id(sample_manifest)
    registry = _build_registry(tmp_path, policy_id)
    try:
        result = promote_ma_crossover_champion(sample_manifest, registry)
        assert result.target_status == StrategyStatus.APPROVED
        assert result.status_applied is True

        record = registry.get_strategy(result.genome_id or "")
        assert record is not None
        assert record["status"] == StrategyStatus.APPROVED.value
    finally:
        registry.close()


def test_promote_ga_champion_honours_active_flag(
    sample_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PAPER_TRADE_GA_MA_CROSSOVER", "active")
    policy_id = _resolve_genome_id(sample_manifest)
    registry = _build_registry(
        tmp_path,
        policy_id,
        stage=PolicyLedgerStage.LIMITED_LIVE,
    )
    try:
        result = promote_ma_crossover_champion(sample_manifest, registry)
        assert result.target_status == StrategyStatus.ACTIVE
        assert result.status_applied is True

        record = registry.get_strategy(result.genome_id or "")
        assert record is not None
        assert record["status"] == StrategyStatus.ACTIVE.value
    finally:
        registry.close()
