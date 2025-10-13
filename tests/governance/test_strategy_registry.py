from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from src.governance.promotion_integrity import PromotionGuard
from src.governance.strategy_registry import (
    StrategyRegistry,
    StrategyRegistryError,
)
from src.understanding.decision_diary import DecisionDiaryStore


def _promotion_guard(
    tmp_path,
    policy_id: str,
    *,
    stage: PolicyLedgerStage = PolicyLedgerStage.PAPER,
    regimes: tuple[str, ...] = ("balanced", "bullish", "bearish"),
    min_decisions: int = 1,
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
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for index, regime in enumerate(regimes):
        for offset in range(min_decisions):
            recorded_at = base + timedelta(minutes=index * 10 + offset)
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
                    "confidence": 0.8,
                    "features": {},
                    "timestamp": recorded_at.isoformat(),
                },
                outcomes={"paper_pnl": 0.0},
                metadata={
                    "release_stage": stage.value,
                    "release_execution": {
                        "stage": stage.value,
                        "route": "paper" if stage is not PolicyLedgerStage.LIMITED_LIVE else "live",
                    },
                },
                recorded_at=recorded_at,
            )
    return PromotionGuard(
        ledger_path=ledger_path,
        diary_path=diary_path,
        required_regimes=regimes,
        min_decisions_per_regime=min_decisions,
    )


def _build_provenance() -> dict[str, object]:
    return {
        "seed_source": "catalogue",
        "generation": 4,
        "population_size": 6,
        "catalogue": {
            "name": "institutional_default",
            "version": "2025.09",
            "size": 5,
            "species": {"trend": 3, "carry": 2},
            "seeded_at": 1_726_000_000.0,
            "source_notes": ["calibrated desk catalogue"],
        },
        "entry": {
            "id": "catalogue/trend-alpha",
            "name": "Trend Surfer Alpha",
            "species": "trend_rider",
            "generation": 4,
            "tags": ["fx", "swing"],
            "performance_metrics": {"sharpe_ratio": 1.3},
        },
    }


@pytest.mark.parametrize("status", ["evolved", "approved"])
def test_strategy_registry_records_catalogue_provenance(tmp_path, status: str) -> None:
    guard = _promotion_guard(tmp_path, "catalogue/trend-alpha::0")
    registry = StrategyRegistry(db_path=str(tmp_path / "registry.db"), promotion_guard=guard)
    genome = SimpleNamespace(
        id="catalogue/trend-alpha::0",
        decision_tree={"nodes": 12},
        name="trend-alpha",
        generation=4,
    )
    fitness_report = {
        "fitness_score": 1.42,
        "max_drawdown": -0.11,
        "sharpe_ratio": 1.2,
        "total_return": 0.22,
        "volatility": 0.05,
        "metadata": {},
    }

    provenance = _build_provenance()
    assert registry.register_champion(genome, dict(fitness_report), provenance=provenance)

    if status != "evolved":
        assert registry.update_strategy_status(genome.id, status)

    stored = registry.get_strategy(genome.id)
    assert stored is not None
    assert stored["seed_source"] == "catalogue"
    assert stored["catalogue_name"] == provenance["catalogue"]["name"]
    assert stored["catalogue_entry_id"] == provenance["entry"]["id"]
    assert stored["status"] == status
    report_metadata = stored["fitness_report"].get("metadata", {})
    assert report_metadata["catalogue_provenance"]["catalogue"]["version"] == "2025.09"

    summary = registry.get_registry_summary()
    assert summary["total_strategies"] == 1
    if status == "approved":
        assert summary["approved_count"] == 1
    else:
        assert summary["evolved_count"] == 1
    assert summary["catalogue_seeded"] == 1
    assert summary["catalogue_entry_count"] == 1
    assert summary["catalogue_missing_provenance"] == 0
    assert summary["seed_source_counts"].get("catalogue") == 1


def test_strategy_registry_surfaces_database_errors(tmp_path) -> None:
    invalid_path = tmp_path / "registry_dir"
    invalid_path.mkdir()

    with pytest.raises(StrategyRegistryError):
        StrategyRegistry(db_path=str(invalid_path))
