from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from src.governance.promotion_integrity import PromotionGuard
from src.governance.strategy_registry import (
    StrategyRegistry,
    StrategyRegistryError,
    StrategyStatus,
)
from src.understanding.decision_diary import DecisionDiaryStore
from tests.util import promotion_checklist_metadata


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


def test_strategy_registry_config_guard_blocks_missing_regimes(tmp_path, monkeypatch) -> None:
    ledger_path = tmp_path / "policy_guard.json"
    diary_path = tmp_path / "decision_guard.json"
    config_path = tmp_path / "promotion_guard.yaml"
    config_path.write_text(
        (
            "promotion_guard:\n"
            f"  ledger_path: \"{ledger_path}\"\n"
            f"  diary_path: \"{diary_path}\"\n"
            "  stage_requirements:\n"
            "    approved: paper\n"
            "    active: limited_live\n"
            "  required_regimes:\n"
            "    - balanced\n"
            "    - bullish\n"
            "  min_decisions_per_regime: 2\n"
            "  regime_gate_statuses:\n"
            "    - approved\n"
            "    - active\n"
        ),
        encoding="utf-8",
    )

    store = PolicyLedgerStore(ledger_path)
    policy_id = "alpha-policy"
    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id=f"dd-{policy_id}-001",
    )

    diary = DecisionDiaryStore(diary_path, publish_on_record=False)
    recorded_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for offset in range(2):
        diary.record(
            policy_id=policy_id,
            decision={
                "tactic_id": policy_id,
                "parameters": {},
                "selected_weight": 1.0,
                "guardrails": {},
                "rationale": "coverage-check",
                "experiments_applied": (),
                "reflection_summary": {},
                "weight_breakdown": {},
            },
            regime_state={
                "regime": "balanced",
                "confidence": 0.75,
                "features": {},
                "timestamp": (recorded_at + timedelta(minutes=offset)).isoformat(),
            },
            outcomes={"paper_pnl": 0.0},
            metadata={
                "release_stage": PolicyLedgerStage.PAPER.value,
                "release_execution": {
                    "stage": PolicyLedgerStage.PAPER.value,
                    "route": "paper",
                },
            },
            recorded_at=recorded_at + timedelta(minutes=offset),
        )

    monkeypatch.setenv("PROMOTION_GUARD_CONFIG", str(config_path))

    registry = StrategyRegistry(db_path=str(tmp_path / "registry-config.db"))
    genome = SimpleNamespace(
        id="alpha-policy",
        decision_tree={"nodes": 4},
        name="alpha-policy",
        generation=2,
    )
    fitness_report = {
        "fitness_score": 1.1,
        "max_drawdown": 0.05,
        "sharpe_ratio": 1.5,
        "total_return": 0.12,
        "volatility": 0.03,
        "metadata": {},
    }

    assert registry.register_champion(genome, dict(fitness_report))

    with pytest.raises(StrategyRegistryError) as excinfo:
        registry.update_strategy_status(genome.id, StrategyStatus.APPROVED.value)
    message = str(excinfo.value)
    assert "missing regimes" in message
    assert "bullish" in message

    for offset in range(2):
        diary.record(
            policy_id=policy_id,
            decision={
                "tactic_id": policy_id,
                "parameters": {},
                "selected_weight": 1.0,
                "guardrails": {},
                "rationale": "coverage-check",
                "experiments_applied": (),
                "reflection_summary": {},
                "weight_breakdown": {},
            },
            regime_state={
                "regime": "bullish",
                "confidence": 0.72,
                "features": {},
                "timestamp": (
                    recorded_at + timedelta(minutes=10 + offset)
                ).isoformat(),
            },
            outcomes={"paper_pnl": 0.01},
            metadata={
                "release_stage": PolicyLedgerStage.PAPER.value,
                "release_execution": {
                    "stage": PolicyLedgerStage.PAPER.value,
                    "route": "paper",
                },
            },
            recorded_at=recorded_at + timedelta(minutes=10 + offset),
        )

    assert registry.update_strategy_status(genome.id, StrategyStatus.APPROVED.value) is True


def test_strategy_registry_guard_blocks_promoted_registration_without_coverage(
    tmp_path, monkeypatch
) -> None:
    ledger_path = tmp_path / "policy_guard.json"
    diary_path = tmp_path / "decision_guard.json"
    config_path = tmp_path / "promotion_guard.yaml"
    config_path.write_text(
        (
            "promotion_guard:\n"
            f"  ledger_path: \"{ledger_path}\"\n"
            f"  diary_path: \"{diary_path}\"\n"
            "  stage_requirements:\n"
            "    approved: paper\n"
            "    active: limited_live\n"
            "  required_regimes:\n"
            "    - balanced\n"
            "    - bullish\n"
            "  min_decisions_per_regime: 2\n"
            "  regime_gate_statuses:\n"
            "    - approved\n"
        ),
        encoding="utf-8",
    )

    store = PolicyLedgerStore(ledger_path)
    policy_id = "alpha-policy"
    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id=f"dd-{policy_id}-001",
    )

    diary = DecisionDiaryStore(diary_path, publish_on_record=False)
    recorded_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for offset in range(2):
        diary.record(
            policy_id=policy_id,
            decision={
                "tactic_id": policy_id,
                "parameters": {},
                "selected_weight": 1.0,
                "guardrails": {},
                "rationale": "coverage-check",
                "experiments_applied": (),
                "reflection_summary": {},
                "weight_breakdown": {},
            },
            regime_state={
                "regime": "balanced",
                "confidence": 0.75,
                "features": {},
                "timestamp": (recorded_at + timedelta(minutes=offset)).isoformat(),
            },
            outcomes={"paper_pnl": 0.0},
            metadata={
                "release_stage": PolicyLedgerStage.PAPER.value,
                "release_execution": {
                    "stage": PolicyLedgerStage.PAPER.value,
                    "route": "paper",
                },
            },
            recorded_at=recorded_at + timedelta(minutes=offset),
        )

    monkeypatch.setenv("PROMOTION_GUARD_CONFIG", str(config_path))

    registry = StrategyRegistry(db_path=str(tmp_path / "registry-config.db"))
    genome = SimpleNamespace(
        id=policy_id,
        decision_tree={"nodes": 4},
        name=policy_id,
        generation=2,
    )
    fitness_report = {
        "fitness_score": 1.1,
        "max_drawdown": 0.05,
        "sharpe_ratio": 1.5,
        "total_return": 0.12,
        "volatility": 0.03,
        "metadata": {},
    }

    with pytest.raises(StrategyRegistryError) as excinfo:
        registry.register_champion(
            genome,
            dict(fitness_report),
            status=StrategyStatus.APPROVED,
        )
    message = str(excinfo.value)
    assert "missing regimes" in message
    assert "bullish" in message

    for offset in range(2):
        diary.record(
            policy_id=policy_id,
            decision={
                "tactic_id": policy_id,
                "parameters": {},
                "selected_weight": 1.0,
                "guardrails": {},
                "rationale": "coverage-check",
                "experiments_applied": (),
                "reflection_summary": {},
                "weight_breakdown": {},
            },
            regime_state={
                "regime": "bullish",
                "confidence": 0.72,
                "features": {},
                "timestamp": (
                    recorded_at + timedelta(minutes=10 + offset)
                ).isoformat(),
            },
            outcomes={"paper_pnl": 0.01},
            metadata={
                "release_stage": PolicyLedgerStage.PAPER.value,
                "release_execution": {
                    "stage": PolicyLedgerStage.PAPER.value,
                    "route": "paper",
                },
            },
            recorded_at=recorded_at + timedelta(minutes=10 + offset),
        )

    assert registry.register_champion(
        genome,
        dict(fitness_report),
        status=StrategyStatus.APPROVED,
    )

    stored = registry.get_strategy(genome.id)
    assert stored is not None
    assert stored["status"] == StrategyStatus.APPROVED.value


def test_strategy_registry_guard_blocks_active_without_paper_green_span(tmp_path) -> None:
    ledger_path = tmp_path / "policy_guard.json"
    diary_path = tmp_path / "decision_guard.json"
    policy_id = "alpha-live"

    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "compliance"),
        evidence_id=f"dd-{policy_id}-pilot",
        metadata=promotion_checklist_metadata(),
    )

    diary = DecisionDiaryStore(diary_path, publish_on_record=False)
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    regimes = ("balanced", "bullish", "bearish")
    for day in range(7):
        recorded_at = base + timedelta(days=day)
        regime = regimes[day % len(regimes)]
        diary.record(
            policy_id=policy_id,
            decision={
                "tactic_id": policy_id,
                "parameters": {},
                "selected_weight": 1.0,
                "guardrails": {},
                "rationale": "span-check",
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
                "release_stage": PolicyLedgerStage.PAPER.value,
                "drift_decision": {
                    "severity": "normal",
                    "force_paper": False,
                },
                "release_execution": {
                    "stage": PolicyLedgerStage.PAPER.value,
                    "route": "paper",
                    "forced": False,
                },
            },
            recorded_at=recorded_at,
        )

    guard = PromotionGuard(
        ledger_path=ledger_path,
        diary_path=diary_path,
        required_regimes=regimes,
        min_decisions_per_regime=1,
    )
    registry = StrategyRegistry(db_path=str(tmp_path / "registry-span.db"), promotion_guard=guard)

    genome = SimpleNamespace(
        id=policy_id,
        decision_tree={"nodes": 3},
        name=policy_id,
        generation=1,
    )
    fitness_report = {
        "fitness_score": 1.05,
        "max_drawdown": 0.04,
        "sharpe_ratio": 1.4,
        "total_return": 0.11,
        "volatility": 0.03,
        "metadata": {},
    }

    assert registry.register_champion(genome, dict(fitness_report))

    with pytest.raises(StrategyRegistryError) as excinfo:
        registry.update_strategy_status(genome.id, StrategyStatus.ACTIVE)

    assert "paper_green_gate_duration_below" in str(excinfo.value)


def test_strategy_registry_surfaces_database_errors(tmp_path) -> None:
    invalid_path = tmp_path / "registry_dir"
    invalid_path.mkdir()

    with pytest.raises(StrategyRegistryError):
        StrategyRegistry(db_path=str(invalid_path))
