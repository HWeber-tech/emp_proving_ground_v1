from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evolution.experiments.promotion import promote_ma_crossover_champion
from src.governance.strategy_registry import StrategyRegistry, StrategyStatus


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


def _build_registry(tmp_path: Path) -> StrategyRegistry:
    db_path = tmp_path / "registry.db"
    return StrategyRegistry(db_path=str(db_path))


def test_promote_ga_champion_registers_evolved_by_default(
    sample_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PAPER_TRADE_GA_MA_CROSSOVER", raising=False)
    registry = _build_registry(tmp_path)
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
    registry = _build_registry(tmp_path)
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
    registry = _build_registry(tmp_path)
    try:
        result = promote_ma_crossover_champion(sample_manifest, registry)
        assert result.target_status == StrategyStatus.ACTIVE
        assert result.status_applied is True

        record = registry.get_strategy(result.genome_id or "")
        assert record is not None
        assert record["status"] == StrategyStatus.ACTIVE.value
    finally:
        registry.close()
