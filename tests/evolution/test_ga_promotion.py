from __future__ import annotations

from pathlib import Path

import pytest

from src.evolution.experiments.promotion import (
    FEATURE_FLAG_ENV,
    maybe_promote_best_genome,
)
from src.governance.strategy_registry import StrategyRegistry


def _sample_manifest() -> dict[str, object]:
    return {
        "experiment": "ma_crossover_ga",
        "generated_at": "2025-09-29T00:00:00Z",
        "seed": 2025,
        "code_version": "abcdef123456",
        "notes": "Synthetic regression test dataset",
        "dataset": {
            "name": "synthetic_trend_v1",
            "metadata": {"length": 512, "mean": 0.12},
        },
        "config": {
            "population_size": 24,
            "generations": 18,
            "elite_count": 4,
            "crossover_rate": 0.7,
            "mutation_rate": 0.25,
            "seed": 2025,
        },
        "replay": {"command": "python scripts/generate_evolution_lab.py --seed 2025"},
        "best_genome": {
            "short_window": 8,
            "long_window": 169,
            "risk_fraction": 0.36,
            "use_var_guard": True,
            "use_drawdown_guard": True,
        },
        "best_metrics": {
            "fitness": 4.166,
            "sharpe": 3.916,
            "sortino": 5.98,
            "max_drawdown": 0.01,
            "total_return": 0.131,
        },
    }


def test_promotion_respects_feature_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(FEATURE_FLAG_ENV, raising=False)
    manifest = _sample_manifest()
    registry_path = tmp_path / "registry.db"

    promoted = maybe_promote_best_genome(manifest, registry_path=registry_path)

    assert promoted is False
    assert not registry_path.exists()


def test_promotion_registers_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(FEATURE_FLAG_ENV, "true")
    manifest = _sample_manifest()
    registry_path = tmp_path / "registry.db"

    promoted = maybe_promote_best_genome(manifest, registry_path=registry_path)

    assert promoted is True
    registry = StrategyRegistry(str(registry_path))
    try:
        record = registry.get_strategy("ma::8-169-0.360")
        assert record is not None
        report = record["fitness_report"]
        assert pytest.approx(report["fitness_score"], rel=1e-6) == 4.166
        metadata = report["metadata"]["evolution_lab"]
        assert metadata["experiment"] == "ma_crossover_ga"
        assert metadata["dataset"]["name"] == "synthetic_trend_v1"
        assert metadata["config"]["population_size"] == 24
    finally:
        registry.conn.close()
