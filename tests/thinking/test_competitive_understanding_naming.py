from __future__ import annotations

from datetime import datetime

import pytest

from src.operational.state_store.adapters import InMemoryStateStore
from src.thinking.competitive.competitive_intelligence_system import (
    AlgorithmSignature,
    CompetitorBehavior,
    CompetitiveIntelligenceSystem,
    CounterStrategy,
)


@pytest.mark.asyncio
async def test_competitive_understanding_stats_alias_matches_legacy() -> None:
    store = InMemoryStateStore()
    system = CompetitiveIntelligenceSystem(store)

    signature = AlgorithmSignature(
        signature_id="sig-1",
        algorithm_type="momentum_bot",
        confidence=0.8,
        characteristics={"feature": "value"},
        first_seen=datetime.utcnow(),
        last_seen=datetime.utcnow(),
        frequency="high",
    )

    behavior = CompetitorBehavior(
        competitor_id="comp-1",
        algorithm_signature=signature,
        behavior_metrics={"hit_rate": 0.7},
        patterns=["momentum"],
        threat_level="medium",
        market_share=0.15,
        performance=0.12,
        first_observed=datetime.utcnow(),
        last_observed=datetime.utcnow(),
    )

    counter = CounterStrategy(
        strategy_id="strat-1",
        target_competitor="comp-1",
        counter_type="hedge",
        parameters={"alpha": 0.1},
        expected_effectiveness=0.5,
        implementation_complexity="medium",
        risk_level="low",
        deployment_timeline="1d",
        timestamp=datetime.utcnow(),
    )

    market_analysis = {"share_delta": 0.05}

    await system._store_understanding_snapshot(
        [signature],
        [behavior],
        [counter],
        market_analysis,
    )

    stats = await system.get_understanding_stats()
    legacy = await system.get_intelligence_stats()

    expected_understanding_counts = {
        "total_understanding_cycles": 1,
        "total_understanding_signatures_detected": 1,
        "total_understanding_competitors_analyzed": 1,
        "total_understanding_counter_strategies_developed": 1,
        "average_understanding_signatures_per_cycle": 1.0,
    }

    for key, value in expected_understanding_counts.items():
        assert key in stats
        if isinstance(value, float):
            assert stats[key] == pytest.approx(value)
            assert legacy[key] == pytest.approx(value)
        else:
            assert stats[key] == value
            assert legacy[key] == value

    alias_map = {
        "total_intelligence_cycles": "total_understanding_cycles",
        "total_signatures_detected": "total_understanding_signatures_detected",
        "total_competitors_analyzed": "total_understanding_competitors_analyzed",
        "total_counter_strategies_developed": "total_understanding_counter_strategies_developed",
        "average_signatures_per_cycle": "average_understanding_signatures_per_cycle",
    }

    for alias, canonical in alias_map.items():
        assert alias not in stats
        expected = legacy[canonical]
        if isinstance(expected, float):
            assert legacy[alias] == pytest.approx(expected)
        else:
            assert legacy[alias] == expected

    assert isinstance(stats["last_understanding"], str)
    assert stats["last_understanding"]
    assert "last_intelligence" not in stats

    assert isinstance(legacy["last_understanding"], str)
    assert isinstance(legacy["last_intelligence"], str)
    assert legacy["last_understanding"]
    assert legacy["last_intelligence"]
