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

    expected_counts = {
        "total_understanding_cycles": 1,
        "total_understanding_signatures_detected": 1,
        "total_understanding_competitors_analyzed": 1,
        "total_understanding_counter_strategies_developed": 1,
        "total_intelligence_cycles": 1,
        "total_signatures_detected": 1,
        "total_competitors_analyzed": 1,
        "total_counter_strategies_developed": 1,
    }

    for key, value in expected_counts.items():
        assert stats[key] == value
        assert legacy[key] == value

    for key in ("last_understanding", "last_intelligence"):
        assert isinstance(stats[key], str)
        assert isinstance(legacy[key], str)
        assert stats[key]
        assert legacy[key]
