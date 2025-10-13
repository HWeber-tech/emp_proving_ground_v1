from __future__ import annotations

import types

import pytest

from src.intelligence import Phase3IntelligenceOrchestrator


class _StubSentientEngine:
    async def adapt_in_real_time(self, market_data, strategy, context):
        return {"strategy": getattr(strategy, "strategy_id", "stub"), "data": market_data}

    def get_status(self):
        return {"status": "sentient-ok"}


class _StubPredictiveModeler:
    async def predict_market_scenarios(self, market_data, horizon):
        return [{"market_data": market_data, "horizon": str(horizon)}]

    def get_status(self):
        return {"status": "predictive-ok"}


class _StubAdversarialTrainer:
    async def train_adversarial_strategies(self, strategies):
        return list(strategies)

    def get_status(self):
        return {"status": "adversarial-ok"}


class _StubRedTeam:
    async def attack_strategy(self, strategy):
        return {"strategy": getattr(strategy, "strategy_id", "stub"), "breach": False}

    def get_status(self):
        return {"status": "red-team-ok"}


class _StubSpecializedEvolution:
    async def evolve_specialized_predators(self):
        return ["predator-alpha"]

    def get_status(self):
        return {"status": "specialized-ok"}


class _StubCompetitiveIntelligence:
    def __init__(self):
        self.called_with: list[object] = []
        self.stats_calls = 0

    async def identify_competitors(self, market_data):
        self.called_with.append(market_data)
        return {"understanding_id": "comp-123", "signals": 2}

    async def get_intelligence_stats(self):
        self.stats_calls += 1
        return {"total_understanding_cycles": 42}


@pytest.mark.asyncio
async def test_phase3_orchestrator_routes_competitive_understanding():
    orchestrator = Phase3IntelligenceOrchestrator()
    orchestrator.sentient_engine = _StubSentientEngine()
    orchestrator.predictive_modeler = _StubPredictiveModeler()
    orchestrator.adversarial_trainer = _StubAdversarialTrainer()
    orchestrator.red_team = _StubRedTeam()
    orchestrator.specialized_evolution = _StubSpecializedEvolution()
    orchestrator.competitive_intelligence = _StubCompetitiveIntelligence()

    market_data = {"price": 101.0}
    strategy = types.SimpleNamespace(strategy_id="strat-1")
    result = await orchestrator.run_intelligence_cycle(market_data, [strategy])

    assert result["competitive_understanding"] == {"understanding_id": "comp-123", "signals": 2}
    assert result["competitive_intelligence"] is result["competitive_understanding"]

    status = await orchestrator.get_phase3_status()
    assert status["competitive_understanding"] == {"total_understanding_cycles": 42}
    assert status["competitive_intelligence"] is status["competitive_understanding"]
