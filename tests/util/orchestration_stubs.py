"""Helpers for installing lightweight Phase 3 orchestrator stubs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from src.operational.state_store import InMemoryStateStore, StateStore


class StubPredictiveModeler:
    def __init__(self, state_store: StateStore) -> None:
        self.state_store = state_store
        self.initialized = False
        self.stopped = False

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def stop(self) -> bool:
        self.stopped = True
        return True

    async def predict_market_scenarios(self, *_: Any, **__: Any) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(probability=0.85, confidence=0.75),
            SimpleNamespace(probability=0.6, confidence=0.55),
        ]


class StubMarketGAN:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.initialized = False
        self.stopped = False
        self.strategy_populations: list[list[str]] = []

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def stop(self) -> bool:
        self.stopped = True
        return True

    async def train_adversarial_strategies(self, strategy_population: list[str]) -> list[str]:
        self.strategy_populations.append(list(strategy_population))
        return ["strat_A", "strat_B"]


class StubRedTeamAI:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.initialized = False
        self.stopped = False
        self.attacks: list[str] = []

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def stop(self) -> bool:
        self.stopped = True
        return True

    async def attack_strategy(self, strategy: str) -> dict[str, Any]:
        self.attacks.append(strategy)
        return {
            "attack_results": [
                {"success": True, "impact": 0.2, "strategy": strategy},
                {"success": False, "impact": 0.05, "strategy": strategy},
            ],
            "weaknesses_found": ["latency", "slippage"],
        }


class StubSpecializedEvolution:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.initialized = False
        self.stopped = False

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def stop(self) -> bool:
        self.stopped = True
        return True


class StubCompetitiveIntelligence:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.initialized = False
        self.stopped = False

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def stop(self) -> bool:
        self.stopped = True
        return True


def _install_package(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, package)
    return package


def _install_module(
    monkeypatch: pytest.MonkeyPatch, name: str, attrs: dict[str, Any]
) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def install_phase3_orchestrator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install stubbed thinking modules and return the Phase 3 orchestrator module."""

    events_module = ModuleType("src.core.events")
    events_module.AnalysisResult = object
    events_module.PerformanceMetrics = object
    events_module.RiskMetrics = object
    events_module.TradeIntent = object
    events_module.MarketData = object
    events_module.__getattr__ = lambda *_: object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.core.events", events_module)

    _install_package(monkeypatch, "src.thinking")
    _install_package(monkeypatch, "src.thinking.adversarial")
    _install_package(monkeypatch, "src.thinking.competitive")
    _install_package(monkeypatch, "src.thinking.ecosystem")
    _install_package(monkeypatch, "src.thinking.prediction")
    _install_package(monkeypatch, "src.ecosystem")
    _install_package(monkeypatch, "src.ecosystem.evolution")

    _install_module(
        monkeypatch,
        "src.thinking.adversarial.market_gan",
        {"MarketGAN": StubMarketGAN},
    )
    _install_module(
        monkeypatch,
        "src.thinking.adversarial.red_team_ai",
        {"RedTeamAI": StubRedTeamAI},
    )
    _install_module(
        monkeypatch,
        "src.thinking.competitive.competitive_intelligence_system",
        {"CompetitiveIntelligenceSystem": StubCompetitiveIntelligence},
    )
    specialized_module = _install_module(
        monkeypatch,
        "src.ecosystem.evolution.specialized_predator_evolution",
        {"SpecializedPredatorEvolution": StubSpecializedEvolution},
    )
    monkeypatch.setitem(
        sys.modules,
        "src.thinking.ecosystem.specialized_predator_evolution",
        specialized_module,
    )
    _install_module(
        monkeypatch,
        "src.thinking.prediction.predictive_market_modeler",
        {"PredictiveMarketModeler": StubPredictiveModeler},
    )

    repo_root = Path(__file__).resolve().parents[2]
    phase3_path = repo_root / "src" / "thinking" / "phase3_orchestrator.py"
    spec = importlib.util.spec_from_file_location("src.thinking.phase3_orchestrator", phase3_path)
    assert spec and spec.loader is not None
    orchestrator_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "src.thinking.phase3_orchestrator", orchestrator_module)
    spec.loader.exec_module(orchestrator_module)
    return orchestrator_module


__all__ = [
    "InMemoryStateStore",
    "StubCompetitiveIntelligence",
    "StubMarketGAN",
    "StubPredictiveModeler",
    "StubRedTeamAI",
    "StubSpecializedEvolution",
    "install_phase3_orchestrator",
]
