from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypedDict, runtime_checkable

from ..types import JSONObject
from .base import DecisionGenome


@runtime_checkable
class HasSpeciesType(Protocol):
    species_type: str


@dataclass
class TradeIntent:
    strategy_id: str
    species_type: str
    symbol: str
    direction: str
    confidence: float
    size: float
    priority: int
    timestamp: datetime


@dataclass
class MarketContext:
    symbol: str
    regime: str
    volatility: float
    trend_strength: float
    volume_anomaly: float
    data: JSONObject | None = None


@runtime_checkable
class CoordinationResult(Protocol):
    approved_intents: list[TradeIntent]
    rejected_intents: list[TradeIntent]
    coordination_score: float
    portfolio_risk: float
    correlation_impact: float


@runtime_checkable
class ICoordinationEngine(Protocol):
    async def resolve_intents(
        self,
        intents: list[TradeIntent],
        market_context: MarketContext,
    ) -> CoordinationResult: ...

    async def prioritize_strategies(
        self,
        strategies: list[HasSpeciesType],
        regime: str,
    ) -> list[HasSpeciesType]: ...

    async def get_portfolio_summary(self) -> dict[str, object]: ...

    async def get_coordination_metrics(self) -> dict[str, float]: ...


class MetricsSummary(TypedDict):
    total_return: float
    sharpe_ratio: float
    diversification_ratio: float
    synergy_score: float


class EcosystemSummary(TypedDict):
    total_optimizations: int
    best_metrics: MetricsSummary | None
    current_species_distribution: dict[str, int]


@runtime_checkable
class IEcosystemOptimizer(Protocol):
    async def optimize_ecosystem(
        self,
        species_populations: Mapping[str, Sequence[DecisionGenome]],
        market_context: MarketContext,
        performance_history: JSONObject,
    ) -> Mapping[str, Sequence[DecisionGenome]]: ...

    async def get_ecosystem_summary(self) -> EcosystemSummary: ...


@runtime_checkable
class ISpecialistGenomeFactory(Protocol):
    def create_genome(self) -> DecisionGenome: ...

    def get_species_name(self) -> str: ...

    def get_parameter_ranges(self) -> dict[str, tuple[float, float]]: ...


__all__ = [
    "HasSpeciesType",
    "TradeIntent",
    "MarketContext",
    "CoordinationResult",
    "ICoordinationEngine",
    "MetricsSummary",
    "EcosystemSummary",
    "IEcosystemOptimizer",
    "ISpecialistGenomeFactory",
]
