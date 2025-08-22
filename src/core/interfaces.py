from collections.abc import Iterable, Mapping, Sequence
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Protocol, Tuple, TypedDict, runtime_checkable, TYPE_CHECKING

from .types import JSONObject
from .event_bus import Event
from .market_data import MarketDataGateway
from .regime import RegimeClassifier, RegimeResult


@runtime_checkable
class Cache(Protocol):
    """Minimal cache facade used as a stable import target.

    Framework-agnostic. Implementations may be backed by dict, Redis, etc.
    """
    def get(self, key: str) -> Optional[object]: ...
    def set(self, key: str, value: object) -> None: ...


@runtime_checkable
class EventBus(Protocol):
    """Deprecated legacy topic/payload publisher.

    Deprecated: Use SupportsEventPublish for async-first Event publishing
    or the TopicBus facade for transitional legacy usage.
    """
    def publish(self, event: str, payload: Mapping[str, object] | None = None, /, **kwargs: object) -> None: ...


@runtime_checkable
class SupportsEventPublish(Protocol):
    """Async-first EventBus protocol (M3): async publish(Event)."""
    async def publish(self, event: Event) -> None: ...


@runtime_checkable
class Logger(Protocol):
    """Minimal logging facade with conventional methods."""
    def info(self, msg: str, /, **kwargs: object) -> None: ...
    def debug(self, msg: str, /, **kwargs: object) -> None: ...
    def warning(self, msg: str, /, **kwargs: object) -> None: ...
    def error(self, msg: str, /, **kwargs: object) -> None: ...


@runtime_checkable
class ConfigProvider(Protocol):
    """Canonical configuration provider Protocol.

    Implementations load and provide structured configuration as JSON-like objects.
    """
    def get(self, key: str, default: object | None = ...) -> object: ...
    def get_section(self, name: str) -> JSONObject: ...
    def get_config(self) -> JSONObject: ...
    def with_overrides(self, updates: JSONObject) -> "ConfigProvider": ...


@runtime_checkable
class RiskManager(Protocol):
    """Canonical risk manager Protocol (minimal shared surface)."""
    def evaluate_portfolio_risk(
        self,
        positions: Mapping[str, float],
        context: Mapping[str, object] | None = ...,
    ) -> float: ...
    def propose_rebalance(
        self,
        positions: Mapping[str, float],
        constraints: Mapping[str, object] | None = ...,
    ) -> Mapping[str, float]: ...
    def update_limits(self, limits: Mapping[str, object]) -> None: ...


@runtime_checkable
class DecisionGenome(Protocol):
    """Structural genome interface used across ecosystem modules.

    Implemented by src.genome.models.genome.DecisionGenome.
    Minimal surface: fields actually consumed by optimizers/factories.
    """
    id: str
    parameters: Mapping[str, float]
    fitness: Optional[float] | None
    generation: int
    species_type: Optional[str] | None


@runtime_checkable
class PopulationManager(Protocol):
    """Protocol for population management (authoritative)."""
    def initialize_population(self, genome_factory: Callable[[], "DecisionGenome"]) -> None: ...
    def get_population(self) -> List[DecisionGenome]: ...
    def get_best_genomes(self, count: int) -> List[DecisionGenome]: ...
    def update_population(self, new_population: List[DecisionGenome]) -> None: ...
    def get_population_statistics(self) -> dict[str, object]: ...
    def advance_generation(self) -> None: ...
    def reset(self) -> None: ...


# Back-compat interface retained for legacy imports; prefer PopulationManager
class IPopulationManager:
    """Interface for population management (legacy class-based interface).

    This is a lightweight runtime-friendly interface to avoid import errors in tests.
    Prefer the PopulationManager Protocol for new code.
    """

    def initialize_population(self, genome_factory: Callable[[], "DecisionGenome"]) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_population(self) -> List[DecisionGenome]:  # pragma: no cover
        raise NotImplementedError

    def get_best_genomes(self, count: int) -> List[DecisionGenome]:  # pragma: no cover
        raise NotImplementedError

    def update_population(self, new_population: List[DecisionGenome]) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_population_statistics(self) -> dict[str, object]:  # pragma: no cover
        raise NotImplementedError

    def advance_generation(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError


# Ecosystem and coordination domain types

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
    approved_intents: List[TradeIntent]
    rejected_intents: List[TradeIntent]
    coordination_score: float
    portfolio_risk: float
    correlation_impact: float


@runtime_checkable
class ICoordinationEngine(Protocol):
    async def resolve_intents(
        self,
        intents: List[TradeIntent],
        market_context: MarketContext,
    ) -> CoordinationResult: ...
    async def prioritize_strategies(
        self,
        strategies: List[HasSpeciesType],
        regime: str,
    ) -> List[HasSpeciesType]: ...
    async def get_portfolio_summary(self) -> dict[str, object]: ...
    async def get_coordination_metrics(self) -> Dict[str, float]: ...


# Ecosystem optimizer summaries
class MetricsSummary(TypedDict):
    total_return: float
    sharpe_ratio: float
    diversification_ratio: float
    synergy_score: float

class EcosystemSummary(TypedDict):
    total_optimizations: int
    best_metrics: MetricsSummary | None
    current_species_distribution: Dict[str, int]

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
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]: ...


# Metrics protocols (can be implemented by prometheus_client metrics or no-op fallbacks)
@runtime_checkable
class CounterLike(Protocol):
    def inc(self, amount: float = 1.0) -> None: ...
    def labels(self, **labels: str) -> "CounterLike": ...


@runtime_checkable
class GaugeLike(Protocol):
    def set(self, value: float) -> None: ...
    def inc(self, amount: float = 1.0) -> None: ...
    def dec(self, amount: float = 1.0) -> None: ...
    def labels(self, **labels: str) -> "GaugeLike": ...


@runtime_checkable
class HistogramLike(Protocol):
    def observe(self, value: float) -> None: ...
    def labels(self, **labels: str) -> "HistogramLike": ...


# Minimal JSON alias for this module to avoid circular imports with src.core.types
JSONValue = object

@runtime_checkable
class ThinkingPattern(Protocol):
    def learn(self, feedback: Mapping[str, object]) -> bool: ...

class SensorySignal(TypedDict, total=False):
    name: str
    value: float
    confidence: float

class AnalysisResult(TypedDict, total=False):
    summary: str
    score: float
    details: Mapping[str, object]

# === End Interfaces hub ===

__all__ = [
    # Core infra
    "Cache",
    "EventBus",
    "SupportsEventPublish",
    "Logger",
    "ConfigProvider",
    "RiskManager",
    # Domain models/interfaces
    "DecisionGenome",
    "PopulationManager",
    "IPopulationManager",
    # Ecosystem and coordination
    "TradeIntent",
    "MarketContext",
    "HasSpeciesType",
    "CoordinationResult",
    "ICoordinationEngine",
    "IEcosystemOptimizer",
    "ISpecialistGenomeFactory",
    "MetricsSummary",
    "EcosystemSummary",
    # Re-exports of canonical ports
    "MarketDataGateway",
    "RegimeClassifier",
    "RegimeResult",
    # Metrics protocols
    "CounterLike",
    "GaugeLike",
    "HistogramLike",
    # Interfaces hub exports
    "ThinkingPattern",
    "SensorySignal",
    "AnalysisResult",
    "JSONValue",
]