from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Optional, Protocol, runtime_checkable

from ..event_bus import Event
from ..market_data import MarketDataGateway
from ..regime import RegimeClassifier, RegimeResult
from ..types import JSONObject


@runtime_checkable
class Cache(Protocol):
    """Minimal cache facade used as a stable import target."""

    def get(self, key: str) -> Optional[object]: ...

    def set(self, key: str, value: object) -> None: ...


@runtime_checkable
class EventBus(Protocol):
    """Deprecated legacy topic/payload publisher."""

    def publish(
        self, event: str, payload: Mapping[str, object] | None = None, /, **kwargs: object
    ) -> None: ...


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
    """Canonical configuration provider Protocol."""

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
        constraints: Mapping[str, object] | None = ...,  # API parity with legacy manager
    ) -> Mapping[str, float]: ...

    def update_limits(self, limits: Mapping[str, object]) -> None: ...


@runtime_checkable
class DecisionGenome(Protocol):
    """Structural genome interface used across ecosystem modules."""

    # Core identity and metadata
    id: str
    parameters: Mapping[str, float]
    fitness: Optional[float] | None
    generation: int
    species_type: Optional[str] | None

    # Widely accessed extended attributes (structural; may exist on concrete genome)
    genome_id: str
    mutation_count: int

    # Sub-structures used by mutation pipelines (typed loosely to avoid import cycles)
    strategy: Any
    risk: Any
    timing: Any
    sensory: Any
    thinking: Any

    # Normalization hook used after mutating weights
    def _normalize_weights(self) -> None: ...


@runtime_checkable
class IMutationStrategy(Protocol):
    def mutate(self, genome: DecisionGenome, mutation_rate: float) -> DecisionGenome: ...


@runtime_checkable
class IExecutionEngine(Protocol):
    async def send_order(
        self, symbol: str, side: str, quantity: float, price: float | None = ...
    ) -> str: ...

    async def cancel_order(self, order_id: str) -> bool: ...

    async def get_position(self, symbol: str) -> Any: ...


@runtime_checkable
class PopulationManager(Protocol):
    """Protocol for population management (authoritative)."""

    def initialize_population(self, genome_factory: Callable[[], DecisionGenome]) -> None: ...

    def get_population(self) -> list[DecisionGenome]: ...

    def get_best_genomes(self, count: int) -> list[DecisionGenome]: ...

    def update_population(self, new_population: list[DecisionGenome]) -> None: ...

    def get_population_statistics(self) -> dict[str, object]: ...

    def advance_generation(self) -> None: ...

    def reset(self) -> None: ...


class IPopulationManager:
    """Interface for population management (legacy class-based interface)."""

    def initialize_population(
        self, genome_factory: Callable[[], DecisionGenome]
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_population(self) -> list[DecisionGenome]:  # pragma: no cover
        raise NotImplementedError

    def get_best_genomes(self, count: int) -> list[DecisionGenome]:  # pragma: no cover
        raise NotImplementedError

    def update_population(self, new_population: list[DecisionGenome]) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_population_statistics(self) -> dict[str, object]:  # pragma: no cover
        raise NotImplementedError

    def advance_generation(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError


__all__ = [
    "Cache",
    "EventBus",
    "SupportsEventPublish",
    "Logger",
    "ConfigProvider",
    "RiskManager",
    "DecisionGenome",
    "IMutationStrategy",
    "IExecutionEngine",
    "PopulationManager",
    "IPopulationManager",
    "MarketDataGateway",
    "RegimeClassifier",
    "RegimeResult",
    "JSONObject",
]
