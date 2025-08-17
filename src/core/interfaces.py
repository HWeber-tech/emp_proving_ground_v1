from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, runtime_checkable

from .event_bus import Event


@runtime_checkable
class Cache(Protocol):
    """Minimal cache facade used as a stable import target.

    Framework-agnostic. Implementations may be backed by dict, Redis, etc.
    """
    def get(self, key: str) -> Optional[Any]:
        ...

    def set(self, key: str, value: Any) -> None:
        ...


@runtime_checkable
class EventBus(Protocol):
    """Deprecated legacy topic/payload publisher.
    
    Deprecated: Use SupportsEventPublish for async-first Event publishing
    or the TopicBus facade for transitional legacy usage.
    """
    def publish(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        ...


@runtime_checkable
class SupportsEventPublish(Protocol):
    """Async-first EventBus protocol (M3): async publish(Event)."""
    async def publish(self, event: Event) -> None:
        ...

@runtime_checkable
class Logger(Protocol):
    """Minimal logging facade with conventional methods."""
    def info(self, msg: str, /, **kwargs: Any) -> None: ...
    def debug(self, msg: str, /, **kwargs: Any) -> None: ...
    def warning(self, msg: str, /, **kwargs: Any) -> None: ...
    def error(self, msg: str, /, **kwargs: Any) -> None: ...
class DecisionGenome:
    """Minimal DecisionGenome placeholder used for typing and population management.
    Real implementations should extend this class and provide fields like
    `fitness` (float) and `species_type` (str).
    """
    fitness: float = 0.0
    species_type: str = "generic"


class IPopulationManager:
    """Interface for population management.

    This is a lightweight runtime-friendly interface to avoid import errors in tests.
    """

    def initialize_population(self, genome_factory: Callable) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_population(self) -> List[DecisionGenome]:  # pragma: no cover
        raise NotImplementedError

    def get_best_genomes(self, count: int) -> List[DecisionGenome]:  # pragma: no cover
        raise NotImplementedError

    def update_population(self, new_population: List[DecisionGenome]) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_population_statistics(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def advance_generation(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError