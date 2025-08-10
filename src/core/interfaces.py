from typing import List, Dict, Any, Callable


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