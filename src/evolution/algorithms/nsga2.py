"""Implementation of the NSGA-II multi-objective evolutionary algorithm."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import random
from typing import Callable, Generic, Sequence, TypeVar

__all__ = [
    "NSGA2",
    "NSGA2Config",
    "NSGA2Result",
    "RankedIndividual",
]

T = TypeVar("T")
Objectives = tuple[float, ...]


@dataclass(slots=True)
class NSGA2Config:
    """Configuration controlling NSGA-II behaviour."""

    population_size: int = 50
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    tournament_size: int = 2
    maximise_objectives: Sequence[bool] | None = None

    def __post_init__(self) -> None:
        if self.population_size <= 1:
            raise ValueError("population_size must be greater than 1")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError("crossover_probability must be in [0, 1]")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in [0, 1]")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")
        if self.maximise_objectives is not None:
            if not isinstance(self.maximise_objectives, Sequence):
                raise TypeError("maximise_objectives must be a sequence of booleans")
            if any(value is None for value in self.maximise_objectives):
                raise ValueError("maximise_objectives cannot contain None")


@dataclass(slots=True)
class RankedIndividual(Generic[T]):
    """Exported view of an individual ranked by NSGA-II."""

    genome: T
    objectives: Objectives
    rank: int
    crowding_distance: float

    def as_dict(self) -> dict[str, object]:
        return {
            "genome": self.genome,
            "objectives": list(self.objectives),
            "rank": int(self.rank),
            "crowding_distance": float(self.crowding_distance),
        }


@dataclass(slots=True)
class NSGA2Result(Generic[T]):
    """Result emitted after executing one NSGA-II generation."""

    population: list[T]
    ranked_population: list[RankedIndividual[T]]
    fronts: list[list[RankedIndividual[T]]]

    def best_front(self) -> list[RankedIndividual[T]]:
        """Return the first non-dominated front."""

        if not self.fronts:
            return []
        return list(self.fronts[0])


@dataclass(slots=True)
class _Individual(Generic[T]):
    genome: T
    objectives: Objectives
    scaled_objectives: Objectives
    rank: int = 0
    crowding_distance: float = 0.0


class NSGA2(Generic[T]):
    """Execute NSGA-II to evolve a multi-objective population."""

    def __init__(
        self,
        evaluate: Callable[[T], Sequence[float]],
        *,
        config: NSGA2Config | None = None,
        crossover: Callable[[T, T], tuple[T, T]] | None = None,
        mutate: Callable[[T], T] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        self._evaluate = evaluate
        self._config = config or NSGA2Config()
        self._rng = rng or random.Random()
        self._crossover = crossover or self._default_crossover
        self._mutate = mutate or self._default_mutation
        self._objective_count: int | None = None
        self._scales: tuple[float, ...] | None = None

    @property
    def config(self) -> NSGA2Config:
        return self._config

    def step(self, population: Sequence[T]) -> NSGA2Result[T]:
        """Run a single NSGA-II generation and return the evolved population."""

        if not population:
            raise ValueError("population cannot be empty")

        evaluated = [self._evaluate_genome(genome) for genome in population]
        self._assign_ranks_and_crowding(evaluated)
        offspring_genomes = self._generate_offspring(evaluated)
        evaluated_offspring = [self._evaluate_genome(genome) for genome in offspring_genomes]
        combined = evaluated + evaluated_offspring
        fronts = self._assign_ranks_and_crowding(combined)
        next_generation = self._select_next_generation(fronts)

        ranked_fronts = [[self._export_ranked(individual) for individual in front] for front in fronts]
        ranked_population = [ranked for front in ranked_fronts for ranked in front]
        next_population = [individual.genome for individual in next_generation]

        return NSGA2Result(next_population, ranked_population, ranked_fronts)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _evaluate_genome(self, genome: T) -> _Individual[T]:
        objectives = self._normalise_objectives(self._evaluate(genome))
        scaled = self._scale_objectives(objectives)
        return _Individual(genome=genome, objectives=objectives, scaled_objectives=scaled)

    def _normalise_objectives(self, values: Sequence[float]) -> Objectives:
        if not isinstance(values, Sequence):
            raise TypeError("evaluate must return a sequence of floats")
        objectives = tuple(float(value) for value in values)
        if not objectives:
            raise ValueError("evaluate must return at least one objective")
        if self._objective_count is None:
            self._objective_count = len(objectives)
            self._scales = self._build_scales(len(objectives))
        elif len(objectives) != self._objective_count:
            raise ValueError("all individuals must return the same number of objectives")
        return objectives

    def _build_scales(self, count: int) -> tuple[float, ...]:
        maximise = self._config.maximise_objectives
        if maximise is None or len(maximise) == 0:
            return tuple(1.0 for _ in range(count))
        if len(maximise) != count:
            raise ValueError("maximise_objectives length must match objective count")
        return tuple(-1.0 if bool(flag) else 1.0 for flag in maximise)

    def _scale_objectives(self, objectives: Objectives) -> Objectives:
        if self._scales is None:
            raise RuntimeError("objective scales have not been initialised")
        return tuple(value * scale for value, scale in zip(objectives, self._scales))

    # ------------------------------------------------------------------
    # Core NSGA-II operations
    # ------------------------------------------------------------------
    def _assign_ranks_and_crowding(self, individuals: list[_Individual[T]]) -> list[list[_Individual[T]]]:
        fronts = self._fast_non_dominated_sort(individuals)
        for front in fronts:
            self._compute_crowding_distance(front)
        return fronts

    def _fast_non_dominated_sort(self, individuals: list[_Individual[T]]) -> list[list[_Individual[T]]]:
        if not individuals:
            return []
        domination_counts = [0 for _ in individuals]
        dominated_sets: list[list[int]] = [[] for _ in individuals]
        fronts: list[list[_Individual[T]]] = []

        for i, p in enumerate(individuals):
            for j, q in enumerate(individuals):
                if i == j:
                    continue
                if self._dominates(p, q):
                    dominated_sets[i].append(j)
                elif self._dominates(q, p):
                    domination_counts[i] += 1

        current_front = [index for index, count in enumerate(domination_counts) if count == 0]
        rank = 0
        while current_front:
            front = [individuals[index] for index in current_front]
            for index in current_front:
                individuals[index].rank = rank
            fronts.append(front)
            next_front_indices: list[int] = []
            for index in current_front:
                for dominated in dominated_sets[index]:
                    domination_counts[dominated] -= 1
                    if domination_counts[dominated] == 0:
                        next_front_indices.append(dominated)
            rank += 1
            current_front = sorted(set(next_front_indices))
        return fronts

    def _dominates(self, left: _Individual[T], right: _Individual[T]) -> bool:
        better_in_any = False
        for l_value, r_value in zip(left.scaled_objectives, right.scaled_objectives):
            if l_value > r_value:
                return False
            if l_value < r_value:
                better_in_any = True
        return better_in_any

    def _compute_crowding_distance(self, front: list[_Individual[T]]) -> None:
        if not front:
            return
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = math.inf
            return
        objective_count = len(front[0].scaled_objectives)
        for individual in front:
            individual.crowding_distance = 0.0
        for objective_index in range(objective_count):
            sorted_front = sorted(front, key=lambda ind: ind.scaled_objectives[objective_index])
            sorted_front[0].crowding_distance = math.inf
            sorted_front[-1].crowding_distance = math.inf
            min_value = sorted_front[0].scaled_objectives[objective_index]
            max_value = sorted_front[-1].scaled_objectives[objective_index]
            if math.isclose(max_value, min_value):
                continue
            denominator = max_value - min_value
            for position in range(1, len(sorted_front) - 1):
                left_value = sorted_front[position - 1].scaled_objectives[objective_index]
                right_value = sorted_front[position + 1].scaled_objectives[objective_index]
                contribution = (right_value - left_value) / denominator
                if math.isfinite(sorted_front[position].crowding_distance):
                    sorted_front[position].crowding_distance += contribution

    def _select_next_generation(self, fronts: list[list[_Individual[T]]]) -> list[_Individual[T]]:
        selected: list[_Individual[T]] = []
        for front in fronts:
            if len(selected) + len(front) <= self._config.population_size:
                selected.extend(front)
                continue
            sorted_front = sorted(front, key=lambda ind: ind.crowding_distance, reverse=True)
            remaining = self._config.population_size - len(selected)
            if remaining > 0:
                selected.extend(sorted_front[:remaining])
            break
        if len(selected) < self._config.population_size and fronts:
            # Defensive: fill using best available individuals.
            remaining = self._config.population_size - len(selected)
            fallback = sorted(fronts[-1], key=lambda ind: (ind.rank, -ind.crowding_distance))
            selected.extend(fallback[:remaining])
        return selected[: self._config.population_size]

    # ------------------------------------------------------------------
    # Variation operators
    # ------------------------------------------------------------------
    def _generate_offspring(self, parents: list[_Individual[T]]) -> list[T]:
        if not parents:
            return []
        offspring: list[T] = []
        while len(offspring) < self._config.population_size:
            parent_one = self._tournament_select(parents)
            parent_two = self._tournament_select(parents)
            if self._rng.random() < self._config.crossover_probability:
                child_one, child_two = self._crossover(parent_one.genome, parent_two.genome)
            else:
                child_one, child_two = self._clone(parent_one.genome), self._clone(parent_two.genome)
            for child in (child_one, child_two):
                if self._rng.random() < self._config.mutation_probability:
                    child = self._mutate(child)
                offspring.append(child)
                if len(offspring) >= self._config.population_size:
                    break
        return offspring

    def _tournament_select(self, population: list[_Individual[T]]) -> _Individual[T]:
        if len(population) == 1:
            return population[0]
        k = max(2, min(self._config.tournament_size, len(population)))
        contenders = self._rng.choices(population, k=k)
        return min(contenders, key=lambda ind: (ind.rank, -ind.crowding_distance))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _clone(self, genome: T) -> T:
        try:
            return copy.deepcopy(genome)
        except Exception:
            return genome

    def _default_crossover(self, left: T, right: T) -> tuple[T, T]:
        return self._clone(left), self._clone(right)

    def _default_mutation(self, genome: T) -> T:
        return self._clone(genome)

    def _export_ranked(self, individual: _Individual[T]) -> RankedIndividual[T]:
        return RankedIndividual(
            genome=individual.genome,
            objectives=individual.objectives,
            rank=individual.rank,
            crowding_distance=individual.crowding_distance,
        )

