from __future__ import annotations

import math
import random

import pytest

from src.evolution.algorithms import NSGA2, NSGA2Config


def test_nsga2_non_dominated_sorting_and_crowding() -> None:
    nsga = NSGA2(
        evaluate=lambda genome: genome,
        config=NSGA2Config(
            population_size=4,
            crossover_probability=0.0,
            mutation_probability=0.0,
        ),
    )

    population = [
        (1.0, 3.0),
        (2.0, 2.0),
        (3.0, 1.0),
        (2.5, 2.5),
    ]

    result = nsga.step(population)

    assert len(result.population) == 4
    best_front = {tuple(individual.genome) for individual in result.fronts[0]}
    assert best_front == {(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)}

    dominated = [tuple(individual.genome) for individual in result.fronts[1]]
    assert dominated == [(2.5, 2.5)]

    distances = [individual.crowding_distance for individual in result.fronts[0]]
    assert any(math.isinf(distance) for distance in distances)


@pytest.mark.parametrize("seed", [7, 17, 101])
def test_nsga2_prefers_offspring_that_dominate_parents(seed: int) -> None:
    rng = random.Random(seed)

    def mutate(genome: tuple[float, float]) -> tuple[float, float]:
        return max(0.0, genome[0] - 10.0), max(0.0, genome[1] - 10.0)

    nsga = NSGA2(
        evaluate=lambda genome: genome,
        config=NSGA2Config(
            population_size=4,
            crossover_probability=0.0,
            mutation_probability=1.0,
        ),
        mutate=mutate,
        rng=rng,
    )

    population = [
        (5.0, 5.0),
        (6.0, 6.0),
        (7.0, 7.0),
        (8.0, 8.0),
    ]

    result = nsga.step(population)

    assert all(individual == (0.0, 0.0) for individual in result.population)
    assert all(individual.rank == 0 for individual in result.fronts[0])


def test_nsga2_handles_mixed_objective_directions() -> None:
    nsga = NSGA2(
        evaluate=lambda genome: (genome["profit"], genome["risk"]),
        config=NSGA2Config(
            population_size=4,
            crossover_probability=0.0,
            mutation_probability=0.0,
            maximise_objectives=(True, False),
        ),
    )

    population = [
        {"name": "baseline", "profit": 10.0, "risk": 5.0},
        {"name": "defensive", "profit": 9.5, "risk": 4.0},
        {"name": "aggressive", "profit": 12.0, "risk": 6.2},
        {"name": "efficient", "profit": 11.0, "risk": 4.2},
    ]

    result = nsga.step(population)

    front_names = {individual.genome["name"] for individual in result.fronts[0]}
    assert front_names == {"defensive", "aggressive", "efficient"}

    dominated_names = {individual.genome["name"] for individual in result.fronts[-1]}
    assert dominated_names == {"baseline"}
