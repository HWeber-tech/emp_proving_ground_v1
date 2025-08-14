from __future__ import annotations

import random
from typing import Tuple


def single_point_crossover(p1: list[float], p2: list[float]) -> Tuple[list[float], list[float]]:
    if not p1 or not p2:
        return p1, p2
    point = random.randint(1, min(len(p1), len(p2)) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]


def gaussian_mutation(individual: list[float], sigma: float = 0.1) -> list[float]:
    import random
    if not individual:
        return individual
    idx = random.randrange(len(individual))
    individual[idx] += random.gauss(0.0, sigma)
    return individual


