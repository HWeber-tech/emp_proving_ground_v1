from __future__ import annotations
from typing import Protocol


class FitnessEvaluator(Protocol):
    def evaluate(self, individual: object) -> float: ...


