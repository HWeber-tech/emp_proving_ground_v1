from __future__ import annotations

from typing import Protocol, Any


class FitnessEvaluator(Protocol):
    def evaluate(self, individual: Any) -> float: ...


