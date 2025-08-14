from __future__ import annotations

from typing import Any, Protocol


class FitnessEvaluator(Protocol):
    def evaluate(self, individual: Any) -> float: ...


