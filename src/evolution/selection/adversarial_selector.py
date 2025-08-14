"""
Stub module to satisfy optional import: src.evolution.selection.adversarial_selector
Runtime-safe no-op implementation for validation flows.
"""

from __future__ import annotations

from typing import Any, List, Sequence


class AdversarialSelector:
    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs

    def select(self, population: Sequence[Any] | None, k: int = 1) -> List[Any]:
        """
        No-op stub: returns the first k items from the population (if provided).
        """
        if not population:
            return []
        try:
            return list(population)[: max(0, k)]
        except Exception:
            return []