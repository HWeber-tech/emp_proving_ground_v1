from __future__ import annotations

from typing import List, Any


class Population:
    def __init__(self, individuals: List[Any] | None = None) -> None:
        self.individuals = individuals or []


