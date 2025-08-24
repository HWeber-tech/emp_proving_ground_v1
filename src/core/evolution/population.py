from __future__ import annotations


class Population:
    def __init__(self, individuals: list[object] | None = None) -> None:
        self.individuals = individuals or []
