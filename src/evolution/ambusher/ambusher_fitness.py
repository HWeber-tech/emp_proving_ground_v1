from __future__ import annotations


class AmbusherFitnessFunction:
    def __init__(self) -> None:
        """Typed shim for ambusher fitness function."""
        pass

    def score(self, x: float) -> float:
        """Compute a fitness score for the given input."""
        return 0.0


__all__ = ["AmbusherFitnessFunction"]