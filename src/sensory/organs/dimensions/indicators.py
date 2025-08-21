from __future__ import annotations



class TechnicalIndicators:
    def compute(self) -> float:
        """Minimal no-op indicator computation."""
        return 0.0

    def calculate_all(self, df: object) -> dict[str, object]:
        """Return a minimal indicators bundle."""
        return {}
        

__all__ = ["TechnicalIndicators"]