from __future__ import annotations


class ICTPatternDetector:
    def update_market_data(self, df: object) -> dict[str, object]:
        """Update internal state from market data (typed no-op)."""
        return {}

    def get_institutional_footprint_score(self, df: object) -> float:
        """Return a simple footprint score (typed no-op)."""
        return 0.0


class OrderFlowDataProvider:
    def fetch(self) -> dict[str, object]:
        """Provide order flow related data (typed no-op)."""
        return {}


__all__ = ["ICTPatternDetector", "OrderFlowDataProvider"]
