from __future__ import annotations


class MarketRegimeDetector:
    def get_temporal_regime(self, df: object) -> dict[str, object]:
        """Typed no-op returning a minimal temporal regime summary."""
        return {"temporal_confidence": 0.0, "temporal_strength": 0.0}

    def detect_market_regime(self, df: object) -> str:
        """Return a canonical string label for the detected market regime."""
        return "unknown"

    def calculate_temporal_confidence(self, df: object) -> float:
        """Calculate a basic temporal confidence metric."""
        return 0.0

    def calculate_temporal_strength(self, df: object) -> float:
        """Calculate a basic temporal strength metric."""
        return 0.0

    def get_chrono_behavior(self, df: object) -> dict[str, object]:
        """Return minimal chrono/session behavior analysis."""
        return {"session": "unknown", "activity": 0.0}


class TemporalAnalyzer:
    def get_temporal_regime(self, df: object) -> dict[str, object]:
        """Analyze temporal regime (typed no-op)."""
        return {"temporal_confidence": 0.0, "temporal_strength": 0.0}


__all__ = ["MarketRegimeDetector", "TemporalAnalyzer"]