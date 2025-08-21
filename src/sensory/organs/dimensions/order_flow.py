from __future__ import annotations


class MarketMicrostructureAnalyzer:
    def analyze_microstructure(self, df: object) -> dict[str, float]:
        """Analyze market microstructure signals (typed no-op)."""
        return {"microstructure_score": 0.0}


class OrderFlowAnalyzer:
    def analyze_institutional_flow(self, df: object) -> dict[str, object]:
        """Analyze institutional order flow (typed no-op)."""
        return {
            "flow_strength": 0.0,
            "institutional_pressure": {"buying_pressure": 0.0, "selling_pressure": 0.0},
        }


__all__ = ["MarketMicrostructureAnalyzer", "OrderFlowAnalyzer"]