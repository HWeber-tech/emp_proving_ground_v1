from __future__ import annotations

__all__ = ["InstitutionalIntelligenceEngine"]


class InstitutionalIntelligenceEngine:
    def analyze_institutional_intelligence(
        self, data: dict[str, float] | None = None
    ) -> dict[str, object]:
        payload: dict[str, float] = dict(data or {})
        participation = float(payload.get("participation", 0.0))
        liquidity = float(payload.get("liquidity", 0.0))
        # Normalize participation [0, 1] to [-1, 1]
        p = max(0.0, min(1.0, participation))
        signal = (p * 2.0) - 1.0

        return {
            "signal": float(signal),
            "participation": float(participation),
            "liquidity": float(liquidity),
            "meta": {"source": "sensory.how"},
        }
