from __future__ import annotations

from typing import Iterable

__all__ = ["AnomalyIntelligenceEngine"]


class AnomalyIntelligenceEngine:
    def analyze_anomaly_intelligence(
        self, series: list[float] | Iterable[float] | None
    ) -> dict[str, object]:
        values = list(series or [])
        try:
            max_abs = max((abs(float(x)) for x in values), default=0.0)
        except Exception:
            max_abs = 0.0
        # Clamp to [0, 1] for a minimal anomaly score
        signal = float(max(0.0, min(1.0, max_abs)))

        return {
            "max_abs": float(max_abs),
            "signal": signal,
            "meta": {"source": "sensory.anomaly"},
        }
