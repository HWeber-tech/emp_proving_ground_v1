from __future__ import annotations

from typing import Any, Dict, Iterable

# Typing-only stub to avoid loading the real module under two names.
# Do NOT import from src.* here to prevent duplicate module path detection.

class AnomalyIntelligenceEngine:
    def analyze_anomaly_intelligence(
        self, series: list[float] | Iterable[float] | None
    ) -> Dict[str, Any]: ...
