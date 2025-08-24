from __future__ import annotations
from typing import Iterable, Dict, Any

# Typing-only stub to shadow the runtime module so mypy doesn't load the .py file
# under two module names. Canonical import path used across the project is src.*.
# This .pyi takes precedence over the .py and eliminates duplicate-module loading.

class AnomalyIntelligenceEngine:
    def analyze_anomaly_intelligence(self, series: list[float] | Iterable[float] | None) -> Dict[str, Any]: ...