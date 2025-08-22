"""
Stub module to satisfy optional import: src.sensory.enhanced.anomaly.manipulation_detection
Runtime-safe no-op implementation for validation flows.
"""

from __future__ import annotations

from typing import Mapping


class ManipulationDetectionSystem:
    async def detect_manipulation(self, data: Mapping[str, object]) -> list[dict[str, object]]:
        # No-op stub: returns empty anomalies list
        return []