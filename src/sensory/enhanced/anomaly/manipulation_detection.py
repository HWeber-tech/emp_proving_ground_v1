"""
Stub module to satisfy optional import: src.sensory.enhanced.anomaly.manipulation_detection
Runtime-safe no-op implementation for validation flows.
"""

from __future__ import annotations

from typing import Any, Dict, List


class ManipulationDetectionSystem:
    async def detect_manipulation(self, data: Any) -> List[Dict[str, Any]]:
        # No-op stub: returns empty anomalies list
        return []