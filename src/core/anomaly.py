"""
Core Anomaly Detection Port (Protocol)
=====================================

Defines a minimal, domain-agnostic interface for anomaly/manipulation detection.

- Do not import sensory/trading here (core must remain dependency-free).
- Domain packages should depend on this Protocol and receive implementations via DI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class AnomalyEvent:
    """Generic anomaly/manipulation event."""
    timestamp: Any
    kind: str
    score: float
    metadata: Dict[str, Any]


@runtime_checkable
class AnomalyDetector(Protocol):
    """Abstract anomaly/manipulation detector."""

    async def detect_manipulation(self, data: Any) -> List[Dict[str, Any]] | List[AnomalyEvent]:
        """
        Detect anomalous/manipulative activity from a tabular/time-series dataset.

        Accepts and returns flexible structures to avoid coupling:
        - Returns a list of dicts or AnomalyEvent objects.
        """
        ...


class NoOpAnomalyDetector:
    """Safe fallback that reports no anomalies."""

    async def detect_manipulation(self, data: Any) -> List[Dict[str, Any]]:
        return []


def is_anomaly_detector(obj: object) -> bool:
    """Runtime duck-typing helper."""
    try:
        return isinstance(obj, AnomalyDetector)
    except Exception:
        return False