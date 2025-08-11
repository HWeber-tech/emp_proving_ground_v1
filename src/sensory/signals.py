from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SensorSignal:
    """Base sensor signal structure used across sensory components."""

    signal_type: str
    value: Any
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class IntegratedSignal:
    """Aggregate signal produced by integrating multiple sensor signals."""

    direction: float
    strength: float
    confidence: float
    contributing: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


