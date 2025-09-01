from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, cast


@dataclass
class SensorSignal:
    """Base sensor signal structure used across sensory components."""

    signal_type: str
    value: object
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Optional[dict[str, object]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = cast(dict[str, object], {})


@dataclass
class IntegratedSignal:
    """Aggregate signal produced by integrating multiple sensor signals."""

    direction: float
    strength: float
    confidence: float
    contributing: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
