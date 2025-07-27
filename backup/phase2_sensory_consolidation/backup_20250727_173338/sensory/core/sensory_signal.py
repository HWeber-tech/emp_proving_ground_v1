"""
EMP Sensory Signal v1.1

Standardized sensory signal format for all sensory organs.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any


@dataclass
class SensorySignal:
    """Standardized sensory signal format."""
    timestamp: datetime
    signal_type: str
    value: float
    confidence: float
    metadata: Dict[str, Any]
