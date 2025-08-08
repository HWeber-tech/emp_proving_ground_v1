#!/usr/bin/env python3
"""
Common signal dataclasses for the 4D+1 sensory cortex and integrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SensorSignal:
    name: str
    strength: float  # -1..1 (bearish..bullish)
    confidence: float  # 0..1


@dataclass
class IntegratedSignal:
    direction: str  # bullish|bearish|neutral
    strength: float
    confidence: float
    contributing: List[str]


