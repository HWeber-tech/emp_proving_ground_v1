"""
Sensory Dimensions Module
========================

This module provides the complete 5D+1 sensory cortex implementation
for the EMP Proving Ground Phase 2 completion.

Dimensions:
- WHY: Macro Predator Intelligence
- HOW: Institutional Footprint Hunter
- WHAT: Pattern Synthesis Engine
- WHEN: Temporal Advantage System
- ANOMALY: Manipulation Detection
- CHAOS: Antifragile Adaptation
"""

from __future__ import annotations

# Re-export canonical dimension implementations

# WHAT dimension: implemented in organs layer
from src.sensory.organs.dimensions.pattern_engine import WhatDimension

# ANOMALY and CHAOS dimensions: implemented in organs layer
from src.sensory.organs.dimensions.anomaly_dimension import AnomalyDimension
from src.sensory.organs.dimensions.chaos_dimension import ChaosDimension

__all__ = [
    "WhatDimension",
    "AnomalyDimension",
    "ChaosDimension",
]
