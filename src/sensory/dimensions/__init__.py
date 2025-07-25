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

from ..organs.dimensions.what_organ import WhatEngine as WhatDimension
from ..organs.dimensions.when_organ import WhenEngine as WhenDimension
from ..organs.dimensions.anomaly_dimension import AnomalyDimension
from ..organs.dimensions.chaos_dimension import ChaosDimension

__all__ = [
    'WhatDimension',
    'WhenDimension', 
    'AnomalyDimension',
    'ChaosDimension'
]
