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

from .what_dimension import WhatDimension
from .when_dimension import WhenDimension
from .anomaly_dimension import AnomalyDimension
from .chaos_dimension import ChaosDimension

__all__ = [
    'WhatDimension',
    'WhenDimension', 
    'AnomalyDimension',
    'ChaosDimension'
]
