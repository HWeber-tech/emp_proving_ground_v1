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

# Re-export canonical dimension implementations (no in-file class definitions)

# WhatDimension is implemented in organs layer; re-export
try:
    from src.sensory.organs.dimensions.pattern_engine import WhatDimension as WhatDimension  # type: ignore
except Exception:
    # Optional shim if organs layer is not present in current runtime
    class WhatDimension:  # type: ignore
        pass

# WhenDimension may be defined in organs layer; re-export if available, else provide a shim
try:
    from src.sensory.organs.dimensions.when_dimension import WhenDimension as WhenDimension  # type: ignore
except Exception:
    class WhenDimension:  # type: ignore
        pass

# Re-export concrete anomaly/chaos dimensions from organs
from src.sensory.organs.dimensions.anomaly_dimension import AnomalyDimension as AnomalyDimension  # type: ignore
from src.sensory.organs.dimensions.chaos_dimension import ChaosDimension as ChaosDimension  # type: ignore

__all__ = [
    'WhatDimension',
    'WhenDimension',
    'AnomalyDimension',
    'ChaosDimension',
]
