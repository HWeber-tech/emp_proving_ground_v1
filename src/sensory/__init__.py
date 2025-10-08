"""
Sensory system module for EMP.

Unified 4D+1 sensory cortex scaffolding and signals.
New code should import `SensorSignal`/`IntegratedSignal` from `src.sensory.signals`
and sensors from `src.sensory.[why|how|what|when|anomaly]`.

Legacy modules under `src.sensory.core` have been retired, and only the
`src.sensory.organs.dimensions.executable_organs` wrappers remain for backward
compatibility with older integration tests. New code should avoid importing
other modules from `src.sensory.organs.dimensions`.
"""

# Re-export canonical signals (no in-file class definitions)
from src.sensory.signals import (
    IntegratedSignal as IntegratedSignal,
)
from src.sensory.signals import (
    SensorSignal as SensorSignal,
)

__all__ = ["SensorSignal", "IntegratedSignal"]
