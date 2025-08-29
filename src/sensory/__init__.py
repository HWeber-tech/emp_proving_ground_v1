"""
Sensory system module for EMP.

Unified 4D+1 sensory cortex scaffolding and signals.
New code should import `SensorSignal`/`IntegratedSignal` from `sensory.signals`
and sensors from `sensory.[why|how|what|when|anomaly]`.

Legacy modules under `sensory.core` and `sensory.organs.dimensions` remain for backward
compatibility but are deprecated.
"""

# Package marker for src.sensory to ensure consistent import resolution by type checkers.
__all__: list[str] = []
