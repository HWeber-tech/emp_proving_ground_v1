"""
Sensory system module for EMP.

Unified 4D+1 sensory cortex scaffolding and signals.
New code should import `SensorSignal`/`IntegratedSignal` from `src.sensory.signals`
and sensors from `src.sensory.[why|how|what|when|anomaly]`.

Legacy modules under `src.sensory.core` and `src.sensory.organs.dimensions` remain for backward
compatibility but are deprecated.
"""

from .signals import SensorSignal, IntegratedSignal  # re-export

__all__ = ['SensorSignal', 'IntegratedSignal']
