#!/usr/bin/env python3
"""
Shim for RealTimeLearningEngine (thinking layer)

Canonical implementation resides at:
- src/sentient/learning/real_time_learning_engine.RealTimeLearningEngine

This module re-exports the canonical class to maintain backward compatibility
for imports that reference src.thinking.learning.real_time_learner.
"""
from __future__ import annotations

from src.sentient.learning.real_time_learning_engine import (
    RealTimeLearningEngine as RealTimeLearningEngine,
)

__all__ = ["RealTimeLearningEngine"]
