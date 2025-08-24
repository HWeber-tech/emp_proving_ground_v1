"""
EMP Thinking Patterns v1.1

Pattern detection and classification modules for the thinking layer.
These patterns analyze sensory signals to identify market conditions,
anomalies, and regime changes.
"""

from __future__ import annotations

from .anomaly_detector import AnomalyDetector
from .cycle_detector import CycleDetector
from .regime_classifier import RegimeClassifier
from .trend_detector import TrendDetector

__all__ = ["TrendDetector", "AnomalyDetector", "RegimeClassifier", "CycleDetector"]
