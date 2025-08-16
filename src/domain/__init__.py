"""
EMP Domain Models v1.1

Shared domain models used across all layers.
Separates domain concerns from infrastructure concerns.
"""

from __future__ import annotations

# Explicit canonical re-exports (no namespace side-effects)
from src.core.instrument import Instrument as Instrument
from src.core.risk.manager import RiskConfig as RiskConfig
from .models import ExecutionReport as ExecutionReport

__all__ = [
    "RiskConfig",
    "Instrument",
    "ExecutionReport",
]

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Domain Models - Shared Business Entities"
