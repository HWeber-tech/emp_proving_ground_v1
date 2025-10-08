"""
EMP Domain Models v1.1

Shared domain models used across all layers.
Separates domain concerns from infrastructure concerns.
"""

from __future__ import annotations

# Explicit canonical re-exports (no namespace side-effects)
from src.core.instrument import Instrument as Instrument
from src.config.risk.risk_config import RiskConfig as RiskConfig

__all__ = [
    "RiskConfig",
    "Instrument",
]

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Domain Models - Shared Business Entities"


def __getattr__(name: str):
    if name == "ExecutionReport":
        raise ModuleNotFoundError(
            "src.domain.ExecutionReport was removed. Use trading telemetry payloads "
            "or define domain DTOs close to their consumers instead."
        )
    raise AttributeError(name)
