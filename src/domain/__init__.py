"""
EMP Domain Models v1.1

Shared domain models used across all layers.
Separates domain concerns from infrastructure concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Any

# Explicit canonical re-exports (no namespace side-effects)
from src.core.instrument import Instrument as Instrument

# For static checking we avoid importing the dynamically-resolved RiskConfig
# (it is loaded at runtime via importlib in src.core.risk.manager). Use an
# Any alias while TYPE_CHECKING so mypy does not attempt to resolve the module.
if TYPE_CHECKING:
    from typing import Any as RiskConfig  # type: ignore[misc]
else:
    class _RiskConfigRT:
        pass

    RiskConfig = _RiskConfigRT

from .models import ExecutionReport as ExecutionReport

__all__: list[str] = [
    "RiskConfig",
    "Instrument",
    "ExecutionReport",
]

__version__: Final[str] = "1.1.0"
__author__ = "EMP System"
__description__ = "Domain Models - Shared Business Entities"
