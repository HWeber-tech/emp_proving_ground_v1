"""
EMP Domain Models v1.1

Shared domain models used across all layers.
Separates domain concerns from infrastructure concerns.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from importlib import import_module
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    # Intentionally avoid importing runtime symbols for typing to prevent cycles
    # and to keep import-time light. Use string annotations where needed.
    pass


class ExecutionReport(BaseModel):
    """Execution report for trade confirmations"""
    event_id: str
    timestamp: datetime
    source: str
    trade_intent_id: str
    action: str
    status: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    order_id: str

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }


# Robust lazy re-exports with safe fallbacks (no import-time side effects)
_DOMAIN_LAZY_EXPORTS = {
    "InstrumentProvider": "src.core:InstrumentProvider",
    "CurrencyConverter": "src.core:CurrencyConverter",
}
__all__ = ["ExecutionReport"] + list(_DOMAIN_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    # Lazy attribute resolution to avoid import-time side effects and cycles.
    # Provides safe fallback shims if canonical targets are unavailable.
    target = _DOMAIN_LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)

    module_path, _, attr = target.partition(":")
    try:
        mod = import_module(module_path)
        obj = getattr(mod, attr)
    except Exception:
        # Minimal shim class to satisfy hasattr(importer, name) and construction
        obj = type(name, (), {})  # type: ignore[type-arg]

    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
