"""
Trading Models Package
======================

This package contains data models for trading entities including orders,
positions, and market data.
"""

from __future__ import annotations

__all__: list[str] = []

# Safe, guarded re-exports to avoid runtime errors if submodules are absent.
try:
    from .order import Order as Order
    from .order import OrderStatus as OrderStatus
    from .order import OrderType as OrderType
except ImportError:  # pragma: no cover
    pass
else:
    __all__.extend(["Order", "OrderStatus", "OrderType"])

try:
    from .position import Position as Position
except ImportError:  # pragma: no cover
    pass
else:
    __all__.append("Position")

try:
    from .trade import Trade as Trade
except ImportError:  # pragma: no cover
    pass
else:
    __all__.append("Trade")
