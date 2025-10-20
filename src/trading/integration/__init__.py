"""Trading integration - Broker interfaces and external connections."""

from __future__ import annotations

__all__ = ["CTraderAdapter"]


def __getattr__(name: str) -> object:
    if name == "CTraderAdapter":
        from .ctrader_adapter import CTraderAdapter

        return CTraderAdapter
    raise AttributeError(name)
