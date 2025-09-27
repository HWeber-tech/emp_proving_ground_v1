"""Volatility targeting helpers for roadmap sizing adapters."""

from __future__ import annotations

from typing import Iterable

__all__ = ["volatility_target_position_size"]


def _effective_volatility(values: Iterable[float]) -> float:
    candidates = [float(value) for value in values if value and value > 0]
    if not candidates:
        raise ValueError("at least one positive volatility estimate is required")
    return sum(candidates) / len(candidates)


def volatility_target_position_size(
    capital: float,
    target_volatility: float,
    *,
    realized_volatility: float | None = None,
    garch_volatility: float | None = None,
    price: float | None = None,
    contract_multiplier: float = 1.0,
    max_leverage: float = 10.0,
    min_size: float = 0.0,
) -> float:
    """Return a position size that targets the supplied volatility budget."""

    capital = max(0.0, float(capital))
    target_volatility = float(target_volatility)
    if capital <= 0:
        return max(min_size, 0.0)
    if target_volatility <= 0:
        raise ValueError("target_volatility must be positive")

    effective_vol = _effective_volatility(
        value
        for value in (realized_volatility, garch_volatility)
        if value is not None
    )

    raw_notional = capital * (target_volatility / effective_vol)
    max_notional = capital * float(max_leverage)
    notional = max(min(raw_notional, max_notional), min_size)

    if price is None:
        return notional

    price = float(price)
    if price <= 0:
        raise ValueError("price must be positive when provided")
    contract_multiplier = max(1.0e-12, float(contract_multiplier))
    units = notional / (price * contract_multiplier)
    return max(min_size, units)

