"""Canonical risk manager facade and factory helpers.

This module owns the backwards-compatible facade that historically lived under
``src.core.risk.manager``.  The roadmap calls for consolidating shim modules
under their canonical packages so downstream code consumes the authoritative
risk implementation directly from :mod:`src.risk`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from src.config.risk.risk_config import RiskConfig

from .risk_manager_impl import RiskManagerImpl

__all__ = ["RiskManager", "get_risk_manager", "create_risk_manager"]

def _coerce_risk_config(config: RiskConfig | Mapping[str, object] | None) -> RiskConfig:
    if config is None:
        raise ValueError("RiskConfig is required for RiskManager")
    if isinstance(config, RiskConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("RiskConfig payload must be a mapping or RiskConfig instance")
    try:
        return RiskConfig.parse_obj(dict(config))
    except ValidationError as exc:
        raise ValueError("Invalid RiskConfig payload for RiskManager") from exc


class RiskManager(RiskManagerImpl):
    """Backwards-compatible facade for :class:`RiskManagerImpl`.

    Legacy callers imported ``RiskManager`` from ``src.core`` and expected a
    synchronous ``validate_trade`` helper.  The facade now lives alongside the
    canonical implementation so integrations resolve a single namespace while
    retaining the strict risk checks provided by :class:`RiskManagerImpl`.
    """

    def __init__(
        self,
        config: RiskConfig | Mapping[str, object] | None = None,
        *,
        initial_balance: float | Decimal | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_config = _coerce_risk_config(config)
        starting_balance = (
            float(initial_balance)
            if initial_balance is not None
            else 10000.0
        )
        super().__init__(
            initial_balance=starting_balance,
            risk_config=resolved_config,
            **kwargs,
        )

    def validate_trade(
        self,
        size: Decimal,
        entry_price: Decimal,
        *,
        symbol: str = "",
        stop_loss_pct: float | Decimal | None = None,
        portfolio_state: Mapping[str, object] | None = None,
    ) -> bool:
        """Synchronously evaluate a trade against configured limits."""

        _ = portfolio_state  # retained for API compatibility

        quantity = float(size)
        price = float(entry_price)
        stop_loss = float(stop_loss_pct) if stop_loss_pct is not None else 0.0

        if quantity <= 0 or price <= 0:
            return False

        if quantity < self._min_position_size or quantity > self._max_position_size:
            return False

        if self._mandatory_stop_loss and not self._research_mode and stop_loss <= 0:
            return False

        effective_stop_loss = stop_loss if stop_loss > 0 else self._risk_per_trade
        risk_amount = quantity * price * effective_stop_loss
        max_allowed_risk = self._compute_risk_budget()

        if max_allowed_risk <= 0:
            return False

        if risk_amount > max_allowed_risk:
            return False

        self._canonicalise_positions()
        projected_risk = self._aggregate_position_risk()
        symbol_key = self._position_key(symbol)
        projected_risk[symbol_key] = projected_risk.get(symbol_key, 0.0) + risk_amount

        aggregate_risk = self.risk_manager.assess_risk(projected_risk)
        if aggregate_risk > 1.0:
            return False

        sector = self._resolve_sector(symbol_key)
        if sector is not None:
            sector_budget = self._sector_budget(sector)
            if sector_budget is not None:
                current_exposure = self._compute_sector_risk(sector)
                projected_exposure = current_exposure + risk_amount
                if sector_budget <= 0.0 and projected_exposure > 0.0:
                    return False
                if sector_budget > 0.0 and projected_exposure > sector_budget:
                    return False

        return True


def create_risk_manager(
    *,
    config: RiskConfig | Mapping[str, object] | None = None,
    initial_balance: float | Decimal | None = None,
    **kwargs: Any,
) -> RiskManager:
    """Explicit factory returning a :class:`RiskManager`."""

    return RiskManager(config=config, initial_balance=initial_balance, **kwargs)


def get_risk_manager(
    config: RiskConfig | Mapping[str, object] | None = None,
    *,
    initial_balance: float | Decimal | None = None,
    **kwargs: Any,
) -> RiskManager:
    """Factory kept for backwards compatibility with legacy callers.

    The roadmap prefers :func:`create_risk_manager`; this wrapper remains only for
    legacy code paths and therefore emits a :class:`DeprecationWarning` when
    invoked.
    """

    warnings.warn(
        "get_risk_manager is deprecated; use create_risk_manager instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return RiskManager(config=config, initial_balance=initial_balance, **kwargs)
