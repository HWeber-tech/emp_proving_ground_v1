from __future__ import annotations

import math
from dataclasses import dataclass

from .market_regime import (
    MarketRegimeAssessment,
    apply_regime_adjustment,
)

from src.data_foundation.config.execution_config import ExecutionConfig


def _safe_float(value: object) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return candidate


@dataclass
class ExecContext:
    spread: float
    top_imbalance: float
    sigma_ann: float
    size_ratio: float


def estimate_slippage_bps(
    ctx: ExecContext,
    cfg: ExecutionConfig,
    *,
    regime_assessment: MarketRegimeAssessment | None = None,
) -> float:
    # Simple monotonic function consistent with test expectations
    s = max(ctx.spread, 0.0)
    imb = max(ctx.top_imbalance, 0.0)
    sig = max(ctx.sigma_ann, 0.0)
    sz = max(ctx.size_ratio, 0.0)
    base = getattr(cfg.slippage, "base_bps", 0.0)
    spread_coef = getattr(cfg.slippage, "spread_coef", 0.0)
    imbalance_coef = getattr(cfg.slippage, "imbalance_coef", 0.0)
    sigma_coef = getattr(cfg.slippage, "sigma_coef", 0.0)
    size_coef = getattr(cfg.slippage, "size_coef", 0.0)
    estimate = float(
        base + spread_coef * s + imbalance_coef * imb + sigma_coef * sig + size_coef * sz
    )
    if regime_assessment is not None:
        return apply_regime_adjustment(estimate, regime_assessment)
    return estimate


def estimate_commission_bps(cfg: ExecutionConfig) -> float:
    return max(0.0, float(cfg.fees.commission_bps))


def calculate_edge_ticks(
    delta_hat: float | int | None,
    spread_ticks: float | int | None,
    *,
    spread_floor: float | int | None = None,
) -> float:
    """Return expected edge in ticks given a normalised price delta.

    Parameters
    ----------
    delta_hat:
        Normalised price change signal (dimensionless).
    spread_ticks:
        Observed bid/ask spread expressed in ticks (or price units when tick
        size is unknown).
    spread_floor:
        Optional minimum spread floor in ticks used when the observed spread is
        narrower than a volatility-derived bound.
    """

    delta = _safe_float(delta_hat)
    spread = _safe_float(spread_ticks)
    floor = _safe_float(spread_floor) if spread_floor is not None else 0.0

    if delta is None or spread is None:
        return 0.0

    effective_spread = max(spread, floor if floor is not None else 0.0, 0.0)
    if effective_spread == 0.0:
        return 0.0
    return delta * effective_spread


def estimate_total_cost_ticks(
    spread_ticks: float | int | None,
    *,
    slippage: float | int | None = None,
    fees: float | int | None = None,
    adverse_selection_penalty: float | int | None = None,
) -> float:
    """Estimate total cost in ticks using the roadmap D.1.1 rule."""

    spread = _safe_float(spread_ticks)
    spread_component = 0.0
    if spread is not None:
        spread_component = 0.5 * max(spread, 0.0)

    total = spread_component
    for component in (slippage, fees, adverse_selection_penalty):
        value = _safe_float(component)
        if value is None:
            continue
        total += value
    return total


__all__ = [
    "ExecContext",
    "estimate_slippage_bps",
    "estimate_commission_bps",
    "calculate_edge_ticks",
    "estimate_total_cost_ticks",
]
