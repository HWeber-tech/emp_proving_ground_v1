from __future__ import annotations

from dataclasses import dataclass

from src.data_foundation.config.execution_config import ExecutionConfig


@dataclass
class ExecContext:
    spread: float
    top_imbalance: float
    sigma_ann: float
    size_ratio: float  # planned size vs baseline (0..1)


def estimate_slippage_bps(ctx: ExecContext, cfg: ExecutionConfig) -> float:
    sl = cfg.slippage
    bps = (
        sl.base_bps +
        sl.spread_coef * ctx.spread +
        sl.imbalance_coef * abs(ctx.top_imbalance) +
        sl.sigma_coef * max(0.0, ctx.sigma_ann) +
        sl.size_coef * max(0.0, ctx.size_ratio)
    )
    return max(0.0, bps)


def estimate_commission_bps(cfg: ExecutionConfig) -> float:
    return max(0.0, cfg.fees.commission_bps)


