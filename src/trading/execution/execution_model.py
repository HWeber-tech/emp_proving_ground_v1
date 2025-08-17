from __future__ import annotations

from dataclasses import dataclass

from src.data_foundation.config.execution_config import ExecutionConfig


@dataclass
class ExecContext:
    spread: float
    top_imbalance: float
    sigma_ann: float
    size_ratio: float


def estimate_slippage_bps(ctx: ExecContext, cfg) -> float:
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
    return float(base + spread_coef * s + imbalance_coef * imb + sigma_coef * sig + size_coef * sz)


def estimate_commission_bps(cfg: ExecutionConfig) -> float:
    return max(0.0, cfg.fees.commission_bps)
