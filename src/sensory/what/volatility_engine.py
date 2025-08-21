from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VolConfig:
    bar_interval_minutes: int = 5
    daily_fit_lookback_days: int = 500
    rv_window_minutes: int = 60
    blend_weight: float = 0.7
    calm_thr: float = 0.08
    storm_thr: float = 0.18
    risk_budget_per_trade: float = 0.003
    k_stop: float = 1.3
    var_confidence: float = 0.95
    ewma_lambda: float = 0.94
    use_regime_gate: bool = False
    block_regime: str = "storm"
    gate_mode: str = "block"
    attenuation_factor: float = 0.3
    brake_scale: float = 0.7


class VolatilityEngine:
    """Typed shim for volatility engine. Canonical import surface.
    Real implementation not yet present in the repository; this preserves type stability.
    """
    def __init__(self, config: VolConfig | None = None) -> None:
        self.config: VolConfig = config or VolConfig()


__all__ = ["VolConfig", "VolatilityEngine"]