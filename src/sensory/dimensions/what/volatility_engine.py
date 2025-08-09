from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from src.data_foundation.schemas import VolSignal
from src.operational.metrics import set_vol_sigma, inc_vol_regime, set_vol_divergence


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
    # Regime gating
    use_regime_gate: bool = False
    block_regime: str = "storm"


class Garch11:
    def __init__(self, omega: float, alpha: float, beta: float):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def fit(returns: list[float]) -> "Garch11 | None":
        # Minimal placeholder fit: use EWMA variance to back out params conservatively
        try:
            lam = 0.94
            var = 0.0
            for r in returns:
                var = lam * var + (1 - lam) * (r * r)
            # crude mapping: omega = var*(1-beta), alpha=(1-beta)/2
            beta = 0.85
            alpha = 0.1
            omega = var * (1 - beta)
            if alpha < 0 or beta < 0 or alpha + beta >= 1:
                return None
            return Garch11(omega, alpha, beta)
        except Exception:
            return None

    def next_var(self, last_var: float, last_resid2: float) -> float:
        return max(self.omega + self.alpha * last_resid2 + self.beta * last_var, 1e-12)


def ewma_var(returns: list[float], lam: float) -> float:
    var = 0.0
    for r in returns:
        var = lam * var + (1 - lam) * (r * r)
    return max(var, 1e-12)


def annualize_sigma2(sigma2: float, bar_minutes: int) -> float:
    # scale factor: number of bars per year
    bars_per_day = int(24 * 60 / bar_minutes)
    scale = math.sqrt(252 * bars_per_day)
    return math.sqrt(sigma2) * scale


def classify_regime(sigma_ann: float, calm: float, storm: float) -> str:
    if sigma_ann < calm:
        return "calm"
    if sigma_ann > storm:
        return "storm"
    return "normal"


def compute_var95_1d(sigma_ann: float, exposure: float, bar_minutes: int) -> float:
    # 1-day sigma from annualized
    bars_per_day = int(24 * 60 / bar_minutes)
    sigma_day = sigma_ann / math.sqrt(252)
    return 1.65 * sigma_day * exposure


def sizing_multiplier(regime: str, sigma_ann: float) -> float:
    # simple cap for now
    if regime == "calm":
        return 1.0
    if regime == "normal":
        return 0.67
    return 0.4


def stop_multiplier() -> float:
    return 1.3


def vol_signal(symbol: str,
               t,
               returns_5m_last_hour: list[float],
               daily_returns: list[float],
               cfg: VolConfig = VolConfig()) -> VolSignal:
    # Fit GARCH daily; fallback to EWMA
    model = Garch11.fit(daily_returns)
    if model is None:
        # EWMA fallback using cfg.ewma_lambda
        sigma2_daily = ewma_var(daily_returns, cfg.ewma_lambda)
    else:
        # one-step variance from last daily residual approximation
        last_var = ewma_var(daily_returns[:-1], cfg.ewma_lambda)
        last_resid2 = daily_returns[-1] ** 2 if daily_returns else 0.0
        sigma2_daily = model.next_var(last_var, last_resid2)

    rv = sum(r * r for r in returns_5m_last_hour)
    sigma2 = cfg.blend_weight * sigma2_daily + (1 - cfg.blend_weight) * rv
    sigma_ann = annualize_sigma2(sigma2, cfg.bar_interval_minutes)
    reg = classify_regime(sigma_ann, cfg.calm_thr, cfg.storm_thr)
    s_mult = sizing_multiplier(reg, sigma_ann)
    v95 = compute_var95_1d(sigma_ann, exposure=1.0, bar_minutes=cfg.bar_interval_minutes)
    try:
        set_vol_sigma(symbol, sigma_ann)
        # crude divergence: absolute diff of annualized vols
        div = abs(annualize_sigma2(rv, cfg.bar_interval_minutes) - annualize_sigma2(sigma2_daily, cfg.bar_interval_minutes))
        set_vol_divergence(symbol, div)
        inc_vol_regime(symbol, reg)
    except Exception:
        pass
    return VolSignal(
        symbol=symbol,
        t=t,
        sigma_ann=float(sigma_ann),
        var95_1d=float(v95),
        regime=reg,
        sizing_multiplier=float(s_mult),
        stop_mult=stop_multiplier(),
        quality=0.9 if model is not None else 0.6,
    )


