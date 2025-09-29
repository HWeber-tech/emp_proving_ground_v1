from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SlippageModel:
    base_bps: float = 0.2
    spread_coef: float = 50.0  # bps per 1.0 spread in price units
    imbalance_coef: float = 2.0  # bps per unit imbalance
    sigma_coef: float = 50.0  # bps per 1.0 annualized sigma
    size_coef: float = 5.0  # bps per unit size_ratio


@dataclass
class FeeModel:
    commission_bps: float = 0.1


@dataclass
class ExecutionRiskLimits:
    """Thresholds applied to pre-trade execution checks."""

    max_slippage_bps: float = 25.0
    max_total_cost_bps: float = 35.0
    max_notional_pct_of_equity: float = 0.25


@dataclass
class MarketRegimeModel:
    """Configuration driving the blended market regime classifier."""

    calm_score_threshold: float = 0.25
    balanced_score_threshold: float = 0.55
    stressed_score_threshold: float = 0.8
    volatility_weight: float = 0.5
    liquidity_weight: float = 0.3
    sentiment_weight: float = 0.2
    calm_volatility: float = 0.12
    dislocated_volatility: float = 0.6
    high_liquidity_ratio: float = 0.65
    low_liquidity_ratio: float = 0.35
    positive_sentiment: float = 0.25
    negative_sentiment: float = -0.25
    risk_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "calm": 0.85,
            "balanced": 1.0,
            "stressed": 1.25,
            "dislocated": 1.6,
        }
    )

    def normalised_weights(self) -> tuple[float, float, float]:
        """Return normalised weights for volatility, liquidity, and sentiment."""

        total = max(
            self.volatility_weight + self.liquidity_weight + self.sentiment_weight, 1e-9
        )
        return (
            self.volatility_weight / total,
            self.liquidity_weight / total,
            self.sentiment_weight / total,
        )


@dataclass
class ExecutionConfig:
    slippage: SlippageModel = field(default_factory=SlippageModel)
    fees: FeeModel = field(default_factory=FeeModel)
    limits: ExecutionRiskLimits = field(default_factory=ExecutionRiskLimits)
    regime_model: MarketRegimeModel = field(default_factory=MarketRegimeModel)


yaml: object | None = None
try:  # pragma: no cover
    # Expose a module-level 'yaml' symbol so tests can monkeypatch ec.yaml
    yaml = importlib.import_module("yaml")
except Exception:  # pragma: no cover
    yaml = None


def load_execution_config(path: Optional[str] = None) -> ExecutionConfig:
    if path is None:
        path = os.environ.get("EXECUTION_CONFIG_PATH", "config/execution/execution.yaml")
    if yaml is None or not os.path.exists(path):
        return ExecutionConfig()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = getattr(yaml, "safe_load")(fh) or {}
        ex = data.get("execution", data)
        sl = ex.get("slippage", {})
        fe = ex.get("fees", {})
        limits_cfg = ex.get("limits") or {}
        limits = None
        if limits_cfg:
            limits = ExecutionRiskLimits(
                max_slippage_bps=float(limits_cfg.get("max_slippage_bps", 25.0)),
                max_total_cost_bps=float(limits_cfg.get("max_total_cost_bps", 35.0)),
                max_notional_pct_of_equity=float(
                    limits_cfg.get("max_notional_pct_of_equity", 0.25)
                ),
            )

        default_regime = MarketRegimeModel()
        regime_cfg = ex.get("regime_model") or {}
        multipliers = dict(default_regime.risk_multipliers)
        user_multipliers = regime_cfg.get("risk_multipliers") or {}
        try:
            multipliers.update({k: float(v) for k, v in dict(user_multipliers).items()})
        except Exception:
            multipliers = dict(default_regime.risk_multipliers)

        return ExecutionConfig(
            slippage=SlippageModel(
                base_bps=float(sl.get("base_bps", 0.2)),
                spread_coef=float(sl.get("spread_coef", 50.0)),
                imbalance_coef=float(sl.get("imbalance_coef", 2.0)),
                sigma_coef=float(sl.get("sigma_coef", 50.0)),
                size_coef=float(sl.get("size_coef", 5.0)),
            ),
            fees=FeeModel(commission_bps=float(fe.get("commission_bps", 0.1))),
            limits=limits or ExecutionRiskLimits(),
            regime_model=MarketRegimeModel(
                calm_score_threshold=float(
                    regime_cfg.get("calm_score_threshold", default_regime.calm_score_threshold)
                ),
                balanced_score_threshold=float(
                    regime_cfg.get(
                        "balanced_score_threshold", default_regime.balanced_score_threshold
                    )
                ),
                stressed_score_threshold=float(
                    regime_cfg.get(
                        "stressed_score_threshold", default_regime.stressed_score_threshold
                    )
                ),
                volatility_weight=float(
                    regime_cfg.get("volatility_weight", default_regime.volatility_weight)
                ),
                liquidity_weight=float(
                    regime_cfg.get("liquidity_weight", default_regime.liquidity_weight)
                ),
                sentiment_weight=float(
                    regime_cfg.get("sentiment_weight", default_regime.sentiment_weight)
                ),
                calm_volatility=float(
                    regime_cfg.get("calm_volatility", default_regime.calm_volatility)
                ),
                dislocated_volatility=float(
                    regime_cfg.get(
                        "dislocated_volatility", default_regime.dislocated_volatility
                    )
                ),
                high_liquidity_ratio=float(
                    regime_cfg.get(
                        "high_liquidity_ratio", default_regime.high_liquidity_ratio
                    )
                ),
                low_liquidity_ratio=float(
                    regime_cfg.get("low_liquidity_ratio", default_regime.low_liquidity_ratio)
                ),
                positive_sentiment=float(
                    regime_cfg.get("positive_sentiment", default_regime.positive_sentiment)
                ),
                negative_sentiment=float(
                    regime_cfg.get("negative_sentiment", default_regime.negative_sentiment)
                ),
                risk_multipliers=multipliers,
            ),
        )
    except Exception:
        return ExecutionConfig()
