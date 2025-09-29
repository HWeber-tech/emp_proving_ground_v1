# Import directly from module to avoid pulling heavy execution package
from src.data_foundation.config.execution_config import ExecutionConfig, FeeModel, SlippageModel
from src.trading.execution.execution_model import (
    ExecContext,
    estimate_commission_bps,
    estimate_slippage_bps,
)
from src.trading.execution.market_regime import (
    MarketRegime,
    MarketRegimeSignals,
    apply_regime_adjustment,
    classify_market_regime,
)


def test_slippage_estimator_increases_with_inputs():
    cfg = ExecutionConfig(
        slippage=SlippageModel(
            base_bps=0.2, spread_coef=10.0, imbalance_coef=1.0, sigma_coef=10.0, size_coef=1.0
        ),
        fees=FeeModel(commission_bps=0.05),
    )
    low = ExecContext(spread=0.0, top_imbalance=0.0, sigma_ann=0.0, size_ratio=0.0)
    high = ExecContext(spread=0.001, top_imbalance=1.0, sigma_ann=0.2, size_ratio=1.0)
    assert estimate_slippage_bps(low, cfg) < estimate_slippage_bps(high, cfg)
    assert estimate_commission_bps(cfg) == 0.05


def test_regime_adjustment_scales_slippage():
    cfg = ExecutionConfig()
    context = ExecContext(spread=0.0005, top_imbalance=0.2, sigma_ann=0.15, size_ratio=0.5)

    signals_calm = MarketRegimeSignals(
        realised_volatility=0.05, order_book_liquidity=0.9, sentiment_score=0.4
    )
    calm_assessment = classify_market_regime(signals_calm, cfg.regime_model)
    assert calm_assessment.regime is MarketRegime.CALM

    signals_stressed = MarketRegimeSignals(
        realised_volatility=0.9, order_book_liquidity=0.1, sentiment_score=-0.6
    )
    stressed_assessment = classify_market_regime(signals_stressed, cfg.regime_model)
    assert stressed_assessment.regime in {MarketRegime.STRESSED, MarketRegime.DISLOCATED}

    base = estimate_slippage_bps(context, cfg)
    calm = estimate_slippage_bps(context, cfg, regime_assessment=calm_assessment)
    stressed = estimate_slippage_bps(
        context, cfg, regime_assessment=stressed_assessment
    )

    assert calm < base < stressed
    assert apply_regime_adjustment(base, stressed_assessment) == stressed
