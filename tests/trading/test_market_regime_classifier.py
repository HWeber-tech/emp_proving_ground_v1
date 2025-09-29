from src.data_foundation.config.execution_config import ExecutionConfig, MarketRegimeModel
from src.trading.execution.market_regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeSignals,
    apply_regime_adjustment,
    classify_market_regime,
)


def test_market_regime_assessment_as_dict():
    cfg = ExecutionConfig()
    signals = MarketRegimeSignals(
        realised_volatility=cfg.regime_model.calm_volatility,
        order_book_liquidity=cfg.regime_model.high_liquidity_ratio,
        sentiment_score=cfg.regime_model.positive_sentiment,
    )

    assessment = classify_market_regime(signals, cfg.regime_model)

    assert isinstance(assessment, MarketRegimeAssessment)
    payload = assessment.as_dict()
    assert payload["regime"] == MarketRegime.CALM.value
    assert 0.0 <= payload["score"] <= 1.0
    assert "volatility_pressure" in payload
    assert "liquidity_pressure" in payload
    assert "sentiment_pressure" in payload
    assert payload["risk_multiplier"] == assessment.risk_multiplier


def test_customised_multipliers_affect_adjustment():
    model = MarketRegimeModel(
        risk_multipliers={"calm": 0.5, "balanced": 1.0, "stressed": 2.0, "dislocated": 3.0}
    )
    signals = MarketRegimeSignals(
        realised_volatility=1.0, order_book_liquidity=0.1, sentiment_score=-0.5
    )

    assessment = classify_market_regime(signals, model)
    assert assessment.regime in {MarketRegime.STRESSED, MarketRegime.DISLOCATED}

    base_slippage = 10.0
    adjusted = apply_regime_adjustment(base_slippage, assessment)
    assert adjusted >= base_slippage
