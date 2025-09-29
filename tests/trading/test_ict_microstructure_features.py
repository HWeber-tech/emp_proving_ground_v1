from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.sensory.organs.dimensions.institutional_tracker import (
    FairValueGap,
    InstitutionalFootprint,
    LiquiditySweep,
)
from src.trading.strategies import (
    MeanReversionStrategy,
    MeanReversionStrategyConfig,
    StrategySignal,
    VolatilityBreakoutConfig,
    VolatilityBreakoutStrategy,
)
from src.trading.strategies.signals.ict_microstructure import (
    ICTMicrostructureAnalyzer,
    ICTMicrostructureFeatures,
)


class _StaticAnalyzer:
    def __init__(self, features: ICTMicrostructureFeatures | None) -> None:
        self._features = features

    async def summarise(self, market_data: dict[str, object], symbol: str) -> ICTMicrostructureFeatures | None:
        return self._features


class _StubFootprintHunter:
    def __init__(self, footprint: InstitutionalFootprint) -> None:
        self._footprint = footprint

    async def analyze_institutional_footprint(self, market_data, symbol):  # noqa: ANN001
        return self._footprint


@pytest.mark.asyncio
async def test_microstructure_analyzer_reduces_footprint() -> None:
    now = datetime.now(tz=timezone.utc)
    footprint = InstitutionalFootprint(
        order_blocks=[],
        fair_value_gaps=[
            FairValueGap(
                type="bullish",
                start_price=1.1000,
                end_price=1.1015,
                gap_range=(1.1000, 1.1015),
                timestamp=now,
                strength=0.8,
                fill_probability=0.6,
                imbalance_ratio=0.0015,
            )
        ],
        liquidity_sweeps=[
            LiquiditySweep(
                direction="down",
                sweep_level=1.099,
                liquidity_pool="equal lows",
                sweep_size=0.0005,
                volume_spike=1.4,
                reversal_probability=0.7,
                institutional_follow_through=True,
            )
        ],
        smart_money_flow=0.4,
        institutional_bias="bullish",
        confidence_score=0.85,
        market_structure="uptrend",
        key_levels=[1.09, 1.10],
    )

    analyzer = ICTMicrostructureAnalyzer(
        footprint_hunter=_StubFootprintHunter(footprint), minimum_candles=3
    )

    payload = {
        "close": [1.0990, 1.1000, 1.1010, 1.1020],
        "open": [1.0985, 1.0995, 1.1005, 1.1015],
        "high": [1.1005, 1.1015, 1.1025, 1.1035],
        "low": [1.0975, 1.0985, 1.0995, 1.1005],
        "volume": [1_000, 1_050, 1_075, 1_120],
        "timestamp": [now - timedelta(minutes=i) for i in range(4)],
    }

    features = await analyzer.summarise({"EURUSD": payload}, "EURUSD")
    assert isinstance(features, ICTMicrostructureFeatures)
    metadata = features.to_metadata()
    assert metadata["recent_fair_value_gap"]["type"] == "bullish"
    assert metadata["recent_liquidity_sweep"]["direction"] == "down"
    assert features.confidence == pytest.approx(0.85, rel=1e-6)


def test_alignment_assessment_handles_buy_and_sell() -> None:
    features = ICTMicrostructureFeatures(
        institutional_bias="bullish",
        smart_money_flow=0.6,
        fair_value_gap_count=2,
        recent_fvg_type="bullish",
        strongest_fvg_strength=0.7,
        liquidity_sweep_count=1,
        recent_liquidity_sweep_direction="down",
        liquidity_sweep_bias="bullish",
        confidence=0.9,
        key_levels=(1.0, 1.1),
    )

    buy_score, buy_breakdown = features.alignment_assessment("BUY")
    assert buy_score > 0
    assert buy_breakdown["institutional_bias"] > 0

    sell_score, sell_breakdown = features.alignment_assessment("SELL")
    assert sell_score < 0
    assert sell_breakdown["institutional_bias"] < 0


@pytest.mark.asyncio
async def test_mean_reversion_confidence_adjusts_with_microstructure() -> None:
    market = {"EURUSD": {"close": [100.0] * 9 + [97.0]}}
    config = MeanReversionStrategyConfig(lookback=10, zscore_entry=1.5)

    base_strategy = MeanReversionStrategy(
        "mr-base",
        ["EURUSD"],
        capital=750_000,
        config=config,
        microstructure_analyzer=_StaticAnalyzer(None),
    )
    base_signal = await base_strategy.generate_signal(market, "EURUSD")
    assert base_signal.action == "BUY"
    assert base_signal.confidence > 0.0

    aligned_features = ICTMicrostructureFeatures(
        institutional_bias="bullish",
        smart_money_flow=0.7,
        fair_value_gap_count=2,
        recent_fvg_type="bullish",
        strongest_fvg_strength=0.8,
        liquidity_sweep_count=1,
        recent_liquidity_sweep_direction="down",
        liquidity_sweep_bias="bullish",
        confidence=0.9,
        key_levels=(1.0, 1.2),
    )
    aligned_strategy = MeanReversionStrategy(
        "mr-aligned",
        ["EURUSD"],
        capital=750_000,
        config=config,
        microstructure_analyzer=_StaticAnalyzer(aligned_features),
    )
    aligned_signal = await aligned_strategy.generate_signal(market, "EURUSD")
    assert aligned_signal.confidence > base_signal.confidence
    assert aligned_signal.metadata["microstructure"]["alignment"]["score"] > 0

    conflict_features = ICTMicrostructureFeatures(
        institutional_bias="bearish",
        smart_money_flow=-0.5,
        fair_value_gap_count=2,
        recent_fvg_type="bearish",
        strongest_fvg_strength=0.8,
        liquidity_sweep_count=1,
        recent_liquidity_sweep_direction="up",
        liquidity_sweep_bias="bearish",
        confidence=0.9,
        key_levels=(1.0, 1.2),
    )
    conflict_strategy = MeanReversionStrategy(
        "mr-conflict",
        ["EURUSD"],
        capital=750_000,
        config=config,
        microstructure_analyzer=_StaticAnalyzer(conflict_features),
    )
    conflict_signal = await conflict_strategy.generate_signal(market, "EURUSD")
    assert conflict_signal.confidence < base_signal.confidence
    assert conflict_signal.metadata["microstructure"]["alignment"]["score"] < 0


@pytest.mark.asyncio
async def test_volatility_breakout_attaches_microstructure_metadata() -> None:
    config = VolatilityBreakoutConfig(
        breakout_lookback=5,
        baseline_lookback=20,
        price_channel_lookback=5,
        volatility_multiplier=1.1,
    )
    market = {
        "EURUSD": {
            "close": [1.0] * 20 + [1.0, 1.05, 1.12, 1.2, 1.28],
        }
    }
    features = ICTMicrostructureFeatures(
        institutional_bias="bullish",
        smart_money_flow=0.5,
        fair_value_gap_count=1,
        recent_fvg_type="bullish",
        strongest_fvg_strength=0.6,
        liquidity_sweep_count=1,
        recent_liquidity_sweep_direction="down",
        liquidity_sweep_bias="bullish",
        confidence=0.75,
        key_levels=(1.0,),
    )

    strategy = VolatilityBreakoutStrategy(
        "vol",
        ["EURUSD"],
        capital=500_000,
        config=config,
        microstructure_analyzer=_StaticAnalyzer(features),
    )

    signal = await strategy.generate_signal(market, "EURUSD")
    assert isinstance(signal, StrategySignal)
    assert "microstructure" in signal.metadata
    assert signal.metadata["microstructure"]["alignment"]["score"] >= 0
