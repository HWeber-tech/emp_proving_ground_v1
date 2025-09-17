"""
Enhanced HOW Dimension - Institutional Footprint Hunter
Phase 2 Implementation: Advanced ICT Pattern Detection

This module provides sophisticated institutional trading pattern analysis including:
- Order block detection using ICT methodology
- Fair value gap identification
- Liquidity sweep detection
- Smart money flow tracking
- Institutional bias determination
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from src.core.base import DimensionalReading, MarketData, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class OrderBlock:
    """Represents an institutional order block"""

    type: str  # 'bullish' or 'bearish'
    price_level: float
    timestamp: datetime
    strength: float  # 0-1 scale
    volume_confirmation: bool
    displacement_size: float
    consolidation_range: Tuple[float, float]
    breaker_level: float
    mitigation_level: float


@dataclass
class FairValueGap:
    """Represents a fair value gap (FVG)"""

    type: str  # 'bullish' or 'bearish'
    start_price: float
    end_price: float
    gap_range: Tuple[float, float]
    timestamp: datetime
    strength: float
    fill_probability: float
    imbalance_ratio: float


@dataclass
class LiquiditySweep:
    """Represents a liquidity sweep event"""

    direction: str  # 'up' or 'down'
    sweep_level: float
    liquidity_pool: str  # 'equal highs', 'equal lows', 'stop hunt'
    sweep_size: float
    volume_spike: float
    reversal_probability: float
    institutional_follow_through: bool


@dataclass
class InstitutionalFootprint:
    """Complete institutional trading footprint"""

    order_blocks: List[OrderBlock]
    fair_value_gaps: List[FairValueGap]
    liquidity_sweeps: List[LiquiditySweep]
    smart_money_flow: float  # -1 to 1 scale
    institutional_bias: str  # 'bullish', 'bearish', 'neutral'
    confidence_score: float
    market_structure: str
    key_levels: List[float]


class InstitutionalFootprintHunter:
    """
    Advanced HOW dimension implementation with ICT pattern detection
    and institutional footprint analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.order_block_detector = OrderBlockDetector()
        self.fvg_detector = FairValueGapDetector()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.smart_money_tracker = SmartMoneyTracker()
        self.logger = logging.getLogger(__name__)

    async def analyze_institutional_footprint(
        self, market_data: List[MarketData], symbol: str = "UNKNOWN"
    ) -> InstitutionalFootprint:
        """
        Analyze institutional trading footprint

        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed

        Returns:
            InstitutionalFootprint: Complete institutional analysis
        """
        try:
            if not market_data:
                return self._get_fallback_footprint()

            # Convert to DataFrame
            df = self._market_data_to_dataframe(market_data)

            # Run all analyses in parallel
            tasks = [
                self.order_block_detector.detect_order_blocks(df),
                self.fvg_detector.detect_fvg(df),
                self.liquidity_analyzer.detect_sweeps(df),
                self.smart_money_tracker.calculate_flow(df),
            ]

            order_blocks, fvgs, sweeps, smart_flow = cast(
                Tuple[List[OrderBlock], List[FairValueGap], List[LiquiditySweep], float],
                await asyncio.gather(*tasks),
            )

            # Determine institutional bias
            institutional_bias = self._determine_institutional_bias(order_blocks, fvgs, smart_flow)

            # Calculate confidence
            confidence = self._calculate_footprint_confidence(order_blocks, fvgs, sweeps)

            # Identify key levels
            key_levels = self._identify_key_levels(order_blocks, fvgs)

            # Determine market structure
            market_structure = self._determine_market_structure(df, order_blocks)

            return InstitutionalFootprint(
                order_blocks=order_blocks,
                fair_value_gaps=fvgs,
                liquidity_sweeps=sweeps,
                smart_money_flow=smart_flow,
                institutional_bias=institutional_bias,
                confidence_score=confidence,
                market_structure=market_structure,
                key_levels=key_levels,
            )

        except Exception as e:
            self.logger.error(f"Institutional footprint analysis failed: {e}")
            return self._get_fallback_footprint()

    def _market_data_to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to pandas DataFrame"""
        data = []
        for md in market_data:
            data.append(
                {
                    "timestamp": md.timestamp,
                    "open": md.open,
                    "high": md.high,
                    "low": md.low,
                    "close": md.close,
                    "volume": md.volume,
                    "bid": md.bid,
                    "ask": md.ask,
                    "spread": md.spread,
                    "mid_price": md.mid_price,
                }
            )

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Add technical indicators
        df["rsi"] = self._calculate_rsi(df["close"])
        df["atr"] = self._calculate_atr(df)
        df["volume_ma"] = df["volume"].rolling(window=20).mean()

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta.gt(0), 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta.lt(0), 0.0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low: pd.Series = df["high"] - df["low"]
        high_close: pd.Series = (df["high"] - df["close"].shift()).abs()
        low_close: pd.Series = (df["low"] - df["close"].shift()).abs()
        ranges = cast(pd.DataFrame, pd.concat([high_low, high_close, low_close], axis=1))
        true_range: pd.Series = cast(pd.Series, ranges.max(axis=1))
        atr = cast(pd.Series, true_range.rolling(window=period).mean())
        return atr

    def _determine_institutional_bias(
        self, order_blocks: List[OrderBlock], fvgs: List[FairValueGap], smart_flow: float
    ) -> str:
        """Determine overall institutional bias"""
        try:
            # Weight different signals
            ob_score = sum(
                ob.strength * (1 if ob.type == "bullish" else -1) for ob in order_blocks
            ) / max(len(order_blocks), 1)

            fvg_score = sum(
                fvg.strength * (1 if fvg.type == "bullish" else -1) for fvg in fvgs
            ) / max(len(fvgs), 1)

            # Combine scores
            combined_score = ob_score * 0.4 + fvg_score * 0.3 + smart_flow * 0.3

            if combined_score > 0.3:
                return "bullish"
            elif combined_score < -0.3:
                return "bearish"
            else:
                return "neutral"

        except Exception:
            return "neutral"

    def _calculate_footprint_confidence(
        self, order_blocks: List[OrderBlock], fvgs: List[FairValueGap], sweeps: List[LiquiditySweep]
    ) -> float:
        """Calculate confidence in footprint analysis"""
        try:
            # Base confidence on number and quality of patterns
            ob_confidence = min(len(order_blocks) * 0.2, 1.0)
            fvg_confidence = min(len(fvgs) * 0.15, 1.0)
            sweep_confidence = min(len(sweeps) * 0.1, 1.0)

            # Average confidence
            total_patterns = len(order_blocks) + len(fvgs) + len(sweeps)
            if total_patterns == 0:
                return 0.1

            return min((ob_confidence + fvg_confidence + sweep_confidence) / 3, 1.0)

        except Exception:
            return 0.1

    def _identify_key_levels(
        self, order_blocks: List[OrderBlock], fvgs: List[FairValueGap]
    ) -> List[float]:
        """Identify key institutional levels"""
        levels = []

        # Add order block levels
        for ob in order_blocks:
            levels.append(ob.price_level)
            levels.append(ob.breaker_level)

        # Add FVG levels
        for fvg in fvgs:
            levels.extend([fvg.start_price, fvg.end_price])

        # Remove duplicates and sort
        levels = list(set(levels))
        levels.sort()

        return levels

    def _determine_market_structure(self, df: pd.DataFrame, order_blocks: List[OrderBlock]) -> str:
        """Determine current market structure"""
        try:
            if len(df) < 50:
                return "insufficient_data"

            # Check for trending vs ranging
            recent_highs = df["high"].tail(20).max()
            recent_lows = df["low"].tail(20).min()
            price_range = recent_highs - recent_lows

            # Calculate volatility
            volatility = df["close"].pct_change().tail(20).std()

            if volatility > 0.02:  # High volatility
                return "volatile"
            elif len(order_blocks) > 3:  # Multiple order blocks suggest trending
                return "trending"
            else:
                return "ranging"

        except Exception:
            return "unknown"

    def _get_fallback_footprint(self) -> InstitutionalFootprint:
        """Fallback footprint when analysis fails"""
        return InstitutionalFootprint(
            order_blocks=[],
            fair_value_gaps=[],
            liquidity_sweeps=[],
            smart_money_flow=0.0,
            institutional_bias="neutral",
            confidence_score=0.1,
            market_structure="unknown",
            key_levels=[],
        )


class OrderBlockDetector:
    """Detects institutional order blocks using ICT methodology"""

    async def detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """Detect order blocks in price data"""
        order_blocks: List[OrderBlock] = []

        try:
            if len(df) < 20:
                return order_blocks

            # Look for strong moves (displacement)
            for i in range(10, len(df) - 5):
                # Calculate displacement
                displacement = abs(df.iloc[i]["close"] - df.iloc[i - 5]["close"])
                avg_range = df["high"].iloc[i - 5 : i].mean() - df["low"].iloc[i - 5 : i].mean()

                if displacement > avg_range * 2:  # Strong displacement
                    # Look for consolidation before the move
                    consolidation_start = max(0, i - 10)
                    consolidation_data = df.iloc[consolidation_start:i]

                    if len(consolidation_data) >= 3:
                        # Determine order block type
                        is_bullish = df.iloc[i]["close"] > df.iloc[i - 5]["close"]

                        # Find order block level
                        if is_bullish:
                            ob_level = consolidation_data["low"].min()
                            breaker_level = consolidation_data["high"].max()
                        else:
                            ob_level = consolidation_data["high"].max()
                            breaker_level = consolidation_data["low"].min()

                        # Calculate strength
                        strength = min(displacement / avg_range / 3, 1.0)

                        # Volume confirmation
                        volume_spike = (
                            df.iloc[i]["volume"] > df["volume"].iloc[i - 5 : i].mean() * 1.5
                        )

                        order_block = OrderBlock(
                            type="bullish" if is_bullish else "bearish",
                            price_level=ob_level,
                            timestamp=df.iloc[i]["timestamp"],
                            strength=strength,
                            volume_confirmation=volume_spike,
                            displacement_size=displacement,
                            consolidation_range=(
                                consolidation_data["low"].min(),
                                consolidation_data["high"].max(),
                            ),
                            breaker_level=breaker_level,
                            mitigation_level=ob_level,
                        )

                        order_blocks.append(order_block)

            # Filter for strongest blocks
            order_blocks.sort(key=lambda x: x.strength, reverse=True)
            return order_blocks[:5]  # Return top 5

        except Exception as e:
            logger.error(f"Error detecting order blocks: {e}")
            return []


class FairValueGapDetector:
    """Detects fair value gaps (FVGs) in price action"""

    async def detect_fvg(self, df: pd.DataFrame) -> List[FairValueGap]:
        """Detect fair value gaps in price data"""
        fvgs: List[FairValueGap] = []

        try:
            if len(df) < 10:
                return fvgs

            for i in range(2, len(df)):
                # ICT FVG definition: 3-candle pattern
                candle1 = df.iloc[i - 2]
                candle2 = df.iloc[i - 1]
                candle3 = df.iloc[i]

                # Bullish FVG
                if (
                    candle1["high"] < candle3["low"]
                    and candle2["low"] > candle1["high"]
                    and candle2["low"] > candle3["low"]
                ):
                    fvg = FairValueGap(
                        type="bullish",
                        start_price=float(candle1["high"]),
                        end_price=float(candle3["low"]),
                        gap_range=(float(candle1["high"]), float(candle3["low"])),
                        timestamp=pd.to_datetime(candle3["timestamp"]).to_pydatetime(),
                        strength=self._calculate_fvg_strength(candle1, candle2, candle3),
                        fill_probability=self._calculate_fill_probability(candle1, candle3),
                        imbalance_ratio=float(
                            (float(candle3["low"]) - float(candle1["high"]))
                            / float(candle1["high"])
                        ),
                    )
                    fvgs.append(fvg)

                # Bearish FVG
                elif (
                    candle1["low"] > candle3["high"]
                    and candle2["high"] < candle1["low"]
                    and candle2["high"] < candle3["high"]
                ):
                    fvg = FairValueGap(
                        type="bearish",
                        start_price=float(candle1["low"]),
                        end_price=float(candle3["high"]),
                        gap_range=(float(candle3["high"]), float(candle1["low"])),
                        timestamp=pd.to_datetime(candle3["timestamp"]).to_pydatetime(),
                        strength=self._calculate_fvg_strength(candle1, candle2, candle3),
                        fill_probability=self._calculate_fill_probability(candle1, candle3),
                        imbalance_ratio=float(
                            (float(candle1["low"]) - float(candle3["high"])) / float(candle1["low"])
                        ),
                    )
                    fvgs.append(fvg)

            # Filter for strongest FVGs
            fvgs.sort(key=lambda x: x.strength, reverse=True)
            return fvgs[:10]

        except Exception as e:
            logger.error(f"Error detecting FVGs: {e}")
            return []

    def _calculate_fvg_strength(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> float:
        """Calculate FVG strength based on imbalance size"""
        try:
            if c1["high"] < c3["low"]:  # Bullish
                imbalance = c3["low"] - c1["high"]
            else:  # Bearish
                imbalance = c1["low"] - c3["high"]

            avg_range = (
                c1["high"] - c1["low"] + c2["high"] - c2["low"] + c3["high"] - c3["low"]
            ) / 3

            return float(min(float(imbalance) / float(avg_range), 1.0))

        except Exception:
            return 0.5

    def _calculate_fill_probability(self, c1: pd.Series, c3: pd.Series) -> float:
        """Calculate probability of FVG being filled"""
        try:
            # Simplified calculation based on gap size
            gap_size = abs(c3["close"] - c1["close"])
            return float(max(0.1, min(0.9, 1.0 - float(gap_size) / float(c1["close"]))))
        except Exception:
            return 0.5


class LiquidityAnalyzer:
    """Analyzes liquidity sweeps and institutional liquidity grabs"""

    async def detect_sweeps(self, df: pd.DataFrame) -> List[LiquiditySweep]:
        """Detect liquidity sweeps in price data"""
        sweeps: List[LiquiditySweep] = []

        try:
            if len(df) < 20:
                return sweeps

            # Look for equal highs/lows
            highs = df["high"].values
            lows = df["low"].values

            for i in range(10, len(df) - 5):
                # Check for liquidity above
                recent_highs = highs[i - 10 : i]
                if len(recent_highs) > 0:
                    equal_highs = self._find_equal_levels(
                        cast(np.ndarray, recent_highs), tolerance=0.001
                    )
                    if equal_highs and float(highs[i]) > float(max(equal_highs)):
                        # Liquidity sweep detected
                        sweep = LiquiditySweep(
                            direction="up",
                            sweep_level=float(highs[i]),
                            liquidity_pool="equal highs",
                            sweep_size=float(highs[i]) - float(max(equal_highs)),
                            volume_spike=float(df.iloc[i]["volume"])
                            / float(df["volume"].iloc[i - 5 : i].mean()),
                            reversal_probability=0.7,
                            institutional_follow_through=bool(
                                float(df.iloc[i + 1]["close"]) < float(highs[i])
                            ),
                        )
                        sweeps.append(sweep)

                # Check for liquidity below
                recent_lows = lows[i - 10 : i]
                if len(recent_lows) > 0:
                    equal_lows = self._find_equal_levels(
                        cast(np.ndarray, recent_lows), tolerance=0.001
                    )
                    if equal_lows and float(lows[i]) < float(min(equal_lows)):
                        # Liquidity sweep detected
                        sweep = LiquiditySweep(
                            direction="down",
                            sweep_level=float(lows[i]),
                            liquidity_pool="equal lows",
                            sweep_size=float(min(equal_lows)) - float(lows[i]),
                            volume_spike=float(df.iloc[i]["volume"])
                            / float(df["volume"].iloc[i - 5 : i].mean()),
                            reversal_probability=0.7,
                            institutional_follow_through=bool(
                                float(df.iloc[i + 1]["close"]) > float(lows[i])
                            ),
                        )
                        sweeps.append(sweep)

            return sweeps[:5]  # Return top 5

        except Exception as e:
            logger.error(f"Error detecting sweeps: {e}")
            return []

    def _find_equal_levels(self, levels: np.ndarray, tolerance: float = 0.001) -> List[float]:
        """Find equal or near-equal price levels"""
        equal_levels = []
        for i, level in enumerate(levels):
            for j in range(i + 1, len(levels)):
                if abs(level - levels[j]) / level <= tolerance:
                    equal_levels.append(level)
        return equal_levels


class SmartMoneyTracker:
    """Tracks smart money flow and institutional positioning"""

    async def calculate_flow(self, df: pd.DataFrame) -> float:
        """Calculate smart money flow indicator"""
        try:
            if len(df) < 20:
                return 0.0

            # Calculate smart money flow index
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            money_flow = typical_price * df["volume"]

            # Positive money flow
            positive_flow = float(money_flow.where(typical_price > typical_price.shift(1), 0).sum())

            # Negative money flow
            negative_flow = float(money_flow.where(typical_price < typical_price.shift(1), 0).sum())

            # Smart money flow ratio
            if negative_flow == 0:
                return 1.0

            flow_ratio = float(positive_flow / negative_flow)

            # Normalize to -1 to 1 scale
            return float(max(-1.0, min(1.0, (flow_ratio - 1.0) / (flow_ratio + 1.0))))

        except Exception as e:
            logger.error(f"Error calculating smart money flow: {e}")
            return 0.0


# Integration adapter for existing system
class EnhancedHowAdapter:
    """Adapter to integrate enhanced HOW dimension with existing system"""

    def __init__(self) -> None:
        self.footprint_hunter = InstitutionalFootprintHunter()

    async def get_enhanced_reading(
        self, market_data: List[MarketData], symbol: str = "UNKNOWN"
    ) -> DimensionalReading:
        """Get enhanced HOW dimensional reading"""
        try:
            footprint = await self.footprint_hunter.analyze_institutional_footprint(
                market_data, symbol
            )

            # Calculate overall signal strength
            signal_strength = footprint.smart_money_flow

            return DimensionalReading(
                dimension="HOW",
                signal_strength=signal_strength,
                confidence=footprint.confidence_score,
                regime=self._determine_regime(footprint),
                context={
                    "institutional_footprint": footprint,
                    "order_blocks": len(footprint.order_blocks),
                    "fair_value_gaps": len(footprint.fair_value_gaps),
                    "liquidity_sweeps": len(footprint.liquidity_sweeps),
                    "institutional_bias": footprint.institutional_bias,
                    "market_structure": footprint.market_structure,
                },
                data_quality=footprint.confidence_score,
                processing_time_ms=0.0,
                evidence={"key_levels": footprint.key_levels},
                warnings=[],
            )

        except Exception as e:
            logger.error(f"Enhanced HOW reading failed: {e}")
            return DimensionalReading(
                dimension="HOW",
                signal_strength=0.0,
                confidence=0.1,
                regime=MarketRegime.UNKNOWN,
                context={},
                data_quality=0.1,
                processing_time_ms=0.0,
                evidence={},
                warnings=["Analysis failed"],
            )

    def _determine_regime(self, footprint: InstitutionalFootprint) -> MarketRegime:
        """Determine market regime based on institutional footprint"""
        if footprint.confidence_score < 0.3:
            return MarketRegime.UNKNOWN

        if footprint.institutional_bias == "bullish":
            return MarketRegime.BULLISH
        elif footprint.institutional_bias == "bearish":
            return MarketRegime.BEARISH
        else:
            return MarketRegime.RANGING


# Example usage
if __name__ == "__main__":

    async def test_footprint_hunter() -> None:
        # Create sample data
        import random
        from datetime import datetime

        market_data = []
        base_price = 100.0

        for i in range(100):
            price = base_price + random.uniform(-5, 5)
            market_data.append(
                MarketData(
                    timestamp=datetime.now() - timedelta(minutes=i),
                    open=price + random.uniform(-1, 1),
                    high=price + random.uniform(0, 2),
                    low=price - random.uniform(0, 2),
                    close=price,
                    volume=random.uniform(1000, 10000),
                    bid=price - 0.01,
                    ask=price + 0.01,
                    spread=0.02,
                    mid_price=price,
                )
            )

        hunter = InstitutionalFootprintHunter()
        result = await hunter.analyze_institutional_footprint(market_data)
        print(f"Institutional Footprint: {result}")

    asyncio.run(test_footprint_hunter())
