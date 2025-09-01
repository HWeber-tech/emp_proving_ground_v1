"""
Advanced Order Book Analyzer
Provides real-time order book analysis, market microstructure insights,
and liquidity assessment for enhanced trading decisions.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.trading.order_management.order_book.snapshot import OrderBookLevel, OrderBookSnapshot

logger = logging.getLogger(__name__)


class OrderBookSide(Enum):
    """Order book side enumeration."""

    BID = "bid"
    ASK = "ask"


@dataclass
class MarketMicrostructure:
    """Market microstructure analysis results."""

    symbol: str
    timestamp: datetime

    # Liquidity metrics
    bid_liquidity: float
    ask_liquidity: float
    total_liquidity: float
    liquidity_imbalance: float

    # Spread analysis
    spread: float
    spread_bps: float
    effective_spread: float

    # Depth analysis
    depth_5_levels: float
    depth_10_levels: float
    depth_20_levels: float

    # Order flow metrics
    order_flow_imbalance: float
    buy_pressure: float
    sell_pressure: float

    # Market impact
    market_impact_1_lot: float
    market_impact_5_lots: float
    market_impact_10_lots: float

    # Volatility indicators
    price_volatility: float
    volume_volatility: float

    # Market regime indicators
    is_tight_spread: bool
    is_high_liquidity: bool
    is_imbalanced: bool


class OrderBookAnalyzer:
    """
    Advanced order book analyzer for real-time market microstructure analysis.
    """

    def __init__(self, max_levels: int = 20, history_window: int = 1000):
        """
        Initialize the order book analyzer.

        Args:
            max_levels: Maximum number of levels to analyze
            history_window: Number of snapshots to keep in history
        """
        self.max_levels = max_levels
        self.history_window = history_window

        # Order book history
        self.order_book_history: Dict[str, deque[OrderBookSnapshot]] = defaultdict(
            lambda: deque(maxlen=history_window)
        )

        # Market microstructure history
        self.microstructure_history: Dict[str, deque[MarketMicrostructure]] = defaultdict(
            lambda: deque(maxlen=history_window)
        )

        # Real-time metrics
        self.current_metrics: Dict[str, MarketMicrostructure] = {}

        # Configuration
        self.spread_thresholds = {
            "tight": 0.0001,  # 1 pip for major pairs
            "normal": 0.0003,  # 3 pips
            "wide": 0.0005,  # 5 pips
        }

        self.liquidity_thresholds = {
            "high": 100.0,  # 100 lots
            "medium": 50.0,  # 50 lots
            "low": 10.0,  # 10 lots
        }

        logger.info(
            f"Order book analyzer initialized with {max_levels} levels and {history_window} history window"
        )

    def update_order_book(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Update order book data for a symbol.

        Args:
            symbol: Trading symbol
            bids: List of (price, volume) tuples for bids
            asks: List of (price, volume) tuples for asks
            timestamp: Timestamp for the update
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Create order book levels
        bid_levels = [
            OrderBookLevel(price=price, volume=volume, orders=1, timestamp=timestamp)
            for price, volume in bids[: self.max_levels]
        ]

        ask_levels = [
            OrderBookLevel(price=price, volume=volume, orders=1, timestamp=timestamp)
            for price, volume in asks[: self.max_levels]
        ]

        # Calculate basic metrics
        best_bid = bid_levels[0].price if bid_levels else 0.0
        best_ask = ask_levels[0].price if ask_levels else 0.0
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0

        # Create snapshot
        snapshot = OrderBookSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            bids=bid_levels,
            asks=ask_levels,
            spread=spread,
            mid_price=mid_price,
            best_bid=best_bid,
            best_ask=best_ask,
        )

        # Store in history
        self.order_book_history[symbol].append(snapshot)

        # Analyze microstructure
        microstructure = self._analyze_microstructure(symbol, snapshot)
        self.microstructure_history[symbol].append(microstructure)
        self.current_metrics[symbol] = microstructure

        logger.debug(
            f"Order book updated for {symbol}: spread={spread:.5f}, mid_price={mid_price:.5f}"
        )

    def _analyze_microstructure(
        self, symbol: str, snapshot: OrderBookSnapshot
    ) -> MarketMicrostructure:
        """Analyze market microstructure from order book snapshot."""

        # Calculate liquidity metrics
        bid_liquidity = sum(level.volume for level in snapshot.bids)
        ask_liquidity = sum(level.volume for level in snapshot.asks)
        total_liquidity = bid_liquidity + ask_liquidity
        liquidity_imbalance = (
            (bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity > 0 else 0.0
        )

        # Spread analysis
        spread = snapshot.spread
        spread_bps = (spread / snapshot.mid_price) * 10000 if snapshot.mid_price > 0 else 0.0
        effective_spread = spread * 2  # Simplified effective spread calculation

        # Depth analysis
        depth_5_levels = self._calculate_depth(snapshot, 5)
        depth_10_levels = self._calculate_depth(snapshot, 10)
        depth_20_levels = self._calculate_depth(snapshot, 20)

        # Order flow metrics
        order_flow_imbalance = self._calculate_order_flow_imbalance(symbol)
        buy_pressure = self._calculate_buy_pressure(symbol)
        sell_pressure = self._calculate_sell_pressure(symbol)

        # Market impact analysis
        market_impact_1_lot = self._calculate_market_impact(snapshot, 1.0)
        market_impact_5_lots = self._calculate_market_impact(snapshot, 5.0)
        market_impact_10_lots = self._calculate_market_impact(snapshot, 10.0)

        # Volatility indicators
        price_volatility = self._calculate_price_volatility(symbol)
        volume_volatility = self._calculate_volume_volatility(symbol)

        # Market regime indicators
        is_tight_spread = spread <= self.spread_thresholds["tight"]
        is_high_liquidity = total_liquidity >= self.liquidity_thresholds["high"]
        is_imbalanced = abs(liquidity_imbalance) > 0.3  # 30% imbalance threshold

        return MarketMicrostructure(
            symbol=symbol,
            timestamp=snapshot.timestamp,
            bid_liquidity=bid_liquidity,
            ask_liquidity=ask_liquidity,
            total_liquidity=total_liquidity,
            liquidity_imbalance=liquidity_imbalance,
            spread=spread,
            spread_bps=spread_bps,
            effective_spread=effective_spread,
            depth_5_levels=depth_5_levels,
            depth_10_levels=depth_10_levels,
            depth_20_levels=depth_20_levels,
            order_flow_imbalance=order_flow_imbalance,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            market_impact_1_lot=market_impact_1_lot,
            market_impact_5_lots=market_impact_5_lots,
            market_impact_10_lots=market_impact_10_lots,
            price_volatility=price_volatility,
            volume_volatility=volume_volatility,
            is_tight_spread=is_tight_spread,
            is_high_liquidity=is_high_liquidity,
            is_imbalanced=is_imbalanced,
        )

    def _calculate_depth(self, snapshot: OrderBookSnapshot, levels: int) -> float:
        """Calculate market depth for specified number of levels."""
        bid_depth = sum(level.volume for level in snapshot.bids[:levels])
        ask_depth = sum(level.volume for level in snapshot.asks[:levels])
        return float(bid_depth + ask_depth)

    def _calculate_order_flow_imbalance(self, symbol: str) -> float:
        """Calculate order flow imbalance from recent history."""
        if len(self.microstructure_history[symbol]) < 2:
            return 0.0

        recent = list(self.microstructure_history[symbol])[-5:]  # Last 5 snapshots
        if not recent:
            return 0.0

        # Calculate average liquidity imbalance
        avg_imbalance = sum(m.liquidity_imbalance for m in recent) / len(recent)
        return avg_imbalance

    def _calculate_buy_pressure(self, symbol: str) -> float:
        """Calculate buy pressure indicator."""
        if len(self.microstructure_history[symbol]) < 2:
            return 0.0

        recent = list(self.microstructure_history[symbol])[-3:]  # Last 3 snapshots
        if not recent:
            return 0.0

        # Calculate trend in bid liquidity
        bid_liquidity_trend = []
        for i in range(1, len(recent)):
            change = recent[i].bid_liquidity - recent[i - 1].bid_liquidity
            bid_liquidity_trend.append(change)

        if not bid_liquidity_trend:
            return 0.0

        return sum(bid_liquidity_trend) / len(bid_liquidity_trend)

    def _calculate_sell_pressure(self, symbol: str) -> float:
        """Calculate sell pressure indicator."""
        if len(self.microstructure_history[symbol]) < 2:
            return 0.0

        recent = list(self.microstructure_history[symbol])[-3:]  # Last 3 snapshots
        if not recent:
            return 0.0

        # Calculate trend in ask liquidity
        ask_liquidity_trend = []
        for i in range(1, len(recent)):
            change = recent[i].ask_liquidity - recent[i - 1].ask_liquidity
            ask_liquidity_trend.append(change)

        if not ask_liquidity_trend:
            return 0.0

        return sum(ask_liquidity_trend) / len(ask_liquidity_trend)

    def _calculate_market_impact(self, snapshot: OrderBookSnapshot, volume: float) -> float:
        """Calculate market impact for a given volume."""
        if volume <= 0:
            return 0.0

        # Simplified market impact calculation
        # In practice, this would be more sophisticated
        remaining_volume = volume
        impact = 0.0

        # Calculate impact on ask side (buy order)
        for level in snapshot.asks:
            if remaining_volume <= 0:
                break

            if remaining_volume <= level.volume:
                impact += remaining_volume * (level.price - snapshot.mid_price)
                remaining_volume = 0
            else:
                impact += level.volume * (level.price - snapshot.mid_price)
                remaining_volume -= level.volume

        return impact / volume if volume > 0 else 0.0

    def _calculate_price_volatility(self, symbol: str) -> float:
        """Calculate price volatility from recent order book history."""
        if len(self.order_book_history[symbol]) < 2:
            return 0.0

        recent = list(self.order_book_history[symbol])[-10:]  # Last 10 snapshots
        if len(recent) < 2:
            return 0.0

        # Calculate mid-price changes
        price_changes = []
        for i in range(1, len(recent)):
            change = recent[i].mid_price - recent[i - 1].mid_price
            price_changes.append(change)

        if not price_changes:
            return 0.0

        return float(np.std(price_changes))

    def _calculate_volume_volatility(self, symbol: str) -> float:
        """Calculate volume volatility from recent order book history."""
        if len(self.microstructure_history[symbol]) < 2:
            return 0.0

        recent = list(self.microstructure_history[symbol])[-10:]  # Last 10 snapshots
        if len(recent) < 2:
            return 0.0

        # Calculate total liquidity changes
        liquidity_changes = []
        for i in range(1, len(recent)):
            change = recent[i].total_liquidity - recent[i - 1].total_liquidity
            liquidity_changes.append(change)

        if not liquidity_changes:
            return 0.0

        return float(np.std(liquidity_changes))

    def get_market_analysis(self, symbol: str) -> dict[str, object]:
        """Get comprehensive market analysis for a symbol."""
        if symbol not in self.current_metrics:
            return {}

        microstructure = self.current_metrics[symbol]

        # Get recent history for trend analysis
        recent_history = list(self.microstructure_history[symbol])[-5:]

        # Calculate trends
        spread_trend = self._calculate_trend([m.spread for m in recent_history])
        liquidity_trend = self._calculate_trend([m.total_liquidity for m in recent_history])
        imbalance_trend = self._calculate_trend([m.liquidity_imbalance for m in recent_history])

        return {
            "symbol": symbol,
            "timestamp": microstructure.timestamp.isoformat(),
            # Current metrics
            "current": {
                "spread": microstructure.spread,
                "spread_bps": microstructure.spread_bps,
                "mid_price": self._get_current_mid_price(symbol),
                "total_liquidity": microstructure.total_liquidity,
                "liquidity_imbalance": microstructure.liquidity_imbalance,
                "order_flow_imbalance": microstructure.order_flow_imbalance,
                "buy_pressure": microstructure.buy_pressure,
                "sell_pressure": microstructure.sell_pressure,
            },
            # Market depth
            "depth": {
                "depth_5_levels": microstructure.depth_5_levels,
                "depth_10_levels": microstructure.depth_10_levels,
                "depth_20_levels": microstructure.depth_20_levels,
            },
            # Market impact
            "market_impact": {
                "impact_1_lot": microstructure.market_impact_1_lot,
                "impact_5_lots": microstructure.market_impact_5_lots,
                "impact_10_lots": microstructure.market_impact_10_lots,
            },
            # Volatility
            "volatility": {
                "price_volatility": microstructure.price_volatility,
                "volume_volatility": microstructure.volume_volatility,
            },
            # Market regime
            "regime": {
                "is_tight_spread": microstructure.is_tight_spread,
                "is_high_liquidity": microstructure.is_high_liquidity,
                "is_imbalanced": microstructure.is_imbalanced,
            },
            # Trends
            "trends": {
                "spread_trend": spread_trend,
                "liquidity_trend": liquidity_trend,
                "imbalance_trend": imbalance_trend,
            },
            # Trading signals
            "signals": self._generate_trading_signals(microstructure),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "stable"

        # Simple trend calculation
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        if not first_half or not second_half:
            return "stable"

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if second_avg > first_avg * 1.05:
            return "increasing"
        elif second_avg < first_avg * 0.95:
            return "decreasing"
        else:
            return "stable"

    def _get_current_mid_price(self, symbol: str) -> float:
        """Get current mid price for a symbol."""
        if symbol not in self.order_book_history or not self.order_book_history[symbol]:
            return 0.0

        latest = self.order_book_history[symbol][-1]
        return float(latest.mid_price)

    def _generate_trading_signals(self, microstructure: MarketMicrostructure) -> dict[str, object]:
        """Generate trading signals based on market microstructure."""
        signals: Dict[str, object] = {
            "liquidity_signal": "neutral",
            "spread_signal": "neutral",
            "imbalance_signal": "neutral",
            "pressure_signal": "neutral",
            "overall_signal": "neutral",
        }

        # Liquidity signal
        if microstructure.is_high_liquidity:
            signals["liquidity_signal"] = "positive"
        elif microstructure.total_liquidity < self.liquidity_thresholds["low"]:
            signals["liquidity_signal"] = "negative"

        # Spread signal
        if microstructure.is_tight_spread:
            signals["spread_signal"] = "positive"
        elif microstructure.spread > self.spread_thresholds["wide"]:
            signals["spread_signal"] = "negative"

        # Imbalance signal
        if microstructure.liquidity_imbalance > 0.2:
            signals["imbalance_signal"] = "buy"  # More bid liquidity
        elif microstructure.liquidity_imbalance < -0.2:
            signals["imbalance_signal"] = "sell"  # More ask liquidity

        # Pressure signal
        if microstructure.buy_pressure > 0 and microstructure.sell_pressure < 0:
            signals["pressure_signal"] = "buy"
        elif microstructure.sell_pressure > 0 and microstructure.buy_pressure < 0:
            signals["pressure_signal"] = "sell"

        # Overall signal (simplified logic)
        positive_signals = sum(1 for signal in signals.values() if signal in ["positive", "buy"])
        negative_signals = sum(1 for signal in signals.values() if signal in ["negative", "sell"])

        if positive_signals > negative_signals:
            signals["overall_signal"] = "buy"
        elif negative_signals > positive_signals:
            signals["overall_signal"] = "sell"

        return signals

    def get_liquidity_analysis(self, symbol: str, volume: float) -> dict[str, object]:
        """Analyze liquidity for a specific volume."""
        if symbol not in self.current_metrics:
            return {}

        microstructure = self.current_metrics[symbol]

        # Calculate available liquidity
        available_bid_liquidity = sum(
            level.volume for level in self.order_book_history[symbol][-1].bids[:5]
        )
        available_ask_liquidity = sum(
            level.volume for level in self.order_book_history[symbol][-1].asks[:5]
        )

        # Calculate execution probability
        bid_execution_prob = min(1.0, available_bid_liquidity / volume) if volume > 0 else 0.0
        ask_execution_prob = min(1.0, available_ask_liquidity / volume) if volume > 0 else 0.0

        return {
            "symbol": symbol,
            "volume": volume,
            "available_bid_liquidity": available_bid_liquidity,
            "available_ask_liquidity": available_ask_liquidity,
            "bid_execution_probability": bid_execution_prob,
            "ask_execution_probability": ask_execution_prob,
            "market_impact": microstructure.market_impact_1_lot * volume,
            "recommended_split": self._calculate_optimal_split(
                volume, available_bid_liquidity, available_ask_liquidity
            ),
        }

    def _calculate_optimal_split(
        self, volume: float, bid_liquidity: float, ask_liquidity: float
    ) -> Dict[str, float]:
        """Calculate optimal order split based on available liquidity."""
        if volume <= 0:
            return {"immediate": 0.0, "rest": 0.0}

        # Use 50% of available liquidity for immediate execution
        immediate_bid = min(volume * 0.5, bid_liquidity * 0.5)
        immediate_ask = min(volume * 0.5, ask_liquidity * 0.5)

        immediate = min(immediate_bid, immediate_ask)
        rest = volume - immediate

        return {"immediate": immediate, "rest": rest}

    def get_order_book_snapshot(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get the latest order book snapshot for a symbol."""
        if symbol not in self.order_book_history or not self.order_book_history[symbol]:
            return None

        return self.order_book_history[symbol][-1]

    def get_microstructure_history(
        self, symbol: str, minutes: int = 60
    ) -> List[MarketMicrostructure]:
        """Get microstructure history for the last N minutes."""
        if symbol not in self.microstructure_history:
            return []

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        history = []

        for microstructure in reversed(self.microstructure_history[symbol]):
            if microstructure.timestamp >= cutoff_time:
                history.append(microstructure)
            else:
                break

        return list(reversed(history))

    def export_order_book_data(self, symbol: str, format: str = "json") -> str:
        """Export order book data for analysis."""
        if symbol not in self.order_book_history:
            return ""

        snapshots: list[dict[str, object]] = []
        data: Dict[str, object] = {"symbol": symbol, "snapshots": snapshots}

        for snapshot in self.order_book_history[symbol]:
            snapshot_data = {
                "timestamp": snapshot.timestamp.isoformat(),
                "spread": snapshot.spread,
                "mid_price": snapshot.mid_price,
                "best_bid": snapshot.best_bid,
                "best_ask": snapshot.best_ask,
                "bids": [(level.price, level.volume) for level in snapshot.bids],
                "asks": [(level.price, level.volume) for level in snapshot.asks],
            }
            snapshots.append(snapshot_data)

        if format == "json":
            import json

            return json.dumps(data, indent=2)
        else:
            return str(data)
