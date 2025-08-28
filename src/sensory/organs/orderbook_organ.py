"""
EMP Orderbook Sensory Organ v1.1

Processes order book data and extracts market microstructure signals
including bid-ask spreads, order flow, and market depth analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from src.core.base import MarketData
from src.core.exceptions import ResourceException as SensoryException
from src.sensory.signals import SensorSignal as SensorySignal

logger = logging.getLogger(__name__)


@dataclass
class OrderbookSignal:
    """Orderbook-related sensory signal."""

    timestamp: datetime
    signal_type: str
    value: float
    confidence: float
    metadata: Dict[str, Any]


if TYPE_CHECKING:
    from typing import Protocol

    class _SensoryOrganProto(Protocol):
        name: str
        config: dict[str, Any]

        def __init__(self, name: str, config: Optional[dict[str, Any]] = ...) -> None: ...


# Minimal runtime placeholder to avoid importing unavailable SensoryOrgan
class SensoryOrgan:
    def __init__(
        self, name: str = "orderbook_organ", config: Optional[dict[str, Any]] = None
    ) -> None:
        self.name = name
        self.config = config or {}
        self.calibrated = False
        self._bid_history: List[float] = []
        self._ask_history: List[float] = []
        self._spread_history: List[float] = []
        self._max_history = self.config.get("max_history", 1000)

    def perceive(self, data: MarketData) -> SensorySignal:
        """Process raw order book data into sensory signals."""
        try:
            # Calculate spread
            spread = data.ask - data.bid
            spread_pips = spread * 10000  # Convert to pips for forex

            # Add to history
            self._bid_history.append(data.bid)
            self._ask_history.append(data.ask)
            self._spread_history.append(spread_pips)

            if len(self._bid_history) > self._max_history:
                self._bid_history.pop(0)
                self._ask_history.pop(0)
                self._spread_history.pop(0)

            # Calculate orderbook signals
            signals = []

            # Spread signal
            if len(self._spread_history) >= 10:
                spread_signal = self._calculate_spread_signal(spread_pips)
                signals.append(spread_signal)

            # Bid-ask pressure signal
            if len(self._bid_history) >= 20:
                pressure_signal = self._calculate_pressure_signal(data)
                signals.append(pressure_signal)

            # Market depth signal
            if len(self._bid_history) >= 10:
                depth_signal = self._calculate_depth_signal(data)
                signals.append(depth_signal)

            # Order flow signal
            if len(self._bid_history) >= 5:
                flow_signal = self._calculate_flow_signal(data)
                signals.append(flow_signal)

            # Combine signals into a single sensory reading
            combined_signal = self._combine_signals(signals)

            return SensorySignal(
                timestamp=data.timestamp,
                signal_type="orderbook_composite",
                value=combined_signal.value,
                confidence=combined_signal.confidence,
                metadata={
                    "signals": [s.__dict__ for s in signals],
                    "spread_pips": spread_pips,
                    "organ_id": "orderbook_organ",
                },
            )

        except Exception as e:
            raise SensoryException(f"Error in orderbook perception: {e}")

    def calibrate(self) -> bool:
        """Calibrate the orderbook organ."""
        try:
            # Reset calibration state
            self.calibrated = False

            # Perform calibration checks
            if len(self._spread_history) < 10:
                logger.warning("Insufficient orderbook history for calibration")
                return False

            # Calculate calibration metrics
            spread_array = np.array(self._spread_history)

            # Check for reasonable spread values
            if np.any(spread_array < 0):
                raise SensoryException("Invalid spread values detected")

            # Check for reasonable spread variation
            spread_std = np.std(spread_array)
            if spread_std == 0:
                raise SensoryException("Zero spread variation detected")

            # Mark as calibrated
            self.calibrated = True
            logger.info("Orderbook organ calibrated successfully")
            return True

        except Exception as e:
            logger.error(f"Orderbook organ calibration failed: {e}")
            return False

    def _calculate_spread_signal(self, current_spread: float) -> OrderbookSignal:
        """Calculate spread signal indicating market liquidity."""
        spreads = np.array(self._spread_history)

        # Calculate average spread
        avg_spread = np.mean(spreads[-20:])  # 20-period average

        # Calculate spread deviation
        spread_deviation = (current_spread - avg_spread) / avg_spread

        # Determine spread signal
        if spread_deviation < -0.2:  # Spread narrowing
            spread_signal = 1.0  # Positive (better liquidity)
        elif spread_deviation > 0.2:  # Spread widening
            spread_signal = -1.0  # Negative (worse liquidity)
        else:
            spread_signal = 0.0  # Normal spread

        # Calculate confidence based on deviation magnitude
        confidence = min(float(abs(spread_deviation) * 2), 1.0)

        return OrderbookSignal(
            timestamp=datetime.now(),
            signal_type="spread",
            value=spread_signal,
            confidence=confidence,
            metadata={
                "current_spread": current_spread,
                "avg_spread": avg_spread,
                "spread_deviation": spread_deviation,
            },
        )

    def _calculate_pressure_signal(self, data: MarketData) -> OrderbookSignal:
        """Calculate bid-ask pressure signal."""
        bids = np.array(self._bid_history)
        asks = np.array(self._ask_history)

        # Calculate recent bid and ask trends
        recent_bids = bids[-10:]
        recent_asks = asks[-10:]

        # Calculate bid strength (how much bids are moving up)
        bid_trend = (recent_bids[-1] - recent_bids[0]) / recent_bids[0]

        # Calculate ask weakness (how much asks are moving down)
        ask_trend = (recent_asks[-1] - recent_asks[0]) / recent_asks[0]

        # Combined pressure signal
        pressure_signal = bid_trend - ask_trend

        # Normalize to [-1, 1] range
        normalized_pressure = float(np.tanh(pressure_signal * 1000))  # Scale for forex

        # Calculate confidence based on trend strength
        confidence = float(min(abs(pressure_signal) * 500, 1.0))

        return OrderbookSignal(
            timestamp=datetime.now(),
            signal_type="pressure",
            value=normalized_pressure,
            confidence=confidence,
            metadata={
                "bid_trend": bid_trend,
                "ask_trend": ask_trend,
                "pressure_signal": pressure_signal,
            },
        )

    def _calculate_depth_signal(self, data: MarketData) -> OrderbookSignal:
        """Calculate market depth signal."""
        # For now, use spread as a proxy for market depth
        # In a real implementation, this would use actual order book depth data
        spreads = np.array(self._spread_history)

        # Calculate depth signal based on spread stability
        recent_spreads = spreads[-10:]
        spread_volatility = float(np.std(recent_spreads))

        # Lower volatility = better depth
        depth_signal = max(0.0, 1.0 - spread_volatility / 2.0)  # Normalize

        # Calculate confidence based on data quality
        confidence = min(len(recent_spreads) / 10, 1.0)

        return OrderbookSignal(
            timestamp=datetime.now(),
            signal_type="depth",
            value=depth_signal,
            confidence=confidence,
            metadata={"spread_volatility": spread_volatility, "depth_quality": depth_signal},
        )

    def _calculate_flow_signal(self, data: MarketData) -> OrderbookSignal:
        """Calculate order flow signal."""
        # For now, use price movement relative to bid/ask as proxy for order flow
        # In a real implementation, this would use actual order flow data
        bids = np.array(self._bid_history)
        asks = np.array(self._ask_history)

        if len(bids) < 2:
            return OrderbookSignal(
                timestamp=data.timestamp, signal_type="flow", value=0.0, confidence=0.0, metadata={}
            )

        # Calculate where current price is relative to bid/ask
        mid_price = (data.bid + data.ask) / 2
        price_position = (data.close - data.bid) / (data.ask - data.bid)

        # Normalize to [-1, 1] range
        # 0 = at bid (selling pressure), 1 = at ask (buying pressure)
        flow_signal = (price_position - 0.5) * 2

        # Calculate confidence based on spread size
        spread = data.ask - data.bid
        confidence = float(min(1.0, 1.0 / (spread * 10000)))  # Smaller spread = higher confidence

        return OrderbookSignal(
            timestamp=data.timestamp,
            signal_type="flow",
            value=flow_signal,
            confidence=confidence,
            metadata={
                "price_position": price_position,
                "mid_price": mid_price,
                "flow_direction": "buy" if flow_signal > 0 else "sell",
            },
        )

    def _combine_signals(self, signals: List[OrderbookSignal]) -> OrderbookSignal:
        """Combine multiple orderbook signals into a composite signal."""
        if not signals:
            return OrderbookSignal(
                timestamp=datetime.now(),
                signal_type="orderbook_composite",
                value=0.0,
                confidence=0.0,
                metadata={},
            )

        # Weighted average of signal values
        total_weight: float = 0.0
        weighted_sum: float = 0.0

        for signal in signals:
            weight = signal.confidence
            total_weight += weight
            weighted_sum += signal.value * weight

        if total_weight > 0:
            composite_value = weighted_sum / total_weight
            composite_confidence = float(np.mean([s.confidence for s in signals]))
        else:
            composite_value = 0.0
            composite_confidence = 0.0

        return OrderbookSignal(
            timestamp=datetime.now(),
            signal_type="orderbook_composite",
            value=composite_value,
            confidence=composite_confidence,
            metadata={
                "signal_count": len(signals),
                "individual_signals": [s.signal_type for s in signals],
            },
        )
