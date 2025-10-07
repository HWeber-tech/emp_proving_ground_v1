"""
CVDDivergenceDetector
Specialized pattern detector for CVD/Price divergence analysis
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any, Literal, Optional, TypeAlias, cast

from src.core.config_access import ConfigurationProvider, NoOpConfigurationProvider
from src.thinking.models import ContextPacket

logger = logging.getLogger(__name__)


class CVDDivergenceDetector:
    """
    Detects divergences between price and Cumulative Volume Delta.

    This detector identifies when price and CVD are moving in opposite directions,
    which often signals exhaustion of the dominant force and potential reversal.
    """

    def __init__(
        self,
        symbol: str,
        lookback: Optional[int] = None,
        config: ConfigurationProvider = NoOpConfigurationProvider(),
    ):
        """
        Initialize the CVD divergence detector.

        Args:
            symbol: The symbol this detector is monitoring
            lookback: Number of data points to analyze (defaults to configuration provider or safe default)
            config: Configuration provider port (defaults to NoOp)
        """
        self.symbol = symbol
        self.config = config
        # Resolve lookback: explicit arg > config value > safe default
        default_lb = self._get_numeric_config("cvd_history_length", 20)
        self.lookback = int(lookback) if lookback is not None else int(default_lb)

        # Rolling history for divergence detection
        self.price_history: deque[float] = deque(maxlen=self.lookback)
        self.cvd_history: deque[float] = deque(maxlen=self.lookback)
        self.timestamp_history: deque[datetime] = deque(maxlen=self.lookback)

        logger.info(f"Initialized CVDDivergenceDetector for {symbol} with lookback={self.lookback}")

    def detect(
        self, current_price: float, current_cvd: float, timestamp: datetime
    ) -> Optional[Literal["bullish", "bearish"]]:
        """
        Detect CVD/Price divergence based on the latest data point.

        Args:
            current_price: Current market price
            current_cvd: Current cumulative volume delta
            timestamp: Current timestamp

        Returns:
            Optional[Literal["bullish", "bearish"]]: Divergence signal or None
        """
        # Add current data to history
        self.price_history.append(current_price)
        self.cvd_history.append(current_cvd)
        self.timestamp_history.append(timestamp)

        # Need minimum data for analysis
        if len(self.price_history) < self.lookback:
            return None

        # Convert deques to lists for analysis
        prices = list(self.price_history)
        cvds = list(self.cvd_history)

        # Find highs and lows in the lookback period
        price_high = max(prices)
        price_low = min(prices)
        cvd_high = max(cvds)
        cvd_low = min(cvds)

        # Calculate confidence based on divergence strength
        confidence = self._calculate_confidence(prices, cvds)

        # Bearish divergence: higher price, lower CVD
        if current_price > price_high and current_cvd < cvd_high:
            logger.debug(
                f"Bearish divergence detected for {self.symbol} - confidence: {confidence:.2f}"
            )
            return "bearish"

        # Bullish divergence: lower price, higher CVD
        if current_price < price_low and current_cvd > cvd_low:
            logger.debug(
                f"Bullish divergence detected for {self.symbol} - confidence: {confidence:.2f}"
            )
            return "bullish"

        return None

    def _get_config_value(self, key: str, default: object) -> object:
        """
        Best-effort fetch from configuration provider:
        - First try get_value(key)
        - Then try get_namespace('system')[key]
        - On any error or None, return default
        """
        try:
            getter = getattr(self.config, "get_value", None)
            val = getter(key, None) if callable(getter) else None
            if val is None:
                ns_getter = getattr(self.config, "get_namespace", None)
                ns = ns_getter("system") if callable(ns_getter) else {}
                val = ns.get(key) if isinstance(ns, dict) else None
        except Exception:
            val = None
        return default if val is None else val

    def _get_numeric_config(self, key: str, default: float | int) -> float:
        """
        Fetch a numeric value from config, coercing to float/int as appropriate.
        Returns the provided default on any error.
        """
        val = self._get_config_value(key, None)
        try:
            if isinstance(default, int):
                if isinstance(val, (int, float, str)):
                    return float(int(val))
                return float(int(default))
            # default was float-like
            if isinstance(val, (int, float, str)):
                return float(val)
            return float(default)
        except Exception:
            # As a last resort, return default coerced to float
            return float(default)

    def _calculate_confidence(self, prices: list[float], cvds: list[float]) -> float:
        """
        Calculate confidence level for divergence detection.

        Args:
            prices: List of price points
            cvds: List of CVD values

        Returns:
            float: Confidence level between 0 and 1
        """
        if len(prices) < 2 or len(cvds) < 2:
            return 0.0

        # Calculate price and CVD trends
        price_trend = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        cvd_trend = (cvds[-1] - cvds[0]) / abs(cvds[0]) if cvds[0] != 0 else 0

        # Divergence strength is the difference in trends
        divergence_strength = abs(price_trend - cvd_trend)

        # Normalize to 0-1 range using configured threshold (default 1.0)
        threshold = self._get_numeric_config("cvd_divergence_threshold", 1.0)
        if threshold <= 0:
            threshold = 1.0
        confidence = min(divergence_strength / float(threshold), 1.0)

        return max(0.0, min(1.0, confidence))

    def create_context_packet(
        self,
        current_price: float,
        current_cvd: float,
        divergence: Optional[Literal["bullish", "bearish"]],
        timestamp: datetime,
    ) -> object:
        """
        Create a ContextPacket with CVD divergence analysis.

        Args:
            current_price: Current market price
            current_cvd: Current cumulative volume delta
            divergence: Detected divergence signal
            timestamp: Current timestamp

        Returns:
            ContextPacket: Enriched context packet
        """
        # Calculate confidence
        confidence = (
            self._calculate_confidence(list(self.price_history), list(self.cvd_history))
            if divergence
            else None
        )

        return cast(Any, ContextPacket)(
            timestamp=timestamp,
            symbol=self.symbol,
            current_price=current_price,
            current_cvd=current_cvd,
            cvd_divergence=divergence,
            divergence_confidence=confidence,
            price_history=list(self.price_history),
            cvd_history=list(self.cvd_history),
            analysis_window=self.lookback,
            metadata={"detector_type": "CVDDivergenceDetector", "lookback_period": self.lookback},
        )

    def reset(self) -> None:
        """Reset the detector's history."""
        self.price_history.clear()
        self.cvd_history.clear()
        self.timestamp_history.clear()
        logger.info(f"Reset CVDDivergenceDetector for {self.symbol}")

    def get_stats(self) -> dict[str, object]:
        """Get current detector statistics."""
        return {
            "symbol": self.symbol,
            "lookback": self.lookback,
            "data_points": len(self.price_history),
            "price_range": (
                (min(self.price_history), max(self.price_history)) if self.price_history else (0, 0)
            ),
            "cvd_range": (
                (min(self.cvd_history), max(self.cvd_history)) if self.cvd_history else (0, 0)
            ),
        }
