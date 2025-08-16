"""
Sensory Cortex v2.2 - Core Utilities

Shared mathematical building blocks and utility functions.
Eliminates redundancy and provides consistent calculations across all dimensions.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_organ import MarketRegime
from src.trading.monitoring.performance_tracker import (
    PerformanceTracker as PerformanceTracker,
)

logger = logging.getLogger(__name__)


class EMA:
    """
    Exponential Moving Average with efficient incremental updates.
    Eliminates the need for storing historical data while maintaining accuracy.
    """

    def __init__(self, period: int, initial_value: Optional[float] = None):
        """
        Initialize EMA calculator.

        Args:
            period: EMA period
            initial_value: Optional initial value, otherwise uses first update
        """
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self.value = initial_value
        self.is_initialized = initial_value is not None

    def update(self, new_value: float) -> float:
        """
        Update EMA with new value.

        Args:
            new_value: New data point

        Returns:
            Updated EMA value
        """
        if not self.is_initialized:
            self.value = new_value
            self.is_initialized = True
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value

        return self.value

    def get_value(self) -> Optional[float]:
        """Get current EMA value."""
        return self.value if self.is_initialized else None

    def reset(self, initial_value: Optional[float] = None) -> None:
        """Reset EMA state."""
        self.value = initial_value
        self.is_initialized = initial_value is not None


class WelfordVar:
    """
    Welford's algorithm for incremental variance calculation.
    Provides numerically stable variance computation without storing historical data.
    """

    def __init__(self):
        """Initialize Welford variance calculator."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from mean

    def update(self, new_value: float) -> Tuple[float, float]:
        """
        Update variance calculation with new value.

        Args:
            new_value: New data point

        Returns:
            Tuple of (mean, variance)
        """
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.m2 += delta * delta2

        variance = self.m2 / self.count if self.count > 0 else 0.0
        return self.mean, variance

    def get_stats(self) -> Tuple[float, float, float]:
        """
        Get current statistics.

        Returns:
            Tuple of (mean, variance, standard_deviation)
        """
        if self.count == 0:
            return 0.0, 0.0, 0.0

        variance = self.m2 / self.count
        std_dev = np.sqrt(variance)
        return self.mean, variance, std_dev

    def reset(self) -> None:
        """Reset variance calculation state."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0


@dataclass
class ConfidenceFactors:
    """
    Factors that contribute to overall confidence calculation.
    Provides transparency into confidence assessment.
    """

    data_quality: float = 1.0  # Quality of input data (0-1)
    signal_clarity: float = 1.0  # How clear/strong the signal is (0-1)
    # Historical performance of this signal (0-1)
    historical_accuracy: float = 1.0
    # Suitability of current market conditions (0-1)
    market_conditions: float = 1.0
    confluence: float = 1.0  # Agreement with other indicators (0-1)
    # Adjustment for current volatility (0-1)
    volatility_adjustment: float = 1.0

    def get_weighted_average(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted average confidence.

        Args:
            weights: Optional custom weights for each factor

        Returns:
            Weighted confidence score (0-1)
        """
        if weights is None:
            # Default equal weighting
            weights = {
                "data_quality": 0.2,
                "signal_clarity": 0.25,
                "historical_accuracy": 0.2,
                "market_conditions": 0.15,
                "confluence": 0.15,
                "volatility_adjustment": 0.05,
            }

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = (
            self.data_quality * weights.get("data_quality", 0)
            + self.signal_clarity * weights.get("signal_clarity", 0)
            + self.historical_accuracy * weights.get("historical_accuracy", 0)
            + self.market_conditions * weights.get("market_conditions", 0)
            + self.confluence * weights.get("confluence", 0)
            + self.volatility_adjustment * weights.get("volatility_adjustment", 0)
        )

        return weighted_sum / total_weight


def compute_confidence(
    signal_strength: float,
    data_quality: float = 1.0,
    historical_accuracy: float = 0.7,
    market_regime: MarketRegime = MarketRegime.CONSOLIDATING,
    volatility: float = 0.5,
    confluence_signals: Optional[List[float]] = None,
    custom_factors: Optional[ConfidenceFactors] = None,
) -> float:
    """
    Universal confidence computation function used across all dimensions.
    Standardizes confidence assessment methodology.

    Args:
        signal_strength: Primary signal strength (-1 to +1)
        data_quality: Quality of input data (0-1)
        historical_accuracy: Historical performance of this signal type (0-1)
        market_regime: Current market regime
        volatility: Current market volatility (0-1)
        confluence_signals: List of supporting signal strengths
        custom_factors: Optional custom confidence factors

    Returns:
        Confidence score (0-1)
    """
    try:
        # Use custom factors if provided, otherwise create default
        if custom_factors is None:
            factors = ConfidenceFactors()
            factors.data_quality = data_quality
            factors.historical_accuracy = historical_accuracy
        else:
            factors = custom_factors

        # Signal clarity based on absolute strength
        factors.signal_clarity = abs(signal_strength)

        # Market conditions adjustment based on regime
        regime_multipliers = {
            MarketRegime.TRENDING_STRONG: 1.0,
            MarketRegime.TRENDING_WEAK: 0.8,
            MarketRegime.CONSOLIDATING: 0.6,
            MarketRegime.EXHAUSTED: 0.4,
            MarketRegime.BREAKOUT: 0.9,
            MarketRegime.REVERSAL: 0.7,
        }
        factors.market_conditions = regime_multipliers.get(market_regime, 0.6)

        # Volatility adjustment (moderate volatility is optimal)
        # Too low volatility = no movement, too high = noise
        optimal_volatility = 0.3
        volatility_penalty = abs(volatility - optimal_volatility) / optimal_volatility
        factors.volatility_adjustment = max(0.1, 1.0 - volatility_penalty)

        # Confluence calculation
        if confluence_signals:
            # Calculate agreement between signals
            signal_agreements = []
            for conf_signal in confluence_signals:
                # Agreement is higher when signals point in same direction
                agreement = 1.0 - abs(signal_strength - conf_signal) / 2.0
                signal_agreements.append(max(0.0, agreement))

            factors.confluence = (
                np.mean(signal_agreements) if signal_agreements else 0.5
            )
        else:
            factors.confluence = 0.5  # Neutral when no confluence data

        # Calculate final confidence
        confidence = factors.get_weighted_average()

        # Apply non-linear scaling to emphasize high-confidence signals
        # This makes the system more decisive when conditions are favorable
        confidence = confidence**0.8  # Slight compression to avoid overconfidence

        # Ensure bounds
        confidence = max(0.0, min(1.0, confidence))

        logger.debug(
            f"Confidence calculation: signal={signal_strength:.3f}, "
            f"factors={factors}, final={confidence:.3f}"
        )

        return confidence

    except Exception as e:
        logger.error(f"Error in confidence computation: {e}")
        return 0.1  # Conservative fallback


def normalize_signal(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to the standard signal range (-1 to +1).

    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value

    Returns:
        Normalized value (-1 to +1)
    """
    if max_val == min_val:
        return 0.0

    # Normalize to 0-1 range first
    normalized = (value - min_val) / (max_val - min_val)

    # Convert to -1 to +1 range
    return 2.0 * normalized - 1.0


def calculate_z_score(value: float, mean: float, std_dev: float) -> float:
    """
    Calculate Z-score for anomaly detection.

    Args:
        value: Current value
        mean: Historical mean
        std_dev: Historical standard deviation

    Returns:
        Z-score
    """
    if std_dev == 0:
        return 0.0
    return (value - mean) / std_dev


def exponential_decay(distance: float, decay_rate: float = 0.1) -> float:
    """
    Calculate exponential decay factor based on distance.
    Useful for time-based or price-based decay calculations.

    Args:
        distance: Distance from reference point
        decay_rate: Decay rate parameter

    Returns:
        Decay factor (0-1)
    """
    return np.exp(-decay_rate * abs(distance))


def calculate_momentum(prices: List[float], period: int = 14) -> float:
    """
    Calculate price momentum over specified period.

    Args:
        prices: List of prices (most recent last)
        period: Momentum calculation period

    Returns:
        Momentum value (-1 to +1)
    """
    if len(prices) < period + 1:
        return 0.0

    current_price = prices[-1]
    past_price = prices[-period - 1]

    if past_price == 0:
        return 0.0

    momentum = (current_price - past_price) / past_price

    # Normalize to reasonable range (assuming max 10% move)
    return np.tanh(momentum * 10)


def calculate_volatility(prices: List[float], period: int = 20) -> float:
    """
    Calculate realized volatility from price series.

    Args:
        prices: List of prices
        period: Volatility calculation period

    Returns:
        Volatility (0-1 normalized)
    """
    if len(prices) < period:
        return 0.5  # Default moderate volatility

    recent_prices = prices[-period:]
    returns = np.diff(np.log(recent_prices))
    volatility = np.std(returns) * np.sqrt(252)  # Annualized

    # Normalize to 0-1 range (assuming max 100% annual volatility)
    return min(1.0, volatility)


def detect_regime_change(
    current_regime: MarketRegime,
    volatility: float,
    momentum: float,
    trend_strength: float,
) -> MarketRegime:
    """
    Detect market regime changes based on multiple factors.

    Args:
        current_regime: Current market regime
        volatility: Current volatility (0-1)
        momentum: Current momentum (-1 to +1)
        trend_strength: Current trend strength (0-1)

    Returns:
        Updated market regime
    """
    abs_momentum = abs(momentum)

    # Regime detection logic
    if volatility > 0.8:
        if abs_momentum > 0.6:
            return MarketRegime.BREAKOUT
        else:
            return MarketRegime.EXHAUSTED
    elif trend_strength > 0.7:
        if abs_momentum > 0.5:
            return MarketRegime.TRENDING_STRONG
        else:
            return MarketRegime.TRENDING_WEAK
    elif volatility < 0.3 and abs_momentum < 0.2:
        return MarketRegime.CONSOLIDATING
    elif current_regime in [MarketRegime.TRENDING_STRONG, MarketRegime.TRENDING_WEAK]:
        if abs_momentum < 0.1:
            return MarketRegime.REVERSAL

    return current_regime  # No change detected


def smooth_signal(
    signal: float, previous_signal: float, smoothing_factor: float = 0.3
) -> float:
    """
    Apply exponential smoothing to reduce signal noise.

    Args:
        signal: Current signal value
        previous_signal: Previous signal value
        smoothing_factor: Smoothing factor (0-1, higher = more smoothing)

    Returns:
        Smoothed signal value
    """
    return smoothing_factor * signal + (1 - smoothing_factor) * previous_signal


def calculate_divergence(
    price_series: List[float], indicator_series: List[float]
) -> float:
    """
    Calculate divergence between price and indicator.

    Args:
        price_series: Price data points
        indicator_series: Indicator data points

    Returns:
        Divergence strength (-1 to +1)
    """
    if len(price_series) < 2 or len(indicator_series) < 2:
        return 0.0

    # Calculate recent trends
    price_trend = (price_series[-1] - price_series[-2]) / price_series[-2]
    indicator_trend = (indicator_series[-1] - indicator_series[-2]) / abs(
        indicator_series[-2]
    )

    # Divergence occurs when trends are opposite
    if price_trend * indicator_trend < 0:
        return -abs(price_trend - indicator_trend)  # Negative for divergence
    else:
        # Positive for confluence
        return abs(price_trend + indicator_trend) / 2


# Utility constants
TRADING_DAYS_PER_YEAR = 252
HOURS_PER_TRADING_DAY = 24
MINUTES_PER_HOUR = 60
SECONDS_PER_MINUTE = 60
