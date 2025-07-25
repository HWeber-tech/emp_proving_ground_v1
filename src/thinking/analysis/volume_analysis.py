"""
Volume Analysis Module
Core CVD (Cumulative Volume Delta) calculation logic
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba not available - using pure Python for CVD calculations")


def calculate_delta(trade_price: float, trade_size: float, best_bid: float, best_ask: float) -> float:
    """
    Calculate the delta for a single trade based on the tick test.
    
    Args:
        trade_price: The price of the trade
        trade_size: The size/volume of the trade
        best_bid: Current best bid price
        best_ask: Current best ask price
        
    Returns:
        float: Positive delta for buyer-initiated trades, negative for seller-initiated
    """
    if trade_price >= best_ask:
        # Aggressor was a BUYER - trade hit the ask
        return trade_size
    elif trade_price <= best_bid:
        # Aggressor was a SELLER - trade hit the bid
        return -trade_size
    else:
        # Trade was inside the spread (passive fill) - no delta
        return 0.0


# Conditionally apply Numba JIT if available
if HAS_NUMBA:
    from src.governance.system_config import config
    
    if config.enable_numba_acceleration:
        calculate_delta = jit(nopython=True)(calculate_delta)
        logger.info("Numba acceleration enabled for CVD calculations")
    else:
        logger.info("Numba acceleration disabled via configuration")
else:
    logger.info("Numba not available - using pure Python for CVD calculations")


class CVDAnalyzer:
    """
    High-level CVD analysis utilities
    """
    
    @staticmethod
    def detect_divergence(
        price_history: list[float],
        cvd_history: list[float],
        lookback: int = 20
    ) -> Optional[str]:
        """
        Detect price/CVD divergence patterns.
        
        Args:
            price_history: List of recent price points
            cvd_history: List of recent CVD values
            lookback: Number of points to look back for divergence
            
        Returns:
            Optional[str]: "bullish", "bearish", or None
        """
        if len(price_history) < lookback or len(cvd_history) < lookback:
            return None
            
        # Get recent window
        recent_prices = price_history[-lookback:]
        recent_cvd = cvd_history[-lookback:]
        
        # Find highs and lows
        price_high = max(recent_prices)
        price_low = min(recent_prices)
        cvd_high = max(recent_cvd)
        cvd_low = min(recent_cvd)
        
        current_price = price_history[-1]
        current_cvd = cvd_history[-1]
        
        # Bearish divergence: higher price, lower CVD
        if current_price > price_high and current_cvd < cvd_high:
            return "bearish"
            
        # Bullish divergence: lower price, higher CVD
        if current_price < price_low and current_cvd > cvd_low:
            return "bullish"
            
        return None
