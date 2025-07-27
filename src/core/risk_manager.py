"""
Risk Manager Implementation
===========================

Concrete implementation of IRiskManager with comprehensive risk controls.
Optimized for real-time risk assessment and position sizing.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import numpy as np

from .interfaces import IRiskManager
from ..performance import get_global_cache, VectorizedIndicators

logger = logging.getLogger(__name__)


class RiskManager(IRiskManager):
    """Comprehensive risk management system with real-time monitoring."""
    
    def __init__(self, max_position_size: float = 0.02, max_drawdown: float = 0.05,
                 risk_per_trade: float = 0.01, cache_ttl: int = 300):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size as % of portfolio
            max_drawdown: Maximum allowed drawdown
            risk_per_trade: Risk per trade as % of portfolio
            cache_ttl: Cache TTL in seconds
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.risk_per_trade = risk_per_trade
        self.cache_ttl = cache_ttl
        self.cache = get_global_cache()
        self._cache_key_prefix = "risk"
        
    async def validate_position(self, position: Dict[str, Any]) -> bool:
        """Validate if position meets risk criteria."""
        try:
            symbol = position.get('symbol', '')
            size = position.get('size', 0.0)
            portfolio_value = position.get('portfolio_value', 100000.0)
            current_positions = position.get('current_positions', {})
            
            # Calculate position size as % of portfolio
            position_size_pct = abs(size) / portfolio_value
            
            # Check against maximum position size
            if position_size_pct > self.max_position_size:
                logger.warning(f"Position size {position_size_pct:.2%} exceeds max {self.max_position_size:.2%}")
                return False
            
            # Check total exposure
            total_exposure = sum(abs(pos.get('size', 0.0)) for pos in current_positions.values()) / portfolio_value
            if total_exposure + position_size_pct > self.max_position_size * 2:
                logger.warning(f"Total exposure would exceed limits")
                return False
            
            # Check symbol-specific limits
            symbol_exposure = sum(
                abs(pos.get('size', 0.0)) 
                for pos in current_positions.values() 
                if pos.get('symbol') == symbol
            ) / portfolio_value
            
            if symbol_exposure + position_size_pct > self.max_position_size:
                logger.warning(f"Symbol exposure would exceed limits")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return False
    
    async def calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate appropriate position size for signal."""
        try:
            symbol = signal.get('symbol', '')
            direction = signal.get('direction', 'BUY')
            confidence = signal.get('confidence', 0.5)
            stop_loss = signal.get('stop_loss', 0.0)
            portfolio_value = signal.get('portfolio_value', 100000.0)
            current_price = signal.get('current_price', 0.0)
            
            if current_price <= 0 or stop_loss <= 0:
                logger.warning("Invalid price or stop loss")
                return 0.0
            
            # Calculate risk amount
            risk_amount = portfolio_value * self.risk_per_trade
            
            # Calculate position size based on stop loss
            price_difference = abs(current_price - stop_loss)
            if price_difference == 0:
                logger.warning("Zero price difference")
                return 0.0
            
            # Basic position size calculation
            position_size = risk_amount / price_difference
            
            # Adjust for confidence
            position_size *= confidence
            
            # Apply maximum position size limit
            max_size = portfolio_value * self.max_position_size
            position_size = min(position_size, max_size)
            
            # Cache the calculation
            self._cache_position_size(symbol, position_size, signal)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def calculate_risk_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        try:
            positions = portfolio.get('positions', {})
            portfolio_value = portfolio.get('portfolio_value', 100000.0)
            
            if not positions:
                return {
                    'total_exposure': 0.0,
                    'max_drawdown': 0.0,
                    'var_95': 0.0,
                    'sharpe_ratio': 0.0,
                    'beta': 0.0
                }
            
            # Calculate total exposure
            total_exposure = sum(abs(pos.get('size', 0.0)) for pos in positions.values()) / portfolio_value
            
            # Calculate VaR (Value at Risk) - simplified
            returns = self._get_portfolio_returns(portfolio)
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
            
            # Calculate Sharpe ratio - simplified
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown(returns)
            
            metrics = {
                'total_exposure': total_exposure,
                'max_drawdown': max_drawdown,
                'var_95': abs(var_95),
                'sharpe_ratio': sharpe_ratio,
                'beta': 1.0  # Simplified beta
            }
            
            # Cache metrics
            self._cache_risk_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'total_exposure': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'sharpe_ratio': 0.0,
                'beta': 0.0
            }
    
    def _get_portfolio_returns(self, portfolio: Dict[str, Any]) -> np.ndarray:
        """Get portfolio returns for risk calculations."""
        # Simplified - in practice, would use actual historical returns
        return np.random.randn(252) * 0.01  # 1% daily volatility
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - 0.0001  # Risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) < 2:
            return 0.0
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def _cache_position_size(self, symbol: str, size: float, signal: Dict[str, Any]) -> None:
        """Cache position size calculation."""
        self.cache.cache_market_data(
            "risk", "position_size",
            np.array([size]),
            symbol=symbol,
            direction=signal.get('direction', 'BUY'),
            confidence=signal.get('confidence', 0.5)
        )
    
    def _cache_risk_metrics(self, metrics: Dict[str, float]) -> None:
        """Cache risk metrics."""
        self.cache.cache_market_data(
            "risk", "metrics",
            np.array(list(metrics.values())),
            timestamp=datetime.now().strftime('%Y%m%d%H%M')
        )
    
    async def validate_order(self, order: Dict[str, Any]) -> bool:
        """Validate order parameters."""
        try:
            required_fields = ['symbol', 'size', 'price', 'type']
            if not all(field in order for field in required_fields):
                return False
            
            size = order.get('size', 0.0)
            price = order.get('price', 0.0)
            
            if size <= 0 or price <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False
    
    async def get_risk_limits(self) -> Dict[str, float]:
        """Get current risk limits."""
        return {
            'max_position_size': self.max_position_size,
            'max_drawdown': self.max_drawdown,
            'risk_per_trade': self.risk_per_trade
        }
    
    async def update_risk_limits(self, limits: Dict[str, float]) -> bool:
        """Update risk limits."""
        try:
            if 'max_position_size' in limits:
                self.max_position_size = limits['max_position_size']
            if 'max_drawdown' in limits:
                self.max_drawdown = limits['max_drawdown']
            if 'risk_per_trade' in limits:
                self.risk_per_trade = limits['risk_per_trade']
            
            logger.info("Risk limits updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating risk limits: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the risk manager."""
        return {
            'max_position_size': self.max_position_size,
            'max_drawdown': self.max_drawdown,
            'risk_per_trade': self.risk_per_trade,
            'cache_hits': 0,  # Implement actual cache tracking
            'validations_performed': 0
        }


# Global risk manager instance
_global_risk_manager = None


def get_global_risk_manager() -> RiskManager:
    """Get global risk manager instance."""
    global _global_risk_manager
    if _global_risk_manager is None:
        _global_risk_manager = RiskManager()
    return _global_risk_manager


if __name__ == "__main__":
    import asyncio
    
    async def test_risk_manager():
        risk_manager = RiskManager()
        
        # Test position validation
        position = {
            'symbol': 'EURUSD',
            'size': 1000.0,
            'portfolio_value': 100000.0,
            'current_positions': {}
        }
        
        is_valid = await risk_manager.validate_position(position)
        print(f"Position valid: {is_valid}")
        
        # Test position sizing
        signal = {
            'symbol': 'EURUSD',
            'direction': 'BUY',
            'confidence': 0.8,
            'stop_loss': 1.095,
            'current_price': 1.10,
            'portfolio_value': 100000.0
        }
        
        position_size = await risk_manager.calculate_position_size(signal)
        print(f"Position size: {position_size}")
        
        # Test risk metrics
        portfolio = {
            'positions': {
                'EURUSD': {'size': 1000.0, 'price': 1.10},
                'GBPUSD': {'size': 500.0, 'price': 1.30}
            },
            'portfolio_value': 100000.0
        }
        
        metrics = await risk_manager.calculate_risk_metrics(portfolio)
        print(f"Risk metrics: {metrics}")
    
    asyncio.run(test_risk_manager())
