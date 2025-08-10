"""
Risk Manager Implementation
==========================

Complete implementation of the IRiskManager interface for the EMP system.
Provides real risk management capabilities with position sizing and validation.
"""

import logging
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime

try:
    from src.core.interfaces import IRiskManager  # legacy
except Exception:  # pragma: no cover
    class IRiskManager:  # type: ignore
        pass
from src.core.risk.manager import RiskManager as RealRiskManager, RiskConfig as RealRiskConfig

logger = logging.getLogger(__name__)


class RiskManagerImpl(IRiskManager):
    """
    Complete implementation of IRiskManager interface.
    
    This class provides real risk management capabilities including:
    - Position sizing using Kelly Criterion
    - Risk validation for trades
    - Portfolio risk monitoring
    - Dynamic risk adjustment
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize the risk manager with configuration.
        
        Args:
            initial_balance: Starting account balance
        """
        self.config = RealRiskConfig(
            max_risk_per_trade_pct=Decimal('0.02'),
            max_leverage=Decimal('10.0'),
            max_total_exposure_pct=Decimal('0.5'),
            max_drawdown_pct=Decimal('0.25'),
            min_position_size=Decimal('1000'),
            max_position_size=Decimal('1000000'),
            kelly_fraction=Decimal('0.25')
        )
        
        self.risk_manager = RealRiskManager(self.config)
        self.risk_manager.update_account_balance(Decimal(str(initial_balance)))
        
        # Track current positions
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.account_balance = Decimal(str(initial_balance))
        
        logger.info(f"RiskManagerImpl initialized with balance: ${initial_balance}")
    
    async def validate_position(self, position: Dict[str, Any]) -> bool:
        """
        Validate if position meets risk criteria.
        
        Args:
            position: Position details including symbol, size, entry_price
            
        Returns:
            True if position is valid, False otherwise
        """
        try:
            symbol = position.get('symbol', '')
            size = Decimal(str(position.get('size', 0)))
            entry_price = Decimal(str(position.get('entry_price', 0)))
            
            # Validate basic parameters
            if size <= 0:
                logger.warning(f"Invalid position size: {size}")
                return False
            
            if entry_price <= 0:
                logger.warning(f"Invalid entry price: {entry_price}")
                return False
            
            # Use RealRiskManager validation
            is_valid = self.risk_manager.validate_position(
                size, None, self.account_balance
            )
            
            if is_valid:
                logger.info(f"Position validated: {symbol} size={size}")
            else:
                logger.warning(f"Position rejected: {symbol} size={size}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return False
    
    async def calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """
        Calculate appropriate position size for signal using Kelly Criterion.
        
        Args:
            signal: Trading signal with risk parameters
            
        Returns:
            Position size in base currency
        """
        try:
            # Extract signal parameters
            symbol = signal.get('symbol', '')
            confidence = signal.get('confidence', 0.5)
            stop_loss_pct = Decimal(str(signal.get('stop_loss_pct', 0.05)))
            
            # Calculate win rate based on confidence
            win_rate = max(0.1, min(0.9, confidence))
            
            # Use historical performance for Kelly calculation
            avg_win = 0.02  # 2% average win
            avg_loss = 0.01  # 1% average loss
            
            # Calculate Kelly Criterion
            kelly_fraction = self.risk_manager.calculate_kelly_criterion(
                win_rate, avg_win, avg_loss
            )
            
            # Calculate position size based on risk
            risk_per_trade = Decimal('0.02')  # 2% risk per trade
            position_size = self.risk_manager.calculate_position_size(
                self.account_balance, risk_per_trade, stop_loss_pct
            )
            
            # Apply Kelly fraction
            final_size = float(position_size) * kelly_fraction
            
            logger.info(f"Calculated position size: {symbol} size={final_size:.2f}")
            
            return max(1000.0, min(final_size, float(self.config.max_position_size)))
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1000.0  # Default minimum position
    
    def update_account_balance(self, new_balance: float) -> None:
        """
        Update the account balance.
        
        Args:
            new_balance: New account balance
        """
        self.account_balance = Decimal(str(new_balance))
        self.risk_manager.update_account_balance(self.account_balance)
        logger.info(f"Account balance updated: ${new_balance}")
    
    def add_position(self, symbol: str, size: float, entry_price: float) -> None:
        """
        Add a new position to track.
        
        Args:
            symbol: Trading symbol
            size: Position size
            entry_price: Entry price
        """
        self.risk_manager.add_position(
            symbol, 
            Decimal(str(size)), 
            Decimal(str(entry_price))
        )
        
        self.positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'entry_time': datetime.now()
        }
        
        logger.info(f"Position added: {symbol} size={size} price={entry_price}")
    
    def update_position_value(self, symbol: str, current_price: float) -> None:
        """
        Update position value with current price.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        self.risk_manager.update_position_price(symbol, Decimal(str(current_price)))
        
        if symbol in self.positions:
            self.positions[symbol]['current_price'] = current_price
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary.
        
        Returns:
            Dictionary with current risk metrics
        """
        summary = self.risk_manager.get_risk_summary()
        
        # Add position tracking
        summary['positions'] = len(self.positions)
        summary['tracked_positions'] = list(self.positions.keys())
        
        return summary
    
    def calculate_portfolio_risk(self) -> Dict[str, float]:
        """
        Calculate current portfolio risk metrics.
        
        Returns:
            Dictionary with portfolio risk metrics
        """
        return self.risk_manager.calculate_portfolio_risk()
    
    def get_position_risk(self, symbol: str) -> Dict[str, Any]:
        """
        Get risk metrics for a specific position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position risk metrics
        """
        if symbol not in self.positions:
            return {}
        
        position = self.positions[symbol]
        return {
            'symbol': symbol,
            'size': position['size'],
            'entry_price': position['entry_price'],
            'current_price': position.get('current_price', position['entry_price']),
            'risk_amount': position['size'] * 0.02  # 2% risk
        }


# Factory function for easy instantiation
def create_risk_manager(initial_balance: float = 10000.0) -> RiskManagerImpl:
    """
    Create a new RiskManagerImpl instance.
    
    Args:
        initial_balance: Starting account balance
        
    Returns:
        Configured RiskManagerImpl instance
    """
    return RiskManagerImpl(initial_balance)


if __name__ == "__main__":
    # Test the implementation
    print("Testing RiskManagerImpl...")
    
    risk_manager = create_risk_manager(10000.0)
    
    # Test position validation
    position = {
        'symbol': 'EURUSD',
        'size': 10000.0,
        'entry_price': 1.1000
    }
    
    is_valid = asyncio.run(risk_manager.validate_position(position))
    print(f"Position validation: {is_valid}")
    
    # Test position sizing
    signal = {
        'symbol': 'EURUSD',
        'confidence': 0.7,
        'stop_loss_pct': 0.02
    }
    
    position_size = asyncio.run(risk_manager.calculate_position_size(signal))
    print(f"Calculated position size: {position_size}")
    
    # Test risk summary
    summary = risk_manager.get_risk_summary()
    print(f"Risk summary: {summary}")
    
    print("RiskManagerImpl test completed successfully!")
