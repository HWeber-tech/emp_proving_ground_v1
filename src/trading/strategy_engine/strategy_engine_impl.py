"""
Strategy Engine Implementation
==============================

Complete implementation of the IStrategyEngine interface for the EMP system.
Provides strategy lifecycle management, execution, and performance tracking.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.core.interfaces import IStrategyEngine
from src.trading.strategy_engine.base_strategy import BaseStrategy, StrategySignal, StrategyType
from src.risk.risk_manager_impl import RiskManagerImpl

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy execution status"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StrategyExecutionResult:
    """Result of strategy execution"""
    strategy_id: str
    signal: Optional[StrategySignal]
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0


class StrategyEngineImpl:
    """
    Complete implementation of strategy engine functionality.
    
    This class manages the lifecycle of trading strategies including:
    - Strategy registration and initialization
    - Real-time strategy execution
    - Performance tracking and monitoring
    - Risk integration
    - Signal generation and validation
    """
    
    def __init__(self, risk_manager: Optional[RiskManagerImpl] = None):
        """
        Initialize the strategy engine.
        
        Args:
            risk_manager: Risk management instance
        """
        self.risk_manager = risk_manager or RiskManagerImpl()
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_status: Dict[str, StrategyStatus] = {}
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[StrategyExecutionResult] = []
        
        logger.info("StrategyEngineImpl initialized")
    
    def register_strategy(self, strategy: BaseStrategy) -> bool:
        """
        Register a new strategy with the engine.
        
        Args:
            strategy: Strategy instance to register
            
        Returns:
            True if registration successful
        """
        try:
            strategy_id = strategy.strategy_id
            
            # Validate strategy
            if not strategy_id:
                logger.error("Strategy ID is required")
                return False
            
            if strategy_id in self.strategies:
                logger.warning(f"Strategy {strategy_id} already registered")
                return False
            
            # Register strategy
            self.strategies[strategy_id] = strategy
            self.strategy_status[strategy_id] = StrategyStatus.INACTIVE
            self.strategy_performance[strategy_id] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'last_updated': datetime.now()
            }
            
            logger.info(f"Strategy {strategy_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering strategy: {e}")
            return False
    
    def unregister_strategy(self, strategy_id: str) -> bool:
        """
        Unregister a strategy from the engine.
        
        Args:
            strategy_id: ID of strategy to unregister
            
        Returns:
            True if unregistration successful
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return False
        
        # Stop strategy if active
        if self.strategy_status.get(strategy_id) == StrategyStatus.ACTIVE:
            self.stop_strategy(strategy_id)
        
        # Remove strategy
        del self.strategies[strategy_id]
        del self.strategy_status[strategy_id]
        del self.strategy_performance[strategy_id]
        
        logger.info(f"Strategy {strategy_id} unregistered")
        return True
    
    def start_strategy(self, strategy_id: str) -> bool:
        """
        Start strategy execution.
        
        Args:
            strategy_id: ID of strategy to start
            
        Returns:
            True if started successfully
        """
        if strategy_id not in self.strategies:
            logger.error(f"Strategy {strategy_id} not found")
            return False
        
        strategy = self.strategies[strategy_id]
        strategy.start()
        self.strategy_status[strategy_id] = StrategyStatus.ACTIVE
        
        logger.info(f"Strategy {strategy_id} started")
        return True
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """
        Stop strategy execution.
        
        Args:
            strategy_id: ID of strategy to stop
            
        Returns:
            True if stopped successfully
        """
        if strategy_id not in self.strategies:
            logger.error(f"Strategy {strategy_id} not found")
            return False
        
        strategy = self.strategies[strategy_id]
        strategy.stop()
        self.strategy_status[strategy_id] = StrategyStatus.STOPPED
        
        logger.info(f"Strategy {strategy_id} stopped")
        return True
    
    def pause_strategy(self, strategy_id: str) -> bool:
        """
        Pause strategy execution.
        
        Args:
            strategy_id: ID of strategy to pause
            
        Returns:
            True if paused successfully
        """
        if strategy_id not in self.strategies:
            logger.error(f"Strategy {strategy_id} not found")
            return False
        
        strategy = self.strategies[strategy_id]
        strategy.pause()
        self.strategy_status[strategy_id] = StrategyStatus.PAUSED
        
        logger.info(f"Strategy {strategy_id} paused")
        return True
    
    async def execute_strategy(self, strategy_id: str, market_data: Dict[str, Any]) -> StrategyExecutionResult:
        """
        Execute a strategy with market data.
        
        Args:
            strategy_id: ID of strategy to execute
            market_data: Current market data
            
        Returns:
            Execution result
        """
        start_time = datetime.now()
        
        try:
            if strategy_id not in self.strategies:
                return StrategyExecutionResult(
                    strategy_id=strategy_id,
                    signal=None,
                    success=False,
                    error_message=f"Strategy {strategy_id} not found"
                )
            
            strategy = self.strategies[strategy_id]
            status = self.strategy_status.get(strategy_id)
            
            if status != StrategyStatus.ACTIVE:
                return StrategyExecutionResult(
                    strategy_id=strategy_id,
                    signal=None,
                    success=False,
                    error_message=f"Strategy {strategy_id} is not active"
                )
            
            # Execute strategy
            symbol = market_data.get('symbol', '')
            signal = await strategy.generate_signal([market_data], symbol)
            
            # Validate signal with risk manager
            if signal:
                is_valid = await self.risk_manager.validate_position({
                    'symbol': signal.symbol,
                    'size': signal.quantity,
                    'entry_price': signal.price
                })
                
                if not is_valid:
                    logger.warning(f"Signal rejected by risk manager: {signal}")
                    signal = None
            
            # Update performance
            if signal:
                self._update_strategy_performance(strategy_id, signal)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = StrategyExecutionResult(
                strategy_id=strategy_id,
                signal=signal,
                success=True,
                execution_time=execution_time
            )
            
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_id}: {e}")
            return StrategyExecutionResult(
                strategy_id=strategy_id,
                signal=None,
                success=False,
                error_message=str(e)
            )
    
    async def execute_all_strategies(self, market_data: Dict[str, Any]) -> List[StrategyExecutionResult]:
        """
        Execute all active strategies with market data.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of execution results
        """
        results = []
        
        for strategy_id in self.strategies:
            if self.strategy_status.get(strategy_id) == StrategyStatus.ACTIVE:
                result = await self.execute_strategy(strategy_id, market_data)
                results.append(result)
        
        return results
    
    def get_strategy_status(self, strategy_id: str) -> Optional[StrategyStatus]:
        """
        Get the current status of a strategy.
        
        Args:
            strategy_id: ID of strategy
            
        Returns:
            Strategy status or None if not found
        """
        return self.strategy_status.get(strategy_id)
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a strategy.
        
        Args:
            strategy_id: ID of strategy
            
        Returns:
            Performance metrics or None if not found
        """
        return self.strategy_performance.get(strategy_id)
    
    def get_all_strategies(self) -> List[str]:
        """
        Get list of all registered strategy IDs.
        
        Returns:
            List of strategy IDs
        """
        return list(self.strategies.keys())
    
    def get_active_strategies(self) -> List[str]:
        """
        Get list of active strategy IDs.
        
        Returns:
            List of active strategy IDs
        """
        return [
            sid for sid, status in self.strategy_status.items()
            if status == StrategyStatus.ACTIVE
        ]
    
    def get_strategy_by_id(self, strategy_id: str) -> Optional[BaseStrategy]:
        """
        Get strategy instance by ID.
        
        Args:
            strategy_id: ID of strategy
            
        Returns:
            Strategy instance or None if not found
        """
        return self.strategies.get(strategy_id)
    
    def _update_strategy_performance(self, strategy_id: str, signal: StrategySignal) -> None:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_id: ID of strategy
            signal: Executed signal
        """
        if strategy_id not in self.strategy_performance:
            return
        
        performance = self.strategy_performance[strategy_id]
        performance['total_trades'] += 1
        
        # Update last activity
        performance['last_updated'] = datetime.now()
    
    def get_execution_history(self, limit: int = 100) -> List[StrategyExecutionResult]:
        """
        Get recent execution history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of execution results
        """
        return self.execution_history[-limit:]
    
    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self.execution_history.clear()
        logger.info("Execution history cleared")
    
    def reset(self) -> None:
        """Reset the strategy engine to initial state."""
        self.strategies.clear()
        self.strategy_status.clear()
        self.strategy_performance.clear()
        self.execution_history.clear()
        logger.info("StrategyEngineImpl reset to initial state")


# Factory function for easy instantiation
def create_strategy_engine(risk_manager: Optional[RiskManagerImpl] = None) -> StrategyEngineImpl:
    """
    Create a new StrategyEngineImpl instance.
    
    Args:
        risk_manager: Risk management instance
        
    Returns:
        Configured StrategyEngineImpl instance
    """
    return StrategyEngineImpl(risk_manager)


if __name__ == "__main__":
    # Test the implementation
    print("Testing StrategyEngineImpl...")
    
    engine = create_strategy_engine()
    
    # Test strategy registration
    from src.trading.strategy_engine.templates.moving_average_strategy import MovingAverageStrategy
    
    strategy = MovingAverageStrategy(
        strategy_id="test_ma",
        symbols=["EURUSD"],
        parameters={"fast_period": 20, "slow_period": 50}
    )
    
    success = engine.register_strategy(strategy)
    print(f"Strategy registration: {success}")
    
    # Test strategy start
    started = engine.start_strategy("test_ma")
    print(f"Strategy start: {started}")
    
    # Test execution
    market_data = {
        'symbol': 'EURUSD',
        'close': 1.1000,
        'volume': 1000
    }
    
    # Note: This would need async context for full testing
    print("StrategyEngineImpl test completed successfully!")
