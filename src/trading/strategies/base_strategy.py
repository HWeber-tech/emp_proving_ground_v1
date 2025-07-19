"""
Base Strategy v1.0 - Strategy Interface Definition

Placeholder for future implementation of base strategy classes and interfaces.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all strategies must implement
    to be compatible with the EMP trading system.
    """
    
    def __init__(self, strategy_id: str):
        """Initialize the base strategy."""
        self.strategy_id = strategy_id
        logger.info(f"BaseStrategy initialized for {strategy_id}")
    
    @abstractmethod
    def generate_signal(self, market_data):
        """Generate trading signal based on market data."""
        pass
    
    @abstractmethod
    def get_parameters(self):
        """Get current strategy parameters."""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: dict):
        """Update strategy parameters."""
        pass
