"""
Sensory Configuration
Configuration for sensory processing and technical analysis
"""

from dataclasses import dataclass
from typing import List

@dataclass
class SensoryConfig:
    """Configuration for sensory processing"""
    
    # Basic settings
    symbol: str = "EURUSD"
    timeframes: List[str] = None
    primary_timeframe: str = "M15"
    
    # Data settings
    max_buffer_size: int = 200
    min_data_points: int = 50
    
    # Indicator periods
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    momentum_period: int = 10
    
    # Database
    database_path: str = "sensory.db"
    
    # Calibration
    calibration_days: int = 30
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["M5", "M15", "H1", "H4"]
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.rsi_period <= 0:
            raise ValueError("rsi_period must be positive")
        if self.bb_period <= 0:
            raise ValueError("bb_period must be positive")
        if self.momentum_period <= 0:
            raise ValueError("momentum_period must be positive")
        return True
