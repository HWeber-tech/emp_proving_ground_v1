"""
Risk Configuration
Configuration for risk management parameters
"""

from dataclasses import dataclass

@dataclass
class RiskConfig:
    """Configuration for risk management"""
    
    # Database
    database_path: str = "risk_management.db"
    
    # Risk parameters
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    max_portfolio_heat: float = 0.30  # 30% max portfolio heat
    max_position_size: float = 0.10  # 10% max position size
    min_position_size: float = 0.001  # Minimum position size
    
    # Kelly Criterion parameters
    kelly_fraction_cap: float = 0.25  # Cap Kelly at 25%
    win_rate_estimate: float = 0.55  # Default win rate
    win_loss_ratio: float = 1.5  # Default win/loss ratio
    
    # Risk assessment
    risk_levels: dict = None
    
    def __post_init__(self):
        if self.risk_levels is None:
            self.risk_levels = {
                'low': {'threshold': 0.2, 'action': 'increase_positions'},
                'medium': {'threshold': 0.4, 'action': 'hold'},
                'high': {'threshold': 0.6, 'action': 'reduce_positions'},
                'extreme': {'threshold': 0.8, 'action': 'close_positions'}
            }
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.max_risk_per_trade <= 0 or self.max_risk_per_trade > 1:
            raise ValueError("max_risk_per_trade must be between 0 and 1")
        if self.max_portfolio_heat <= 0 or self.max_portfolio_heat > 1:
            raise ValueError("max_portfolio_heat must be between 0 and 1")
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("max_position_size must be between 0 and 1")
        return True
