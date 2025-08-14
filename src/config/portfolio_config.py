"""
Portfolio Configuration
Configuration for portfolio monitoring
"""

from dataclasses import dataclass


@dataclass
class PortfolioConfig:
    """Configuration for portfolio monitoring"""
    
    # Database
    database_path: str = "portfolio.db"
    
    # Initial settings
    initial_balance: float = 10000.0
    
    # Performance tracking
    save_snapshots: bool = True
    snapshot_interval_minutes: int = 15
    
    # Risk limits
    max_positions: int = 10
    max_position_size_pct: float = 0.10  # 10% max per position
    max_total_exposure: float = 0.80  # 80% max total exposure
    
    # Reporting
    performance_report_days: int = 30
    detailed_logging: bool = True
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if self.max_position_size_pct <= 0 or self.max_position_size_pct > 1:
            raise ValueError("max_position_size_pct must be between 0 and 1")
        return True
