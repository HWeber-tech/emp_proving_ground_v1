"""System configuration management for EMP Professional Predator."""

import os
from typing import Dict, Any, Optional


class SystemConfig:
    """Manages system configuration for the EMP Professional Predator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize system configuration."""
        self.config_path = config_path
        self.connection_protocol = "fix"  # Default to FIX protocol
        self.environment = "demo"
        self.account_number = os.getenv("ICMARKETS_ACCOUNT", "9533708")
        self.password = os.getenv("ICMARKETS_PASSWORD", "WNSE5822")
        
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return {
            "connection_protocol": self.connection_protocol,
            "environment": self.environment,
            "account_number": self.account_number,
            "password": self.password,
        }
        
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        if not self.account_number:
            raise ValueError("Account number is required")
        if not self.password:
            raise ValueError("Password is required")
        return True
        
    def get_icmarkets_config(self) -> Dict[str, Any]:
        """Get IC Markets specific configuration."""
        return {
            "account_number": self.account_number,
            "password": self.password,
            "environment": self.environment,
            "host": "demo-uk-eqx-01.p.c-trader.com",
            "price_port": 5211,
            "trade_port": 5212,
        }
