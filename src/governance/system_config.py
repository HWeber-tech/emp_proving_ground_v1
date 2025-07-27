"""System configuration management for EMP Professional Predator."""

import os
from typing import Dict, Any, Optional


class SystemConfig:
    """Manages system configuration for the EMP Professional Predator."""
    
    def __init__(self, config_path: Optional[str] = None, CONNECTION_PROTOCOL: Optional[str] = None):
        """Initialize system configuration."""
        self.config_path = config_path
        
        # Handle both parameter and environment variable for protocol
        protocol_from_env = os.getenv("CONNECTION_PROTOCOL", "fix")
        self.connection_protocol = CONNECTION_PROTOCOL or protocol_from_env
        
        # Add uppercase property for backward compatibility
        self.CONNECTION_PROTOCOL = self.connection_protocol
        
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
