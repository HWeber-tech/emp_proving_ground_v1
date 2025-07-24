"""
System Configuration Module
Provides centralized configuration for all system components
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class SystemConfig:
    """System configuration settings"""
    
    # cTrader Configuration
    ctrader_demo_host: str = "demo.ctraderapi.com"
    ctrader_live_host: str = "live.ctraderapi.com"
    ctrader_port: int = 5035
    ctrader_account_id: int = 0
    ctrader_client_id: str = ""
    ctrader_client_secret: str = ""
    ctrader_access_token: str = ""
    ctrader_refresh_token: str = ""
    
    # FIX Protocol Configuration
    fix_price_sender_comp_id: str = ""
    fix_price_username: str = ""
    fix_price_password: str = ""
    fix_trade_sender_comp_id: str = ""
    fix_trade_username: str = ""
    fix_trade_password: str = ""
    
    # Master Switch Configuration
    connection_protocol: str = "fix"  # "fix" or "openapi"
    
    # CVD Configuration
    enable_numba_acceleration: bool = True
    cvd_history_length: int = 100
    cvd_divergence_threshold: float = 0.01
    
    # Default symbols for trading
    default_symbols: Optional[List[str]] = None
    
    # Database settings
    database_url: str = "sqlite:///emp.db"
    
    # Logging settings
    log_level: str = "INFO"
    
    # Trading settings
    max_risk_per_trade: Decimal = Decimal("0.02")
    max_total_exposure: Decimal = Decimal("0.10")
    max_drawdown: Decimal = Decimal("0.20")
    
    def __post_init__(self):
        if self.default_symbols is None:
            self.default_symbols = [
                "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
                "XAUUSD", "XAGUSD"
            ]
    
    def validate_credentials(self) -> bool:
        """Validate cTrader credentials are provided"""
        return bool(
            self.ctrader_client_id and
            self.ctrader_client_secret and
            self.ctrader_access_token
        )


# Global configuration instance
config = SystemConfig()

# Load from environment variables
config.ctrader_client_id = os.getenv("CTRADER_CLIENT_ID", "")
config.ctrader_client_secret = os.getenv("CTRADER_CLIENT_SECRET", "")
config.ctrader_access_token = os.getenv("CTRADER_ACCESS_TOKEN", "")
config.ctrader_account_id = int(os.getenv("CTRADER_ACCOUNT_ID", "0"))
config.database_url = os.getenv("DATABASE_URL", "sqlite:///emp.db")
config.log_level = os.getenv("LOG_LEVEL", "INFO")

# FIX Protocol credentials
config.fix_price_sender_comp_id = os.getenv("FIX_PRICE_SENDER_COMP_ID", "")
config.fix_price_username = os.getenv("FIX_PRICE_USERNAME", "")
config.fix_price_password = os.getenv("FIX_PRICE_PASSWORD", "")
config.fix_trade_sender_comp_id = os.getenv("FIX_TRADE_SENDER_COMP_ID", "")
config.fix_trade_username = os.getenv("FIX_TRADE_USERNAME", "")
config.fix_trade_password = os.getenv("FIX_TRADE_PASSWORD", "")

# Master switch configuration
config.connection_protocol = os.getenv("CONNECTION_PROTOCOL", "fix")

# CVD configuration
config.enable_numba_acceleration = os.getenv("ENABLE_NUMBA_ACCELERATION", "true").lower() == "true"
config.cvd_history_length = int(os.getenv("CVD_HISTORY_LENGTH", "100"))
config.cvd_divergence_threshold = float(os.getenv("CVD_DIVERGENCE_THRESHOLD", "0.01"))
