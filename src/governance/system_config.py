"""
System Configuration Module
Provides centralized configuration for all system components
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from decimal import Decimal
from typing import Literal


class SystemConfig(BaseModel):
    """System configuration settings"""
    
    # cTrader Configuration
    ctrader_demo_host: str = Field(default="demo.ctraderapi.com", description="cTrader demo host")
    ctrader_live_host: str = Field(default="live.ctraderapi.com", description="cTrader live host")
    ctrader_port: int = Field(default=5035, description="cTrader port")
    ctrader_account_id: int = Field(default=0, description="cTrader account ID")
    ctrader_client_id: str = Field(default="", description="cTrader client ID")
    ctrader_client_secret: str = Field(default="", description="cTrader client secret")
    ctrader_access_token: str = Field(default="", description="cTrader access token")
    ctrader_refresh_token: str = Field(default="", description="cTrader refresh token")
    
    # FIX Protocol Configuration
    fix_price_sender_comp_id: str = Field(default="", description="FIX price sender comp ID")
    fix_price_username: str = Field(default="", description="FIX price username")
    fix_price_password: str = Field(default="", description="FIX price password")
    fix_trade_sender_comp_id: str = Field(default="", description="FIX trade sender comp ID")
    fix_trade_username: str = Field(default="", description="FIX trade username")
    fix_trade_password: str = Field(default="", description="FIX trade password")
    
    # Master Switch - The Professional Upgrade
    CONNECTION_PROTOCOL: Literal["fix", "openapi"] = Field(
        default="fix",
        description="The communication protocol to use for live trading and data"
    )
    
    # Default symbols for trading
    default_symbols: Optional[List[str]] = Field(
        default_factory=lambda: [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "XAUUSD", "XAGUSD"
        ],
        description="Default trading symbols"
    )
    
    # Database settings
    database_url: str = Field(default="sqlite:///emp.db", description="Database URL")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Trading settings
    max_risk_per_trade: Decimal = Field(default=Decimal("0.02"), description="Max risk per trade")
    max_total_exposure: Decimal = Field(default=Decimal("0.10"), description="Max total exposure")
    max_drawdown: Decimal = Field(default=Decimal("0.20"), description="Max drawdown")
    
    # FIX Symbol Mapping (symbol name -> FIX symbolId)
    fix_symbol_map: Dict[str, int] = Field(
        default_factory=lambda: {
            "EURUSD": 1,
            "GBPUSD": 2,
            "USDJPY": 3,
            "AUDUSD": 4,
            "USDCAD": 5,
            "XAUUSD": 6,
            "XAGUSD": 7
        },
        description="FIX symbol mapping"
    )
    
    def validate_credentials(self) -> bool:
        """Validate cTrader credentials are provided"""
        return bool(
            self.ctrader_client_id and
            self.ctrader_client_secret and
            self.ctrader_access_token
        )

    class Config:
        env_prefix = ""
        case_sensitive = False


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
