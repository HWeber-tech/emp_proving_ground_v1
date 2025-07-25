"""IC Markets configuration management."""

import os
from typing import Dict, Any


class ICMarketsConfig:
    """Configuration manager for IC Markets FIX API."""
    
    def __init__(self, environment: str = "demo", account_number: str = None):
        """Initialize IC Markets configuration."""
        self.environment = environment
        self.account_number = account_number or os.getenv("ICMARKETS_ACCOUNT", "9533708")
        self.password = os.getenv("ICMARKETS_PASSWORD", "WNSE5822")
        
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        if not self.account_number:
            raise ValueError("Account number is required")
        if not self.password:
            raise ValueError("Password is required")
        return True
        
    def _get_host(self) -> str:
        """Get the appropriate host based on environment."""
        return "demo-uk-eqx-01.p.c-trader.com"
        
    def _get_port(self, session_type: str) -> int:
        """Get the appropriate port based on session type."""
        if session_type == "price":
            return 5211
        elif session_type == "trade":
            return 5212
        else:
            raise ValueError(f"Unknown session type: {session_type}")
            
    def get_price_session_config(self) -> Dict[str, Any]:
        """Get price session configuration."""
        return {
            "host": self._get_host(),
            "port": self._get_port("price"),
            "account": self.account_number,
            "password": self.password,
            "sender_comp_id": f"demo.icmarkets.{self.account_number}",
            "target_comp_id": "cServer",
            "target_sub_id": "QUOTE"
        }
        
    def get_trade_session_config(self) -> Dict[str, Any]:
        """Get trade session configuration."""
        return {
            "host": self._get_host(),
            "port": self._get_port("trade"),
            "account": self.account_number,
            "password": self.password,
            "sender_comp_id": f"demo.icmarkets.{self.account_number}",
            "target_comp_id": "cServer",
            "target_sub_id": "TRADE"
        }
