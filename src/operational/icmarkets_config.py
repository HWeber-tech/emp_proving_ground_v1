#!/usr/bin/env python3
"""
Final FIX Configuration for IC Markets
Combining account-specific host from screenshot with TargetSubID fix
"""

class ICMarketsConfig:
    """IC Markets FIX API configuration with verified working settings."""
    
    def __init__(self, environment="demo", account_number=None):
        self.environment = environment
        self.account_number = account_number
        
        # Account-specific endpoints from cTrader screenshot
        if environment == "demo":
            self.price_host = "demo-uk-eqx-01.p.c-trader.com"  # From screenshot
            self.trade_host = "demo-uk-eqx-01.p.c-trader.com"  # From screenshot
            self.price_port = 5211  # From screenshot (SSL port)
            self.trade_port = 5212  # From screenshot (SSL port)
        else:
            # Live environment would use different endpoints
            self.price_host = "live-uk-eqx-01.p.c-trader.com"  # Assumed pattern
            self.trade_host = "live-uk-eqx-01.p.c-trader.com"  # Assumed pattern
            self.price_port = 5211  # SSL port for live
            self.trade_port = 5212  # SSL port for live
            
        # FIX protocol settings
        self.fix_version = "FIX.4.4"
        self.heartbeat_interval = 30
        
        # Message identifiers from screenshot
        self.sender_comp_id = f"demo.icmarkets.{account_number}" if environment == "demo" else f"icmarkets.{account_number}"
        self.target_comp_id = "cServer"  # From screenshot
        
        # Session identifiers - based on broker error messages
        self.price_sender_sub_id = "QUOTE"  # From screenshot
        self.price_target_sub_id = "QUOTE"  # From screenshot - works for price session
        self.trade_sender_sub_id = "TRADE"  # From screenshot
        self.trade_target_sub_id = "TRADE"  # CORRECTED: broker expects TRADE for trade session
        
        # Authentication (to be set separately)
        self.username = account_number
        self.password = None  # FIX API password from screenshot
        
        # SSL settings - using SSL ports from screenshot
        self.use_ssl = True  # Using SSL ports from screenshot
        
    def set_fix_api_password(self, password):
        """Set the FIX API password from cTrader interface."""
        self.password = password
        
    def get_price_config(self):
        """Get configuration for price connection."""
        return {
            "host": self.price_host,
            "port": self.price_port,
            "sender_comp_id": self.sender_comp_id,
            "target_comp_id": self.target_comp_id,
            "sender_sub_id": self.price_sender_sub_id,
            "target_sub_id": self.price_target_sub_id,
            "username": self.username,
            "password": self.password,
            "fix_version": self.fix_version,
            "heartbeat_interval": self.heartbeat_interval,
            "use_ssl": self.use_ssl
        }
        
    def get_trade_config(self):
        """Get configuration for trade connection."""
        return {
            "host": self.trade_host,
            "port": self.trade_port,
            "sender_comp_id": self.sender_comp_id,
            "target_comp_id": self.target_comp_id,
            "sender_sub_id": self.trade_sender_sub_id,
            "target_sub_id": self.trade_target_sub_id,
            "username": self.username,
            "password": self.password,
            "fix_version": self.fix_version,
            "heartbeat_interval": self.heartbeat_interval,
            "use_ssl": self.use_ssl
        }
        
    def validate_config(self):
        """Validate configuration parameters."""
        if not self.account_number:
            raise ValueError("Account number is required")
        if not self.password:
            raise ValueError("FIX API password is required")
        if not self.username:
            raise ValueError("Username is required")
            
    def __str__(self):
        return f"ICMarketsConfig(environment={self.environment}, account={self.account_number}, host={self.price_host})"


# Symbol mapping based on SecurityList results
SYMBOL_ID_MAPPING = {
    # Numeric ID -> Symbol Name (from SecurityList response)
    1: "EURUSD",
    2: "GBPUSD", 
    3: "USDJPY",
    4: "USDCHF",
    5: "AUDUSD",
    6: "USDCAD",
    7: "NZDUSD",
    8: "EURGBP",
    9: "EURJPY",
    10: "GBPJPY",
    # Add more as discovered from SecurityList
}

def get_symbol_name(symbol_id):
    """Get symbol name for a numeric symbol ID."""
    return SYMBOL_ID_MAPPING.get(symbol_id, f"SYMBOL_{symbol_id}")

def get_symbol_id(symbol_name):
    """Get numeric symbol ID for a symbol name."""
    for sid, name in SYMBOL_ID_MAPPING.items():
        if name == symbol_name:
            return sid
    return None

