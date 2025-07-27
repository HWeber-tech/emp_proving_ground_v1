"""
IC Markets FIX Configuration
Production-ready configuration for IC Markets cTrader FIX API
"""

import os
from typing import Dict, Any

class ICMarketsConfig:
    """Configuration manager for IC Markets FIX connections."""
    
    def __init__(self, environment: str = "demo", account_number: str = None):
        self.environment = environment
        self.account_number = account_number or os.getenv("ICMARKETS_ACCOUNT", "9533708")
        self.password = os.getenv("ICMARKETS_PASSWORD", "WNSE5822")
        
    def get_price_session_config(self) -> Dict[str, Any]:
        """Get configuration for price data session."""
        return {
            # Connection settings
            'SocketConnectHost': self._get_host(),
            'SocketConnectPort': self._get_port('price'),
            'SocketUseSSL': 'Y',
            
            # Session identification
            'BeginString': 'FIX.4.4',
            'SenderCompID': f'icmarkets.{self.account_number}',
            'TargetCompID': 'cServer',
            'TargetSubID': 'QUOTE',
            
            # Authentication
            'Username': self.account_number,
            'Password': self.password,
            
            # Session settings
            'HeartBtInt': '30',
            'ReconnectInterval': '5',
            'ResetOnLogon': 'Y',
            'ResetOnLogout': 'Y',
            'ResetOnDisconnect': 'Y',
            
            # Message storage
            'PersistMessages': 'Y',
            'FileStorePath': f'./data/fix/{self.environment}/price',
            'FileLogPath': f'./logs/fix/{self.environment}/price',
            
            # Validation
            'UseDataDictionary': 'Y',
            'DataDictionary': './config/fix/FIX44.xml',
            'ValidateUserDefinedFields': 'N',
            'AllowUnknownMsgFields': 'Y',
        }
    
    def get_trade_session_config(self) -> Dict[str, Any]:
        """Get configuration for trading session."""
        return {
            # Connection settings
            'SocketConnectHost': self._get_host(),
            'SocketConnectPort': self._get_port('trade'),
            'SocketUseSSL': 'Y',
            
            # Session identification
            'BeginString': 'FIX.4.4',
            'SenderCompID': f'icmarkets.{self.account_number}',
            'TargetCompID': 'cServer',
            'TargetSubID': 'TRADE',
            
            # Authentication
            'Username': self.account_number,
            'Password': self.password,
            
            # Session settings
            'HeartBtInt': '30',
            'ReconnectInterval': '5',
            'ResetOnLogon': 'Y',
            'ResetOnLogout': 'Y',
            'ResetOnDisconnect': 'Y',
            
            # Message storage
            'PersistMessages': 'Y',
            'FileStorePath': f'./data/fix/{self.environment}/trade',
            'FileLogPath': f'./logs/fix/{self.environment}/trade',
            
            # Validation
            'UseDataDictionary': 'Y',
            'DataDictionary': './config/fix/FIX44.xml',
            'ValidateUserDefinedFields': 'N',
            'AllowUnknownMsgFields': 'Y',
        }
    
    def _get_host(self) -> str:
        """Get the appropriate host based on environment."""
        hosts = {
            'demo': 'demo-uk-eqx-01.p.c-trader.com',
            'live': 'h24.p.ctrader.com'
        }
        return hosts.get(self.environment, hosts['demo'])
    
    def _get_port(self, session_type: str) -> int:
        """Get the appropriate port based on session type."""
        ports = {
            'price': 5211,
            'trade': 5212
        }
        return ports.get(session_type, 5211)
    
    def validate_config(self) -> bool:
        """Validate that all required configuration is present."""
        if not self.account_number:
            raise ValueError("IC Markets account number is required")
        if not self.password:
            raise ValueError("IC Markets password is required")
        return True
    
    def create_config_files(self) -> tuple:
        """Create QuickFIX configuration files."""
        import configparser
        
        # Price session config
        price_config = configparser.ConfigParser()
        price_config['DEFAULT'] = self.get_price_session_config()
        
        # Trade session config
        trade_config = configparser.ConfigParser()
        trade_config['DEFAULT'] = self.get_trade_session_config()
        
        return price_config, trade_config


# Environment-specific configurations
DEMO_CONFIG = ICMarketsConfig(environment="demo")
LIVE_CONFIG = ICMarketsConfig(environment="live")
