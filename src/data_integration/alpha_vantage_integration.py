"""
Alpha Vantage Integration Module - Phase 1.5 Implementation

This module provides comprehensive Alpha Vantage API integration for premium market data.
It includes real-time quotes, technical indicators, and fundamental data.

Author: EMP Development Team
Date: July 18, 2024
Phase: 1.5 - Advanced Data Sources
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import aiohttp
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.sensory.core.base import MarketData
from .data_validation import MarketDataValidator, ValidationLevel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AlphaVantageConfig:
    """Configuration for Alpha Vantage API"""
    api_key: str
    base_url: str = "https://www.alphavantage.co/query"
    rate_limit: int = 5  # requests per minute (free tier)
    timeout: int = 30
    retry_attempts: int = 3


class AlphaVantageProvider:
    """Alpha Vantage data provider for premium market data"""
    
    def __init__(self, config: Optional[AlphaVantageConfig] = None):
        self.config = config or AlphaVantageConfig(
            api_key=os.getenv('ALPHA_VANTAGE_API_KEY', '')
        )
        self.rate_limiter = asyncio.Semaphore(self.config.rate_limit)
        self.validator = MarketDataValidator()
        self.last_request_time = 0
        
        if not self.config.api_key:
            logger.warning("Alpha Vantage API key not found. Premium features will be disabled.")
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = datetime.now().timestamp()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < 60 / self.config.rate_limit:
            wait_time = (60 / self.config.rate_limit) - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = datetime.now().timestamp()
    
    async def _make_request(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting and error handling"""
        if not self.config.api_key:
            logger.warning("Alpha Vantage API key not configured")
            return None
        
        try:
            await self._rate_limit()
            
            params['apikey'] = self.config.api_key
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.base_url, params=params, timeout=self.config.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API errors
                        if 'Error Message' in data:
                            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                            return None
                        
                        if 'Note' in data:
                            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                            return None
                        
                        return data
                    else:
                        logger.error(f"Alpha Vantage HTTP error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error making Alpha Vantage request: {e}")
            return None
    
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote from Alpha Vantage"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        
        data = await self._make_request(params)
        
        if data and 'Global Quote' in data:
            quote = data['Global Quote']
            
            try:
                market_data = MarketData(
                    timestamp=datetime.now(),
                    bid=float(quote.get('05. price', 0)),
                    ask=float(quote.get('05. price', 0)),  # Alpha Vantage doesn't provide separate bid/ask
                    volume=int(quote.get('06. volume', 0)),
                    volatility=0.0  # Will be calculated separately
                )
                
                # Validate the data
                validation_result = self.validator.validate_market_data(market_data, ValidationLevel.STRICT)
                
                if validation_result.is_valid:
                    logger.info(f"Retrieved Alpha Vantage data for {symbol}: {market_data}")
                    return market_data
                else:
                    logger.warning(f"Alpha Vantage data validation failed for {symbol}: {validation_result.issues}")
                    return None
                    
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing Alpha Vantage data for {symbol}: {e}")
                return None
        
        return None
    
    async def get_intraday_data(self, symbol: str, interval: str = "5min") -> Optional[pd.DataFrame]:
        """Get intraday data from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': 'compact'
        }
        
        data = await self._make_request(params)
        
        if data and f'Time Series ({interval})' in data:
            time_series = data[f'Time Series ({interval})']
            
            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp')
            
            logger.info(f"Retrieved {len(df)} intraday data points for {symbol}")
            return df
        
        return None
    
    async def get_technical_indicator(self, symbol: str, indicator: str = "GENERIC", 
                                     interval: str = "daily", time_period: int = 14) -> Optional[Dict[str, Any]]:
        """Get provider-computed indicator (treated as external feature)."""
        params = {
            'function': indicator,
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': 'close'
        }
        
        data = await self._make_request(params)
        
        if data and f'Technical Analysis: {indicator}' in data:
            analysis = data[f'Technical Analysis: {indicator}']
            
            # Convert to more usable format
            result = {
                'indicator': indicator,
                'symbol': symbol,
                'interval': interval,
                'time_period': time_period,
                'data': {}
            }
            
            for timestamp, values in analysis.items():
                result['data'][timestamp] = float(values[f'{indicator}'])
            
            logger.info(f"Retrieved external indicator data for {symbol}")
            return result
        
        return None
    
    async def get_fundamental_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data from Alpha Vantage"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        
        data = await self._make_request(params)
        
        if data and 'Symbol' in data:
            logger.info(f"Retrieved fundamental data for {symbol}")
            return data
        
        return None
    
    async def get_earnings_calendar(self, horizon: str = "3month") -> Optional[pd.DataFrame]:
        """Get earnings calendar from Alpha Vantage"""
        params = {
            'function': 'EARNINGS_CALENDAR',
            'horizon': horizon
        }
        
        data = await self._make_request(params)
        
        if data:
            # Earnings calendar returns CSV data
            try:
                df = pd.read_csv(pd.StringIO(data))
                logger.info(f"Retrieved earnings calendar with {len(df)} entries")
                return df
            except Exception as e:
                logger.error(f"Error parsing earnings calendar: {e}")
                return None
        
        return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status and configuration"""
        return {
            'api_key_configured': bool(self.config.api_key),
            'rate_limit': self.config.rate_limit,
            'base_url': self.config.base_url,
            'timeout': self.config.timeout
        }


# Example usage and testing
async def test_alpha_vantage_integration():
    """Test Alpha Vantage integration"""
    provider = AlphaVantageProvider()
    
    # Test API status
    status = provider.get_api_status()
    print(f"Alpha Vantage API Status: {status}")
    
    if not status['api_key_configured']:
        print("⚠️ Alpha Vantage API key not configured. Tests will be skipped.")
        return
    
    # Test real-time quote
    print("\nTesting real-time quote...")
    quote = await provider.get_real_time_quote("AAPL")
    if quote:
        print(f"✅ Real-time quote: {quote}")
    else:
        print("❌ Real-time quote failed")
    
    # Test technical indicator
    print("\nTesting technical indicator...")
    indicator_data = await provider.get_technical_indicator("AAPL", "GENERIC")
    if indicator_data:
        print(f"✅ External indicator data retrieved: {len(indicator_data['data'])} data points")
    else:
        print("❌ External indicator retrieval failed")
    
    # Test fundamental data
    print("\nTesting fundamental data...")
    fundamental = await provider.get_fundamental_data("AAPL")
    if fundamental:
        print(f"✅ Fundamental data: {fundamental.get('Company Name', 'N/A')}")
    else:
        print("❌ Fundamental data failed")


if __name__ == "__main__":
    asyncio.run(test_alpha_vantage_integration()) 
