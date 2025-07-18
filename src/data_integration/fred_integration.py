"""
FRED API Integration Module - Phase 1.5 Implementation

This module provides comprehensive FRED (Federal Reserve Economic Data) API integration.
It includes economic indicators, GDP data, inflation metrics, and unemployment statistics.

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

from .data_validation import MarketDataValidator, ValidationLevel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class FREDConfig:
    """Configuration for FRED API"""
    api_key: str
    base_url: str = "https://api.stlouisfed.org/fred"
    rate_limit: int = 120  # requests per minute
    timeout: int = 30
    retry_attempts: int = 3


class FREDProvider:
    """FRED API data provider for economic indicators"""
    
    def __init__(self, config: Optional[FREDConfig] = None):
        self.config = config or FREDConfig(
            api_key=os.getenv('FRED_API_KEY', '')
        )
        self.rate_limiter = asyncio.Semaphore(self.config.rate_limit)
        self.validator = MarketDataValidator()
        self.last_request_time = 0
        
        if not self.config.api_key:
            logger.warning("FRED API key not found. Economic data will be disabled.")
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = datetime.now().timestamp()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < 60 / self.config.rate_limit:
            wait_time = (60 / self.config.rate_limit) - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = datetime.now().timestamp()
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting and error handling"""
        if not self.config.api_key:
            logger.warning("FRED API key not configured")
            return None
        
        try:
            await self._rate_limit()
            
            params['api_key'] = self.config.api_key
            params['file_type'] = 'json'
            
            url = f"{self.config.base_url}/{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=self.config.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API errors
                        if 'error_code' in data:
                            logger.error(f"FRED API error: {data.get('error_message', 'Unknown error')}")
                            return None
                        
                        return data
                    else:
                        logger.error(f"FRED HTTP error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error making FRED request: {e}")
            return None
    
    async def get_series_observations(self, series_id: str, limit: int = 100, 
                                    sort_order: str = 'desc') -> Optional[pd.DataFrame]:
        """Get series observations from FRED"""
        params = {
            'series_id': series_id,
            'limit': limit,
            'sort_order': sort_order
        }
        
        data = await self._make_request('series/observations', params)
        
        if data and 'observations' in data:
            observations = data['observations']
            
            # Convert to DataFrame
            df_data = []
            for obs in observations:
                try:
                    df_data.append({
                        'date': pd.to_datetime(obs['date']),
                        'value': pd.to_numeric(obs['value'], errors='coerce'),
                        'realtime_start': obs.get('realtime_start'),
                        'realtime_end': obs.get('realtime_end')
                    })
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing observation: {e}")
                    continue
            
            if df_data:
                df = pd.DataFrame(df_data)
                df = df.sort_values('date')
                df = df.dropna(subset=['value'])
                
                logger.info(f"Retrieved {len(df)} observations for series {series_id}")
                return df
        
        return None
    
    async def get_gdp_data(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get GDP data (Gross Domestic Product)"""
        return await self.get_series_observations('GDP', limit)
    
    async def get_inflation_data(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get inflation data (Consumer Price Index)"""
        return await self.get_series_observations('CPIAUCSL', limit)
    
    async def get_unemployment_data(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get unemployment data (Unemployment Rate)"""
        return await self.get_series_observations('UNRATE', limit)
    
    async def get_interest_rate_data(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get federal funds rate data"""
        return await self.get_series_observations('FEDFUNDS', limit)
    
    async def get_consumer_sentiment_data(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get consumer sentiment data"""
        return await self.get_series_observations('UMCSENT', limit)
    
    async def get_housing_data(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get housing starts data"""
        return await self.get_series_observations('HOUST', limit)
    
    async def get_series_info(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Get series information and metadata"""
        params = {
            'series_id': series_id
        }
        
        data = await self._make_request('series', params)
        
        if data and 'seriess' in data and data['seriess']:
            series_info = data['seriess'][0]
            logger.info(f"Retrieved series info for {series_id}")
            return series_info
        
        return None
    
    async def search_series(self, search_text: str, limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """Search for series by text"""
        params = {
            'search_text': search_text,
            'limit': limit
        }
        
        data = await self._make_request('series/search', params)
        
        if data and 'seriess' in data:
            series_list = data['seriess']
            logger.info(f"Found {len(series_list)} series matching '{search_text}'")
            return series_list
        
        return None
    
    async def get_category_series(self, category_id: int, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """Get series in a category"""
        params = {
            'category_id': category_id,
            'limit': limit
        }
        
        data = await self._make_request('category/series', params)
        
        if data and 'seriess' in data:
            series_list = data['seriess']
            logger.info(f"Retrieved {len(series_list)} series from category {category_id}")
            return series_list
        
        return None
    
    async def get_economic_calendar(self, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Get economic calendar data"""
        params = {}
        
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        data = await self._make_request('releases', params)
        
        if data and 'releases' in data:
            releases = data['releases']
            
            # Convert to DataFrame
            df_data = []
            for release in releases:
                try:
                    df_data.append({
                        'release_id': release['id'],
                        'name': release['name'],
                        'press_release': release.get('press_release', False),
                        'link': release.get('link', ''),
                        'notes': release.get('notes', '')
                    })
                except KeyError as e:
                    logger.warning(f"Error parsing release: {e}")
                    continue
            
            if df_data:
                df = pd.DataFrame(df_data)
                logger.info(f"Retrieved {len(df)} economic releases")
                return df
        
        return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status and configuration"""
        return {
            'api_key_configured': bool(self.config.api_key),
            'rate_limit': self.config.rate_limit,
            'base_url': self.config.base_url,
            'timeout': self.config.timeout
        }
    
    async def get_economic_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive economic dashboard data"""
        dashboard = {
            'gdp': None,
            'inflation': None,
            'unemployment': None,
            'interest_rate': None,
            'consumer_sentiment': None,
            'housing': None,
            'timestamp': datetime.now()
        }
        
        # Get all economic indicators
        tasks = [
            self.get_gdp_data(10),
            self.get_inflation_data(10),
            self.get_unemployment_data(10),
            self.get_interest_rate_data(10),
            self.get_consumer_sentiment_data(10),
            self.get_housing_data(10)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        dashboard['gdp'] = results[0] if not isinstance(results[0], Exception) else None
        dashboard['inflation'] = results[1] if not isinstance(results[1], Exception) else None
        dashboard['unemployment'] = results[2] if not isinstance(results[2], Exception) else None
        dashboard['interest_rate'] = results[3] if not isinstance(results[3], Exception) else None
        dashboard['consumer_sentiment'] = results[4] if not isinstance(results[4], Exception) else None
        dashboard['housing'] = results[5] if not isinstance(results[5], Exception) else None
        
        logger.info("Economic dashboard data retrieved")
        return dashboard


# Example usage and testing
async def test_fred_integration():
    """Test FRED API integration"""
    provider = FREDProvider()
    
    # Test API status
    status = provider.get_api_status()
    print(f"FRED API Status: {status}")
    
    if not status['api_key_configured']:
        print("⚠️ FRED API key not configured. Tests will be skipped.")
        return
    
    # Test GDP data
    print("\nTesting GDP data...")
    gdp_data = await provider.get_gdp_data(5)
    if gdp_data is not None:
        print(f"✅ GDP data: {len(gdp_data)} records")
        print(f"   Latest GDP: {gdp_data.iloc[-1]['value']:.2f} ({gdp_data.iloc[-1]['date'].strftime('%Y-%m-%d')})")
    else:
        print("❌ GDP data failed")
    
    # Test inflation data
    print("\nTesting inflation data...")
    inflation_data = await provider.get_inflation_data(5)
    if inflation_data is not None:
        print(f"✅ Inflation data: {len(inflation_data)} records")
        print(f"   Latest CPI: {inflation_data.iloc[-1]['value']:.2f} ({inflation_data.iloc[-1]['date'].strftime('%Y-%m-%d')})")
    else:
        print("❌ Inflation data failed")
    
    # Test unemployment data
    print("\nTesting unemployment data...")
    unemployment_data = await provider.get_unemployment_data(5)
    if unemployment_data is not None:
        print(f"✅ Unemployment data: {len(unemployment_data)} records")
        print(f"   Latest rate: {unemployment_data.iloc[-1]['value']:.2f}% ({unemployment_data.iloc[-1]['date'].strftime('%Y-%m-%d')})")
    else:
        print("❌ Unemployment data failed")
    
    # Test economic dashboard
    print("\nTesting economic dashboard...")
    dashboard = await provider.get_economic_dashboard()
    if dashboard:
        print(f"✅ Economic dashboard: {sum(1 for v in dashboard.values() if v is not None)} indicators loaded")
    else:
        print("❌ Economic dashboard failed")


if __name__ == "__main__":
    asyncio.run(test_fred_integration()) 