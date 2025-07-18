"""
Real Data Integration Module - Phase 1 Implementation

This module provides real data integration capabilities for the EMP system.
It replaces mock data sources with actual market data APIs.

Phase 1 Data Sources:
- Yahoo Finance (yfinance) - Real-time and historical market data
- Alpha Vantage - Premium market data and technical indicators
- FRED API - Economic indicators and fundamental data
- NewsAPI - Sentiment analysis and news data

Author: EMP Development Team
Date: July 18, 2024
Phase: 1 - Real Data Foundation
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import aiohttp
import asyncio_throttle

from src.sensory.core.base import MarketData, InstrumentMeta, MarketRegime

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    source_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: Optional[int] = None
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation"""
    completeness: float  # 0-1, percentage of expected data points
    accuracy: float      # 0-1, data accuracy score
    latency: float       # seconds, data latency
    freshness: float     # 0-1, how recent the data is
    consistency: float   # 0-1, data consistency score


class YahooFinanceDataProvider:
    """Yahoo Finance data provider for real market data"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.session = None
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        
    async def get_market_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> Optional[MarketData]:
        """Get real-time market data from Yahoo Finance"""
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Get real-time data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return None
            
            # Get latest data point
            latest = data.iloc[-1]
            
            # Create MarketData object
            market_data = MarketData(
                timestamp=latest.name.to_pydatetime(),
                bid=latest['Low'],  # Use low as bid approximation
                ask=latest['High'], # Use high as ask approximation
                volume=latest['Volume'],
                volatility=self._calculate_volatility(data)
            )
            
            logger.info(f"Retrieved real market data for {symbol}: {market_data}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return None
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility from price data"""
        if len(data) < 2:
            return 0.0
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        # Calculate volatility (standard deviation of returns)
        volatility = returns.std()
        
        return float(volatility)
    
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No historical data received for {symbol}")
                return None
            
            logger.info(f"Retrieved {len(data)} historical data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None


class AlphaVantageDataProvider:
    """Alpha Vantage data provider for premium market data"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = config.base_url or "https://www.alphavantage.co/query"
        self.rate_limiter = asyncio_throttle.Throttler(rate_limit=5, period=60)  # 5 requests per minute
        
        if not self.api_key:
            logger.warning("Alpha Vantage API key not found. Premium features will be disabled.")
    
    async def get_real_time_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time data from Alpha Vantage"""
        if not self.api_key:
            logger.warning("Alpha Vantage API key not configured")
            return None
        
        try:
            async with self.rate_limiter:
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'Global Quote' in data:
                                quote = data['Global Quote']
                                
                                market_data = MarketData(
                                    timestamp=datetime.now(),
                                    bid=float(quote.get('05. price', 0)),
                                    ask=float(quote.get('05. price', 0)),
                                    volume=int(quote.get('06. volume', 0)),
                                    volatility=0.0  # Will be calculated separately
                                )
                                
                                logger.info(f"Retrieved Alpha Vantage data for {symbol}")
                                return market_data
                        
                        logger.error(f"Alpha Vantage API error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None
    
    async def get_technical_indicators(self, symbol: str, indicator: str = "RSI") -> Optional[Dict[str, Any]]:
        """Get technical indicators from Alpha Vantage"""
        if not self.api_key:
            return None
        
        try:
            async with self.rate_limiter:
                params = {
                    'function': indicator,
                    'symbol': symbol,
                    'interval': 'daily',
                    'time_period': 14,
                    'series_type': 'close',
                    'apikey': self.api_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Retrieved {indicator} data for {symbol}")
                            return data
                        
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching technical indicators for {symbol}: {e}")
            return None


class FREDDataProvider:
    """FRED API data provider for economic indicators"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv('FRED_API_KEY')
        self.base_url = config.base_url or "https://api.stlouisfed.org/fred/series/observations"
        
        if not self.api_key:
            logger.warning("FRED API key not found. Economic data will be disabled.")
    
    async def get_economic_indicator(self, series_id: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get economic indicator data from FRED"""
        if not self.api_key:
            logger.warning("FRED API key not configured")
            return None
        
        try:
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'observations' in data:
                            observations = data['observations']
                            
                            # Convert to DataFrame
                            df = pd.DataFrame(observations)
                            df['date'] = pd.to_datetime(df['date'])
                            df['value'] = pd.to_numeric(df['value'], errors='coerce')
                            
                            logger.info(f"Retrieved {len(df)} economic data points for {series_id}")
                            return df
                    
                    logger.error(f"FRED API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {e}")
            return None
    
    async def get_gdp_data(self) -> Optional[pd.DataFrame]:
        """Get GDP data"""
        return await self.get_economic_indicator('GDP')
    
    async def get_inflation_data(self) -> Optional[pd.DataFrame]:
        """Get inflation data (CPI)"""
        return await self.get_economic_indicator('CPIAUCSL')
    
    async def get_unemployment_data(self) -> Optional[pd.DataFrame]:
        """Get unemployment data"""
        return await self.get_economic_indicator('UNRATE')


class NewsAPIDataProvider:
    """NewsAPI data provider for sentiment analysis"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv('NEWS_API_KEY')
        self.base_url = config.base_url or "https://newsapi.org/v2/everything"
        
        if not self.api_key:
            logger.warning("NewsAPI key not found. Sentiment analysis will be disabled.")
    
    async def get_market_sentiment(self, query: str = "forex trading", days: int = 7) -> Optional[Dict[str, Any]]:
        """Get market sentiment from news articles"""
        if not self.api_key:
            logger.warning("NewsAPI key not configured")
            return None
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'apiKey': self.api_key,
                'language': 'en',
                'pageSize': 100
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'articles' in data:
                            articles = data['articles']
                            
                            # Calculate sentiment metrics
                            sentiment_score = self._calculate_sentiment(articles)
                            
                            result = {
                                'articles_count': len(articles),
                                'sentiment_score': sentiment_score,
                                'articles': articles[:10],  # Top 10 articles
                                'query': query,
                                'date_range': f"{from_date} to {datetime.now().strftime('%Y-%m-%d')}"
                            }
                            
                            logger.info(f"Retrieved sentiment data: {sentiment_score:.3f} for query '{query}'")
                            return result
                    
                    logger.error(f"NewsAPI error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching NewsAPI data: {e}")
            return None
    
    def _calculate_sentiment(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate sentiment score from articles"""
        if not articles:
            return 0.0
        
        # Simple sentiment calculation based on article titles
        # In production, this would use a proper NLP sentiment analysis model
        
        positive_words = ['bullish', 'surge', 'rally', 'gain', 'positive', 'up', 'higher']
        negative_words = ['bearish', 'drop', 'fall', 'decline', 'negative', 'down', 'lower']
        
        total_score = 0.0
        article_count = 0
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = f"{title} {description}"
            
            # Count positive and negative words
            positive_count = sum(1 for word in positive_words if word in content)
            negative_count = sum(1 for word in negative_words if word in content)
            
            # Calculate article sentiment (-1 to 1)
            if positive_count + negative_count > 0:
                article_sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                total_score += article_sentiment
                article_count += 1
        
        # Return average sentiment
        return total_score / article_count if article_count > 0 else 0.0


# Import additional data providers
try:
    from .alpha_vantage_integration import AlphaVantageProvider, AlphaVantageConfig
    from .fred_integration import FREDProvider, FREDConfig
    from .newsapi_integration import NewsAPIProvider, NewsAPIConfig
    ADVANCED_PROVIDERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced data providers not available: {e}")
    ADVANCED_PROVIDERS_AVAILABLE = False


class RealDataManager:
    """Enhanced real data manager with all data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.data_quality = {}
        self.fallback_enabled = config.get('fallback_to_mock', True)
        
        # Initialize data providers
        self._initialize_providers()
        
        logger.info("Enhanced Real Data Manager initialized")
    
    def _initialize_providers(self):
        """Initialize all data providers"""
        try:
            # Yahoo Finance (always available, no API key required)
            yahoo_config = DataSourceConfig(
                source_name="yahoo_finance",
                rate_limit=100,  # Yahoo Finance has generous limits
                timeout=30
            )
            self.providers['yahoo_finance'] = YahooFinanceDataProvider(yahoo_config)
            
            # Alpha Vantage (requires API key)
            if ADVANCED_PROVIDERS_AVAILABLE:
                alpha_config = AlphaVantageConfig(
                    api_key=os.getenv('ALPHA_VANTAGE_API_KEY', ''),
                    rate_limit=5,  # 5 requests per minute for free tier
                    timeout=30
                )
                if alpha_config.api_key:
                    self.providers['alpha_vantage'] = AlphaVantageProvider(alpha_config)
                    logger.info("Alpha Vantage provider initialized")
                else:
                    logger.warning("Alpha Vantage API key not found")
                
                # FRED API (requires API key)
                fred_config = FREDConfig(
                    api_key=os.getenv('FRED_API_KEY', ''),
                    rate_limit=120,  # 120 requests per minute
                    timeout=30
                )
                if fred_config.api_key:
                    self.providers['fred'] = FREDProvider(fred_config)
                    logger.info("FRED API provider initialized")
                else:
                    logger.warning("FRED API key not found")
                
                # NewsAPI (requires API key)
                news_config = NewsAPIConfig(
                    api_key=os.getenv('NEWS_API_KEY', ''),
                    rate_limit=100,  # 100 requests per day for free tier
                    timeout=30
                )
                if news_config.api_key:
                    self.providers['newsapi'] = NewsAPIProvider(news_config)
                    logger.info("NewsAPI provider initialized")
                else:
                    logger.warning("NewsAPI key not found")
            else:
                # Fallback to basic implementations
                alpha_config = DataSourceConfig(
                    source_name="alpha_vantage",
                    api_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
                    base_url="https://www.alphavantage.co/query",
                    rate_limit=5,
                    timeout=30
                )
                if alpha_config.api_key:
                    self.providers['alpha_vantage'] = AlphaVantageDataProvider(alpha_config)
                
                fred_config = DataSourceConfig(
                    source_name="fred",
                    api_key=os.getenv('FRED_API_KEY'),
                    base_url="https://api.stlouisfed.org/fred/series/observations",
                    rate_limit=120,
                    timeout=30
                )
                if fred_config.api_key:
                    self.providers['fred'] = FREDDataProvider(fred_config)
                
                news_config = DataSourceConfig(
                    source_name="newsapi",
                    api_key=os.getenv('NEWS_API_KEY'),
                    base_url="https://newsapi.org/v2/everything",
                    rate_limit=100,
                    timeout=30
                )
                if news_config.api_key:
                    self.providers['newsapi'] = NewsAPIDataProvider(news_config)
            
            logger.info(f"Initialized {len(self.providers)} data providers: {list(self.providers.keys())}")
            
        except Exception as e:
            logger.error(f"Error initializing data providers: {e}")
    
    async def get_market_data(self, symbol: str, source: str = "yahoo_finance") -> Optional[MarketData]:
        """Get market data from specified source with fallback"""
        try:
            if source in self.providers:
                provider = self.providers[source]
                
                if isinstance(provider, YahooFinanceDataProvider):
                    data = await provider.get_market_data(symbol)
                elif isinstance(provider, AlphaVantageDataProvider):
                    data = await provider.get_real_time_data(symbol)
                elif ADVANCED_PROVIDERS_AVAILABLE and isinstance(provider, AlphaVantageProvider):
                    data = await provider.get_real_time_quote(symbol)
                else:
                    logger.warning(f"Provider {source} does not support market data")
                    data = None
                
                if data:
                    # Update data quality metrics
                    self._update_data_quality(source, data)
                    return data
            
            # Fallback to Yahoo Finance if primary source fails
            if source != "yahoo_finance" and "yahoo_finance" in self.providers:
                logger.info(f"Falling back to Yahoo Finance for {symbol}")
                return await self.get_market_data(symbol, "yahoo_finance")
            
            # Fallback to mock data if enabled
            if self.fallback_enabled:
                logger.warning(f"All real data sources failed for {symbol}, using mock data")
                return self._generate_mock_data(symbol)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def get_technical_indicators(self, symbol: str, indicator: str = "RSI", source: str = "alpha_vantage") -> Optional[Dict[str, Any]]:
        """Get technical indicators from specified source"""
        if source in self.providers:
            provider = self.providers[source]
            
            if isinstance(provider, AlphaVantageDataProvider):
                return await provider.get_technical_indicators(symbol, indicator)
            elif ADVANCED_PROVIDERS_AVAILABLE and isinstance(provider, AlphaVantageProvider):
                return await provider.get_technical_indicator(symbol, indicator)
        
        logger.warning(f"Technical indicators not available from {source}")
        return None
    
    async def get_economic_data(self, indicator: str = "GDP", source: str = "fred") -> Optional[pd.DataFrame]:
        """Get economic data from specified source"""
        if source in self.providers:
            provider = self.providers[source]
            
            if isinstance(provider, FREDDataProvider):
                if indicator == "GDP":
                    return await provider.get_gdp_data()
                elif indicator == "INFLATION":
                    return await provider.get_inflation_data()
                elif indicator == "UNEMPLOYMENT":
                    return await provider.get_unemployment_data()
                else:
                    return await provider.get_economic_indicator(indicator)
            elif ADVANCED_PROVIDERS_AVAILABLE and isinstance(provider, FREDProvider):
                if indicator == "GDP":
                    return await provider.get_gdp_data()
                elif indicator == "INFLATION":
                    return await provider.get_inflation_data()
                elif indicator == "UNEMPLOYMENT":
                    return await provider.get_unemployment_data()
                elif indicator == "INTEREST_RATE":
                    return await provider.get_interest_rate_data()
                elif indicator == "CONSUMER_SENTIMENT":
                    return await provider.get_consumer_sentiment_data()
                elif indicator == "HOUSING":
                    return await provider.get_housing_data()
                else:
                    return await provider.get_series_observations(indicator)
        
        logger.warning(f"Economic data not available from {source}")
        return None
    
    async def get_sentiment_data(self, query: str = "forex trading", source: str = "newsapi") -> Optional[Dict[str, Any]]:
        """Get sentiment data from specified source"""
        if source in self.providers:
            provider = self.providers[source]
            
            if isinstance(provider, NewsAPIDataProvider):
                return await provider.get_market_sentiment(query)
            elif ADVANCED_PROVIDERS_AVAILABLE and isinstance(provider, NewsAPIProvider):
                return await provider.get_market_sentiment(query)
        
        logger.warning(f"Sentiment data not available from {source}")
        return None
    
    async def get_advanced_data(self, data_type: str, **kwargs) -> Optional[Any]:
        """Get advanced data from appropriate provider"""
        try:
            if data_type == "technical_indicators":
                symbol = kwargs.get('symbol', 'AAPL')
                indicator = kwargs.get('indicator', 'RSI')
                source = kwargs.get('source', 'alpha_vantage')
                return await self.get_technical_indicators(symbol, indicator, source)
            
            elif data_type == "economic_dashboard":
                source = kwargs.get('source', 'fred')
                if source in self.providers and ADVANCED_PROVIDERS_AVAILABLE:
                    provider = self.providers[source]
                    if isinstance(provider, FREDProvider):
                        return await provider.get_economic_dashboard()
                return None
            
            elif data_type == "sentiment_trends":
                queries = kwargs.get('queries', ['forex', 'stocks', 'crypto'])
                source = kwargs.get('source', 'newsapi')
                if source in self.providers and ADVANCED_PROVIDERS_AVAILABLE:
                    provider = self.providers[source]
                    if isinstance(provider, NewsAPIProvider):
                        return await provider.get_sentiment_trends(queries)
                return None
            
            elif data_type == "intraday_data":
                symbol = kwargs.get('symbol', 'AAPL')
                interval = kwargs.get('interval', '5min')
                source = kwargs.get('source', 'alpha_vantage')
                if source in self.providers and ADVANCED_PROVIDERS_AVAILABLE:
                    provider = self.providers[source]
                    if isinstance(provider, AlphaVantageProvider):
                        return await provider.get_intraday_data(symbol, interval)
                return None
            
            else:
                logger.warning(f"Unknown advanced data type: {data_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting advanced data {data_type}: {e}")
            return None
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {}
        
        for name, provider in self.providers.items():
            if hasattr(provider, 'get_api_status'):
                status[name] = provider.get_api_status()
            else:
                status[name] = {
                    'available': True,
                    'api_key_configured': hasattr(provider, 'api_key') and bool(provider.api_key)
                }
        
        return status
    
    def _update_data_quality(self, source: str, data: MarketData):
        """Update data quality metrics"""
        if source not in self.data_quality:
            self.data_quality[source] = DataQualityMetrics(
                completeness=1.0,
                accuracy=1.0,
                latency=0.0,
                freshness=1.0,
                consistency=1.0
            )
        
        # Update freshness (how recent the data is)
        time_diff = datetime.now() - data.timestamp
        freshness = max(0.0, 1.0 - (time_diff.total_seconds() / 3600))  # 1 hour max
        
        self.data_quality[source].freshness = freshness
    
    def _generate_mock_data(self, symbol: str) -> MarketData:
        """Generate mock data as fallback"""
        return MarketData(
            timestamp=datetime.now(),
            bid=1.0950 + np.random.normal(0, 0.001),
            ask=1.0952 + np.random.normal(0, 0.001),
            volume=1000 + np.random.exponential(500),
            volatility=0.01 + np.random.exponential(0.005)
        )
    
    def get_data_quality_report(self) -> Dict[str, DataQualityMetrics]:
        """Get data quality report for all sources"""
        return self.data_quality.copy()
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources"""
        return list(self.providers.keys())


# Example usage and testing
async def test_real_data_integration():
    """Test real data integration"""
    config = {
        'fallback_to_mock': True,
        'cache_duration': 300
    }
    
    data_manager = RealDataManager(config)
    
    # Test market data
    print("Testing market data...")
    market_data = await data_manager.get_market_data("EURUSD=X")
    if market_data:
        print(f"✅ Market data retrieved: {market_data}")
    else:
        print("❌ Market data failed")
    
    # Test economic data
    print("\nTesting economic data...")
    gdp_data = await data_manager.get_economic_data("GDP")
    if gdp_data is not None:
        print(f"✅ Economic data retrieved: {len(gdp_data)} records")
    else:
        print("❌ Economic data failed")
    
    # Test sentiment data
    print("\nTesting sentiment data...")
    sentiment_data = await data_manager.get_sentiment_data("forex trading")
    if sentiment_data:
        print(f"✅ Sentiment data retrieved: {sentiment_data['sentiment_score']:.3f}")
    else:
        print("❌ Sentiment data failed")
    
    # Print quality report
    print(f"\nData Quality Report:")
    for source, metrics in data_manager.get_data_quality_report().items():
        print(f"  {source}: Freshness={metrics.freshness:.3f}, Completeness={metrics.completeness:.3f}")


if __name__ == "__main__":
    asyncio.run(test_real_data_integration()) 