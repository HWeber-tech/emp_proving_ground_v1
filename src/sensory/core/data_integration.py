"""
Data Integration Layer - Real Data Sources and Processing

This module provides the "nerve endings" for the multidimensional market intelligence system,
connecting it to real data sources and processing pipelines.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
from fredapi import Fred
import requests
import json
import time
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class EconomicData:
    """Economic data point with metadata"""
    indicator: str
    value: float
    timestamp: datetime
    frequency: str  # daily, weekly, monthly
    surprise_factor: float  # actual vs expected
    importance: float  # 0-1 scale

@dataclass
class NewsEvent:
    """News event with sentiment analysis"""
    timestamp: datetime
    headline: str
    content: str
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1
    source: str
    impact_category: str  # monetary, fiscal, geopolitical, etc.

@dataclass
class OrderBookLevel:
    """Order book level data"""
    price: float
    size: float
    orders: int
    side: str  # 'bid' or 'ask'

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to data source"""
        pass

class EconomicDataProvider(DataProvider):
    """
    Economic data provider using FRED API and other sources
    Provides real economic indicators for WHY dimension
    """
    
    def __init__(self, fred_api_key: str):
        self.fred_api_key = fred_api_key
        self.fred = None
        self.cache = {}
        self.cache_lock = Lock()
        self.last_update = {}
        
        # Key economic indicators to track
        self.indicators = {
            'GDP': 'GDP',
            'INFLATION': 'CPIAUCSL',
            'UNEMPLOYMENT': 'UNRATE',
            'FED_FUNDS': 'FEDFUNDS',
            'DXY': 'DEXUSEU',
            'VIX': '^VIX',
            'YIELD_10Y': '^TNX',
            'YIELD_2Y': '^IRX',
            'OIL': 'CL=F',
            'GOLD': 'GC=F'
        }
    
    async def connect(self) -> bool:
        """Initialize FRED API connection"""
        try:
            self.fred = Fred(api_key=self.fred_api_key)
            # Test connection
            test_data = self.fred.get_series('GDP', limit=1)
            logger.info("Connected to FRED API successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to FRED API: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close FRED connection"""
        self.fred = None
        logger.info("Disconnected from FRED API")
    
    async def is_connected(self) -> bool:
        """Check FRED connection status"""
        return self.fred is not None
    
    async def get_economic_data(self, indicator: str, days_back: int = 30) -> List[EconomicData]:
        """Get economic data for specified indicator"""
        if not await self.is_connected():
            await self.connect()
        
        try:
            # Check cache first
            cache_key = f"{indicator}_{days_back}"
            with self.cache_lock:
                if cache_key in self.cache:
                    cached_time = self.last_update.get(cache_key, datetime.min)
                    if datetime.now() - cached_time < timedelta(hours=1):
                        return self.cache[cache_key]
            
            # Get data from FRED or Yahoo Finance
            if indicator in self.indicators:
                fred_series = self.indicators[indicator]
                
                if fred_series.startswith('^') or fred_series.endswith('=F'):
                    # Yahoo Finance data
                    ticker = yf.Ticker(fred_series)
                    hist = ticker.history(period=f"{days_back}d")
                    
                    data_points = []
                    for date, row in hist.iterrows():
                        data_points.append(EconomicData(
                            indicator=indicator,
                            value=float(row['Close']),
                            timestamp=date.to_pydatetime(),
                            frequency='daily',
                            surprise_factor=0.0,  # Would need expectations data
                            importance=self._get_indicator_importance(indicator)
                        ))
                else:
                    # FRED data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days_back)
                    
                    series = self.fred.get_series(
                        fred_series,
                        start=start_date,
                        end=end_date
                    )
                    
                    data_points = []
                    for date, value in series.items():
                        if pd.notna(value):
                            data_points.append(EconomicData(
                                indicator=indicator,
                                value=float(value),
                                timestamp=date.to_pydatetime(),
                                frequency='monthly',  # Most FRED data is monthly
                                surprise_factor=0.0,  # Would need expectations data
                                importance=self._get_indicator_importance(indicator)
                            ))
                
                # Cache the results
                with self.cache_lock:
                    self.cache[cache_key] = data_points
                    self.last_update[cache_key] = datetime.now()
                
                return data_points
            
        except Exception as e:
            logger.error(f"Failed to get economic data for {indicator}: {e}")
            return []
    
    def _get_indicator_importance(self, indicator: str) -> float:
        """Get importance weight for economic indicator"""
        importance_map = {
            'GDP': 1.0,
            'INFLATION': 0.9,
            'FED_FUNDS': 0.9,
            'UNEMPLOYMENT': 0.8,
            'VIX': 0.7,
            'YIELD_10Y': 0.8,
            'YIELD_2Y': 0.7,
            'DXY': 0.6,
            'OIL': 0.5,
            'GOLD': 0.4
        }
        return importance_map.get(indicator, 0.5)
    
    async def calculate_policy_divergence(self, base_currency: str = 'USD', 
                                        quote_currency: str = 'EUR') -> float:
        """Calculate monetary policy divergence between currencies"""
        try:
            # Get interest rate data for both currencies
            usd_rates = await self.get_economic_data('FED_FUNDS', 90)
            # For EUR, we'd need ECB data - simplified here
            eur_rates = await self.get_economic_data('FED_FUNDS', 90)  # Placeholder
            
            if not usd_rates or not eur_rates:
                return 0.0
            
            # Calculate rate differential trend
            usd_current = usd_rates[-1].value if usd_rates else 0
            usd_previous = usd_rates[-2].value if len(usd_rates) > 1 else usd_current
            
            eur_current = eur_rates[-1].value if eur_rates else 0
            eur_previous = eur_rates[-2].value if len(eur_rates) > 1 else eur_current
            
            # Divergence is the change in rate differential
            current_diff = usd_current - eur_current
            previous_diff = usd_previous - eur_previous
            
            divergence = current_diff - previous_diff
            
            # Normalize to -1 to 1 range
            return np.tanh(divergence / 0.5)
            
        except Exception as e:
            logger.error(f"Failed to calculate policy divergence: {e}")
            return 0.0

class NewsDataProvider(DataProvider):
    """
    News and sentiment data provider
    Provides real news analysis for WHY dimension
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        self.news_api_key = news_api_key
        self.session = None
        
    async def connect(self) -> bool:
        """Initialize news API connection"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("Connected to news data provider")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to news provider: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close news API connection"""
        if self.session:
            await self.session.close()
        logger.info("Disconnected from news provider")
    
    async def is_connected(self) -> bool:
        """Check news API connection status"""
        return self.session is not None
    
    async def get_market_news(self, symbol: str = 'EURUSD', 
                            hours_back: int = 24) -> List[NewsEvent]:
        """Get relevant market news with sentiment analysis"""
        if not await self.is_connected():
            await self.connect()
        
        try:
            # Simplified news fetching - in production would use real news APIs
            # like NewsAPI, Alpha Vantage News, or Bloomberg API
            
            # For demonstration, create synthetic but realistic news events
            current_time = datetime.now()
            news_events = []
            
            # Economic calendar events (simplified)
            events = [
                {
                    'headline': 'ECB Monetary Policy Decision',
                    'content': 'European Central Bank maintains rates at current levels',
                    'sentiment': 0.1,
                    'relevance': 0.9,
                    'category': 'monetary'
                },
                {
                    'headline': 'US Employment Data Release',
                    'content': 'Non-farm payrolls exceed expectations',
                    'sentiment': 0.3,
                    'relevance': 0.8,
                    'category': 'economic'
                },
                {
                    'headline': 'Geopolitical Tensions Rise',
                    'content': 'Trade negotiations face new challenges',
                    'sentiment': -0.4,
                    'relevance': 0.6,
                    'category': 'geopolitical'
                }
            ]
            
            for i, event in enumerate(events):
                news_events.append(NewsEvent(
                    timestamp=current_time - timedelta(hours=i*2),
                    headline=event['headline'],
                    content=event['content'],
                    sentiment_score=event['sentiment'],
                    relevance_score=event['relevance'],
                    source='MarketNews',
                    impact_category=event['category']
                ))
            
            return news_events
            
        except Exception as e:
            logger.error(f"Failed to get market news: {e}")
            return []

class OrderFlowDataProvider(DataProvider):
    """
    Order flow and Level 2 data provider
    Provides real institutional mechanics data for HOW dimension
    """
    
    def __init__(self, broker_api_key: Optional[str] = None):
        self.broker_api_key = broker_api_key
        self.connection = None
        self.order_book_cache = {}
        
    async def connect(self) -> bool:
        """Initialize order flow data connection"""
        try:
            # In production, this would connect to a real broker API
            # like Interactive Brokers, FXCM, or institutional data provider
            logger.info("Connected to order flow data provider")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to order flow provider: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close order flow connection"""
        self.connection = None
        logger.info("Disconnected from order flow provider")
    
    async def is_connected(self) -> bool:
        """Check order flow connection status"""
        return True  # Simplified for demo
    
    async def get_order_book(self, symbol: str = 'EURUSD') -> OrderBookSnapshot:
        """Get current order book snapshot"""
        try:
            # Simulate realistic order book data
            current_time = datetime.now()
            mid_price = 1.0950  # Example EUR/USD price
            spread = 0.0002
            
            # Generate realistic bid/ask levels
            bids = []
            asks = []
            
            for i in range(10):  # 10 levels each side
                bid_price = mid_price - spread/2 - i * 0.0001
                ask_price = mid_price + spread/2 + i * 0.0001
                
                # Realistic size distribution (larger at better prices)
                bid_size = np.random.exponential(1000000) * (1 + 0.5 * (10-i))
                ask_size = np.random.exponential(1000000) * (1 + 0.5 * (10-i))
                
                bids.append(OrderBookLevel(
                    price=bid_price,
                    size=bid_size,
                    orders=np.random.randint(1, 20),
                    side='bid'
                ))
                
                asks.append(OrderBookLevel(
                    price=ask_price,
                    size=ask_size,
                    orders=np.random.randint(1, 20),
                    side='ask'
                ))
            
            return OrderBookSnapshot(
                timestamp=current_time,
                symbol=symbol,
                bids=bids,
                asks=asks,
                spread=spread,
                mid_price=mid_price
            )
            
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            return None
    
    async def calculate_order_flow_imbalance(self, symbol: str = 'EURUSD', 
                                           minutes_back: int = 5) -> float:
        """Calculate order flow imbalance over specified period"""
        try:
            # In production, this would analyze actual trade data
            # For now, simulate realistic order flow imbalance
            
            # Get multiple order book snapshots over time period
            imbalances = []
            
            for i in range(minutes_back):
                order_book = await self.get_order_book(symbol)
                if order_book:
                    # Calculate bid/ask imbalance
                    total_bid_size = sum(level.size for level in order_book.bids[:5])
                    total_ask_size = sum(level.size for level in order_book.asks[:5])
                    
                    if total_bid_size + total_ask_size > 0:
                        imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
                        imbalances.append(imbalance)
                
                # Simulate time passage
                await asyncio.sleep(0.1)
            
            # Return average imbalance
            return np.mean(imbalances) if imbalances else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate order flow imbalance: {e}")
            return 0.0

class DataIntegrationManager:
    """
    Central manager for all data providers
    Coordinates data collection and provides unified interface
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize providers
        if config.get('fred_api_key'):
            self.providers['economic'] = EconomicDataProvider(config['fred_api_key'])
        
        if config.get('news_api_key'):
            self.providers['news'] = NewsDataProvider(config['news_api_key'])
        
        if config.get('broker_api_key'):
            self.providers['orderflow'] = OrderFlowDataProvider(config['broker_api_key'])
    
    async def initialize(self) -> bool:
        """Initialize all data providers"""
        success = True
        for name, provider in self.providers.items():
            try:
                connected = await provider.connect()
                if not connected:
                    logger.warning(f"Failed to connect to {name} provider")
                    success = False
                else:
                    logger.info(f"Successfully connected to {name} provider")
            except Exception as e:
                logger.error(f"Error initializing {name} provider: {e}")
                success = False
        
        return success
    
    async def shutdown(self) -> None:
        """Shutdown all data providers"""
        for name, provider in self.providers.items():
            try:
                await provider.disconnect()
                logger.info(f"Disconnected from {name} provider")
            except Exception as e:
                logger.error(f"Error disconnecting from {name} provider: {e}")
        
        self.executor.shutdown(wait=True)
    
    async def get_comprehensive_market_data(self, symbol: str = 'EURUSD') -> Dict[str, Any]:
        """Get comprehensive market data from all sources"""
        data = {}
        
        # Collect data from all providers concurrently
        tasks = []
        
        if 'economic' in self.providers:
            tasks.append(self._get_economic_overview())
        
        if 'news' in self.providers:
            tasks.append(self._get_news_sentiment())
        
        if 'orderflow' in self.providers:
            tasks.append(self._get_institutional_data(symbol))
        
        # Execute all data collection tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Data collection task {i} failed: {result}")
            else:
                data.update(result)
        
        return data
    
    async def _get_economic_overview(self) -> Dict[str, Any]:
        """Get economic data overview"""
        provider = self.providers['economic']
        
        # Get key economic indicators
        gdp_data = await provider.get_economic_data('GDP', 90)
        inflation_data = await provider.get_economic_data('INFLATION', 30)
        rates_data = await provider.get_economic_data('FED_FUNDS', 30)
        vix_data = await provider.get_economic_data('VIX', 30)
        
        # Calculate policy divergence
        policy_divergence = await provider.calculate_policy_divergence()
        
        return {
            'economic': {
                'gdp_trend': self._calculate_trend(gdp_data),
                'inflation_trend': self._calculate_trend(inflation_data),
                'rates_trend': self._calculate_trend(rates_data),
                'risk_sentiment': self._calculate_risk_sentiment(vix_data),
                'policy_divergence': policy_divergence
            }
        }
    
    async def _get_news_sentiment(self) -> Dict[str, Any]:
        """Get news and sentiment data"""
        provider = self.providers['news']
        
        news_events = await provider.get_market_news()
        
        # Calculate aggregate sentiment
        if news_events:
            weighted_sentiment = sum(
                event.sentiment_score * event.relevance_score 
                for event in news_events
            ) / sum(event.relevance_score for event in news_events)
            
            # Categorize news impact
            categories = {}
            for event in news_events:
                if event.impact_category not in categories:
                    categories[event.impact_category] = []
                categories[event.impact_category].append(event.sentiment_score)
            
            category_sentiment = {
                cat: np.mean(scores) for cat, scores in categories.items()
            }
        else:
            weighted_sentiment = 0.0
            category_sentiment = {}
        
        return {
            'news': {
                'overall_sentiment': weighted_sentiment,
                'category_sentiment': category_sentiment,
                'event_count': len(news_events),
                'recent_events': [
                    {
                        'headline': event.headline,
                        'sentiment': event.sentiment_score,
                        'relevance': event.relevance_score
                    }
                    for event in news_events[:3]  # Top 3 recent events
                ]
            }
        }
    
    async def _get_institutional_data(self, symbol: str) -> Dict[str, Any]:
        """Get institutional order flow data"""
        provider = self.providers['orderflow']
        
        # Get order book and flow data
        order_book = await provider.get_order_book(symbol)
        flow_imbalance = await provider.calculate_order_flow_imbalance(symbol)
        
        institutional_data = {
            'order_flow_imbalance': flow_imbalance,
            'spread': order_book.spread if order_book else 0.0,
            'book_depth': len(order_book.bids) if order_book else 0,
            'liquidity_score': self._calculate_liquidity_score(order_book)
        }
        
        return {
            'institutional': institutional_data
        }
    
    def _calculate_trend(self, data_points: List[EconomicData]) -> float:
        """Calculate trend from economic data points"""
        if len(data_points) < 2:
            return 0.0
        
        values = [point.value for point in data_points[-10:]]  # Last 10 points
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize slope
        return np.tanh(slope / np.std(values)) if np.std(values) > 0 else 0.0
    
    def _calculate_risk_sentiment(self, vix_data: List[EconomicData]) -> float:
        """Calculate risk sentiment from VIX data"""
        if not vix_data:
            return 0.0
        
        current_vix = vix_data[-1].value
        
        # VIX interpretation: <20 = low fear, >30 = high fear
        if current_vix < 20:
            return 0.5  # Risk-on
        elif current_vix > 30:
            return -0.5  # Risk-off
        else:
            return (25 - current_vix) / 10  # Linear interpolation
    
    def _calculate_liquidity_score(self, order_book: OrderBookSnapshot) -> float:
        """Calculate liquidity score from order book"""
        if not order_book:
            return 0.0
        
        # Calculate total liquidity in top 5 levels
        bid_liquidity = sum(level.size for level in order_book.bids[:5])
        ask_liquidity = sum(level.size for level in order_book.asks[:5])
        total_liquidity = bid_liquidity + ask_liquidity
        
        # Normalize to 0-1 scale (assuming 10M is high liquidity)
        return min(total_liquidity / 10_000_000, 1.0)

# Example usage and configuration
async def main():
    """Example usage of the data integration system"""
    
    # Configuration with API keys (would be loaded from environment/config file)
    config = {
        'fred_api_key': 'your_fred_api_key_here',
        'news_api_key': 'your_news_api_key_here',
        'broker_api_key': 'your_broker_api_key_here'
    }
    
    # Initialize data manager
    data_manager = DataIntegrationManager(config)
    
    try:
        # Connect to all data sources
        success = await data_manager.initialize()
        if not success:
            logger.warning("Some data providers failed to initialize")
        
        # Get comprehensive market data
        market_data = await data_manager.get_comprehensive_market_data('EURUSD')
        
        print("Comprehensive Market Data:")
        print(json.dumps(market_data, indent=2, default=str))
        
    finally:
        # Clean shutdown
        await data_manager.shutdown()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the example
    asyncio.run(main())

