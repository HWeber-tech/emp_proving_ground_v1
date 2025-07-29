"""
NewsAPI Integration Module - Phase 1.5 Implementation

This module provides comprehensive NewsAPI integration for market sentiment analysis.
It includes news sentiment scoring, market-related articles, and trend analysis.

Author: EMP Development Team
Date: July 18, 2024
Phase: 1.5 - Advanced Data Sources
"""

import asyncio
import logging
import os
import re
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
class NewsAPIConfig:
    """Configuration for NewsAPI"""
    api_key: str
    base_url: str = "https://newsapi.org/v2"
    rate_limit: int = 100  # requests per day (free tier)
    timeout: int = 30
    retry_attempts: int = 3


class NewsAPIProvider:
    """NewsAPI data provider for market sentiment analysis"""
    
    def __init__(self, config: Optional[NewsAPIConfig] = None):
        self.config = config or NewsAPIConfig(
            api_key=os.getenv('NEWS_API_KEY', '')
        )
        self.rate_limiter = asyncio.Semaphore(self.config.rate_limit)
        self.validator = MarketDataValidator()
        self.last_request_time = 0
        self.request_count = 0
        self.request_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        if not self.config.api_key:
            logger.warning("NewsAPI key not found. Sentiment analysis will be disabled.")
    
    def _reset_request_count(self):
        """Reset request count at midnight"""
        now = datetime.now()
        if now >= self.request_reset_time:
            self.request_count = 0
            self.request_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        self._reset_request_count()
        
        if self.request_count >= self.config.rate_limit:
            wait_time = (self.request_reset_time - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.warning(f"NewsAPI rate limit reached. Waiting {wait_time:.0f} seconds.")
                await asyncio.sleep(wait_time)
                self._reset_request_count()
        
        self.request_count += 1
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting and error handling"""
        if not self.config.api_key:
            logger.warning("NewsAPI key not configured")
            return None
        
        try:
            await self._rate_limit()
            
            params['apiKey'] = self.config.api_key
            
            url = f"{self.config.base_url}/{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=self.config.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API errors
                        if data.get('status') == 'error':
                            logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                            return None
                        
                        return data
                    else:
                        logger.error(f"NewsAPI HTTP error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error making NewsAPI request: {e}")
            return None
    
    async def get_market_news(self, query: str = "forex trading", days: int = 7, 
                            language: str = "en", sort_by: str = "publishedAt") -> Optional[Dict[str, Any]]:
        """Get market-related news articles"""
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'sortBy': sort_by,
            'language': language,
            'pageSize': 100
        }
        
        data = await self._make_request('everything', params)
        
        if data and 'articles' in data:
            articles = data['articles']
            
            # Process articles
            processed_articles = []
            for article in articles:
                try:
                    processed_article = {
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'published_at': pd.to_datetime(article.get('publishedAt')),
                        'source': article.get('source', {}).get('name', ''),
                        'author': article.get('author', ''),
                        'sentiment_score': 0.0  # Will be calculated
                    }
                    processed_articles.append(processed_article)
                except Exception as e:
                    logger.warning(f"Error processing article: {e}")
                    continue
            
            result = {
                'query': query,
                'total_results': data.get('totalResults', 0),
                'articles': processed_articles,
                'date_range': f"{from_date} to {datetime.now().strftime('%Y-%m-%d')}"
            }
            
            logger.info(f"Retrieved {len(processed_articles)} articles for query '{query}'")
            return result
        
        return None
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score for text"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Define sentiment words
        positive_words = [
            'bullish', 'surge', 'rally', 'gain', 'positive', 'up', 'higher', 'strong',
            'growth', 'profit', 'success', 'win', 'victory', 'optimistic', 'confident',
            'recovery', 'rebound', 'soar', 'jump', 'climb', 'advance', 'improve'
        ]
        
        negative_words = [
            'bearish', 'drop', 'fall', 'decline', 'negative', 'down', 'lower', 'weak',
            'loss', 'crash', 'plunge', 'dive', 'sink', 'pessimistic', 'worried',
            'concern', 'risk', 'danger', 'threat', 'crisis', 'panic', 'fear'
        ]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        total_words = positive_count + negative_count
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_words
        return sentiment_score
    
    async def get_market_sentiment(self, query: str = "forex trading", days: int = 7) -> Optional[Dict[str, Any]]:
        """Get market sentiment analysis"""
        news_data = await self.get_market_news(query, days)
        
        if not news_data or not news_data['articles']:
            return None
        
        articles = news_data['articles']
        
        # Calculate sentiment for each article
        for article in articles:
            title_sentiment = self._calculate_sentiment_score(article['title'])
            description_sentiment = self._calculate_sentiment_score(article['description'])
            content_sentiment = self._calculate_sentiment_score(article['content'])
            
            # Weighted average (title most important, then description, then content)
            article['sentiment_score'] = (
                title_sentiment * 0.5 + 
                description_sentiment * 0.3 + 
                content_sentiment * 0.2
            )
        
        # Calculate overall sentiment metrics
        sentiment_scores = [article['sentiment_score'] for article in articles]
        
        sentiment_analysis = {
            'query': query,
            'total_articles': len(articles),
            'average_sentiment': np.mean(sentiment_scores),
            'sentiment_std': np.std(sentiment_scores),
            'positive_articles': sum(1 for score in sentiment_scores if score > 0.1),
            'negative_articles': sum(1 for score in sentiment_scores if score < -0.1),
            'neutral_articles': sum(1 for score in sentiment_scores if -0.1 <= score <= 0.1),
            'sentiment_trend': 'bullish' if np.mean(sentiment_scores) > 0.1 else 'bearish' if np.mean(sentiment_scores) < -0.1 else 'neutral',
            'articles': articles[:10],  # Top 10 articles
            'date_range': news_data['date_range']
        }
        
        logger.info(f"Sentiment analysis for '{query}': {sentiment_analysis['sentiment_trend']} ({sentiment_analysis['average_sentiment']:.3f})")
        return sentiment_analysis
    
    async def get_top_headlines(self, category: str = "business", country: str = "us", 
                               language: str = "en") -> Optional[Dict[str, Any]]:
        """Get top headlines by category"""
        params = {
            'category': category,
            'country': country,
            'language': language,
            'pageSize': 100
        }
        
        data = await self._make_request('top-headlines', params)
        
        if data and 'articles' in data:
            articles = data['articles']
            
            # Process articles
            processed_articles = []
            for article in articles:
                try:
                    processed_article = {
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published_at': pd.to_datetime(article.get('publishedAt')),
                        'source': article.get('source', {}).get('name', ''),
                        'author': article.get('author', ''),
                        'sentiment_score': self._calculate_sentiment_score(article.get('title', '') + ' ' + article.get('description', ''))
                    }
                    processed_articles.append(processed_article)
                except Exception as e:
                    logger.warning(f"Error processing headline: {e}")
                    continue
            
            result = {
                'category': category,
                'country': country,
                'total_results': data.get('totalResults', 0),
                'articles': processed_articles
            }
            
            logger.info(f"Retrieved {len(processed_articles)} top headlines for {category}")
            return result
        
        return None
    
    async def get_sources(self, category: str = "business", language: str = "en", 
                         country: str = "us") -> Optional[List[Dict[str, Any]]]:
        """Get available news sources"""
        params = {
            'category': category,
            'language': language,
            'country': country
        }
        
        data = await self._make_request('sources', params)
        
        if data and 'sources' in data:
            sources = data['sources']
            logger.info(f"Retrieved {len(sources)} news sources")
            return sources
        
        return None
    
    async def get_sentiment_trends(self, queries: List[str], days: int = 7) -> Optional[Dict[str, Any]]:
        """Get sentiment trends for multiple queries"""
        trends = {}
        
        for query in queries:
            sentiment_data = await self.get_market_sentiment(query, days)
            if sentiment_data:
                trends[query] = {
                    'average_sentiment': sentiment_data['average_sentiment'],
                    'sentiment_trend': sentiment_data['sentiment_trend'],
                    'total_articles': sentiment_data['total_articles']
                }
        
        if trends:
            overall_sentiment = np.mean([t['average_sentiment'] for t in trends.values()])
            overall_trend = 'bullish' if overall_sentiment > 0.1 else 'bearish' if overall_sentiment < -0.1 else 'neutral'
            
            result = {
                'overall_sentiment': overall_sentiment,
                'overall_trend': overall_trend,
                'trends': trends,
                'date_range': f"{(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}"
            }
            
            logger.info(f"Sentiment trends: {overall_trend} ({overall_sentiment:.3f})")
            return result
        
        return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status and configuration"""
        return {
            'api_key_configured': bool(self.config.api_key),
            'rate_limit': self.config.rate_limit,
            'base_url': self.config.base_url,
            'timeout': self.config.timeout,
            'requests_used': self.request_count,
            'requests_remaining': max(0, self.config.rate_limit - self.request_count),
            'reset_time': self.request_reset_time.isoformat()
        }


# Example usage and testing
async def test_newsapi_integration():
    """Test NewsAPI integration"""
    provider = NewsAPIProvider()
    
    # Test API status
    status = provider.get_api_status()
    print(f"NewsAPI Status: {status}")
    
    if not status['api_key_configured']:
        print("⚠️ NewsAPI key not configured. Tests will be skipped.")
        return
    
    # Test market sentiment
    print("\nTesting market sentiment...")
    sentiment = await provider.get_market_sentiment("forex trading", days=3)
    if sentiment:
        print(f"✅ Market sentiment: {sentiment['sentiment_trend']} ({sentiment['average_sentiment']:.3f})")
        print(f"   Articles: {sentiment['total_articles']} (Positive: {sentiment['positive_articles']}, Negative: {sentiment['negative_articles']})")
    else:
        print("❌ Market sentiment failed")
    
    # Test top headlines
    print("\nTesting top headlines...")
    headlines = await provider.get_top_headlines("business", "us")
    if headlines:
        print(f"✅ Top headlines: {len(headlines['articles'])} articles")
        avg_sentiment = np.mean([article['sentiment_score'] for article in headlines['articles']])
        print(f"   Average sentiment: {avg_sentiment:.3f}")
    else:
        print("❌ Top headlines failed")
    
    # Test sentiment trends
    print("\nTesting sentiment trends...")
    trends = await provider.get_sentiment_trends(["forex", "stocks", "crypto"], days=3)
    if trends:
        print(f"✅ Sentiment trends: {trends['overall_trend']} ({trends['overall_sentiment']:.3f})")
        for query, trend in trends['trends'].items():
            print(f"   {query}: {trend['sentiment_trend']} ({trend['average_sentiment']:.3f})")
    else:
        print("❌ Sentiment trends failed")


if __name__ == "__main__":
    asyncio.run(test_newsapi_integration()) 
