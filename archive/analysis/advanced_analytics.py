"""
Advanced Analytics System
Provides sentiment analysis, news integration, advanced technical indicators,
and market correlation analysis for enhanced trading decisions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import requests
import json
from textblob import TextBlob
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class SentimentType(Enum):
    """Sentiment type enumeration."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class NewsSource(Enum):
    """News source enumeration."""
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    CNBC = "cnbc"
    MARKETWATCH = "marketwatch"
    YAHOO_FINANCE = "yahoo_finance"

@dataclass
class NewsItem:
    """News item structure."""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float
    sentiment_type: SentimentType
    relevance_score: float
    keywords: List[str]

@dataclass
class SentimentAnalysis:
    """Sentiment analysis results."""
    symbol: str
    timestamp: datetime
    overall_sentiment: float
    sentiment_type: SentimentType
    news_count: int
    positive_news: int
    negative_news: int
    neutral_news: int
    top_keywords: List[str]
    sentiment_trend: str  # 'improving', 'declining', 'stable'

@dataclass
class AdvancedIndicators:
    """Advanced technical indicators."""
    symbol: str
    timestamp: datetime
    
    # Momentum indicators
    rsi: float
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    cci: float
    
    # Volatility indicators
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    atr: float
    keltner_upper: float
    keltner_lower: float
    
    # Trend indicators
    macd: float
    macd_signal: float
    macd_histogram: float
    adx: float
    di_plus: float
    di_minus: float
    
    # Volume indicators
    obv: float
    vwap: float
    money_flow_index: float
    
    # Custom indicators
    support_level: float
    resistance_level: float
    pivot_point: float
    fibonacci_retracement: Dict[str, float]

@dataclass
class MarketCorrelation:
    """Market correlation analysis."""
    symbol: str
    timestamp: datetime
    correlations: Dict[str, float]
    beta: float
    alpha: float
    sharpe_ratio: float
    volatility: float
    correlation_matrix: pd.DataFrame
    sector_correlation: float
    market_correlation: float

class AdvancedAnalytics:
    """
    Advanced analytics system for comprehensive market analysis.
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the advanced analytics system.
        
        Args:
            api_keys: Dictionary of API keys for news and data sources
        """
        self.api_keys = api_keys or {}
        self.news_cache: Dict[str, List[NewsItem]] = {}
        self.sentiment_cache: Dict[str, SentimentAnalysis] = {}
        self.indicators_cache: Dict[str, AdvancedIndicators] = {}
        self.correlation_cache: Dict[str, MarketCorrelation] = {}
        
        # Configuration
        self.news_cache_duration = timedelta(hours=1)
        self.sentiment_cache_duration = timedelta(minutes=30)
        self.indicators_cache_duration = timedelta(minutes=15)
        self.correlation_cache_duration = timedelta(hours=2)
        
        # Market symbols for correlation analysis
        self.market_symbols = [
            'SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'USO', 'VIX'
        ]
        
        logger.info("Advanced analytics system initialized")
    
    def analyze_sentiment(self, symbol: str, force_refresh: bool = False) -> SentimentAnalysis:
        """
        Analyze market sentiment for a symbol.
        
        Args:
            symbol: Trading symbol
            force_refresh: Force refresh of cached data
            
        Returns:
            Sentiment analysis results
        """
        # Check cache
        if not force_refresh and symbol in self.sentiment_cache:
            cached = self.sentiment_cache[symbol]
            if datetime.now() - cached.timestamp < self.sentiment_cache_duration:
                return cached
        
        try:
            # Get news data
            news_items = self._fetch_news(symbol)
            
            if not news_items:
                # Return neutral sentiment if no news
                sentiment = SentimentAnalysis(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    overall_sentiment=0.0,
                    sentiment_type=SentimentType.NEUTRAL,
                    news_count=0,
                    positive_news=0,
                    negative_news=0,
                    neutral_news=0,
                    top_keywords=[],
                    sentiment_trend='stable'
                )
                self.sentiment_cache[symbol] = sentiment
                return sentiment
            
            # Analyze sentiment
            sentiment_scores = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            all_keywords = []
            
            for news in news_items:
                # Analyze title and content
                title_sentiment = self._analyze_text_sentiment(news.title)
                content_sentiment = self._analyze_text_sentiment(news.content)
                
                # Combined sentiment
                combined_sentiment = (title_sentiment * 0.7) + (content_sentiment * 0.3)
                sentiment_scores.append(combined_sentiment)
                
                # Count sentiment types
                if combined_sentiment > 0.1:
                    positive_count += 1
                elif combined_sentiment < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1
                
                all_keywords.extend(news.keywords)
            
            # Calculate overall sentiment
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            
            # Determine sentiment type
            if overall_sentiment > 0.1:
                sentiment_type = SentimentType.POSITIVE
            elif overall_sentiment < -0.1:
                sentiment_type = SentimentType.NEGATIVE
            else:
                sentiment_type = SentimentType.NEUTRAL
            
            # Get top keywords
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_keywords = [kw[0] for kw in top_keywords]
            
            # Determine sentiment trend
            sentiment_trend = self._determine_sentiment_trend(symbol, overall_sentiment)
            
            sentiment = SentimentAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_sentiment=overall_sentiment,
                sentiment_type=sentiment_type,
                news_count=len(news_items),
                positive_news=positive_count,
                negative_news=negative_count,
                neutral_news=neutral_count,
                top_keywords=top_keywords,
                sentiment_trend=sentiment_trend
            )
            
            self.sentiment_cache[symbol] = sentiment
            logger.info(f"Sentiment analysis for {symbol}: {sentiment_type.value} ({overall_sentiment:.3f})")
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return self._create_neutral_sentiment(symbol)
    
    def calculate_advanced_indicators(self, symbol: str, data: pd.DataFrame, 
                                    force_refresh: bool = False) -> AdvancedIndicators:
        """
        Calculate advanced technical indicators.
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            force_refresh: Force refresh of cached data
            
        Returns:
            Advanced indicators
        """
        # Check cache
        if not force_refresh and symbol in self.indicators_cache:
            cached = self.indicators_cache[symbol]
            if datetime.now() - cached.timestamp < self.indicators_cache_duration:
                return cached
        
        try:
            if data.empty or len(data) < 50:
                return self._create_empty_indicators(symbol)
            
            # Ensure required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Missing required columns for {symbol}")
                return self._create_empty_indicators(symbol)
            
            # Calculate indicators
            indicators = AdvancedIndicators(
                symbol=symbol,
                timestamp=datetime.now(),
                
                # Momentum indicators
                rsi=self._calculate_rsi(data['Close']),
                stochastic_k=self._calculate_stochastic_k(data),
                stochastic_d=self._calculate_stochastic_d(data),
                williams_r=self._calculate_williams_r(data),
                cci=self._calculate_cci(data),
                
                # Volatility indicators
                bollinger_upper=self._calculate_bollinger_bands(data['Close'])[0],
                bollinger_middle=self._calculate_bollinger_bands(data['Close'])[1],
                bollinger_lower=self._calculate_bollinger_bands(data['Close'])[2],
                atr=self._calculate_atr(data),
                keltner_upper=self._calculate_keltner_channels(data)[0],
                keltner_lower=self._calculate_keltner_channels(data)[1],
                
                # Trend indicators
                macd=self._calculate_macd(data['Close'])[0],
                macd_signal=self._calculate_macd(data['Close'])[1],
                macd_histogram=self._calculate_macd(data['Close'])[2],
                adx=self._calculate_adx(data),
                di_plus=self._calculate_directional_indicators(data)[0],
                di_minus=self._calculate_directional_indicators(data)[1],
                
                # Volume indicators
                obv=self._calculate_obv(data),
                vwap=self._calculate_vwap(data),
                money_flow_index=self._calculate_money_flow_index(data),
                
                # Custom indicators
                support_level=self._calculate_support_level(data),
                resistance_level=self._calculate_resistance_level(data),
                pivot_point=self._calculate_pivot_point(data),
                fibonacci_retracement=self._calculate_fibonacci_retracement(data)
            )
            
            self.indicators_cache[symbol] = indicators
            logger.debug(f"Advanced indicators calculated for {symbol}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating advanced indicators for {symbol}: {e}")
            return self._create_empty_indicators(symbol)
    
    def analyze_market_correlation(self, symbol: str, data: pd.DataFrame,
                                 force_refresh: bool = False) -> MarketCorrelation:
        """
        Analyze market correlations for a symbol.
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            force_refresh: Force refresh of cached data
            
        Returns:
            Market correlation analysis
        """
        # Check cache
        if not force_refresh and symbol in self.correlation_cache:
            cached = self.correlation_cache[symbol]
            if datetime.now() - cached.timestamp < self.correlation_cache_duration:
                return cached
        
        try:
            if data.empty or len(data) < 30:
                return self._create_empty_correlation(symbol)
            
            # Get market data
            market_data = self._fetch_market_data()
            if market_data.empty:
                return self._create_empty_correlation(symbol)
            
            # Calculate returns
            symbol_returns = data['Close'].pct_change().dropna()
            
            # Calculate correlations
            correlations = {}
            for market_symbol in self.market_symbols:
                if market_symbol in market_data.columns:
                    market_returns = market_data[market_symbol].pct_change().dropna()
                    # Align data
                    aligned_data = pd.concat([symbol_returns, market_returns], axis=1).dropna()
                    if len(aligned_data) > 10:
                        correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                        correlations[market_symbol] = correlation
            
            # Calculate beta and alpha (using SPY as market proxy)
            if 'SPY' in correlations:
                spy_returns = market_data['SPY'].pct_change().dropna()
                aligned_data = pd.concat([symbol_returns, spy_returns], axis=1).dropna()
                if len(aligned_data) > 10:
                    beta = self._calculate_beta(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                    alpha = self._calculate_alpha(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1], beta)
                else:
                    beta = alpha = 0.0
            else:
                beta = alpha = 0.0
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(symbol_returns)
            
            # Calculate volatility
            volatility = symbol_returns.std() * np.sqrt(252)
            
            # Create correlation matrix
            correlation_matrix = self._create_correlation_matrix(symbol_returns, market_data)
            
            # Calculate sector and market correlations
            sector_correlation = correlations.get('QQQ', 0.0)  # Tech sector
            market_correlation = correlations.get('SPY', 0.0)  # Market
            
            correlation = MarketCorrelation(
                symbol=symbol,
                timestamp=datetime.now(),
                correlations=correlations,
                beta=beta,
                alpha=alpha,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                correlation_matrix=correlation_matrix,
                sector_correlation=sector_correlation,
                market_correlation=market_correlation
            )
            
            self.correlation_cache[symbol] = correlation
            logger.debug(f"Market correlation analysis for {symbol}: beta={beta:.3f}, sharpe={sharpe_ratio:.3f}")
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error analyzing market correlation for {symbol}: {e}")
            return self._create_empty_correlation(symbol)
    
    def get_comprehensive_analysis(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive analysis combining all analytics.
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Get all analyses
            sentiment = self.analyze_sentiment(symbol)
            indicators = self.calculate_advanced_indicators(symbol, data)
            correlation = self.analyze_market_correlation(symbol, data)
            
            # Combine into comprehensive analysis
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                
                'sentiment': {
                    'overall_sentiment': sentiment.overall_sentiment,
                    'sentiment_type': sentiment.sentiment_type.value,
                    'news_count': sentiment.news_count,
                    'positive_news': sentiment.positive_news,
                    'negative_news': sentiment.negative_news,
                    'sentiment_trend': sentiment.sentiment_trend,
                    'top_keywords': sentiment.top_keywords
                },
                
                'technical_indicators': {
                    'momentum': {
                        'rsi': indicators.rsi,
                        'stochastic_k': indicators.stochastic_k,
                        'stochastic_d': indicators.stochastic_d,
                        'williams_r': indicators.williams_r,
                        'cci': indicators.cci
                    },
                    'volatility': {
                        'bollinger_upper': indicators.bollinger_upper,
                        'bollinger_middle': indicators.bollinger_middle,
                        'bollinger_lower': indicators.bollinger_lower,
                        'atr': indicators.atr,
                        'keltner_upper': indicators.keltner_upper,
                        'keltner_lower': indicators.keltner_lower
                    },
                    'trend': {
                        'macd': indicators.macd,
                        'macd_signal': indicators.macd_signal,
                        'macd_histogram': indicators.macd_histogram,
                        'adx': indicators.adx,
                        'di_plus': indicators.di_plus,
                        'di_minus': indicators.di_minus
                    },
                    'volume': {
                        'obv': indicators.obv,
                        'vwap': indicators.vwap,
                        'money_flow_index': indicators.money_flow_index
                    },
                    'support_resistance': {
                        'support_level': indicators.support_level,
                        'resistance_level': indicators.resistance_level,
                        'pivot_point': indicators.pivot_point
                    }
                },
                
                'market_correlation': {
                    'beta': correlation.beta,
                    'alpha': correlation.alpha,
                    'sharpe_ratio': correlation.sharpe_ratio,
                    'volatility': correlation.volatility,
                    'sector_correlation': correlation.sector_correlation,
                    'market_correlation': correlation.market_correlation,
                    'correlations': correlation.correlations
                },
                
                'trading_signals': self._generate_trading_signals(sentiment, indicators, correlation)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis for {symbol}: {e}")
            return {}
    
    def _fetch_news(self, symbol: str) -> List[NewsItem]:
        """Fetch news for a symbol (mock implementation)."""
        # Mock news data - in real implementation, this would use news APIs
        mock_news = [
            NewsItem(
                title=f"Positive outlook for {symbol} as earnings beat expectations",
                content=f"Company {symbol} reported strong quarterly results...",
                source="Mock News",
                url="https://mock-news.com/article1",
                published_at=datetime.now() - timedelta(hours=2),
                sentiment_score=0.3,
                sentiment_type=SentimentType.POSITIVE,
                relevance_score=0.8,
                keywords=[symbol, "earnings", "positive", "growth"]
            ),
            NewsItem(
                title=f"Market analysts bullish on {symbol}",
                content=f"Leading analysts have upgraded their rating...",
                source="Mock News",
                url="https://mock-news.com/article2",
                published_at=datetime.now() - timedelta(hours=4),
                sentiment_score=0.2,
                sentiment_type=SentimentType.POSITIVE,
                relevance_score=0.7,
                keywords=[symbol, "analysts", "bullish", "upgrade"]
            )
        ]
        
        return mock_news
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _determine_sentiment_trend(self, symbol: str, current_sentiment: float) -> str:
        """Determine sentiment trend (simplified)."""
        # In real implementation, this would compare with historical sentiment
        if current_sentiment > 0.1:
            return 'improving'
        elif current_sentiment < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _create_neutral_sentiment(self, symbol: str) -> SentimentAnalysis:
        """Create neutral sentiment analysis."""
        return SentimentAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_sentiment=0.0,
            sentiment_type=SentimentType.NEUTRAL,
            news_count=0,
            positive_news=0,
            negative_news=0,
            neutral_news=0,
            top_keywords=[],
            sentiment_trend='stable'
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_stochastic_k(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Stochastic %K."""
        try:
            low_min = data['Low'].rolling(window=period).min()
            high_max = data['High'].rolling(window=period).max()
            k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            return k.iloc[-1] if not pd.isna(k.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_stochastic_d(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Stochastic %D."""
        try:
            k = pd.Series([self._calculate_stochastic_k(data.iloc[:i+1]) for i in range(len(data))])
            d = k.rolling(window=3).mean()
            return d.iloc[-1] if not pd.isna(d.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Williams %R."""
        try:
            low_min = data['Low'].rolling(window=period).min()
            high_max = data['High'].rolling(window=period).max()
            wr = -100 * ((high_max - data['Close']) / (high_max - low_min))
            return wr.iloc[-1] if not pd.isna(wr.iloc[-1]) else -50.0
        except:
            return -50.0
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate CCI."""
        try:
            tp = (data['High'] + data['Low'] + data['Close']) / 3
            sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (tp - sma) / (0.015 * mad)
            return cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_keltner_channels(self, data: pd.DataFrame, period: int = 20) -> Tuple[float, float]:
        """Calculate Keltner Channels."""
        try:
            tp = (data['High'] + data['Low'] + data['Close']) / 3
            atr = self._calculate_atr(data, period)
            upper = tp + (2 * atr)
            lower = tp - (2 * atr)
            return upper.iloc[-1], lower.iloc[-1]
        except:
            return 0.0, 0.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX."""
        try:
            # Simplified ADX calculation
            high_diff = data['High'].diff()
            low_diff = data['Low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            tr = self._calculate_atr(data, period)
            plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / tr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / tr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = pd.Series(dx).rolling(period).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0
        except:
            return 25.0
    
    def _calculate_directional_indicators(self, data: pd.DataFrame, period: int = 14) -> Tuple[float, float]:
        """Calculate Directional Indicators."""
        try:
            # Simplified calculation
            high_diff = data['High'].diff()
            low_diff = data['Low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            tr = self._calculate_atr(data, period)
            plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / tr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / tr)
            
            return plus_di.iloc[-1], minus_di.iloc[-1]
        except:
            return 25.0, 25.0
    
    def _calculate_obv(self, data: pd.DataFrame) -> float:
        """Calculate On-Balance Volume."""
        try:
            if len(data) < 2:
                return data['Volume'].iloc[0] if len(data) > 0 else 0.0
            
            obv = pd.Series(index=data.index, dtype=float)
            obv.iloc[0] = data['Volume'].iloc[0]
            
            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            result = obv.iloc[-1]
            return result if not pd.isna(result) else 0.0
        except:
            return 0.0
    
    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        """Calculate VWAP."""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            return vwap.iloc[-1]
        except:
            return 0.0
    
    def _calculate_money_flow_index(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Money Flow Index."""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            money_flow = typical_price * data['Volume']
            
            positive_flow = pd.Series(0.0, index=data.index)
            negative_flow = pd.Series(0.0, index=data.index)
            
            for i in range(1, len(data)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = money_flow.iloc[i]
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    negative_flow.iloc[i] = money_flow.iloc[i]
            
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            return mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_support_level(self, data: pd.DataFrame) -> float:
        """Calculate support level."""
        try:
            # Simplified support calculation
            recent_lows = data['Low'].tail(20)
            return recent_lows.min()
        except:
            return 0.0
    
    def _calculate_resistance_level(self, data: pd.DataFrame) -> float:
        """Calculate resistance level."""
        try:
            # Simplified resistance calculation
            recent_highs = data['High'].tail(20)
            return recent_highs.max()
        except:
            return 0.0
    
    def _calculate_pivot_point(self, data: pd.DataFrame) -> float:
        """Calculate pivot point."""
        try:
            high = data['High'].iloc[-1]
            low = data['Low'].iloc[-1]
            close = data['Close'].iloc[-1]
            return (high + low + close) / 3
        except:
            return 0.0
    
    def _calculate_fibonacci_retracement(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        try:
            high = data['High'].max()
            low = data['Low'].min()
            diff = high - low
            
            return {
                '0.0': low,
                '0.236': low + 0.236 * diff,
                '0.382': low + 0.382 * diff,
                '0.5': low + 0.5 * diff,
                '0.618': low + 0.618 * diff,
                '0.786': low + 0.786 * diff,
                '1.0': high
            }
        except:
            return {}
    
    def _fetch_market_data(self) -> pd.DataFrame:
        """Fetch market data for correlation analysis."""
        try:
            # Mock market data - in real implementation, this would fetch from APIs
            dates = pd.date_range(start=datetime.now() - timedelta(days=60), end=datetime.now(), freq='D')
            market_data = pd.DataFrame(index=dates)
            
            for symbol in self.market_symbols:
                market_data[symbol] = np.random.randn(len(dates)).cumsum() + 100
            
            return market_data
        except:
            return pd.DataFrame()
    
    def _calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta."""
        try:
            covariance = returns.cov(market_returns)
            market_variance = market_returns.var()
            return covariance / market_variance if market_variance != 0 else 1.0
        except:
            return 1.0
    
    def _calculate_alpha(self, returns: pd.Series, market_returns: pd.Series, beta: float) -> float:
        """Calculate alpha."""
        try:
            return returns.mean() - (beta * market_returns.mean())
        except:
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        try:
            excess_returns = returns - risk_free_rate / 252
            return excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0.0
        except:
            return 0.0
    
    def _create_correlation_matrix(self, symbol_returns: pd.Series, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create correlation matrix."""
        try:
            # Combine symbol returns with market data
            combined_data = pd.concat([symbol_returns, market_data], axis=1)
            return combined_data.corr()
        except:
            return pd.DataFrame()
    
    def _create_empty_indicators(self, symbol: str) -> AdvancedIndicators:
        """Create empty indicators."""
        return AdvancedIndicators(
            symbol=symbol,
            timestamp=datetime.now(),
            rsi=50.0, stochastic_k=50.0, stochastic_d=50.0, williams_r=-50.0, cci=0.0,
            bollinger_upper=0.0, bollinger_middle=0.0, bollinger_lower=0.0,
            atr=0.0, keltner_upper=0.0, keltner_lower=0.0,
            macd=0.0, macd_signal=0.0, macd_histogram=0.0,
            adx=25.0, di_plus=25.0, di_minus=25.0,
            obv=0.0, vwap=0.0, money_flow_index=50.0,
            support_level=0.0, resistance_level=0.0, pivot_point=0.0,
            fibonacci_retracement={}
        )
    
    def _create_empty_correlation(self, symbol: str) -> MarketCorrelation:
        """Create empty correlation."""
        return MarketCorrelation(
            symbol=symbol,
            timestamp=datetime.now(),
            correlations={},
            beta=1.0,
            alpha=0.0,
            sharpe_ratio=0.0,
            volatility=0.0,
            correlation_matrix=pd.DataFrame(),
            sector_correlation=0.0,
            market_correlation=0.0
        )
    
    def _generate_trading_signals(self, sentiment: SentimentAnalysis, 
                                indicators: AdvancedIndicators, 
                                correlation: MarketCorrelation) -> Dict[str, Any]:
        """Generate trading signals based on all analyses."""
        signals = {
            'sentiment_signal': 'neutral',
            'technical_signal': 'neutral',
            'correlation_signal': 'neutral',
            'overall_signal': 'neutral',
            'confidence': 0.5
        }
        
        # Sentiment signal
        if sentiment.sentiment_type == SentimentType.POSITIVE:
            signals['sentiment_signal'] = 'buy'
        elif sentiment.sentiment_type == SentimentType.NEGATIVE:
            signals['sentiment_signal'] = 'sell'
        
        # Technical signal
        if indicators.rsi < 30 and indicators.stochastic_k < 20:
            signals['technical_signal'] = 'buy'
        elif indicators.rsi > 70 and indicators.stochastic_k > 80:
            signals['technical_signal'] = 'sell'
        
        # Correlation signal
        if correlation.beta < 0.8:  # Low beta, defensive
            signals['correlation_signal'] = 'buy'
        elif correlation.beta > 1.2:  # High beta, aggressive
            signals['correlation_signal'] = 'sell'
        
        # Overall signal
        buy_signals = sum(1 for signal in signals.values() if signal == 'buy')
        sell_signals = sum(1 for signal in signals.values() if signal == 'sell')
        
        if buy_signals > sell_signals:
            signals['overall_signal'] = 'buy'
            signals['confidence'] = min(0.9, 0.5 + (buy_signals * 0.1))
        elif sell_signals > buy_signals:
            signals['overall_signal'] = 'sell'
            signals['confidence'] = min(0.9, 0.5 + (sell_signals * 0.1))
        
        return signals 