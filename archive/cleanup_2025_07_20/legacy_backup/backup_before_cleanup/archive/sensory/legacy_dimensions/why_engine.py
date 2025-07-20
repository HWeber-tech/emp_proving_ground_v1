"""
Sensory Cortex v2.2 - WHY Dimension Engine (Fundamental Analysis)

Masterful implementation of fundamental analysis using real economic data.
Eliminates all mock data generation and implements sophisticated macro analysis.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import aiohttp
import json

from src.sensory.core.base import (
    DimensionalSensor, DimensionalReading, MarketData, InstrumentMeta,
    MarketRegime, EconomicEvent, EventTier
)
from src.sensory.core.utils import (
    EMA, WelfordVar, compute_confidence, normalize_signal,
    calculate_momentum, PerformanceTracker
)

logger = logging.getLogger(__name__)


class EconomicDataProvider:
    """
    Real economic data provider with CSV fallback for backtesting.
    Eliminates all random data generation.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize economic data provider.
        
        Args:
            data_dir: Directory containing CSV data files
        """
        self.data_dir = data_dir
        self.yield_curve_data = None
        self.risk_indexes_data = None
        self.policy_rates_data = None
        self._load_csv_data()
        
    def _load_csv_data(self) -> None:
        """Load deterministic CSV data for backtesting."""
        try:
            # Load yield curve data
            yield_curve_path = self.data_dir / "yield_curve.csv"
            if yield_curve_path.exists():
                self.yield_curve_data = pd.read_csv(yield_curve_path)
                self.yield_curve_data['date'] = pd.to_datetime(self.yield_curve_data['date'])
                logger.info(f"Loaded {len(self.yield_curve_data)} yield curve records")
            
            # Load risk indexes data
            risk_indexes_path = self.data_dir / "risk_indexes.csv"
            if risk_indexes_path.exists():
                self.risk_indexes_data = pd.read_csv(risk_indexes_path)
                self.risk_indexes_data['date'] = pd.to_datetime(self.risk_indexes_data['date'])
                logger.info(f"Loaded {len(self.risk_indexes_data)} risk index records")
            
            # Load policy rates data
            policy_rates_path = self.data_dir / "policy_rates.csv"
            if policy_rates_path.exists():
                self.policy_rates_data = pd.read_csv(policy_rates_path)
                self.policy_rates_data['date'] = pd.to_datetime(self.policy_rates_data['date'])
                logger.info(f"Loaded {len(self.policy_rates_data)} policy rate records")
                
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise
    
    async def get_yield_curve(self, date: datetime) -> Optional[Dict[str, float]]:
        """
        Get yield curve data for specific date.
        
        Args:
            date: Target date
            
        Returns:
            Yield curve data or None if not available
        """
        if self.yield_curve_data is None:
            return None
        
        # Find closest date
        target_date = date.date()
        closest_row = self.yield_curve_data.iloc[
            (self.yield_curve_data['date'].dt.date - target_date).abs().argsort()[:1]
        ]
        
        if len(closest_row) == 0:
            return None
        
        row = closest_row.iloc[0]
        return {
            '3M': row['3M'],
            '6M': row['6M'],
            '1Y': row['1Y'],
            '2Y': row['2Y'],
            '5Y': row['5Y'],
            '10Y': row['10Y'],
            '30Y': row['30Y']
        }
    
    async def get_risk_indexes(self, date: datetime) -> Optional[Dict[str, float]]:
        """
        Get risk index data for specific date.
        
        Args:
            date: Target date
            
        Returns:
            Risk index data or None if not available
        """
        if self.risk_indexes_data is None:
            return None
        
        # Find closest date
        target_date = date.date()
        closest_row = self.risk_indexes_data.iloc[
            (self.risk_indexes_data['date'].dt.date - target_date).abs().argsort()[:1]
        ]
        
        if len(closest_row) == 0:
            return None
        
        row = closest_row.iloc[0]
        return {
            'VIX': row['VIX'],
            'DXY': row['DXY'],
            'GOLD': row['GOLD'],
            'OIL': row['OIL'],
            'SPX': row['SPX'],
            'EURUSD': row['EURUSD'],
            'GBPUSD': row['GBPUSD'],
            'USDJPY': row['USDJPY']
        }
    
    async def get_policy_rates(self, date: datetime) -> Optional[Dict[str, float]]:
        """
        Get central bank policy rates for specific date.
        
        Args:
            date: Target date
            
        Returns:
            Policy rates data or None if not available
        """
        if self.policy_rates_data is None:
            return None
        
        # Find closest date
        target_date = date.date()
        closest_row = self.policy_rates_data.iloc[
            (self.policy_rates_data['date'].dt.date - target_date).abs().argsort()[:1]
        ]
        
        if len(closest_row) == 0:
            return None
        
        row = closest_row.iloc[0]
        return {
            'FED_FUNDS': row['FED_FUNDS'],
            'ECB_RATE': row['ECB_RATE'],
            'BOE_RATE': row['BOE_RATE'],
            'BOJ_RATE': row['BOJ_RATE'],
            'SNB_RATE': row['SNB_RATE'],
            'BOC_RATE': row['BOC_RATE'],
            'RBA_RATE': row['RBA_RATE'],
            'RBNZ_RATE': row['RBNZ_RATE']
        }


class YieldCurveAnalyzer:
    """
    Sophisticated yield curve analysis for fundamental insights.
    """
    
    def __init__(self):
        """Initialize yield curve analyzer."""
        self.curve_history: List[Dict[str, float]] = []
        self.slope_ema = EMA(20)
        self.curvature_ema = EMA(20)
        
    def analyze_curve(self, curve_data: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze yield curve for fundamental signals.
        
        Args:
            curve_data: Yield curve data
            
        Returns:
            Analysis results
        """
        self.curve_history.append(curve_data)
        if len(self.curve_history) > 100:
            self.curve_history.pop(0)
        
        # Calculate curve slope (10Y - 2Y spread)
        slope = curve_data['10Y'] - curve_data['2Y']
        self.slope_ema.update(slope)
        
        # Calculate curve curvature (belly vs wings)
        belly = curve_data['5Y']
        wings = (curve_data['2Y'] + curve_data['10Y']) / 2
        curvature = belly - wings
        self.curvature_ema.update(curvature)
        
        # Inversion detection
        inversion_score = 0.0
        if curve_data['2Y'] > curve_data['10Y']:
            inversion_score = (curve_data['2Y'] - curve_data['10Y']) / curve_data['10Y']
        
        # Steepening/flattening trend
        slope_trend = 0.0
        if len(self.curve_history) >= 2:
            prev_slope = self.curve_history[-2]['10Y'] - self.curve_history[-2]['2Y']
            slope_trend = (slope - prev_slope) / abs(prev_slope) if prev_slope != 0 else 0.0
        
        return {
            'slope': slope,
            'slope_ema': self.slope_ema.get_value() or 0.0,
            'curvature': curvature,
            'curvature_ema': self.curvature_ema.get_value() or 0.0,
            'inversion_score': inversion_score,
            'slope_trend': slope_trend,
            'steepness_percentile': self._calculate_steepness_percentile(slope)
        }
    
    def _calculate_steepness_percentile(self, current_slope: float) -> float:
        """Calculate current slope percentile vs historical."""
        if len(self.curve_history) < 20:
            return 0.5
        
        historical_slopes = [
            data['10Y'] - data['2Y'] for data in self.curve_history[-50:]
        ]
        
        percentile = np.percentile(historical_slopes, 
                                 [p for p in range(101) if np.percentile(historical_slopes, p) <= current_slope][-1] if historical_slopes else 50)
        
        return percentile / 100.0


class CurrencyStrengthAnalyzer:
    """
    Multi-factor currency strength analysis.
    """
    
    def __init__(self):
        """Initialize currency strength analyzer."""
        self.strength_history: Dict[str, List[float]] = {}
        self.momentum_emas: Dict[str, EMA] = {}
        
    def analyze_strength(self, risk_data: Dict[str, float], policy_data: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze relative currency strength.
        
        Args:
            risk_data: Risk index data
            policy_data: Policy rates data
            
        Returns:
            Currency strength scores
        """
        # Currency pairs and their components
        currencies = {
            'USD': {
                'rate': policy_data['FED_FUNDS'],
                'pairs': {'EURUSD': -1, 'GBPUSD': -1, 'USDJPY': 1},  # -1 = USD is quote, 1 = USD is base
                'safe_haven_factor': 0.3
            },
            'EUR': {
                'rate': policy_data['ECB_RATE'],
                'pairs': {'EURUSD': 1},
                'safe_haven_factor': 0.1
            },
            'GBP': {
                'rate': policy_data['BOE_RATE'],
                'pairs': {'GBPUSD': 1},
                'safe_haven_factor': 0.0
            },
            'JPY': {
                'rate': policy_data['BOJ_RATE'],
                'pairs': {'USDJPY': -1},
                'safe_haven_factor': 0.4
            }
        }
        
        strength_scores = {}
        
        for currency, data in currencies.items():
            # Initialize momentum EMA if needed
            if currency not in self.momentum_emas:
                self.momentum_emas[currency] = EMA(14)
            
            # Rate differential component
            rate_differential = data['rate'] - policy_data['FED_FUNDS'] if currency != 'USD' else 0.0
            
            # Price momentum component
            price_momentum = 0.0
            pair_count = 0
            for pair, direction in data['pairs'].items():
                if pair in risk_data:
                    # Calculate momentum for this pair
                    if currency not in self.strength_history:
                        self.strength_history[currency] = []
                    
                    current_price = risk_data[pair]
                    if len(self.strength_history[currency]) > 0:
                        prev_price = self.strength_history[currency][-1]
                        momentum = (current_price - prev_price) / prev_price * direction
                        price_momentum += momentum
                        pair_count += 1
            
            if pair_count > 0:
                price_momentum /= pair_count
            
            # Risk sentiment component
            risk_sentiment = 0.0
            if 'VIX' in risk_data:
                vix_normalized = (risk_data['VIX'] - 20) / 20  # Normalize around 20
                risk_sentiment = -vix_normalized * data['safe_haven_factor']
            
            # Combine components
            total_strength = (
                rate_differential * 0.4 +
                price_momentum * 0.4 +
                risk_sentiment * 0.2
            )
            
            # Update momentum EMA
            self.momentum_emas[currency].update(total_strength)
            
            # Store for history
            if currency not in self.strength_history:
                self.strength_history[currency] = []
            self.strength_history[currency].append(risk_data.get(f'{currency}USD', 1.0))
            if len(self.strength_history[currency]) > 100:
                self.strength_history[currency].pop(0)
            
            strength_scores[currency] = normalize_signal(total_strength, -0.1, 0.1)
        
        return strength_scores


class WHYEngine(DimensionalSensor):
    """
    Masterful WHY dimension engine for fundamental analysis.
    Eliminates all mock data and implements sophisticated macro analysis.
    """
    
    def __init__(self, instrument_meta: InstrumentMeta, data_dir: Optional[Path] = None):
        """
        Initialize WHY engine.
        
        Args:
            instrument_meta: Instrument metadata
            data_dir: Directory containing economic data files
        """
        super().__init__(instrument_meta)
        
        # Initialize data provider
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_provider = EconomicDataProvider(data_dir)
        
        # Initialize analyzers
        self.yield_curve_analyzer = YieldCurveAnalyzer()
        self.currency_strength_analyzer = CurrencyStrengthAnalyzer()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # State variables
        self.last_analysis_time = None
        self.analysis_cache = {}
        self.regime_detector = EMA(20)
        
        logger.info(f"WHY Engine initialized for {instrument_meta.symbol}")
    
    async def update(self, market_data: MarketData) -> DimensionalReading:
        """
        Process market data and generate fundamental analysis.
        
        Args:
            market_data: Latest market data
            
        Returns:
            Dimensional reading with fundamental analysis
        """
        start_time = datetime.utcnow()
        
        try:
            # Get economic data
            yield_curve = await self.data_provider.get_yield_curve(market_data.timestamp)
            risk_indexes = await self.data_provider.get_risk_indexes(market_data.timestamp)
            policy_rates = await self.data_provider.get_policy_rates(market_data.timestamp)
            
            # Validate data availability
            data_quality = self._assess_data_quality(yield_curve, risk_indexes, policy_rates)
            
            if data_quality < 0.3:
                logger.warning("Insufficient economic data quality for analysis")
                return self._create_low_confidence_reading(market_data.timestamp, data_quality)
            
            # Perform fundamental analysis
            analysis_results = await self._perform_fundamental_analysis(
                yield_curve, risk_indexes, policy_rates, market_data
            )
            
            # Generate signal and confidence
            signal_strength = self._calculate_signal_strength(analysis_results)
            confidence = self._calculate_confidence(analysis_results, data_quality)
            
            # Detect market regime
            regime = self._detect_market_regime(analysis_results)
            
            # Create dimensional reading
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            reading = DimensionalReading(
                dimension="WHY",
                timestamp=market_data.timestamp,
                signal_strength=signal_strength,
                confidence=confidence,
                regime=regime,
                context=analysis_results,
                data_quality=data_quality,
                processing_time_ms=processing_time,
                evidence=self._extract_evidence(analysis_results),
                warnings=self._generate_warnings(analysis_results)
            )
            
            self.last_reading = reading
            self.is_initialized = True
            
            logger.debug(f"WHY analysis complete: signal={signal_strength:.3f}, "
                        f"confidence={confidence:.3f}, regime={regime}")
            
            return reading
            
        except Exception as e:
            logger.error(f"Error in WHY engine update: {e}")
            return self._create_error_reading(market_data.timestamp, str(e))
    
    async def _perform_fundamental_analysis(
        self,
        yield_curve: Dict[str, float],
        risk_indexes: Dict[str, float],
        policy_rates: Dict[str, float],
        market_data: MarketData
    ) -> Dict[str, any]:
        """
        Perform comprehensive fundamental analysis.
        
        Args:
            yield_curve: Yield curve data
            risk_indexes: Risk index data
            policy_rates: Policy rates data
            market_data: Current market data
            
        Returns:
            Analysis results
        """
        results = {}
        
        # Yield curve analysis
        if yield_curve:
            results['yield_curve'] = self.yield_curve_analyzer.analyze_curve(yield_curve)
        
        # Currency strength analysis
        if risk_indexes and policy_rates:
            results['currency_strength'] = self.currency_strength_analyzer.analyze_strength(
                risk_indexes, policy_rates
            )
        
        # Risk sentiment analysis
        if risk_indexes:
            results['risk_sentiment'] = self._analyze_risk_sentiment(risk_indexes)
        
        # Policy divergence analysis
        if policy_rates:
            results['policy_divergence'] = self._analyze_policy_divergence(
                policy_rates, self.instrument_meta.symbol
            )
        
        # Economic momentum
        results['economic_momentum'] = self._calculate_economic_momentum(results)
        
        return results
    
    def _analyze_risk_sentiment(self, risk_data: Dict[str, float]) -> Dict[str, float]:
        """Analyze risk sentiment from market indicators."""
        vix = risk_data.get('VIX', 20.0)
        gold = risk_data.get('GOLD', 2000.0)
        spx = risk_data.get('SPX', 4500.0)
        
        # VIX analysis (fear gauge)
        vix_signal = normalize_signal(vix, 10, 40)  # 10-40 typical range
        
        # Gold analysis (safe haven)
        gold_momentum = 0.0
        if hasattr(self, '_prev_gold'):
            gold_momentum = (gold - self._prev_gold) / self._prev_gold
        self._prev_gold = gold
        
        # Equity analysis
        spx_momentum = 0.0
        if hasattr(self, '_prev_spx'):
            spx_momentum = (spx - self._prev_spx) / self._prev_spx
        self._prev_spx = spx
        
        return {
            'vix_signal': vix_signal,
            'gold_momentum': normalize_signal(gold_momentum, -0.05, 0.05),
            'equity_momentum': normalize_signal(spx_momentum, -0.05, 0.05),
            'overall_risk_sentiment': (vix_signal + normalize_signal(gold_momentum, -0.05, 0.05)) / 2
        }
    
    def _analyze_policy_divergence(self, policy_rates: Dict[str, float], symbol: str) -> Dict[str, float]:
        """Analyze central bank policy divergence."""
        # Extract currencies from symbol
        if len(symbol) == 6:
            base_currency = symbol[:3]
            quote_currency = symbol[3:]
        else:
            return {'divergence': 0.0, 'base_rate': 0.0, 'quote_rate': 0.0}
        
        # Map currencies to central bank rates
        rate_mapping = {
            'USD': 'FED_FUNDS',
            'EUR': 'ECB_RATE',
            'GBP': 'BOE_RATE',
            'JPY': 'BOJ_RATE',
            'CHF': 'SNB_RATE',
            'CAD': 'BOC_RATE',
            'AUD': 'RBA_RATE',
            'NZD': 'RBNZ_RATE'
        }
        
        base_rate_key = rate_mapping.get(base_currency)
        quote_rate_key = rate_mapping.get(quote_currency)
        
        if not base_rate_key or not quote_rate_key:
            return {'divergence': 0.0, 'base_rate': 0.0, 'quote_rate': 0.0}
        
        base_rate = policy_rates.get(base_rate_key, 0.0)
        quote_rate = policy_rates.get(quote_rate_key, 0.0)
        
        # Calculate rate differential
        rate_differential = base_rate - quote_rate
        
        return {
            'divergence': normalize_signal(rate_differential, -5.0, 5.0),
            'base_rate': base_rate,
            'quote_rate': quote_rate,
            'differential': rate_differential
        }
    
    def _calculate_economic_momentum(self, analysis_results: Dict[str, any]) -> float:
        """Calculate overall economic momentum score."""
        momentum_factors = []
        
        # Yield curve momentum
        if 'yield_curve' in analysis_results:
            curve_data = analysis_results['yield_curve']
            momentum_factors.append(curve_data.get('slope_trend', 0.0))
        
        # Currency strength momentum
        if 'currency_strength' in analysis_results:
            strength_data = analysis_results['currency_strength']
            avg_strength = np.mean(list(strength_data.values()))
            momentum_factors.append(avg_strength)
        
        # Risk sentiment momentum
        if 'risk_sentiment' in analysis_results:
            risk_data = analysis_results['risk_sentiment']
            momentum_factors.append(risk_data.get('overall_risk_sentiment', 0.0))
        
        if not momentum_factors:
            return 0.0
        
        return np.mean(momentum_factors)
    
    def _calculate_signal_strength(self, analysis_results: Dict[str, any]) -> float:
        """Calculate overall signal strength from analysis results."""
        signals = []
        
        # Policy divergence signal (strongest factor)
        if 'policy_divergence' in analysis_results:
            signals.append(analysis_results['policy_divergence']['divergence'] * 0.4)
        
        # Currency strength signal
        if 'currency_strength' in analysis_results:
            strength_data = analysis_results['currency_strength']
            # For EURUSD, compare EUR vs USD strength
            if self.instrument_meta.symbol == 'EURUSD':
                eur_strength = strength_data.get('EUR', 0.0)
                usd_strength = strength_data.get('USD', 0.0)
                signals.append((eur_strength - usd_strength) * 0.3)
        
        # Yield curve signal
        if 'yield_curve' in analysis_results:
            curve_data = analysis_results['yield_curve']
            inversion_signal = -curve_data.get('inversion_score', 0.0)  # Inversion is bearish
            signals.append(inversion_signal * 0.2)
        
        # Economic momentum
        momentum = analysis_results.get('economic_momentum', 0.0)
        signals.append(momentum * 0.1)
        
        if not signals:
            return 0.0
        
        return np.clip(sum(signals), -1.0, 1.0)
    
    def _calculate_confidence(self, analysis_results: Dict[str, any], data_quality: float) -> float:
        """Calculate confidence in the fundamental analysis."""
        # Base confidence on data quality
        base_confidence = data_quality
        
        # Adjust for signal clarity
        signal_strength = abs(self._calculate_signal_strength(analysis_results))
        signal_clarity = signal_strength
        
        # Adjust for confluence (agreement between different factors)
        confluence_factors = []
        if 'policy_divergence' in analysis_results:
            confluence_factors.append(analysis_results['policy_divergence']['divergence'])
        if 'currency_strength' in analysis_results and self.instrument_meta.symbol == 'EURUSD':
            strength_data = analysis_results['currency_strength']
            eur_strength = strength_data.get('EUR', 0.0)
            usd_strength = strength_data.get('USD', 0.0)
            confluence_factors.append(eur_strength - usd_strength)
        
        confluence = 1.0
        if len(confluence_factors) >= 2:
            # Check if signals agree (same sign)
            signs = [1 if x > 0 else -1 if x < 0 else 0 for x in confluence_factors]
            agreement = len(set(signs)) == 1 if signs else False
            confluence = 0.8 if agreement else 0.4
        
        return compute_confidence(
            signal_strength=signal_strength,
            data_quality=base_confidence,
            historical_accuracy=self.performance_tracker.get_accuracy(),
            confluence_signals=confluence_factors
        )
    
    def _detect_market_regime(self, analysis_results: Dict[str, any]) -> MarketRegime:
        """Detect current market regime from fundamental factors."""
        # Risk sentiment analysis
        risk_sentiment = 0.0
        if 'risk_sentiment' in analysis_results:
            risk_sentiment = analysis_results['risk_sentiment'].get('overall_risk_sentiment', 0.0)
        
        # Volatility proxy from VIX
        volatility = 0.5  # Default moderate
        if 'risk_sentiment' in analysis_results:
            vix_signal = analysis_results['risk_sentiment'].get('vix_signal', 0.0)
            volatility = (vix_signal + 1.0) / 2.0  # Convert -1,1 to 0,1
        
        # Economic momentum
        momentum = analysis_results.get('economic_momentum', 0.0)
        
        # Regime detection logic
        if volatility > 0.8:
            return MarketRegime.EXHAUSTED
        elif abs(momentum) > 0.6 and volatility < 0.4:
            return MarketRegime.TRENDING_STRONG
        elif abs(momentum) > 0.3:
            return MarketRegime.TRENDING_WEAK
        elif volatility < 0.3 and abs(momentum) < 0.2:
            return MarketRegime.CONSOLIDATING
        else:
            return MarketRegime.CONSOLIDATING
    
    def _assess_data_quality(self, yield_curve, risk_indexes, policy_rates) -> float:
        """Assess quality of available economic data."""
        quality_score = 0.0
        total_weight = 0.0
        
        # Yield curve data quality
        if yield_curve:
            quality_score += 0.4
        total_weight += 0.4
        
        # Risk indexes data quality
        if risk_indexes:
            quality_score += 0.4
        total_weight += 0.4
        
        # Policy rates data quality
        if policy_rates:
            quality_score += 0.2
        total_weight += 0.2
        
        return quality_score / total_weight if total_weight > 0 else 0.0
    
    def _extract_evidence(self, analysis_results: Dict[str, any]) -> Dict[str, float]:
        """Extract evidence scores for transparency."""
        evidence = {}
        
        if 'policy_divergence' in analysis_results:
            evidence['policy_divergence'] = abs(analysis_results['policy_divergence']['divergence'])
        
        if 'yield_curve' in analysis_results:
            evidence['yield_curve_slope'] = abs(analysis_results['yield_curve'].get('slope_trend', 0.0))
        
        if 'risk_sentiment' in analysis_results:
            evidence['risk_sentiment'] = abs(analysis_results['risk_sentiment'].get('overall_risk_sentiment', 0.0))
        
        evidence['economic_momentum'] = abs(analysis_results.get('economic_momentum', 0.0))
        
        return evidence
    
    def _generate_warnings(self, analysis_results: Dict[str, any]) -> List[str]:
        """Generate warnings about analysis quality or concerns."""
        warnings = []
        
        # Check for yield curve inversion
        if 'yield_curve' in analysis_results:
            inversion_score = analysis_results['yield_curve'].get('inversion_score', 0.0)
            if inversion_score > 0.01:  # 1bp inversion
                warnings.append(f"Yield curve inversion detected: {inversion_score:.3f}")
        
        # Check for extreme VIX levels
        if 'risk_sentiment' in analysis_results:
            vix_signal = analysis_results['risk_sentiment'].get('vix_signal', 0.0)
            if abs(vix_signal) > 0.8:
                warnings.append(f"Extreme VIX level detected: {vix_signal:.3f}")
        
        return warnings
    
    def _create_low_confidence_reading(self, timestamp: datetime, data_quality: float) -> DimensionalReading:
        """Create reading when data quality is insufficient."""
        return DimensionalReading(
            dimension="WHY",
            timestamp=timestamp,
            signal_strength=0.0,
            confidence=0.1,
            regime=MarketRegime.CONSOLIDATING,
            context={'error': 'insufficient_data_quality'},
            data_quality=data_quality,
            processing_time_ms=0.0,
            evidence={},
            warnings=['Insufficient economic data quality for reliable analysis']
        )
    
    def _create_error_reading(self, timestamp: datetime, error_msg: str) -> DimensionalReading:
        """Create reading when error occurs."""
        return DimensionalReading(
            dimension="WHY",
            timestamp=timestamp,
            signal_strength=0.0,
            confidence=0.0,
            regime=MarketRegime.CONSOLIDATING,
            context={'error': error_msg},
            data_quality=0.0,
            processing_time_ms=0.0,
            evidence={},
            warnings=[f'Analysis error: {error_msg}']
        )
    
    def snapshot(self) -> DimensionalReading:
        """Return current dimensional state."""
        if self.last_reading:
            return self.last_reading
        
        return DimensionalReading(
            dimension="WHY",
            timestamp=datetime.utcnow(),
            signal_strength=0.0,
            confidence=0.0,
            regime=MarketRegime.CONSOLIDATING,
            context={},
            data_quality=0.0,
            processing_time_ms=0.0,
            evidence={},
            warnings=['Engine not initialized']
        )
    
    def reset(self) -> None:
        """Reset engine state."""
        self.last_reading = None
        self.is_initialized = False
        self.last_analysis_time = None
        self.analysis_cache.clear()
        
        # Reset analyzers
        self.yield_curve_analyzer = YieldCurveAnalyzer()
        self.currency_strength_analyzer = CurrencyStrengthAnalyzer()
        self.regime_detector = EMA(20)
        
        logger.info("WHY Engine reset completed")

