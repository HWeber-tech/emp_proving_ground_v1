#!/usr/bin/env python3
"""
Niche Detection System
======================

Identifies and segments market conditions into distinct niches where
specialized predator strategies can excel.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class MarketNiche:
    """Represents a detected market niche."""
    niche_id: str
    regime_type: str
    volatility_range: Tuple[float, float]
    volume_range: Tuple[float, float]
    trend_strength: float
    duration: int
    opportunity_score: float
    risk_level: str
    preferred_species: List[str]


class NicheDetector:
    """Advanced market niche detection system."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=5, random_state=42)
        self.niche_history = []
        
    async def detect_niches(self, market_data: Dict[str, Any]) -> Dict[str, MarketNiche]:
        """Detect and segment market into different niches."""
        if not market_data or 'data' not in market_data:
            return {}
        
        df = pd.DataFrame(market_data['data'])
        if len(df) < 50:
            return {}
        
        # Calculate market features
        features = self._calculate_market_features(df)
        
        # Detect regimes
        regimes = self._detect_regimes(features)
        
        # Identify niches within regimes
        niches = self._identify_niches(features, regimes)
        
        # Score opportunities
        scored_niches = self._score_opportunities(niches)
        
        return scored_niches
    
    def _calculate_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive market features."""
        features = pd.DataFrame()
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std() * np.sqrt(252)
        features['trend_strength'] = self._calculate_trend_strength(df)
        
        # Volume features
        features['volume_ma'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        features['volume_volatility'] = df['volume'].pct_change().rolling(20).std()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['close'])
        features['atr'] = self._calculate_atr(df)
        features['momentum'] = self._calculate_momentum(df)
        
        # Market microstructure
        features['spread'] = (df['high'] - df['low']) / df['close']
        features['efficiency'] = np.abs(df['close'].diff()) / (df['high'] - df['low'])
        
        return features.dropna()
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using linear regression."""
        def rolling_trend(window):
            if len(window) < 10:
                return 0
            x = np.arange(len(window))
            slope, _ = np.polyfit(x, window, 1)
            return slope / np.std(window) if np.std(window) > 0 else 0
        
        return df['close'].rolling(20).apply(
            lambda x: rolling_trend(x[-10:]) if len(x) >= 10 else 0
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum indicator."""
        return df['close'].pct_change(10)
    
    def _detect_regimes(self, features: pd.DataFrame) -> pd.Series:
        """Detect market regimes using clustering."""
        # Select key features for clustering
        cluster_features = features[['volatility', 'trend_strength', 'volume_ratio']].dropna()
        
        if len(cluster_features) < 10:
            return pd.Series(['neutral'] * len(features), index=features.index)
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(cluster_features)
        
        # Cluster market states
        clusters = self.clusterer.fit_predict(scaled_features)
        
        # Map clusters to regimes
        regime_map = {
            0: 'trending_bull',
            1: 'trending_bear',
            2: 'ranging',
            3: 'volatile',
            4: 'quiet'
        }
        
        regimes = pd.Series([regime_map.get(c, 'neutral') for c in clusters], 
                          index=cluster_features.index)
        
        # Extend to full length
        full_regimes = pd.Series(['neutral'] * len(features), index=features.index)
        full_regimes.update(regimes)
        
        return full_regimes
    
    def _identify_niches(self, features: pd.DataFrame, regimes: pd.Series) -> List[Dict[str, Any]]:
        """Identify specific niches within market regimes."""
        niches = []
        
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_data = features[regime_mask]
            
            if len(regime_data) < 10:
                continue
            
            # Calculate regime characteristics
            volatility_range = (regime_data['volatility'].quantile(0.25),
                              regime_data['volatility'].quantile(0.75))
            volume_range = (regime_data['volume_ratio'].quantile(0.25),
                          regime_data['volume_ratio'].quantile(0.75))
            trend_strength = regime_data['trend_strength'].mean()
            
            # Determine preferred species for this niche
            preferred_species = self._determine_preferred_species(regime, volatility_range)
            
            niche = {
                'regime_type': regime,
                'volatility_range': volatility_range,
                'volume_range': volume_range,
                'trend_strength': trend_strength,
                'duration': len(regime_data),
                'preferred_species': preferred_species
            }
            
            niches.append(niche)
        
        return niches
    
    def _determine_preferred_species(self, regime: str, volatility_range: Tuple[float, float]) -> List[str]:
        """Determine which species are best suited for this regime."""
        preferred = []
        
        vol_low, vol_high = volatility_range
        
        if regime == 'trending_bull':
            preferred = ['stalker', 'alpha', 'pack_hunter']
        elif regime == 'trending_bear':
            preferred = ['stalker', 'pack_hunter']
        elif regime == 'ranging':
            if vol_high < 0.02:
                preferred = ['ambusher', 'scavenger']
            else:
                preferred = ['pack_hunter', 'scavenger']
        elif regime == 'volatile':
            if vol_high > 0.05:
                preferred = ['scavenger', 'alpha']
            else:
                preferred = ['pack_hunter', 'ambusher']
        elif regime == 'quiet':
            preferred = ['ambusher', 'scavenger']
        
        return preferred
    
    def _score_opportunities(self, niches: List[Dict[str, Any]]) -> Dict[str, MarketNiche]:
        """Score niches based on opportunity potential."""
        scored_niches = {}
        
        for i, niche_data in enumerate(niches):
            # Calculate opportunity score
            volatility_score = min(niche_data['volatility_range'][1] * 10, 1.0)
            volume_score = min(niche_data['volume_range'][1] * 0.5, 1.0)
            trend_score = abs(niche_data['trend_strength']) * 2
            
            opportunity_score = (volatility_score + volume_score + trend_score) / 3
            
            # Determine risk level
            if niche_data['volatility_range'][1] > 0.05:
                risk_level = 'high'
            elif niche_data['volatility_range'][1] > 0.02:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            niche = MarketNiche(
                niche_id=f"niche_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                regime_type=niche_data['regime_type'],
                volatility_range=niche_data['volatility_range'],
                volume_range=niche_data['volume_range'],
                trend_strength=niche_data['trend_strength'],
                duration=niche_data['duration'],
                opportunity_score=opportunity_score,
                risk_level=risk_level,
                preferred_species=niche_data['preferred_species']
            )
            
            scored_niches[niche.niche_id] = niche
        
        return scored_niches
    
    async def get_current_regime(self, market_data: Dict[str, Any]) -> str:
        """Get current market regime classification."""
        niches = await self.detect_niches(market_data)
        
        if not niches:
            return 'neutral'
        
        # Return the most recent/active niche
        return list(niches.values())[0].regime_type
    
    async def get_species_recommendations(self, market_data: Dict[str, Any]) -> List[str]:
        """Get species recommendations for current market conditions."""
        niches = await self.detect_niches(market_data)
        
        if not niches:
            return ['pack_hunter']  # Default fallback
        
        # Get recommendations from highest-scoring niche
        best_niche = max(niches.values(), key=lambda x: x.opportunity_score)
        return best_niche.preferred_species
    
    def get_niche_history(self) -> List[MarketNiche]:
        """Get historical niche data."""
        return self.niche_history


# Example usage
async def test_niche_detection():
    """Test the niche detection system."""
    import numpy as np
    
    # Generate test market data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    # Create trending market
    trend = np.linspace(1.0, 1.1, 100)
    noise = np.random.normal(0, 0.001, 100)
    prices = trend + noise
    
    test_data = {
        'data': pd.DataFrame({
            'open': prices,
            'high': prices + 0.001,
            'low': prices - 0.001,
            'close': prices,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
    }
    
    detector = NicheDetector()
    niches = await detector.detect_niches(test_data)
    
    print(f"Detected {len(niches)} niches:")
    for niche_id, niche in niches.items():
        print(f"  {niche.regime_type}: score={niche.opportunity_score:.2f}, "
              f"species={niche.preferred_species}")


if __name__ == "__main__":
    asyncio.run(test_niche_detection())
