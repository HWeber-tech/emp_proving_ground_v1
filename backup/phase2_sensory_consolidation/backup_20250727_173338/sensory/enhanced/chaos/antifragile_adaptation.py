"""
Chaos Adaptation System - CHAOS Dimension
========================================

Antifragile adaptation and chaos handling.
Provides the CHAOS dimension of the 5D+1 sensory cortex.

Author: EMP Development Team
Phase: 2 - Truth-First Completion
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BlackSwanDetection:
    """Black swan event detection"""
    detected: bool
    probability: float
    severity: str
    affected_markets: List[str]
    duration_estimate: timedelta


@dataclass
class VolatilityHarvesting:
    """Volatility harvesting opportunities"""
    opportunity_score: float
    optimal_strategies: List[str]
    risk_adjusted_return: float
    implementation_complexity: str


@dataclass
class CrisisAlpha:
    """Crisis alpha opportunities"""
    alpha_potential: float
    crisis_type: str
    strategy_recommendations: List[str]
    confidence: float


@dataclass
class ChaosAdaptation:
    """Antifragile adaptation and chaos handling results"""
    black_swan: BlackSwanDetection
    volatility_harvesting: VolatilityHarvesting
    crisis_alpha: CrisisAlpha
    regime_change: Dict[str, Any]
    antifragile_strategies: List[str]
    overall_adaptation_score: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class ChaosAdaptationSystem:
    """
    Implements the CHAOS dimension of the 5D+1 sensory cortex.
    Provides antifragile adaptation and chaos handling capabilities.
    """
    
    def __init__(self):
        self.black_swan_detector = BlackSwanDetector()
        self.volatility_harvester = VolatilityHarvester()
        self.crisis_alpha_finder = CrisisAlphaFinder()
        self.regime_change_detector = RegimeChangeDetector()
        self.logger = logging.getLogger(__name__)
    
    async def assess_chaos_opportunities(self, market_data: pd.DataFrame) -> ChaosAdaptation:
        """
        Assess chaos opportunities and antifragile adaptation
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            ChaosAdaptation: Chaos adaptation results
        """
        return await self.analyze_chaos_adaptation(market_data)
    
    async def analyze_chaos_adaptation(self, market_data: pd.DataFrame) -> ChaosAdaptation:
        """
        Analyze chaos adaptation and antifragile opportunities
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            ChaosAdaptation: Chaos adaptation results
        """
        try:
            # Detect black swan events
            black_swan = await self.black_swan_detector.detect_black_swan(market_data)
            
            # Identify volatility harvesting opportunities
            volatility_harvesting = await self.volatility_harvester.identify_opportunities(market_data)
            
            # Find crisis alpha opportunities
            crisis_alpha = await self.crisis_alpha_finder.find_opportunities(market_data)
            
            # Detect regime changes
            regime_change = await self.regime_change_detector.detect_changes(market_data)
            
            # Generate antifragile strategies
            antifragile_strategies = await self._generate_antifragile_strategies(
                black_swan, volatility_harvesting, crisis_alpha
            )
            
            # Calculate overall adaptation score
            adaptation_score = self._calculate_adaptation_score(
                black_swan, volatility_harvesting, crisis_alpha
            )
            
            # Calculate confidence
            confidence = self._calculate_chaos_confidence(
                black_swan, volatility_harvesting, crisis_alpha
            )
            
            return ChaosAdaptation(
                black_swan=black_swan,
                volatility_harvesting=volatility_harvesting,
                crisis_alpha=crisis_alpha,
                regime_change=regime_change,
                antifragile_strategies=antifragile_strategies,
                overall_adaptation_score=adaptation_score,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Chaos adaptation analysis failed: {e}")
            return self._get_fallback_chaos_adaptation()
    
    def _calculate_adaptation_score(self, black_swan: BlackSwanDetection, 
                                  volatility_harvesting: VolatilityHarvesting,
                                  crisis_alpha: CrisisAlpha) -> float:
        """Calculate overall adaptation score"""
        factors = [
            1.0 - black_swan.probability,
            volatility_harvesting.opportunity_score,
            crisis_alpha.alpha_potential
        ]
        return min(max(np.mean(factors), 0.0), 1.0)
    
    def _calculate_chaos_confidence(self, black_swan: BlackSwanDetection,
                                  volatility_harvesting: VolatilityHarvesting,
                                  crisis_alpha: CrisisAlpha) -> float:
        """Calculate chaos confidence score"""
        confidences = [
            0.8 if black_swan.detected else 0.9,
            volatility_harvesting.opportunity_score,
            crisis_alpha.confidence
        ]
        return min(max(np.mean(confidences), 0.0), 1.0)
    
    async def _generate_antifragile_strategies(self, black_swan: BlackSwanDetection,
                                             volatility_harvesting: VolatilityHarvesting,
                                             crisis_alpha: CrisisAlpha) -> List[str]:
        """Generate antifragile strategies based on chaos analysis"""
        strategies = []
        
        if black_swan.detected:
            strategies.extend([
                'increase_cash_position',
                'reduce_leverage',
                'implement_tail_risk_hedging'
            ])
        
        if volatility_harvesting.opportunity_score > 0.5:
            strategies.extend([
                'volatility_selling_strategies',
                'gamma_scalping',
                'variance_swap_positions'
            ])
        
        if crisis_alpha.alpha_potential > 0.3:
            strategies.extend([
                'crisis_alpha_strategies',
                'momentum_reversal_trades',
                'liquidity_premium_capture'
            ])
        
        if not strategies:
            strategies = ['maintain_diversified_portfolio', 'monitor_market_conditions']
        
        return strategies
    
    def _get_fallback_chaos_adaptation(self) -> ChaosAdaptation:
        """Return fallback chaos adaptation"""
        return ChaosAdaptation(
            black_swan=BlackSwanDetection(
                detected=False,
                probability=0.1,
                severity='low',
                affected_markets=[],
                duration_estimate=timedelta(hours=1)
            ),
            volatility_harvesting=VolatilityHarvesting(
                opportunity_score=0.2,
                optimal_strategies=['basic_volatility_selling'],
                risk_adjusted_return=0.05,
                implementation_complexity='low'
            ),
            crisis_alpha=CrisisAlpha(
                alpha_potential=0.1,
                crisis_type='none',
                strategy_recommendations=['maintain_positions'],
                confidence=0.8
            ),
            regime_change={'detected': False, 'type': 'none'},
            antifragile_strategies=['maintain_diversified_portfolio'],
            overall_adaptation_score=0.3,
            confidence=0.7
        )


class BlackSwanDetector:
    """Detects black swan events"""
    
    async def detect_black_swan(self, data: pd.DataFrame) -> BlackSwanDetection:
        """Detect black swan events"""
        try:
            if data.empty:
                return self._get_fallback_black_swan()
            
            # Simple black swan detection
            volatility = self._calculate_volatility(data)
            price_change = self._calculate_price_change(data)
            
            # Detect extreme events
            is_black_swan = volatility > 0.1 or abs(price_change) > 0.15
            
            return BlackSwanDetection(
                detected=is_black_swan,
                probability=0.3 if is_black_swan else 0.05,
                severity='high' if is_black_swan else 'low',
                affected_markets=['equity', 'fx', 'crypto'] if is_black_swan else [],
                duration_estimate=timedelta(days=3) if is_black_swan else timedelta(hours=1)
            )
            
        except Exception as e:
            logger.error(f"Black swan detection failed: {e}")
            return self._get_fallback_black_swan()
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility for black swan detection"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        return returns.std() if len(returns) > 0 else 0.0
    
    def _calculate_price_change(self, data: pd.DataFrame) -> float:
        """Calculate price change for black swan detection"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        
        return (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
    
    def _get_fallback_black_swan(self) -> BlackSwanDetection:
        """Return fallback black swan detection"""
        return BlackSwanDetection(
            detected=False,
            probability=0.05,
            severity='low',
            affected_markets=[],
            duration_estimate=timedelta(hours=1)
        )


class VolatilityHarvester:
    """Identifies volatility harvesting opportunities"""
    
    async def identify_opportunities(self, data: pd.DataFrame) -> VolatilityHarvesting:
        """Identify volatility harvesting opportunities"""
        try:
            if data.empty:
                return self._get_fallback_volatility_harvesting()
            
            # Simple volatility harvesting identification
            volatility = self._calculate_volatility(data)
            
            opportunity_score = min(volatility * 10, 1.0)
            
            return VolatilityHarvesting(
                opportunity_score=opportunity_score,
                optimal_strategies=['volatility_selling', 'gamma_scalping'] if opportunity_score > 0.5 else ['basic_strategies'],
                risk_adjusted_return=opportunity_score * 0.1,
                implementation_complexity='medium' if opportunity_score > 0.5 else 'low'
            )
            
        except Exception as e:
            logger.error(f"Volatility harvesting identification failed: {e}")
            return self._get_fallback_volatility_harvesting()
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility for harvesting opportunities"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        return returns.std() if len(returns) > 0 else 0.0
    
    def _get_fallback_volatility_harvesting(self) -> VolatilityHarvesting:
        """Return fallback volatility harvesting"""
        return VolatilityHarvesting(
            opportunity_score=0.2,
            optimal_strategies=['basic_volatility_selling'],
            risk_adjusted_return=0.05,
            implementation_complexity='low'
        )


class CrisisAlphaFinder:
    """Finds crisis alpha opportunities"""
    
    async def find_opportunities(self, data: pd.DataFrame) -> CrisisAlpha:
        """Find crisis alpha opportunities"""
        try:
            if data.empty:
                return self._get_fallback_crisis_alpha()
            
            # Simple crisis alpha identification
            volatility = self._calculate_volatility(data)
            price_change = self._calculate_price_change(data)
            
            alpha_potential = min(abs(price_change) * 2, 1.0)
            
            return CrisisAlpha(
                alpha_potential=alpha_potential,
                crisis_type='volatility_spike' if volatility > 0.05 else 'none',
                strategy_recommendations=['momentum_reversal', 'liquidity_premium'] if alpha_potential > 0.3 else ['hold'],
                confidence=0.7 if alpha_potential > 0.3 else 0.9
            )
            
        except Exception as e:
            logger.error(f"Crisis alpha finding failed: {e}")
            return self._get_fallback_crisis_alpha()
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility for crisis alpha"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        return returns.std() if len(returns) > 0 else 0.0
    
    def _calculate_price_change(self, data: pd.DataFrame) -> float:
        """Calculate price change for crisis alpha"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        
        return (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
    
    def _get_fallback_crisis_alpha(self) -> CrisisAlpha:
        """Return fallback crisis alpha"""
        return CrisisAlpha(
            alpha_potential=0.1,
            crisis_type='none',
            strategy_recommendations=['maintain_positions'],
            confidence=0.8
        )


class RegimeChangeDetector:
    """Detects regime changes"""
    
    async def detect_changes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect regime changes"""
        try:
            if data.empty:
                return {'detected': False, 'type': 'none'}
            
            # Simple regime change detection
            volatility = self._calculate_volatility(data)
            trend_strength = self._calculate_trend_strength(data)
            
            detected = volatility > 0.05 or abs(trend_strength) > 0.1
            
            return {
                'detected': detected,
                'type': 'volatility_regime' if volatility > 0.05 else 'trend_regime' if abs(trend_strength) > 0.1 else 'none',
                'strength': max(volatility, abs(trend_strength))
            }
            
        except Exception as e:
            logger.error(f"Regime change detection failed: {e}")
            return {'detected': False, 'type': 'none'}
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility for regime change detection"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        return returns.std() if len(returns) > 0 else 0.0
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength for regime change detection"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        
        prices = data['close']
        if len(prices) < 5:
            return 0.0
        
        # Simple trend strength calculation
        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
