"""
CHAOS Dimension - Antifragile Adaptation System
=============================================

Advanced antifragile adaptation engine for the 5D+1 sensory cortex.
Implements sophisticated algorithms for:
- Black swan event detection
- Volatility harvesting opportunities
- Crisis alpha potential
- Regime change detection
- Antifragile strategy adaptation

Author: EMP Development Team
Phase: 2 - Truth-First Completion
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class BlackSwanDetection:
    """Black swan event detection results"""
    detected: bool
    confidence: float
    event_type: str  # 'market_crash', 'flash_crash', 'liquidity_crisis'
    severity: float
    expected_impact: float
    duration_estimate: timedelta


@dataclass
class VolatilityHarvesting:
    """Volatility harvesting opportunities"""
    opportunity_detected: bool
    confidence: float
    optimal_strategy: str
    expected_return: float
    risk_adjusted_return: float
    holding_period: timedelta


@dataclass
class CrisisAlpha:
    """Crisis alpha potential"""
    alpha_detected: bool
    confidence: float
    alpha_source: str  # 'volatility_premium', 'liquidity_premium', 'risk_premium'
    expected_alpha: float
    implementation_cost: float
    regulatory_considerations: List[str]


@dataclass
class RegimeChange:
    """Regime change detection"""
    change_detected: bool
    confidence: float
    from_regime: str
    to_regime: str
    transition_probability: float
    adaptation_required: bool


@dataclass
class AntifragileStrategy:
    """Antifragile strategy recommendations"""
    strategy_type: str
    confidence: float
    expected_benefit: float
    risk_level: str
    implementation_complexity: str
    monitoring_requirements: List[str]


@dataclass
class ChaosAdaptation:
    """Complete chaos adaptation analysis"""
    black_swan: BlackSwanDetection
    volatility_harvesting: VolatilityHarvesting
    crisis_alpha: CrisisAlpha
    regime_change: RegimeChange
    antifragile_strategies: List[AntifragileStrategy]
    overall_adaptation_score: float
    confidence: float


class ChaosAdaptationSystem:
    """Advanced antifragile adaptation system."""
    
    def __init__(self):
        self.black_swan_detector = BlackSwanDetector()
        self.volatility_harvester = VolatilityHarvester()
        self.crisis_alpha_detector = CrisisAlphaDetector()
        self.regime_detector = RegimeChangeDetector()
        self.strategy_optimizer = AntifragileStrategyOptimizer()
        
    async def assess_chaos_opportunities(self, market_data: pd.DataFrame) -> ChaosAdaptation:
        """Comprehensive chaos adaptation analysis."""
        try:
            if market_data.empty:
                return self._get_fallback_adaptation()
            
            # Detect black swan events
            black_swan = await self.black_swan_detector.detect_black_swan(market_data)
            
            # Identify volatility harvesting opportunities
            volatility_harvesting = await self.volatility_harvester.identify_opportunities(market_data)
            
            # Detect crisis alpha potential
            crisis_alpha = await self.crisis_alpha_detector.detect_alpha(market_data)
            
            # Detect regime changes
            regime_change = await self.regime_detector.detect_change(market_data)
            
            # Generate antifragile strategies
            antifragile_strategies = await self.strategy_optimizer.generate_strategies(
                black_swan, volatility_harvesting, crisis_alpha, regime_change
            )
            
            # Calculate overall adaptation score
            adaptation_score = self._calculate_adaptation_score(
                black_swan, volatility_harvesting, crisis_alpha, regime_change, antifragile_strategies
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(
                black_swan, volatility_harvesting, crisis_alpha, regime_change, antifragile_strategies
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
            logger.error(f"Chaos adaptation analysis failed: {e}")
            return self._get_fallback_adaptation()
    
    def _calculate_adaptation_score(self, black_swan: BlackSwanDetection,
                                  volatility_harvesting: VolatilityHarvesting,
                                  crisis_alpha: CrisisAlpha,
                                  regime_change: RegimeChange,
                                  strategies: List[AntifragileStrategy]) -> float:
        """Calculate overall adaptation score"""
        factors = [
            1.0 if black_swan.detected else 0.0,
            1.0 if volatility_harvesting.opportunity_detected else 0.0,
            1.0 if crisis_alpha.alpha_detected else 0.0,
            1.0 if regime_change.change_detected else 0.0,
            min(len(strategies) / 3.0, 1.0)
        ]
        return np.mean(factors)
    
    def _calculate_confidence(self, black_swan: BlackSwanDetection,
                            volatility_harvesting: VolatilityHarvesting,
                            crisis_alpha: CrisisAlpha,
                            regime_change: RegimeChange,
                            strategies: List[AntifragileStrategy]) -> float:
        """Calculate overall confidence"""
        confidences = [
            black_swan.confidence,
            volatility_harvesting.confidence,
            crisis_alpha.confidence,
            regime_change.confidence,
            np.mean([s.confidence for s in strategies]) if strategies else 0.0
        ]
        return np.mean([c for c in confidences if c > 0])
    
    def _get_fallback_adaptation(self) -> ChaosAdaptation:
        """Return fallback chaos adaptation"""
        return ChaosAdaptation(
            black_swan=BlackSwanDetection(
                detected=False,
                confidence=0.0,
                event_type='none',
                severity=0.0,
                expected_impact=0.0,
                duration_estimate=timedelta(hours=1)
            ),
            volatility_harvesting=VolatilityHarvesting(
                opportunity_detected=False,
                confidence=0.0,
                optimal_strategy='hold',
                expected_return=0.0,
                risk_adjusted_return=0.0,
                holding_period=timedelta(hours=1)
            ),
            crisis_alpha=CrisisAlpha(
                alpha_detected=False,
                confidence=0.0,
                alpha_source='none',
                expected_alpha=0.0,
                implementation_cost=0.0,
                regulatory_considerations=[]
            ),
            regime_change=RegimeChange(
                change_detected=False,
                confidence=0.0,
                from_regime='normal',
                to_regime='normal',
                transition_probability=0.0,
                adaptation_required=False
            ),
            antifragile_strategies=[],
            overall_adaptation_score=0.0,
            confidence=0.1
        )


class BlackSwanDetector:
    """Black swan event detection"""
    
    async def detect_black_swan(self, market_data: pd.DataFrame) -> BlackSwanDetection:
        """Detect black swan events in market data"""
        try:
            if market_data.empty or len(market_data) < 20:
                return BlackSwanDetection(
                    detected=False,
                    confidence=0.0,
                    event_type='none',
                    severity=0.0,
                    expected_impact=0.0,
                    duration_estimate=timedelta(hours=1)
                )
            
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < 10:
                return BlackSwanDetection(
                    detected=False,
                    confidence=0.0,
                    event_type='none',
                    severity=0.0,
                    expected_impact=0.0,
                    duration_estimate=timedelta(hours=1)
                )
            
            # Detect extreme returns (beyond 3 standard deviations)
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return == 0:
                return BlackSwanDetection(
                    detected=False,
                    confidence=0.0,
                    event_type='none',
                    severity=0.0,
                    expected_impact=0.0,
                    duration_estimate=timedelta(hours=1)
                )
            
            z_scores = abs(returns - mean_return) / std_return
            extreme_returns = z_scores > 3.0
            
            if extreme_returns.any():
                max_z_score = z_scores.max()
                severity = min(max_z_score / 5.0, 1.0)
                
                # Determine event type
                if max_z_score > 5.0:
                    event_type = 'market_crash'
                elif max_z_score > 4.0:
                    event_type = 'flash_crash'
                else:
                    event_type = 'liquidity_crisis'
                
                return BlackSwanDetection(
                    detected=True,
                    confidence=min(severity * 0.8, 1.0),
                    event_type=event_type,
                    severity=severity,
                    expected_impact=severity * 0.1,
                    duration_estimate=timedelta(hours=24)
                )
            
            return BlackSwanDetection(
                detected=False,
                confidence=0.0,
                event_type='none',
                severity=0.0,
                expected_impact=0.0,
                duration_estimate=timedelta(hours=1)
            )
            
        except Exception as e:
            logger.error(f"Black swan detection failed: {e}")
            return BlackSwanDetection(
                detected=False,
                confidence=0.0,
                event_type='none',
                severity=0.0,
                expected_impact=0.0,
                duration_estimate=timedelta(hours=1)
            )


class VolatilityHarvester:
    """Volatility harvesting opportunity identification"""
    
    async def identify_opportunities(self, market_data: pd.DataFrame) -> VolatilityHarvesting:
        """Identify volatility harvesting opportunities"""
        try:
            if market_data.empty or len(market_data) < 20:
                return VolatilityHarvesting(
                    opportunity_detected=False,
                    confidence=0.0,
                    optimal_strategy='hold',
                    expected_return=0.0,
                    risk_adjusted_return=0.0,
                    holding_period=timedelta(hours=1)
                )
            
            # Calculate volatility metrics
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < 10:
                return VolatilityHarvesting(
                    opportunity_detected=False,
                    confidence=0.0,
                    optimal_strategy='hold',
                    expected_return=0.0,
                    risk_adjusted_return=0.0,
                    holding_period=timedelta(hours=1)
                )
            
            current_volatility = returns.rolling(10).std().iloc[-1]
            historical_volatility = returns.std()
            
            # Detect volatility spikes
            volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1.0
            
            if volatility_ratio > 2.0:
                confidence = min((volatility_ratio - 1.0) / 2.0, 1.0)
                
                # Determine optimal strategy
                if volatility_ratio > 3.0:
                    optimal_strategy = 'straddle'
                elif volatility_ratio > 2.5:
                    optimal_strategy = 'strangle'
                else:
                    optimal_strategy = 'iron_condor'
                
                # Calculate expected returns
                expected_return = min(volatility_ratio * 0.05, 0.2)
                risk_adjusted_return = expected_return / volatility_ratio if volatility_ratio > 0 else 0.0
                
                return VolatilityHarvesting(
                    opportunity_detected=True,
                    confidence=confidence,
                    optimal_strategy=optimal_strategy,
                    expected_return=expected_return,
                    risk_adjusted_return=risk_adjusted_return,
                    holding_period=timedelta(days=7)
                )
            
            return VolatilityHarvesting(
                opportunity_detected=False,
                confidence=0.0,
                optimal_strategy='hold',
                expected_return=0.0,
                risk_adjusted_return=0.0,
                holding_period=timedelta(hours=1)
            )
            
        except Exception as e:
            logger.error(f"Volatility harvesting identification failed: {e}")
            return VolatilityHarvesting(
                opportunity_detected=False,
                confidence=0.0,
                optimal_strategy='hold',
                expected_return=0.0,
                risk_adjusted_return=0.0,
                holding_period=timedelta(hours=1)
            )


class CrisisAlphaDetector:
    """Crisis alpha detection"""
    
    async def detect_alpha(self, market_data: pd.DataFrame) -> CrisisAlpha:
        """Detect crisis alpha potential"""
        try:
            if market_data.empty or len(market_data) < 30:
                return CrisisAlpha(
                    alpha_detected=False,
                    confidence=0.0,
                    alpha_source='none',
                    expected_alpha=0.0,
                    implementation_cost=0.0,
                    regulatory_considerations=[]
                )
            
            # Calculate crisis indicators
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < 20:
                return CrisisAlpha(
                    alpha_detected=False,
                    confidence=0.0,
                    alpha_source='none',
                    expected_alpha=0.0,
                    implementation_cost=0.0,
                    regulatory_considerations=[]
                )
            
            # Detect crisis conditions
            rolling_returns = returns.rolling(10)
            crisis_threshold = -0.05
            
            crisis_periods = (rolling_returns.mean() < crisis_threshold).sum()
            total_periods = len(returns) - 9
            
            crisis_ratio = crisis_periods / total_periods if total_periods > 0 else 0.0
            
            if crisis_ratio > 0.1:
                confidence = min(crisis_ratio * 2.0, 1.0)
                
                # Determine alpha source
                if crisis_ratio > 0.3:
                    alpha_source = 'volatility_premium'
                elif crisis_ratio > 0.2:
                    alpha_source = 'liquidity_premium'
                else:
                    alpha_source = 'risk_premium'
                
                # Calculate expected alpha
                expected_alpha = min(crisis_ratio * 0.1, 0.3)
                
                return CrisisAlpha(
                    alpha_detected=True,
                    confidence=confidence,
                    alpha_source=alpha_source,
                    expected_alpha=expected_alpha,
                    implementation_cost=expected_alpha * 0.1,
                    regulatory_considerations=['increased_reporting', 'risk_limits']
                )
            
            return CrisisAlpha(
                alpha_detected=False,
                confidence=0.0,
                alpha_source='none',
                expected_alpha=0.0,
                implementation_cost=0.0,
                regulatory_considerations=[]
            )
            
        except Exception as e:
            logger.error(f"Crisis alpha detection failed: {e}")
            return CrisisAlpha(
                alpha_detected=False,
                confidence=0.0,
                alpha_source='none',
                expected_alpha=0.0,
                implementation_cost=0.0,
                regulatory_considerations=[]
            )


class RegimeChangeDetector:
    """Regime change detection"""
    
    async def detect_change(self, market_data: pd.DataFrame) -> RegimeChange:
        """Detect regime changes"""
        try:
            if market_data.empty or len(market_data) < 50:
                return RegimeChange(
                    change_detected=False,
                    confidence=0.0,
                    from_regime='normal',
                    to_regime='normal',
                    transition_probability=0.0,
                    adaptation_required=False
                )
            
            # Calculate regime indicators
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < 40:
                return RegimeChange(
                    change_detected=False,
                    confidence=0.0,
                    from_regime='normal',
                    to_regime='normal',
                    transition_probability=0.0,
                    adaptation_required=False
                )
            
            # Detect volatility regime changes
            vol_20 = returns.rolling(20).std()
            vol_40 = returns.rolling(40).std()
            
            vol_ratio = vol_20.iloc[-1] / vol_40.iloc[-1] if vol_40.iloc[-1] > 0 else 1.0
            
            # Detect trend changes
            sma_20 = market_data['close'].rolling(20).mean()
            sma_40 = market_data['close'].rolling(40).mean()
            
            trend_change = abs(sma_20.iloc[-1] - sma_40.iloc[-1]) / sma_40.iloc[-1] if sma_40.iloc[-1] > 0 else 0.0
            
            # Combined regime change detection
            change_score = (abs(vol_ratio - 1.0) + trend_change) / 2
            
            if change_score > 0.3:
                confidence = min(change_score, 1.0)
                
                # Determine regime transition
                if vol_ratio > 1.5:
                    from_regime = 'low_volatility'
                    to_regime = 'high_volatility'
                elif vol_ratio < 0.7:
                    from_regime = 'high_volatility'
                    to_regime = 'low_volatility'
                elif trend_change > 0.2:
                    from_regime = 'trending'
                    to_regime = 'ranging'
                else:
                    from_regime = 'normal'
                    to_regime = 'normal'
                
                return RegimeChange(
                    change_detected=True,
                    confidence=confidence,
                    from_regime=from_regime,
                    to_regime=to_regime,
                    transition_probability=confidence,
                    adaptation_required=True
                )
            
            return RegimeChange(
                change_detected=False,
                confidence=0.0,
                from_regime='normal',
                to_regime='normal',
                transition_probability=0.0,
                adaptation_required=False
            )
            
        except Exception as e:
            logger.error(f"Regime change detection failed: {e}")
            return RegimeChange(
                change_detected=False,
                confidence=0.0,
                from_regime='normal',
                to_regime='normal',
                transition_probability=0.0,
                adaptation_required=False
            )


class AntifragileStrategyOptimizer:
    """Antifragile strategy optimization"""
    
    async def generate_strategies(self, black_swan: BlackSwanDetection,
                                volatility_harvesting: VolatilityHarvesting,
                                crisis_alpha: CrisisAlpha,
                                regime_change: RegimeChange) -> List[AntifragileStrategy]:
        """Generate antifragile strategies based on analysis"""
        strategies = []
        
        # Strategy based on black swan detection
        if black_swan.detected:
            strategies.append(AntifragileStrategy(
                strategy_type='tail_hedge',
                confidence=black_swan.confidence * 0.8,
                expected_benefit=black_swan.expected_impact * 2.0,
                risk_level='high',
                implementation_complexity='medium',
                monitoring_requirements=['daily_rebalancing', 'volatility_monitoring']
            ))
        
        # Strategy based on volatility harvesting
        if volatility_harvesting.opportunity_detected:
            strategies.append(AntifragileStrategy(
                strategy_type='volatility_arbitrage',
                confidence=volatility_harvesting.confidence,
                expected_benefit=volatility_harvesting.expected_return,
                risk_level='medium',
                implementation_complexity='high',
                monitoring_requirements=['real_time_volatility', 'greeks_monitoring']
            ))
        
        # Strategy based on crisis alpha
        if crisis_alpha.alpha_detected:
            strategies.append(AntifragileStrategy(
                strategy_type='crisis_alpha',
                confidence=crisis_alpha.confidence,
                expected_benefit=crisis_alpha.expected_alpha,
                risk_level='medium',
                implementation_complexity='medium',
                monitoring_requirements=['market_regime', 'liquidity_monitoring']
            ))
        
        # Strategy based on regime change
        if regime_change.change_detected:
            strategies.append(AntifragileStrategy(
                strategy_type='regime_adaptation',
                confidence=regime_change.confidence,
                expected_benefit=0.05,
                risk_level='low',
                implementation_complexity='low',
                monitoring_requirements=['regime_indicators', 'performance_tracking']
            ))
        
        # Default antifragile strategy
        if not strategies:
            strategies.append(AntifragileStrategy(
                strategy_type='barbell_portfolio',
                confidence=0.7,
                expected_benefit=0.03,
                risk_level='low',
                implementation_complexity='low',
                monitoring_requirements=['portfolio_balance', 'risk_metrics']
            ))
        
        return strategies
