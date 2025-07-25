"""
Competitive Intelligence System
Identify and counter competing algorithmic traders.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
import uuid

from src.core.events import AlgorithmSignature, CompetitorBehavior, CounterStrategy
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)


class AlgorithmFingerprinter:
    """Identifies algorithmic patterns in market data."""
    
    def __init__(self):
        self.known_patterns = {
            'momentum_bot': {
                'indicators': ['rsi', 'macd', 'volume'],
                'frequency': 'high',
                'position_size': 'medium'
            },
            'mean_reversion_bot': {
                'indicators': ['bollinger_bands', 'rsi', 'volume'],
                'frequency': 'medium',
                'position_size': 'large'
            },
            'arbitrage_bot': {
                'indicators': ['spread', 'liquidity', 'latency'],
                'frequency': 'very_high',
                'position_size': 'small'
            },
            'trend_following_bot': {
                'indicators': ['sma', 'ema', 'adx'],
                'frequency': 'low',
                'position_size': 'large'
            },
            'scalping_bot': {
                'indicators': ['order_book', 'volume', 'momentum'],
                'frequency': 'very_high',
                'position_size': 'very_small'
            }
        }
    
    async def identify_signatures(
        self,
        market_data: Dict[str, Any],
        known_patterns: Dict[str, Any]
    ) -> List[AlgorithmSignature]:
        """Identify algorithmic signatures in market data."""
        try:
            signatures = []
            
            # Analyze trading patterns
            patterns = await self._analyze_trading_patterns(market_data)
            
            # Match against known patterns
            for pattern_type, pattern_data in patterns.items():
                if self._matches_known_pattern(pattern_data, known_patterns):
                    signature = AlgorithmSignature(
                        signature_id=str(uuid.uuid4()),
                        algorithm_type=pattern_type,
                        confidence=self._calculate_confidence(pattern_data),
                        characteristics=pattern_data,
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow(),
                        frequency=pattern_data.get('frequency', 'medium')
                    )
                    signatures.append(signature)
            
            logger.info(
                f"Identified {len(signatures)} algorithmic signatures: "
                f"{[s.algorithm_type for s in signatures]}"
            )
            
            return signatures
            
        except Exception as e:
            logger.error(f"Error identifying signatures: {e}")
            return []
    
    async def _analyze_trading_patterns(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze trading patterns in market data."""
        try:
            # Simulate pattern analysis
            patterns = {
                'momentum_bot': {
                    'rsi_usage': np.random.normal(0.8, 0.1),
                    'macd_usage': np.random.normal(0.7, 0.1),
                    'frequency': 'high',
                    'position_size': 'medium'
                },
                'mean_reversion_bot': {
                    'bollinger_usage': np.random.normal(0.9, 0.05),
                    'rsi_usage': np.random.normal(0.6, 0.1),
                    'frequency': 'medium',
                    'position_size': 'large'
                }
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing trading patterns: {e}")
            return {}
    
    def _matches_known_pattern(
        self,
        pattern: Dict[str, Any],
        known_patterns: Dict[str, Any]
    ) -> bool:
        """Check if pattern matches known algorithm type."""
        try:
            # Simple matching logic
            threshold = 0.7
            for algo_type, known_pattern in known_patterns.items():
                similarity = self._calculate_pattern_similarity(pattern, known_pattern)
                if similarity > threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error matching pattern: {e}")
            return False
    
    def _calculate_pattern_similarity(
        self,
        pattern: Dict[str, Any],
        known: Dict[str, Any]
    ) -> float:
        """Calculate similarity between pattern and known algorithm."""
        try:
            # Simple similarity calculation
            indicators_match = 0
            total_indicators = 0
            
            for key in ['indicators', 'frequency', 'position_size']:
                if key in pattern and key in known:
                    total_indicators += 1
                    if pattern[key] == known[key]:
                        indicators_match += 1
            
            return indicators_match / total_indicators if total_indicators > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0
    
    def _calculate_confidence(self, pattern: Dict[str, Any]) -> Decimal:
        """Calculate confidence in signature identification."""
        try:
            # Simple confidence calculation
            base_confidence = 0.7
            if pattern.get('frequency') == 'high':
                base_confidence += 0.1
            if pattern.get('position_size') == 'large':
                base_confidence += 0.1
            
            return Decimal(str(min(1.0, base_confidence)))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return Decimal('0.5')


class BehaviorAnalyzer:
    """Analyzes competitor behavior patterns."""
    
    def __init__(self):
        self.behavior_metrics = [
            'trade_frequency',
            'position_size_pattern',
            'timing_precision',
            'risk_management',
            'market_impact'
        ]
    
    async def analyze_behavior(
        self,
        signature: AlgorithmSignature,
        historical_data: Dict[str, Any]
    ) -> CompetitorBehavior:
        """Analyze competitor behavior patterns."""
        try:
            # Analyze behavior over time
            behavior_data = await self._extract_behavior_data(
                signature,
                historical_data
            )
            
            # Calculate behavior metrics
            metrics = self._calculate_behavior_metrics(behavior_data)
            
            # Identify behavior patterns
            patterns = self._identify_behavior_patterns(behavior_data)
            
            # Assess threat level
            threat_level = self._assess_threat_level(metrics, patterns)
            
            behavior = CompetitorBehavior(
                competitor_id=str(uuid.uuid4()),
                algorithm_signature=signature,
                behavior_metrics=metrics,
                patterns=patterns,
                threat_level=threat_level,
                market_share=self._estimate_market_share(behavior_data),
                performance=self._estimate_performance(behavior_data),
                first_observed=datetime.utcnow(),
                last_observed=datetime.utcnow()
            )
            
            logger.info(
                f"Analyzed competitor behavior: "
                f"{signature.algorithm_type} with threat level {threat_level}"
            )
            
            return behavior
            
        except Exception as e:
            logger.error(f"Error analyzing behavior: {e}")
            return CompetitorBehavior(
                competitor_id=str(uuid.uuid4()),
                algorithm_signature=signature,
                behavior_metrics={},
                patterns=[],
                threat_level='low',
                market_share=Decimal('0.0'),
                performance=Decimal('0.0')
            )
    
    async def _extract_behavior_data(
        self,
        signature: AlgorithmSignature,
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract behavior data from historical data."""
        try:
            # Simulate behavior extraction
            return {
                'trade_frequency': np.random.normal(100, 20),
                'avg_position_size': np.random.normal(1000, 200),
                'win_rate': np.random.normal(0.65, 0.1),
                'avg_hold_time': np.random.normal(30, 10),
                'market_impact': np.random.normal(0.001, 0.0005)
            }
            
        except Exception as e:
            logger.error(f"Error extracting behavior data: {e}")
            return {}
    
    def _calculate_behavior_metrics(
        self,
        behavior_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate behavior metrics."""
        try:
            metrics = {
                'efficiency': behavior_data.get('win_rate', 0.5),
                'aggressiveness': min(behavior_data.get('trade_frequency', 50) / 200, 1.0),
                'sophistication': min(behavior_data.get('market_impact', 0.001) / 0.005, 1.0),
                'consistency': 1.0 - abs(behavior_data.get('win_rate', 0.5) - 0.5)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating behavior metrics: {e}")
            return {}
    
    def _identify_behavior_patterns(
        self,
        behavior_data: Dict[str, Any]
    ) -> List[str]:
        """Identify behavior patterns."""
        try:
            patterns = []
            
            if behavior_data.get('trade_frequency', 0) > 150:
                patterns.append('high_frequency')
            
            if behavior_data.get('avg_position_size', 0) > 1500:
                patterns.append('large_positions')
            
            if behavior_data.get('win_rate', 0) > 0.7:
                patterns.append('high_accuracy')
            
            if behavior_data.get('market_impact', 0) < 0.0005:
                patterns.append('low_impact')
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying behavior patterns: {e}")
            return []
    
    def _assess_threat_level(
        self,
        metrics: Dict[str, float],
        patterns: List[str]
    ) -> str:
        """Assess threat level based on metrics and patterns."""
        try:
            threat_score = 0
            
            # Calculate threat score
            threat_score += metrics.get('efficiency', 0) * 0.3
            threat_score += metrics.get('aggressiveness', 0) * 0.2
            threat_score += metrics.get('sophistication', 0) * 0.3
            threat_score += metrics.get('consistency', 0) * 0.2
            
            # Adjust based on patterns
            if 'high_frequency' in patterns:
                threat_score += 0.1
            if 'high_accuracy' in patterns:
                threat_score += 0.2
            
            # Determine threat level
            if threat_score > 0.8:
                return 'high'
            elif threat_score > 0.6:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing threat level: {e}")
            return 'low'
    
    def _estimate_market_share(
        self,
        behavior_data: Dict[str, Any]
    ) -> Decimal:
        """Estimate competitor's market share."""
        try:
            # Simple market share estimation
            volume = behavior_data.get('avg_position_size', 0) * behavior_data.get('trade_frequency', 0)
            market_share = min(volume / 1000000, 0.1)  # Cap at 10%
            
            return Decimal(str(market_share))
            
        except Exception as e:
            logger.error(f"Error estimating market share: {e}")
            return Decimal('0.0')
    
    def _estimate_performance(
        self,
        behavior_data: Dict[str, Any]
    ) -> Decimal:
        """Estimate competitor's performance."""
        try:
            # Simple performance estimation
            win_rate = behavior_data.get('win_rate', 0.5)
            avg_return = behavior_data.get('avg_return', 0.001)
            performance = win_rate * avg_return * 252  # Annualized
            
            return Decimal(str(min(performance, 1.0)))
            
        except Exception as e:
            logger.error(f"Error estimating performance: {e}")
            return Decimal('0.0')


class CounterStrategyDeveloper:
    """Develops counter-strategies against competitors."""
    
    def __init__(self):
        self.counter_strategies = {
            'momentum_bot': {
                'counter_type': 'mean_reversion',
                'effectiveness': 0.7,
                'complexity': 'medium'
            },
            'mean_reversion_bot': {
                'counter_type': 'momentum_breakout',
                'effectiveness': 0.6,
                'complexity': 'high'
            },
            'arbitrage_bot': {
                'counter_type': 'latency_advantage',
                'effectiveness': 0.8,
                'complexity': 'high'
            },
            'trend_following_bot': {
                'counter_type': 'range_trading',
                'effectiveness': 0.5,
                'complexity': 'medium'
            },
            'scalping_bot': {
                'counter_type': 'liquidity_manipulation',
                'effectiveness': 0.4,
                'complexity': 'very_high'
            }
        }
    
    async def develop_counter(
        self,
        behavior: CompetitorBehavior,
        our_capabilities: Dict[str, Any]
    ) -> CounterStrategy:
        """Develop counter-strategy against competitor."""
        try:
            algorithm_type = behavior.algorithm_signature.algorithm_type
            
            # Get base counter-strategy
            base_counter = self.counter_strategies.get(algorithm_type, {
                'counter_type': 'generic',
                'effectiveness': 0.3,
                'complexity': 'low'
            })
            
            # Customize based on competitor behavior
            customized_counter = self._customize_counter_strategy(
                base_counter,
                behavior,
                our_capabilities
            )
            
            # Calculate expected impact
            expected_impact = self._calculate_expected_impact(
                customized_counter,
                behavior
            )
            
            counter_strategy = CounterStrategy(
                strategy_id=str(uuid.uuid4()),
                target_competitor=behavior.competitor_id,
                counter_type=customized_counter['counter_type'],
                parameters=customized_counter['parameters'],
                expected_effectiveness=Decimal(str(expected_impact)),
                implementation_complexity=customized_counter['complexity'],
                risk_level=customized_counter['risk_level'],
                deployment_timeline=customized_counter['timeline'],
                timestamp=datetime.utcnow()
            )
            
            logger.info(
                f"Developed counter-strategy against {algorithm_type}: "
                f"{customized_counter['counter_type']} "
                f"with {expected_impact:.2f} expected effectiveness"
            )
            
            return counter_strategy
            
        except Exception as e:
            logger.error(f"Error developing counter-strategy: {e}")
            return CounterStrategy(
                strategy_id=str(uuid.uuid4()),
                target_competitor=behavior.competitor_id,
                counter_type='generic',
                parameters={},
                expected_effectiveness=Decimal('0.3'),
                implementation_complexity='low',
                risk_level='low',
                deployment_timeline='immediate',
                timestamp=datetime.utcnow()
            )
    
    def _customize_counter_strategy(
        self,
        base_counter: Dict[str, Any],
        behavior: CompetitorBehavior,
        our_capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize counter-strategy based on competitor and our capabilities."""
        try:
            customized = base_counter.copy()
            
            # Adjust parameters based on competitor behavior
            if behavior.threat_level == 'high':
                customized['effectiveness'] *= 1.2
                customized['complexity'] = 'high'
            
            # Adjust based on our capabilities
            if our_capabilities.get('latency_advantage', False):
                customized['parameters']['latency_optimization'] = True
            
            # Set risk level
            threat_score = float(behavior.market_share) * float(behavior.performance)
            if threat_score > 0.05:
                customized['risk_level'] = 'high'
            elif threat_score > 0.02:
                customized['risk_level'] = 'medium'
            else:
                customized['risk_level'] = 'low'
            
            # Set timeline
            complexity_map = {
                'low': 'immediate',
                'medium': 'short_term',
                'high': 'medium_term',
                'very_high': 'long_term'
            }
            customized['timeline'] = complexity_map.get(
                customized['complexity'],
                'medium_term'
            )
            
            return customized
            
        except Exception as e:
            logger.error(f"Error customizing counter-strategy: {e}")
            return base_counter
    
    def _calculate_expected_impact(
        self,
        counter: Dict[str, Any],
        behavior: CompetitorBehavior
    ) -> float:
        """Calculate expected impact of counter-strategy."""
        try:
            base_effectiveness = counter.get('effectiveness', 0.3)
            
            # Adjust based on competitor threat level
            threat_multiplier = {
                'low': 1.0,
                'medium': 0.8,
                'high': 0.6
            }
            
            effectiveness = base_effectiveness * threat_multiplier.get(
                behavior.threat_level,
                1.0
            )
            
            return min(effectiveness, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating expected impact: {e}")
            return 0.3


class MarketShareTracker:
    """Tracks market share changes and competitive dynamics."""
    
    def __init__(self):
        self.tracking_metrics = [
            'volume_share',
            'profit_share',
            'frequency_share',
            'impact_share'
        ]
    
    async def analyze_share_changes(
        self,
        competitor_behaviors: List[CompetitorBehavior],
        our_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze market share changes."""
        try:
            # Calculate current market shares
            total_market = sum([
                float(cb.market_share) for cb in competitor_behaviors
            ]) + float(our_performance.get('market_share', 0))
            
            # Analyze share trends
            share_trends = self._calculate_share_trends(
                competitor_behaviors,
                our_performance
            )
            
            # Identify competitive threats
            threats = self._identify_competitive_threats(
                competitor_behaviors,
                our_performance
            )
            
            # Calculate competitive position
            position = self._calculate_competitive_position(
                competitor_behaviors,
                our_performance,
                total_market
            )
            
            analysis = {
                'total_market_size': total_market,
                'our_share': float(our_performance.get('market_share', 0)),
                'competitor_shares': {
                    cb.competitor_id: float(cb.market_share)
                    for cb in competitor_behaviors
                },
                'share_trends': share_trends,
                'threats': threats,
                'competitive_position': position,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Market share analysis complete: "
                f"Our share {analysis['our_share']:.3f}, "
                f"{len(competitor_behaviors)} competitors"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing share changes: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _calculate_share_trends(
        self,
        competitors: List[CompetitorBehavior],
        our_performance: Dict[str, Any]
    ) -> Dict[str, str]:
        """Calculate share trends."""
        try:
            trends = {}
            
            # Simulate trend analysis
            for competitor in competitors:
                trend = np.random.choice(['increasing', 'stable', 'decreasing'])
                trends[competitor.competitor_id] = trend
            
            our_trend = np.random.choice(['increasing', 'stable', 'decreasing'])
            trends['our_system'] = our_trend
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating share trends: {e}")
            return {}
    
    def _identify_competitive_threats(
        self,
        competitors: List[CompetitorBehavior],
        our_performance: Dict[str, Any]
    ) -> List[str]:
        """Identify competitive threats."""
        try:
            threats = []
            
            for competitor in competitors:
                if competitor.threat_level == 'high':
                    threats.append(f"High threat from {competitor.algorithm_signature.algorithm_type}")
                
                if float(competitor.market_share) > 0.05:
                    threats.append(f"Large market share by {competitor.competitor_id}")
                
                if float(competitor.performance) > 0.5:
                    threats.append(f"High performance by {competitor.competitor_id}")
            
            return threats
            
        except Exception as e:
            logger.error(f"Error identifying threats: {e}")
            return []
    
    def _calculate_competitive_position(
        self,
        competitors: List[CompetitorBehavior],
        our_performance: Dict[str, Any],
        total_market: float
    ) -> str:
        """Calculate competitive position."""
        try:
            our_share = float(our_performance.get('market_share', 0))
            
            if total_market == 0:
                return 'unknown'
            
            our_percentage = our_share / total_market
            
            if our_percentage > 0.3:
                return 'dominant'
            elif our_percentage > 0.15:
                return 'strong'
            elif our_percentage > 0.05:
                return 'competitive'
            else:
                return 'minor'
                
        except Exception as e:
            logger.error(f"Error calculating competitive position: {e}")
            return 'unknown'


class CompetitiveIntelligenceSystem:
    """
    Identify and counter competing algorithmic traders.
    
    Features:
    - Algorithm identification and fingerprinting
    - Competitor behavior analysis
    - Counter-strategy development
    - Market share tracking
    """
    
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.algorithm_fingerprinter = AlgorithmFingerprinter()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.counter_strategy_developer = CounterStrategyDeveloper()
        self.market_share_tracker = MarketShareTracker()
        
        self._intelligence_history_key = "emp:competitive_intelligence"
    
    async def identify_competitors(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify and analyze competing algorithmic traders.
        
        Args:
            market_data: Current market data
            
        Returns:
            Comprehensive competitive intelligence report
        """
        try:
            # Step 1: Identify algorithmic signatures
            known_patterns = await self._get_known_patterns()
            signatures = await self.algorithm_fingerprinter.identify_signatures(
                market_data,
                known_patterns
            )
            
            if not signatures:
                logger.warning("No algorithmic signatures detected")
                return {'error': 'No signatures detected'}
            
            # Step 2: Analyze competitor behaviors
            historical_data = await self._get_historical_data()
            competitor_behaviors = []
            
            for signature in signatures:
                behavior = await self.behavior_analyzer.analyze_behavior(
                    signature,
                    historical_data
                )
                competitor_behaviors.append(behavior)
            
            # Step 3: Develop counter-strategies
            our_capabilities = await self._get_our_capabilities()
            counter_strategies = []
            
            for behavior in competitor_behaviors:
                counter = await self.counter_strategy_developer.develop_counter(
                    behavior,
                    our_capabilities
                )
                counter_strategies.append(counter)
            
            # Step 4: Analyze market share
            our_performance = await self._get_our_performance()
            market_share_analysis = await self.market_share_tracker.analyze_share_changes(
                competitor_behaviors,
                our_performance
            )
            
            # Step 5: Store intelligence
            await self._store_intelligence(
                signatures,
                competitor_behaviors,
                counter_strategies,
                market_share_analysis
            )
            
            report = {
                'intelligence_id': str(uuid.uuid4()),
                'signatures_detected': len(signatures),
                'competitors_analyzed': len(competitor_behaviors),
                'counter_strategies_developed': len(counter_strategies),
                'market_share_analysis': market_share_analysis,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Competitive intelligence complete: "
                f"{len(signatures)} signatures, "
                f"{len(competitor_behaviors)} competitors, "
                f"{len(counter_strategies)} counter-strategies"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error in competitive intelligence: {e}")
            return {'error': str(e)}
    
    async def _get_known_patterns(self) -> Dict[str, Any]:
        """Get known algorithmic patterns."""
        try:
            return self.algorithm_fingerprinter.known_patterns
        except Exception as e:
            logger.error(f"Error getting known patterns: {e}")
            return {}
    
    async def _get_historical_data(self) -> Dict[str, Any]:
        """Get historical market data."""
        try:
            # This would be enhanced with actual historical data
            return {
                'price_data': [],
                'volume_data': [],
                'order_book_data': [],
                'trade_data': []
            }
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return {}
    
    async def _get_our_capabilities(self) -> Dict[str, Any]:
        """Get our system capabilities."""
        try:
            # This would be enhanced with actual capabilities
            return {
                'latency_advantage': True,
                'data_sources': ['real_time', 'historical', 'news'],
                'strategy_types': ['momentum', 'mean_reversion', 'arbitrage'],
                'risk_management': 'advanced'
            }
        except Exception as e:
            logger.error(f"Error getting our capabilities: {e}")
            return {}
    
    async def _get_our_performance(self) -> Dict[str, Any]:
        """Get our system performance."""
        try:
            # This would be enhanced with actual performance
            return {
                'market_share': Decimal('0.15'),
                'total_return': 0.18,
                'sharpe_ratio': 1.4,
                'win_rate': 0.68,
                'max_drawdown': 0.08
            }
        except Exception as e:
            logger.error(f"Error getting our performance: {e}")
            return {}
    
    async def _store_intelligence(
        self,
        signatures: List[AlgorithmSignature],
        behaviors: List[CompetitorBehavior],
        counters: List[CounterStrategy],
        market_analysis: Dict[str, Any]
    ) -> None:
        """Store competitive intelligence."""
        try:
            intelligence = {
                'signatures': [s.dict() for s in signatures],
                'behaviors': [b.dict() for b in behaviors],
                'counter_strategies': [c.dict() for c in counters],
                'market_analysis': market_analysis,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            key = f"{self._intelligence_history_key}:{datetime.utcnow().date()}"
            await self.state_store.set(
                key,
                str(intelligence),
                expire=86400 * 30  # 30 days
            )
            
        except Exception as e:
            logger.error(f"Error storing intelligence: {e}")
    
    async def get_intelligence_stats(self) -> Dict[str, Any]:
        """Get competitive intelligence statistics."""
        try:
            keys = await self.state_store.keys(f"{self._intelligence_history_key}:*")
            
            total_signatures = 0
            total_competitors = 0
            total_counters = 0
            
            for key in keys:
                data = await self.state_store.get(key)
                if data:
                    record = eval(data)
                    total_signatures += len(record.get('signatures', []))
                    total_competitors += len(record.get('behaviors', []))
                    total_counters += len(record.get('counter_strategies', []))
            
            return {
                'total_intelligence_cycles': len(keys),
                'total_signatures_detected': total_signatures,
                'total_competitors_analyzed': total_competitors,
                'total_counter_strategies_developed': total_counters,
                'average_signatures_per_cycle': total_signatures / len(keys) if keys else 0,
                'last_intelligence': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting intelligence stats: {e}")
            return {
                'total_intelligence_cycles': 0,
                'total_signatures_detected': 0,
                'total_competitors_analyzed': 0,
                'last_intelligence': None
            }
