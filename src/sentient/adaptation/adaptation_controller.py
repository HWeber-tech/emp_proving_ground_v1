#!/usr/bin/env python3
"""
AdaptationController - Epic 1: The Predator's Instinct
Generates tactical adaptations based on recalled memories to modify genome behavior in real-time.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

class AdaptationType(Enum):
    """Types of tactical adaptations."""
    RISK_ADJUSTMENT = "risk_adjustment"
    POSITION_SIZE = "position_size"
    ENTRY_TIMING = "entry_timing"
    EXIT_TIMING = "exit_timing"
    STRATEGY_SWITCH = "strategy_switch"
    LIQUIDITY_FOCUS = "liquidity_focus"

@dataclass
class TacticalAdaptation:
    """A tactical adaptation to modify genome behavior."""
    adaptation_id: str
    adaptation_type: AdaptationType
    parameters: Dict[str, Any]
    confidence: float
    timestamp: datetime
    source_memories: List[str]
    expected_impact: Dict[str, float]
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence adaptation."""
        return self.confidence > 0.8
    
    @property
    def impact_score(self) -> float:
        """Calculate overall impact score."""
        return sum(self.expected_impact.values()) / len(self.expected_impact) if self.expected_impact else 0

class AdaptationController:
    """Generates tactical adaptations based on recalled memories."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_confidence = config.get('min_confidence', 0.7)
        self.max_adaptations = config.get('max_adaptations', 10)
        self.adaptation_history: List[TacticalAdaptation] = []
        
    async def generate_adaptations(self, 
                                 memories: List[Dict[str, Any]], 
                                 current_context: Dict[str, Any]) -> List[TacticalAdaptation]:
        """Generate tactical adaptations based on recalled memories."""
        logger.info(f"Generating adaptations from {len(memories)} memories")
        
        adaptations = []
        
        # Analyze memory patterns
        patterns = self._analyze_memory_patterns(memories)
        
        # Generate adaptations based on patterns
        adaptations.extend(self._generate_risk_adaptations(patterns, current_context))
        adaptations.extend(self._generate_position_size_adaptations(patterns, current_context))
        adaptations.extend(self._generate_timing_adaptations(patterns, current_context))
        adaptations.extend(self._generate_strategy_adaptations(patterns, current_context))
        adaptations.extend(self._generate_liquidity_adaptations(patterns, current_context))
        
        # Filter by confidence
        adaptations = [a for a in adaptations if a.confidence >= self.min_confidence]
        
        # Limit number of adaptations
        adaptations = adaptations[:self.max_adaptations]
        
        # Add to history
        self.adaptation_history.extend(adaptations)
        
        logger.info(f"Generated {len(adaptations)} tactical adaptations")
        return adaptations
    
    def _analyze_memory_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in recalled memories."""
        if not memories:
            return {}
            
        # Extract outcomes
        outcomes = [m['metadata'].get('outcome', {}) for m in memories]
        contexts = [m['metadata'].get('context', {}) for m in memories]
        
        # Calculate statistics
        pnls = [o.get('pnl', 0) for o in outcomes]
        durations = [o.get('duration', 0) for o in outcomes]
        
        # Pattern analysis
        profitable_memories = [m for m in memories if m['metadata'].get('outcome', {}).get('pnl', 0) > 0]
        loss_memories = [m for m in memories if m['metadata'].get('outcome', {}).get('pnl', 0) < 0]
        
        return {
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'win_rate': len(profitable_memories) / len(memories) if memories else 0,
            'profit_ratio': len(profitable_memories) / len(loss_memories) if loss_memories else 1,
            'volatility': np.std(pnls) if pnls else 0,
            'memory_count': len(memories)
        }
    
    def _generate_risk_adaptations(self, patterns: Dict[str, Any], context: Dict[str, Any]) -> List[TacticalAdaptation]:
        """Generate risk-related adaptations."""
        adaptations = []
        
        # Risk adjustment based on win rate
        if patterns.get('win_rate', 0) < 0.4:
            adaptations.append(TacticalAdaptation(
                adaptation_id=f"risk_reduce_{datetime.utcnow().timestamp()}",
                adaptation_type=AdaptationType.RISK_ADJUSTMENT,
                parameters={'risk_multiplier': 0.7, 'max_drawdown_limit': 0.02},
                confidence=min(0.9, 1 - patterns.get('win_rate', 0)),
                timestamp=datetime.utcnow(),
                source_memories=[],
                expected_impact={'risk_reduction': 0.3, 'profit_stability': 0.2}
            ))
        elif patterns.get('win_rate', 0) > 0.7:
            adaptations.append(TacticalAdaptation(
                adaptation_id=f"risk_increase_{datetime.utcnow().timestamp()}",
                adaptation_type=AdaptationType.RISK_ADJUSTMENT,
                parameters={'risk_multiplier': 1.3, 'max_drawdown_limit': 0.05},
                confidence=patterns.get('win_rate', 0),
                timestamp=datetime.utcnow(),
                source_memories=[],
                expected_impact={'profit_increase': 0.4, 'volatility_increase': 0.1}
            ))
            
        return adaptations
    
    def _generate_position_size_adaptations(self, patterns: Dict[str, Any], context: Dict[str, Any]) -> List[TacticalAdaptation]:
        """Generate position size adaptations."""
        adaptations = []
        
        # Position sizing based on volatility
        volatility = patterns.get('volatility', 0)
        if volatility > 0.01:
            adaptations.append(TacticalAdaptation(
                adaptation_id=f"size_reduce_{datetime.utcnow().timestamp()}",
                adaptation_type=AdaptationType.POSITION_SIZE,
                parameters={'size_multiplier': 0.5, 'max_position': 0.02},
                confidence=min(0.9, volatility * 100),
                timestamp=datetime.utcnow(),
                source_memories=[],
                expected_impact={'risk_reduction': 0.5, 'drawdown_reduction': 0.3}
            ))
            
        return adaptations
    
    def _generate_timing_adaptations(self, patterns: Dict[str, Any], context: Dict[str, Any]) -> List[TacticalAdaptation]:
        """Generate timing-related adaptations."""
        adaptations = []
        
        # Entry timing based on average duration
        avg_duration = patterns.get('avg_duration', 0)
        if avg_duration > 3600:  # More than 1 hour
            adaptations.append(TacticalAdaptation(
                adaptation_id=f"entry_early_{datetime.utcnow().timestamp()}",
                adaptation_type=AdaptationType.ENTRY_TIMING,
                parameters={'entry_delay': -300, 'momentum_threshold': 0.001},
                confidence=0.8,
                timestamp=datetime.utcnow(),
                source_memories=[],
                expected_impact={'profit_increase': 0.2, 'duration_reduction': 0.3}
            ))
            
        return adaptations
    
    def _generate_strategy_adaptations(self, patterns: Dict[str, Any], context: Dict[str, Any]) -> List[TacticalAdaptation]:
        """Generate strategy switching adaptations."""
        adaptations = []
        
        # Strategy switching based on market conditions
        market_volatility = context.get('volatility', 0)
        if market_volatility > 0.02:
            adaptations.append(TacticalAdaptation(
                adaptation_id=f"strategy_volatile_{datetime.utcnow().timestamp()}",
                adaptation_type=AdaptationType.STRATEGY_SWITCH,
                parameters={'strategy': 'momentum', 'timeframe': '5m'},
                confidence=0.75,
                timestamp=datetime.utcnow(),
                source_memories=[],
                expected_impact={'profit_increase': 0.3, 'risk_adjustment': 0.2}
            ))
            
        return adaptations
    
    def _generate_liquidity_adaptations(self, patterns: Dict[str, Any], context: Dict[str, Any]) -> List[TacticalAdaptation]:
        """Generate liquidity-focused adaptations."""
        adaptations = []
        
        # Liquidity grab detection
        if context.get('liquidity_depth', 0) < 1000000:  # Low liquidity
            adaptations.append(TacticalAdaptation(
                adaptation_id=f"liquidity_focus_{datetime.utcnow().timestamp()}",
                adaptation_type=AdaptationType.LIQUIDITY_FOCUS,
                parameters={'liquidity_threshold': 500000, 'grab_detection': True},
                confidence=0.85,
                timestamp=datetime.utcnow(),
                source_memories=[],
                expected_impact={'opportunity_detection': 0.4, 'profit_increase': 0.3}
            ))
            
        return adaptations
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptations."""
        if not self.adaptation_history:
            return {'total_adaptations': 0}
            
        adaptations = self.adaptation_history
        
        return {
            'total_adaptations': len(adaptations),
            'recent_adaptations': len([a for a in adaptations[-100:] if a.is_high_confidence]),
            'adaptation_types': {t.value: len([a for a in adaptations if a.adaptation_type == t]) 
                               for t in AdaptationType},
            'avg_confidence': np.mean([a.confidence for a in adaptations]),
            'latest_adaptation': adaptations[-1] if adaptations else None
        }
