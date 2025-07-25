#!/usr/bin/env python3
"""
AmbusherFitnessFunction - Epic 2: Evolving "The Ambusher"
Specialized fitness function for liquidity grab and stop cascade events.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AmbushEventType(Enum):
    """Types of ambush events."""
    LIQUIDITY_GRAB = "liquidity_grab"
    STOP_CASCADE = "stop_cascade"
    ICEBERG_DETECTION = "iceberg_detection"
    MOMENTUM_BURST = "momentum_burst"

@dataclass
class AmbushEvent:
    """A detected ambush event."""
    event_type: AmbushEventType
    timestamp: datetime
    price_level: float
    volume: float
    liquidity_depth: float
    expected_profit: float
    confidence: float
    metadata: Dict[str, Any]

class AmbusherFitnessFunction:
    """Fitness function specialized for ambush strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Fitness weights
        self.profit_weight = config.get('profit_weight', 0.4)
        self.accuracy_weight = config.get('accuracy_weight', 0.3)
        self.timing_weight = config.get('timing_weight', 0.2)
        self.risk_weight = config.get('risk_weight', 0.1)
        
        # Thresholds
        self.min_liquidity_threshold = config.get('min_liquidity_threshold', 1000000)
        self.min_volume_threshold = config.get('min_volume_threshold', 100000)
        self.max_drawdown = config.get('max_drawdown', 0.05)
        
    def calculate_fitness(self, 
                         genome: Dict[str, Any], 
                         market_data: Dict[str, Any], 
                         trade_history: List[Dict[str, Any]]) -> float:
        """Calculate fitness score for ambush strategies."""
        
        # Detect ambush events
        events = self._detect_ambush_events(market_data, genome)
        
        if not events:
            return 0.0
            
        # Calculate component scores
        profit_score = self._calculate_profit_score(events, trade_history)
        accuracy_score = self._calculate_accuracy_score(events, trade_history)
        timing_score = self._calculate_timing_score(events, trade_history)
        risk_score = self._calculate_risk_score(trade_history)
        
        # Weighted fitness
        fitness = (
            self.profit_weight * profit_score +
            self.accuracy_weight * accuracy_score +
            self.timing_weight * timing_score +
            self.risk_weight * risk_score
        )
        
        logger.info(f"Ambusher fitness: {fitness:.4f} (profit: {profit_score:.4f}, "
                   f"accuracy: {accuracy_score:.4f}, timing: {timing_score:.4f}, "
                   f"risk: {risk_score:.4f})")
        
        return max(0.0, fitness)
        
    def _detect_ambush_events(self, market_data: Dict[str, Any], genome: Dict[str, Any]) -> List[AmbushEvent]:
        """Detect potential ambush events in market data."""
        events = []
        
        # Get market data
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])
        order_book = market_data.get('order_book', {})
        
        if not prices or not volumes:
            return events
            
        # Detect liquidity grabs
        liquidity_grabs = self._detect_liquidity_grabs(prices, volumes, order_book, genome)
        events.extend(liquidity_grabs)
        
        # Detect stop cascades
        stop_cascades = self._detect_stop_cascades(prices, volumes, order_book, genome)
        events.extend(stop_cascades)
        
        # Detect iceberg orders
        icebergs = self._detect_iceberg_orders(order_book, genome)
        events.extend(icebergs)
        
        # Detect momentum bursts
        momentum_bursts = self._detect_momentum_bursts(prices, volumes, genome)
        events.extend(momentum_bursts)
        
        return events
        
    def _detect_liquidity_grabs(self, prices: List[float], volumes: List[float], 
                              order_book: Dict[str, Any], genome: Dict[str, Any]) -> List[AmbushEvent]:
        """Detect liquidity grab events."""
        events = []
        
        # Parameters from genome
        grab_threshold = genome.get('liquidity_grab_threshold', 0.001)
        volume_threshold = genome.get('volume_threshold', 2.0)
        
        for i in range(1, len(prices)):
            price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
            volume_ratio = volumes[i] / np.mean(volumes[max(0, i-10):i]) if i > 0 else 1
            
            if price_change > grab_threshold and volume_ratio > volume_threshold:
                event = AmbushEvent(
                    event_type=AmbushEventType.LIQUIDITY_GRAB,
                    timestamp=datetime.utcnow(),
                    price_level=prices[i],
                    volume=volumes[i],
                    liquidity_depth=order_book.get('depth', 0),
                    expected_profit=price_change * 0.5,  # Conservative estimate
                    confidence=min(0.9, price_change * 1000),
                    metadata={
                        'price_change': price_change,
                        'volume_ratio': volume_ratio,
                        'direction': 'up' if prices[i] > prices[i-1] else 'down'
                    }
                )
                events.append(event)
                
        return events
        
    def _detect_stop_cascades(self, prices: List[float], volumes: List[float], 
                            order_book: Dict[str, Any], genome: Dict[str, Any]) -> List[AmbushEvent]:
        """Detect stop cascade events."""
        events = []
        
        # Parameters from genome
        cascade_threshold = genome.get('cascade_threshold', 0.002)
        consecutive_moves = genome.get('consecutive_moves', 3)
        
        for i in range(consecutive_moves, len(prices)):
            moves = [prices[j] - prices[j-1] for j in range(i-consecutive_moves+1, i+1)]
            
            # Check for consecutive moves in same direction
            if all(m > 0 for m in moves) or all(m < 0 for m in moves):
                total_move = abs(prices[i] - prices[i-consecutive_moves]) / prices[i-consecutive_moves]
                
                if total_move > cascade_threshold:
                    event = AmbushEvent(
                        event_type=AmbushEventType.STOP_CASCADE,
                        timestamp=datetime.utcnow(),
                        price_level=prices[i],
                        volume=np.mean(volumes[i-consecutive_moves:i+1]),
                        liquidity_depth=order_book.get('depth', 0),
                        expected_profit=total_move * 0.7,
                        confidence=min(0.95, total_move * 500),
                        metadata={
                            'total_move': total_move,
                            'consecutive_moves': consecutive_moves,
                            'direction': 'up' if moves[-1] > 0 else 'down'
                        }
                    )
                    events.append(event)
                    
        return events
        
    def _detect_iceberg_orders(self, order_book: Dict[str, Any], genome: Dict[str, Any]) -> List[AmbushEvent]:
        """Detect iceberg order events."""
        events = []
        
        # Parameters from genome
        iceberg_threshold = genome.get('iceberg_threshold', 1000000)
        
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        # Check for large hidden orders
        for side, orders in [('bid', bids), ('ask', asks)]:
            for level in orders:
                if level.get('volume', 0) > iceberg_threshold:
                    event = AmbushEvent(
                        event_type=AmbushEventType.ICEBERG_DETECTION,
                        timestamp=datetime.utcnow(),
                        price_level=level.get('price', 0),
                        volume=level.get('volume', 0),
                        liquidity_depth=order_book.get('depth', 0),
                        expected_profit=0.001,  # Small but consistent
                        confidence=0.7,
                        metadata={
                            'side': side,
                            'level': level
                        }
                    )
                    events.append(event)
                    
        return events
        
    def _detect_momentum_bursts(self, prices: List[float], volumes: List[float], 
                              genome: Dict[str, Any]) -> List[AmbushEvent]:
        """Detect momentum burst events."""
        events = []
        
        # Parameters from genome
        momentum_threshold = genome.get('momentum_threshold', 0.0015)
        volume_spike = genome.get('volume_spike', 3.0)
        
        for i in range(1, len(prices)):
            price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
            volume_ratio = volumes[i] / np.mean(volumes[max(0, i-5):i]) if i > 0 else 1
            
            if price_change > momentum_threshold and volume_ratio > volume_spike:
                event = AmbushEvent(
                    event_type=AmbushEventType.MOMENTUM_BURST,
                    timestamp=datetime.utcnow(),
                    price_level=prices[i],
                    volume=volumes[i],
                    liquidity_depth=0,  # Not applicable for momentum
                    expected_profit=price_change * 0.6,
                    confidence=min(0.8, price_change * 500),
                    metadata={
                        'price_change': price_change,
                        'volume_ratio': volume_ratio,
                        'direction': 'up' if prices[i] > prices[i-1] else 'down'
                    }
                )
                events.append(event)
                
        return events
        
    def _calculate_profit_score(self, events: List[AmbushEvent], trade_history: List[Dict[str, Any]]) -> float:
        """Calculate profit score from events and trades."""
        if not events:
            return 0.0
            
        total_expected_profit = sum(event.expected_profit for event in events)
        actual_profits = [trade.get('pnl', 0) for trade in trade_history if trade.get('event_type') in [e.value for e in AmbushEventType]]
        
        if actual_profits:
            return min(1.0, sum(actual_profits) / total_expected_profit) if total_expected_profit > 0 else 0.0
            
        return min(1.0, total_expected_profit * 100)  # Scale for expected
        
    def _calculate_accuracy_score(self, events: List[AmbushEvent], trade_history: List[Dict[str, Any]]) -> float:
        """Calculate accuracy score for event detection."""
        if not events or not trade_history:
            return 0.0
            
        successful_trades = [t for t in trade_history if t.get('pnl', 0) > 0]
        total_trades = len(trade_history)
        
        return len(successful_trades) / total_trades if total_trades > 0 else 0.0
        
    def _calculate_timing_score(self, events: List[AmbushEvent], trade_history: List[Dict[str, Any]]) -> float:
        """Calculate timing score for optimal entry/exit."""
        if not trade_history:
            return 0.0
            
        # Calculate average time to profit
        profitable_trades = [t for t in trade_history if t.get('pnl', 0) > 0]
        if not profitable_trades:
            return 0.0
            
        avg_time_to_profit = np.mean([t.get('duration', 0) for t in profitable_trades])
        optimal_time = 300  # 5 minutes
        
        return max(0.0, 1.0 - (avg_time_to_profit / optimal_time))
        
    def _calculate_risk_score(self, trade_history: List[Dict[str, Any]]) -> float:
        """Calculate risk management score."""
        if not trade_history:
            return 0.0
            
        # Calculate max drawdown
        drawdowns = [t.get('max_drawdown', 0) for t in trade_history]
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Risk score based on drawdown vs threshold
        if max_drawdown <= self.max_drawdown:
            return 1.0
        else:
            return max(0.0, 1.0 - (max_drawdown - self.max_drawdown) / self.max_drawdown)
        
    def get_fitness_breakdown(self, genome: Dict[str, Any], 
                            market_data: Dict[str, Any], 
                            trade_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get detailed fitness breakdown."""
        events = self._detect_ambush_events(market_data, genome)
        
        return {
            'profit_score': self._calculate_profit_score(events, trade_history),
            'accuracy_score': self._calculate_accuracy_score(events, trade_history),
            'timing_score': self._calculate_timing_score(events, trade_history),
            'risk_score': self._calculate_risk_score(trade_history),
            'total_events': len(events),
            'fitness': self.calculate_fitness(genome, market_data, trade_history)
        }
