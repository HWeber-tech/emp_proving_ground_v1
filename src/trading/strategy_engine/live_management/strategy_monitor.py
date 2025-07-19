"""
Strategy Monitor

Real-time strategy monitoring and health checks.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StrategyHealth:
    """Strategy health metrics"""
    strategy_id: str
    status: str
    last_signal_time: datetime
    signal_frequency: float
    performance_score: float
    risk_score: float
    memory_usage: float
    cpu_usage: float
    errors: List[str]
    warnings: List[str]


class StrategyMonitor:
    """
    Real-time Strategy Monitoring System
    
    Implements comprehensive strategy monitoring with:
    - Health checks
    - Performance monitoring
    - Risk monitoring
    - Error tracking
    """
    
    def __init__(self):
        self.monitored_strategies: Dict[str, StrategyHealth] = {}
        self.health_history: Dict[str, List[StrategyHealth]] = {}
        
        logger.info("StrategyMonitor initialized")
    
    def register_strategy(self, strategy_id: str) -> None:
        """Register a strategy for monitoring"""
        health = StrategyHealth(
            strategy_id=strategy_id,
            status="active",
            last_signal_time=datetime.utcnow(),
            signal_frequency=0.0,
            performance_score=1.0,
            risk_score=0.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            errors=[],
            warnings=[]
        )
        
        self.monitored_strategies[strategy_id] = health
        self.health_history[strategy_id] = []
        
        logger.info(f"Strategy {strategy_id} registered for monitoring")
    
    def update_strategy_health(self, strategy_id: str, 
                             signal_generated: bool = False,
                             performance_metrics: Optional[Dict[str, Any]] = None,
                             risk_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Update strategy health metrics"""
        
        if strategy_id not in self.monitored_strategies:
            self.register_strategy(strategy_id)
        
        health = self.monitored_strategies[strategy_id]
        
        # Update signal frequency
        if signal_generated:
            health.last_signal_time = datetime.utcnow()
            health.signal_frequency = self._calculate_signal_frequency(strategy_id)
        
        # Update performance score
        if performance_metrics:
            health.performance_score = self._calculate_performance_score(performance_metrics)
        
        # Update risk score
        if risk_metrics:
            health.risk_score = self._calculate_risk_score(risk_metrics)
        
        # Update system metrics
        health.memory_usage = self._get_memory_usage()
        health.cpu_usage = self._get_cpu_usage()
        
        # Store health history
        self.health_history[strategy_id].append(health)
        
        # Keep only recent history
        if len(self.health_history[strategy_id]) > 1000:
            self.health_history[strategy_id] = self.health_history[strategy_id][-1000:]
    
    def _calculate_signal_frequency(self, strategy_id: str) -> float:
        """Calculate signal frequency for strategy"""
        history = self.health_history.get(strategy_id, [])
        
        if len(history) < 2:
            return 0.0
        
        # Calculate average signals per hour
        recent_history = history[-100:]  # Last 100 updates
        signal_count = sum(1 for h in recent_history if h.signal_frequency > 0)
        
        return signal_count / len(recent_history) if recent_history else 0.0
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score from metrics"""
        score = 1.0
        
        # Adjust based on various metrics
        if 'sharpe_ratio' in metrics:
            sharpe = metrics['sharpe_ratio']
            if sharpe < 0:
                score *= 0.5
            elif sharpe > 2:
                score *= 1.2
        
        if 'max_drawdown' in metrics:
            drawdown = metrics['max_drawdown']
            if drawdown > 0.1:  # 10% drawdown
                score *= 0.8
            elif drawdown > 0.2:  # 20% drawdown
                score *= 0.6
        
        if 'win_rate' in metrics:
            win_rate = metrics['win_rate']
            if win_rate < 0.4:
                score *= 0.7
            elif win_rate > 0.6:
                score *= 1.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_risk_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate risk score from metrics"""
        risk_score = 0.0
        
        # Higher risk for higher volatility
        if 'volatility' in metrics:
            volatility = metrics['volatility']
            risk_score += min(volatility * 10, 0.5)
        
        # Higher risk for larger drawdowns
        if 'max_drawdown' in metrics:
            drawdown = metrics['max_drawdown']
            risk_score += min(drawdown * 5, 0.3)
        
        # Higher risk for lower Sharpe ratio
        if 'sharpe_ratio' in metrics:
            sharpe = metrics['sharpe_ratio']
            if sharpe < 0:
                risk_score += 0.2
            elif sharpe < 1:
                risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent() / 100.0
        except ImportError:
            return 0.0
    
    def get_strategy_health(self, strategy_id: str) -> Optional[StrategyHealth]:
        """Get current health for a strategy"""
        return self.monitored_strategies.get(strategy_id)
    
    def get_all_health(self) -> Dict[str, StrategyHealth]:
        """Get health for all monitored strategies"""
        return self.monitored_strategies.copy()
    
    def check_strategy_alerts(self, strategy_id: str) -> List[str]:
        """Check for alerts for a strategy"""
        alerts = []
        health = self.get_strategy_health(strategy_id)
        
        if not health:
            return alerts
        
        # Performance alerts
        if health.performance_score < 0.5:
            alerts.append(f"Low performance score: {health.performance_score:.2f}")
        
        # Risk alerts
        if health.risk_score > 0.7:
            alerts.append(f"High risk score: {health.risk_score:.2f}")
        
        # Signal frequency alerts
        if health.signal_frequency < 0.01:  # Very low signal frequency
            alerts.append("Very low signal frequency")
        
        # System resource alerts
        if health.memory_usage > 0.8:
            alerts.append(f"High memory usage: {health.memory_usage:.1%}")
        
        if health.cpu_usage > 0.8:
            alerts.append(f"High CPU usage: {health.cpu_usage:.1%}")
        
        # Time-based alerts
        time_since_signal = datetime.utcnow() - health.last_signal_time
        if time_since_signal > timedelta(hours=24):
            alerts.append(f"No signals for {time_since_signal.days} days")
        
        return alerts
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system-wide monitoring summary"""
        total_strategies = len(self.monitored_strategies)
        active_strategies = sum(1 for h in self.monitored_strategies.values() 
                              if h.status == "active")
        
        avg_performance = np.mean([h.performance_score for h in self.monitored_strategies.values()])
        avg_risk = np.mean([h.risk_score for h in self.monitored_strategies.values()])
        
        total_alerts = sum(len(self.check_strategy_alerts(sid)) 
                          for sid in self.monitored_strategies.keys())
        
        return {
            'total_strategies': total_strategies,
            'active_strategies': active_strategies,
            'average_performance': avg_performance,
            'average_risk': avg_risk,
            'total_alerts': total_alerts,
            'system_health': 'healthy' if total_alerts == 0 else 'degraded'
        } 