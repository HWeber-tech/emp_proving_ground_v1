"""
EMP Strategy Registry v1.1

Manages approved strategies, their lifecycle, and provides
registry services for strategy deployment and monitoring.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.genome.models.genome import DecisionGenome
from src.core.exceptions import GovernanceException
from src.core.event_bus import event_bus

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Status of registered strategies."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class StrategyType(Enum):
    """Types of trading strategies."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MACHINE_LEARNING = "machine_learning"
    EVOLVED = "evolved"
    HYBRID = "hybrid"


@dataclass
class StrategyMetadata:
    """Metadata for a registered strategy."""
    strategy_id: str
    name: str
    description: str
    version: str
    author: str
    created_at: datetime
    updated_at: datetime
    strategy_type: StrategyType
    status: StrategyStatus
    tags: List[str]
    risk_level: str
    expected_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    instruments: List[str]
    timeframes: List[str]
    parameters: Dict[str, Any]
    constraints: Dict[str, Any]
    dependencies: List[str]
    documentation_url: Optional[str] = None
    source_code_url: Optional[str] = None


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_id: str
    timestamp: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    recovery_time: float
    risk_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class StrategyRegistry:
    """Registry for managing approved strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.strategies: Dict[str, StrategyMetadata] = {}
        self.performance_history: Dict[str, List[StrategyPerformance]] = {}
        self.active_strategies: Set[str] = set()
        self.deprecated_strategies: Set[str] = set()
        
        # Registry configuration
        self.max_strategies = self.config.get('max_strategies', 1000)
        self.performance_history_days = self.config.get('performance_history_days', 365)
        self.auto_archive_days = self.config.get('auto_archive_days', 30)
        
        logger.info("Strategy Registry initialized")
        
    async def register_strategy(self, genome: DecisionGenome, metadata: Dict[str, Any]) -> str:
        """Register a new strategy from a genome."""
        try:
            # Generate strategy ID
            strategy_id = f"strategy_{genome.genome_id}_{datetime.now().timestamp()}"
            
            # Create strategy metadata
            strategy_metadata = StrategyMetadata(
                strategy_id=strategy_id,
                name=metadata.get('name', f"Evolved Strategy {strategy_id}"),
                description=metadata.get('description', 'Evolved trading strategy'),
                version=metadata.get('version', '1.0.0'),
                author=metadata.get('author', 'EMP Evolution Engine'),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                strategy_type=StrategyType.EVOLVED,
                status=StrategyStatus.DRAFT,
                tags=metadata.get('tags', ['evolved', 'genetic']),
                risk_level=metadata.get('risk_level', 'moderate'),
                expected_return=metadata.get('expected_return', 0.0),
                max_drawdown=metadata.get('max_drawdown', 0.0),
                sharpe_ratio=metadata.get('sharpe_ratio', 0.0),
                sortino_ratio=metadata.get('sortino_ratio', 0.0),
                win_rate=metadata.get('win_rate', 0.0),
                profit_factor=metadata.get('profit_factor', 0.0),
                total_trades=metadata.get('total_trades', 0),
                avg_trade_duration=metadata.get('avg_trade_duration', 0.0),
                instruments=metadata.get('instruments', ['EURUSD']),
                timeframes=metadata.get('timeframes', ['1H']),
                parameters=genome.to_dict(),
                constraints=metadata.get('constraints', {}),
                dependencies=metadata.get('dependencies', []),
                documentation_url=metadata.get('documentation_url'),
                source_code_url=metadata.get('source_code_url')
            )
            
            # Store strategy
            self.strategies[strategy_id] = strategy_metadata
            self.performance_history[strategy_id] = []
            
            # Emit strategy registered event
            await event_bus.publish('registry.strategy.registered', {
                'strategy_id': strategy_id,
                'genome_id': genome.genome_id,
                'status': strategy_metadata.status.value
            })
            
            logger.info(f"Registered strategy {strategy_id}")
            return strategy_id
            
        except Exception as e:
            raise GovernanceException(f"Error registering strategy: {e}")
            
    async def approve_strategy(self, strategy_id: str, approver: str, 
                             comments: str = "") -> bool:
        """Approve a strategy for deployment."""
        try:
            if strategy_id not in self.strategies:
                raise GovernanceException(f"Strategy {strategy_id} not found")
                
            strategy = self.strategies[strategy_id]
            
            # Update status
            strategy.status = StrategyStatus.APPROVED
            strategy.updated_at = datetime.now()
            
            # Emit strategy approved event
            await event_bus.publish('registry.strategy.approved', {
                'strategy_id': strategy_id,
                'approver': approver,
                'comments': comments
            })
            
            logger.info(f"Strategy {strategy_id} approved by {approver}")
            return True
            
        except Exception as e:
            raise GovernanceException(f"Error approving strategy: {e}")
            
    async def activate_strategy(self, strategy_id: str, activator: str) -> bool:
        """Activate a strategy for live trading."""
        try:
            if strategy_id not in self.strategies:
                raise GovernanceException(f"Strategy {strategy_id} not found")
                
            strategy = self.strategies[strategy_id]
            
            # Check if strategy is approved
            if strategy.status != StrategyStatus.APPROVED:
                raise GovernanceException(f"Strategy {strategy_id} is not approved")
                
            # Update status
            strategy.status = StrategyStatus.ACTIVE
            strategy.updated_at = datetime.now()
            self.active_strategies.add(strategy_id)
            
            # Emit strategy activated event
            await event_bus.publish('registry.strategy.activated', {
                'strategy_id': strategy_id,
                'activator': activator
            })
            
            logger.info(f"Strategy {strategy_id} activated by {activator}")
            return True
            
        except Exception as e:
            raise GovernanceException(f"Error activating strategy: {e}")
            
    async def suspend_strategy(self, strategy_id: str, suspender: str, 
                             reason: str) -> bool:
        """Suspend an active strategy."""
        try:
            if strategy_id not in self.strategies:
                raise GovernanceException(f"Strategy {strategy_id} not found")
                
            strategy = self.strategies[strategy_id]
            
            # Update status
            strategy.status = StrategyStatus.SUSPENDED
            strategy.updated_at = datetime.now()
            self.active_strategies.discard(strategy_id)
            
            # Emit strategy suspended event
            await event_bus.publish('registry.strategy.suspended', {
                'strategy_id': strategy_id,
                'suspender': suspender,
                'reason': reason
            })
            
            logger.info(f"Strategy {strategy_id} suspended by {suspender}")
            return True
            
        except Exception as e:
            raise GovernanceException(f"Error suspending strategy: {e}")
            
    async def deprecate_strategy(self, strategy_id: str, deprecator: str, 
                               reason: str) -> bool:
        """Deprecate a strategy."""
        try:
            if strategy_id not in self.strategies:
                raise GovernanceException(f"Strategy {strategy_id} not found")
                
            strategy = self.strategies[strategy_id]
            
            # Update status
            strategy.status = StrategyStatus.DEPRECATED
            strategy.updated_at = datetime.now()
            self.active_strategies.discard(strategy_id)
            self.deprecated_strategies.add(strategy_id)
            
            # Emit strategy deprecated event
            await event_bus.publish('registry.strategy.deprecated', {
                'strategy_id': strategy_id,
                'deprecator': deprecator,
                'reason': reason
            })
            
            logger.info(f"Strategy {strategy_id} deprecated by {deprecator}")
            return True
            
        except Exception as e:
            raise GovernanceException(f"Error deprecating strategy: {e}")
            
    async def update_performance(self, strategy_id: str, 
                               performance: StrategyPerformance) -> bool:
        """Update performance metrics for a strategy."""
        try:
            if strategy_id not in self.strategies:
                raise GovernanceException(f"Strategy {strategy_id} not found")
                
            # Add performance record
            self.performance_history[strategy_id].append(performance)
            
            # Update strategy metadata with latest performance
            strategy = self.strategies[strategy_id]
            strategy.sharpe_ratio = performance.sharpe_ratio
            strategy.sortino_ratio = performance.sortino_ratio
            strategy.win_rate = performance.win_rate
            strategy.profit_factor = performance.profit_factor
            strategy.total_trades = performance.total_trades
            strategy.updated_at = datetime.now()
            
            # Clean up old performance data
            await self._cleanup_old_performance(strategy_id)
            
            # Emit performance updated event
            await event_bus.publish('registry.performance.updated', {
                'strategy_id': strategy_id,
                'total_return': performance.total_return,
                'sharpe_ratio': performance.sharpe_ratio
            })
            
            return True
            
        except Exception as e:
            raise GovernanceException(f"Error updating performance: {e}")
            
    async def _cleanup_old_performance(self, strategy_id: str):
        """Clean up old performance data."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.performance_history_days)
            
            if strategy_id in self.performance_history:
                self.performance_history[strategy_id] = [
                    p for p in self.performance_history[strategy_id]
                    if p.timestamp > cutoff_date
                ]
                
        except Exception as e:
            logger.error(f"Error cleaning up old performance data: {e}")
            
    def get_strategy(self, strategy_id: str) -> Optional[StrategyMetadata]:
        """Get strategy by ID."""
        return self.strategies.get(strategy_id)
        
    def get_strategies_by_status(self, status: StrategyStatus) -> List[StrategyMetadata]:
        """Get strategies by status."""
        return [s for s in self.strategies.values() if s.status == status]
        
    def get_active_strategies(self) -> List[StrategyMetadata]:
        """Get all active strategies."""
        return [s for s in self.strategies.values() if s.status == StrategyStatus.ACTIVE]
        
    def get_strategies_by_type(self, strategy_type: StrategyType) -> List[StrategyMetadata]:
        """Get strategies by type."""
        return [s for s in self.strategies.values() if s.strategy_type == strategy_type]
        
    def get_strategies_by_instrument(self, instrument: str) -> List[StrategyMetadata]:
        """Get strategies that trade a specific instrument."""
        return [s for s in self.strategies.values() if instrument in s.instruments]
        
    def get_performance_history(self, strategy_id: str, 
                              days: Optional[int] = None) -> List[StrategyPerformance]:
        """Get performance history for a strategy."""
        if strategy_id not in self.performance_history:
            return []
            
        history = self.performance_history[strategy_id]
        
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            history = [p for p in history if p.timestamp > cutoff_date]
            
        return history
        
    def get_latest_performance(self, strategy_id: str) -> Optional[StrategyPerformance]:
        """Get latest performance for a strategy."""
        history = self.get_performance_history(strategy_id)
        return history[-1] if history else None
        
    def search_strategies(self, query: str) -> List[StrategyMetadata]:
        """Search strategies by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for strategy in self.strategies.values():
            if (query_lower in strategy.name.lower() or
                query_lower in strategy.description.lower() or
                any(query_lower in tag.lower() for tag in strategy.tags)):
                results.append(strategy)
                
        return results
        
    async def auto_archive_strategies(self):
        """Automatically archive old deprecated strategies."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.auto_archive_days)
            archived_count = 0
            
            for strategy_id, strategy in self.strategies.items():
                if (strategy.status == StrategyStatus.DEPRECATED and
                    strategy.updated_at < cutoff_date):
                    strategy.status = StrategyStatus.ARCHIVED
                    strategy.updated_at = datetime.now()
                    archived_count += 1
                    
            if archived_count > 0:
                logger.info(f"Auto-archived {archived_count} deprecated strategies")
                
        except Exception as e:
            logger.error(f"Error auto-archiving strategies: {e}")
            
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        status_counts = {}
        for status in StrategyStatus:
            status_counts[status.value] = len(self.get_strategies_by_status(status))
            
        type_counts = {}
        for strategy_type in StrategyType:
            type_counts[strategy_type.value] = len(self.get_strategies_by_type(strategy_type))
            
        return {
            'total_strategies': len(self.strategies),
            'active_strategies': len(self.active_strategies),
            'deprecated_strategies': len(self.deprecated_strategies),
            'status_counts': status_counts,
            'type_counts': type_counts,
            'performance_history_records': sum(len(h) for h in self.performance_history.values()),
            'max_strategies': self.max_strategies,
            'performance_history_days': self.performance_history_days
        }
        
    def export_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Export strategy data."""
        if strategy_id not in self.strategies:
            raise GovernanceException(f"Strategy {strategy_id} not found")
            
        strategy = self.strategies[strategy_id]
        performance_history = self.get_performance_history(strategy_id)
        
        return {
            'strategy': strategy.__dict__,
            'performance_history': [p.__dict__ for p in performance_history],
            'export_timestamp': datetime.now().isoformat()
        }
        
    def import_strategy(self, strategy_data: Dict[str, Any]) -> str:
        """Import strategy data."""
        try:
            strategy_dict = strategy_data['strategy']
            
            # Convert string dates back to datetime
            strategy_dict['created_at'] = datetime.fromisoformat(strategy_dict['created_at'])
            strategy_dict['updated_at'] = datetime.fromisoformat(strategy_dict['updated_at'])
            
            # Convert enum values
            strategy_dict['strategy_type'] = StrategyType(strategy_dict['strategy_type'])
            strategy_dict['status'] = StrategyStatus(strategy_dict['status'])
            
            # Create strategy metadata
            strategy = StrategyMetadata(**strategy_dict)
            
            # Store strategy
            self.strategies[strategy.strategy_id] = strategy
            
            # Import performance history
            if 'performance_history' in strategy_data:
                performance_history = []
                for p_dict in strategy_data['performance_history']:
                    p_dict['timestamp'] = datetime.fromisoformat(str(p_dict['timestamp']))
                    performance = StrategyPerformance(**p_dict)
                    performance_history.append(performance)
                    
                self.performance_history[strategy.strategy_id] = performance_history
                
            logger.info(f"Imported strategy {strategy.strategy_id}")
            return strategy.strategy_id
            
        except Exception as e:
            raise GovernanceException(f"Error importing strategy: {e}") 