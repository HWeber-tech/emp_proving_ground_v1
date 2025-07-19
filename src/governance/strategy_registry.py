"""
EMP Strategy Registry v1.1

Manages strategy lifecycle and champion genomes for the governance layer
in the EMP Ultimate Architecture v1.1.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.events import FitnessReport, GovernanceDecision

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy lifecycle status."""
    REGISTERED = "registered"
    APPROVED = "approved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ApprovalLevel(Enum):
    """Approval levels for strategies."""
    AUTO = "auto"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


@dataclass
class StrategyRecord:
    """Record for a registered strategy."""
    strategy_id: str
    genome_id: str
    name: str
    description: str
    created_at: datetime
    status: StrategyStatus
    approval_level: ApprovalLevel
    fitness_score: float
    performance_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.approved_at:
            data['approved_at'] = self.approved_at.isoformat()
        if self.last_updated:
            data['last_updated'] = self.last_updated.isoformat()
        data['status'] = self.status.value
        data['approval_level'] = self.approval_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyRecord':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('approved_at'):
            data['approved_at'] = datetime.fromisoformat(data['approved_at'])
        if data.get('last_updated'):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        data['status'] = StrategyStatus(data['status'])
        data['approval_level'] = ApprovalLevel(data['approval_level'])
        return cls(**data)


class StrategyRegistry:
    """Registry for managing trading strategies."""
    
    def __init__(self, registry_file: str = "data/strategy_registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.strategies: Dict[str, StrategyRecord] = {}
        self.champion_strategies: List[str] = []
        self.max_champions: int = 10
        
        logger.info(f"Strategy Registry initialized with file: {registry_file}")
        self._load_registry()
        
    def _load_registry(self):
        """Load registry from file."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    
                # Load strategies
                for strategy_data in data.get('strategies', []):
                    strategy = StrategyRecord.from_dict(strategy_data)
                    self.strategies[strategy.strategy_id] = strategy
                    
                # Load champion list
                self.champion_strategies = data.get('champions', [])
                
                logger.info(f"Loaded {len(self.strategies)} strategies from registry")
            else:
                logger.info("No existing registry file found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            
    def _save_registry(self):
        """Save registry to file."""
        try:
            data = {
                'strategies': [strategy.to_dict() for strategy in self.strategies.values()],
                'champions': self.champion_strategies,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Registry saved to file")
            
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            
    def register_strategy(self, strategy_id: str, genome_id: str, name: str, 
                         description: str, fitness_score: float,
                         performance_metrics: Dict[str, Any],
                         risk_metrics: Dict[str, Any],
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a new strategy."""
        try:
            if strategy_id in self.strategies:
                logger.warning(f"Strategy {strategy_id} already registered")
                return False
                
            # Determine approval level based on risk metrics
            approval_level = self._determine_approval_level(risk_metrics)
            
            strategy = StrategyRecord(
                strategy_id=strategy_id,
                genome_id=genome_id,
                name=name,
                description=description,
                created_at=datetime.now(),
                status=StrategyStatus.REGISTERED,
                approval_level=approval_level,
                fitness_score=fitness_score,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                metadata=metadata or {}
            )
            
            self.strategies[strategy_id] = strategy
            self._save_registry()
            
            logger.info(f"Registered strategy: {strategy_id} ({name})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering strategy: {e}")
            return False
            
    def approve_strategy(self, strategy_id: str, approver: str, 
                        auto_approve: bool = False) -> bool:
        """Approve a strategy."""
        try:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
                
            strategy = self.strategies[strategy_id]
            
            if strategy.status != StrategyStatus.REGISTERED:
                logger.warning(f"Strategy {strategy_id} is not in REGISTERED status")
                return False
                
            # Check if auto-approval is allowed
            if not auto_approve and strategy.approval_level in [ApprovalLevel.HIGH_RISK, ApprovalLevel.CRITICAL]:
                logger.info(f"Strategy {strategy_id} requires manual approval")
                return False
                
            strategy.status = StrategyStatus.APPROVED
            strategy.approved_by = approver
            strategy.approved_at = datetime.now()
            strategy.last_updated = datetime.now()
            
            self._save_registry()
            
            logger.info(f"Approved strategy: {strategy_id} by {approver}")
            return True
            
        except Exception as e:
            logger.error(f"Error approving strategy: {e}")
            return False
            
    def activate_strategy(self, strategy_id: str) -> bool:
        """Activate an approved strategy."""
        try:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
                
            strategy = self.strategies[strategy_id]
            
            if strategy.status != StrategyStatus.APPROVED:
                logger.warning(f"Strategy {strategy_id} is not in APPROVED status")
                return False
                
            strategy.status = StrategyStatus.ACTIVE
            strategy.last_updated = datetime.now()
            
            self._save_registry()
            
            logger.info(f"Activated strategy: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error activating strategy: {e}")
            return False
            
    def suspend_strategy(self, strategy_id: str, reason: str = "") -> bool:
        """Suspend an active strategy."""
        try:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
                
            strategy = self.strategies[strategy_id]
            
            if strategy.status != StrategyStatus.ACTIVE:
                logger.warning(f"Strategy {strategy_id} is not in ACTIVE status")
                return False
                
            strategy.status = StrategyStatus.SUSPENDED
            strategy.last_updated = datetime.now()
            strategy.metadata['suspension_reason'] = reason
            
            self._save_registry()
            
            logger.info(f"Suspended strategy: {strategy_id} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error suspending strategy: {e}")
            return False
            
    def update_champion_strategies(self, fitness_reports: List[FitnessReport]) -> bool:
        """Update champion strategies based on fitness reports."""
        try:
            # Create temporary list of strategies with fitness scores
            strategy_scores = []
            
            for report in fitness_reports:
                if report.strategy_id in self.strategies:
                    strategy_scores.append((report.strategy_id, report.fitness_score))
                    
            # Sort by fitness score (descending)
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Update champion list
            new_champions = [strategy_id for strategy_id, _ in strategy_scores[:self.max_champions]]
            
            if new_champions != self.champion_strategies:
                self.champion_strategies = new_champions
                self._save_registry()
                logger.info(f"Updated champion strategies: {new_champions}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating champion strategies: {e}")
            return False
            
    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)
        
    def get_strategies_by_status(self, status: StrategyStatus) -> List[StrategyRecord]:
        """Get all strategies with a specific status."""
        return [s for s in self.strategies.values() if s.status == status]
        
    def get_active_strategies(self) -> List[StrategyRecord]:
        """Get all active strategies."""
        return self.get_strategies_by_status(StrategyStatus.ACTIVE)
        
    def get_champion_strategies(self) -> List[StrategyRecord]:
        """Get champion strategies."""
        return [self.strategies[sid] for sid in self.champion_strategies if sid in self.strategies]
        
    def get_strategies_needing_approval(self) -> List[StrategyRecord]:
        """Get strategies that need manual approval."""
        return [s for s in self.strategies.values() 
                if s.status == StrategyStatus.REGISTERED 
                and s.approval_level in [ApprovalLevel.HIGH_RISK, ApprovalLevel.CRITICAL]]
        
    def _determine_approval_level(self, risk_metrics: Dict[str, Any]) -> ApprovalLevel:
        """Determine approval level based on risk metrics."""
        try:
            max_dd = risk_metrics.get('max_drawdown', 0)
            var_95 = abs(risk_metrics.get('var_95', 0))
            volatility = risk_metrics.get('volatility', 0)
            
            # Critical risk
            if max_dd > 0.25 or var_95 > 0.15 or volatility > 0.40:
                return ApprovalLevel.CRITICAL
                
            # High risk
            elif max_dd > 0.15 or var_95 > 0.10 or volatility > 0.30:
                return ApprovalLevel.HIGH_RISK
                
            # Medium risk
            elif max_dd > 0.10 or var_95 > 0.05 or volatility > 0.20:
                return ApprovalLevel.MEDIUM_RISK
                
            # Low risk
            elif max_dd > 0.05 or var_95 > 0.02 or volatility > 0.10:
                return ApprovalLevel.LOW_RISK
                
            # Auto approval
            else:
                return ApprovalLevel.AUTO
                
        except Exception as e:
            logger.error(f"Error determining approval level: {e}")
            return ApprovalLevel.MEDIUM_RISK
            
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get registry summary statistics."""
        status_counts = {}
        for status in StrategyStatus:
            status_counts[status.value] = len(self.get_strategies_by_status(status))
            
        return {
            'total_strategies': len(self.strategies),
            'champion_strategies': len(self.champion_strategies),
            'status_counts': status_counts,
            'pending_approval': len(self.get_strategies_needing_approval()),
            'active_strategies': len(self.get_active_strategies())
        } 