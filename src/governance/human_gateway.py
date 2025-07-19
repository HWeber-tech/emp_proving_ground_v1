"""
EMP Human Approval Gateway v1.1

Provides human oversight and approval mechanisms for evolved strategies,
ensuring compliance with governance rules and risk limits.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.genome.models.genome import DecisionGenome
from src.core.exceptions import GovernanceException
from src.core.event_bus import event_bus

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNDER_REVIEW = "under_review"


class ApprovalLevel(Enum):
    """Levels of approval required."""
    AUTOMATIC = "automatic"
    SUPERVISOR = "supervisor"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


@dataclass
class ApprovalRequest:
    """Approval request for a strategy or genome."""
    request_id: str
    genome_id: str
    genome: DecisionGenome
    requester: str
    timestamp: datetime
    approval_level: ApprovalLevel
    status: ApprovalStatus
    risk_assessment: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    compliance_check: Dict[str, Any]
    comments: str = ""
    approver: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    expiry_time: Optional[datetime] = None


@dataclass
class GovernanceRule:
    """Governance rule for approval decisions."""
    rule_id: str
    name: str
    description: str
    rule_type: str  # 'risk', 'performance', 'compliance', 'operational'
    conditions: Dict[str, Any]
    action: str  # 'approve', 'reject', 'escalate', 'require_review'
    priority: int
    enabled: bool = True


class HumanApprovalGateway:
    """Gateway for human approval of evolved strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        self.governance_rules: List[GovernanceRule] = []
        self.approvers: Dict[ApprovalLevel, List[str]] = {}
        self.approval_callbacks: Dict[str, Callable] = {}
        
        # Configuration
        self.auto_approval_threshold = self.config.get('auto_approval_threshold', 0.8)
        self.approval_timeout_hours = self.config.get('approval_timeout_hours', 24)
        self.max_risk_score = self.config.get('max_risk_score', 0.7)
        self.min_performance_score = self.config.get('min_performance_score', 0.6)
        
        # Initialize default governance rules
        self._initialize_default_rules()
        
        logger.info("Human Approval Gateway initialized")
        
    def _initialize_default_rules(self):
        """Initialize default governance rules."""
        default_rules = [
            GovernanceRule(
                rule_id="risk_limit",
                name="Risk Limit Check",
                description="Reject strategies exceeding maximum risk threshold",
                rule_type="risk",
                conditions={"max_risk_score": self.max_risk_score},
                action="reject",
                priority=1
            ),
            GovernanceRule(
                rule_id="performance_threshold",
                name="Performance Threshold",
                description="Require review for strategies below performance threshold",
                rule_type="performance",
                conditions={"min_performance_score": self.min_performance_score},
                action="require_review",
                priority=2
            ),
            GovernanceRule(
                rule_id="auto_approval",
                name="Auto Approval",
                description="Auto-approve high-performing, low-risk strategies",
                rule_type="operational",
                conditions={"fitness_threshold": self.auto_approval_threshold},
                action="approve",
                priority=3
            ),
            GovernanceRule(
                rule_id="compliance_check",
                name="Compliance Check",
                description="Ensure strategy compliance with regulations",
                rule_type="compliance",
                conditions={"compliance_required": True},
                action="escalate",
                priority=4
            )
        ]
        
        for rule in default_rules:
            self.add_governance_rule(rule)
            
    async def request_approval(self, genome: DecisionGenome, requester: str,
                             risk_assessment: Dict[str, Any],
                             performance_metrics: Dict[str, Any],
                             compliance_check: Dict[str, Any]) -> str:
        """Request approval for a genome."""
        try:
            # Generate request ID
            request_id = f"approval_{genome.genome_id}_{datetime.now().timestamp()}"
            
            # Determine approval level
            approval_level = self._determine_approval_level(
                genome, risk_assessment, performance_metrics, compliance_check
            )
            
            # Create approval request
            request = ApprovalRequest(
                request_id=request_id,
                genome_id=genome.genome_id,
                genome=genome,
                requester=requester,
                timestamp=datetime.now(),
                approval_level=approval_level,
                status=ApprovalStatus.PENDING,
                risk_assessment=risk_assessment,
                performance_metrics=performance_metrics,
                compliance_check=compliance_check,
                expiry_time=datetime.now() + timedelta(hours=self.approval_timeout_hours)
            )
            
            # Store request
            self.approval_requests[request_id] = request
            
            # Apply governance rules
            await self._apply_governance_rules(request)
            
            # Emit approval requested event
            await event_bus.publish('governance.approval.requested', {
                'request_id': request_id,
                'genome_id': genome.genome_id,
                'approval_level': approval_level.value,
                'requester': requester
            })
            
            logger.info(f"Approval requested for genome {genome.genome_id} at level {approval_level.value}")
            return request_id
            
        except Exception as e:
            raise GovernanceException(f"Error requesting approval: {e}")
            
    async def approve_request(self, request_id: str, approver: str, 
                            comments: str = "") -> bool:
        """Approve an approval request."""
        try:
            if request_id not in self.approval_requests:
                raise GovernanceException(f"Approval request {request_id} not found")
                
            request = self.approval_requests[request_id]
            
            # Check if request is still pending
            if request.status != ApprovalStatus.PENDING:
                raise GovernanceException(f"Request {request_id} is not pending")
                
            # Check if request has expired
            if request.expiry_time and datetime.now() > request.expiry_time:
                request.status = ApprovalStatus.EXPIRED
                raise GovernanceException(f"Request {request_id} has expired")
                
            # Update request
            request.status = ApprovalStatus.APPROVED
            request.approver = approver
            request.approval_timestamp = datetime.now()
            request.comments = comments
            
            # Emit approval granted event
            await event_bus.publish('governance.approval.granted', {
                'request_id': request_id,
                'genome_id': request.genome_id,
                'approver': approver,
                'approval_level': request.approval_level.value
            })
            
            # Execute approval callback
            if request_id in self.approval_callbacks:
                await self.approval_callbacks[request_id](request)
                
            logger.info(f"Approval granted for request {request_id} by {approver}")
            return True
            
        except Exception as e:
            raise GovernanceException(f"Error approving request: {e}")
            
    async def reject_request(self, request_id: str, approver: str, 
                           rejection_reason: str) -> bool:
        """Reject an approval request."""
        try:
            if request_id not in self.approval_requests:
                raise GovernanceException(f"Approval request {request_id} not found")
                
            request = self.approval_requests[request_id]
            
            # Check if request is still pending
            if request.status != ApprovalStatus.PENDING:
                raise GovernanceException(f"Request {request_id} is not pending")
                
            # Update request
            request.status = ApprovalStatus.REJECTED
            request.approver = approver
            request.approval_timestamp = datetime.now()
            request.rejection_reason = rejection_reason
            
            # Emit approval rejected event
            await event_bus.publish('governance.approval.rejected', {
                'request_id': request_id,
                'genome_id': request.genome_id,
                'approver': approver,
                'rejection_reason': rejection_reason
            })
            
            logger.info(f"Approval rejected for request {request_id} by {approver}")
            return True
            
        except Exception as e:
            raise GovernanceException(f"Error rejecting request: {e}")
            
    def _determine_approval_level(self, genome: DecisionGenome,
                                risk_assessment: Dict[str, Any],
                                performance_metrics: Dict[str, Any],
                                compliance_check: Dict[str, Any]) -> ApprovalLevel:
        """Determine required approval level based on risk and performance."""
        try:
            # Extract key metrics
            risk_score = risk_assessment.get('overall_risk', 0.5)
            fitness_score = genome.fitness_score
            compliance_score = compliance_check.get('compliance_score', 1.0)
            
            # High risk or low compliance requires executive approval
            if risk_score > 0.8 or compliance_score < 0.7:
                return ApprovalLevel.EXECUTIVE
                
            # Medium risk requires director approval
            if risk_score > 0.6:
                return ApprovalLevel.DIRECTOR
                
            # Low performance requires manager approval
            if fitness_score < 0.5:
                return ApprovalLevel.MANAGER
                
            # Standard approval for supervisor
            if fitness_score < 0.7:
                return ApprovalLevel.SUPERVISOR
                
            # Automatic approval for high-performing, low-risk strategies
            return ApprovalLevel.AUTOMATIC
            
        except Exception as e:
            logger.error(f"Error determining approval level: {e}")
            return ApprovalLevel.MANAGER  # Default to manager approval
            
    async def _apply_governance_rules(self, request: ApprovalRequest):
        """Apply governance rules to approval request."""
        try:
            # Sort rules by priority
            sorted_rules = sorted(self.governance_rules, key=lambda r: r.priority)
            
            for rule in sorted_rules:
                if not rule.enabled:
                    continue
                    
                # Check if rule conditions are met
                if self._check_rule_conditions(rule, request):
                    # Apply rule action
                    await self._apply_rule_action(rule, request)
                    
                    # Stop processing if rule results in final decision
                    if request.status in [ApprovalStatus.APPROVED, ApprovalStatus.REJECTED]:
                        break
                        
        except Exception as e:
            logger.error(f"Error applying governance rules: {e}")
            
    def _check_rule_conditions(self, rule: GovernanceRule, request: ApprovalRequest) -> bool:
        """Check if rule conditions are met."""
        try:
            conditions = rule.conditions
            
            if rule.rule_type == "risk":
                risk_score = request.risk_assessment.get('overall_risk', 0.0)
                max_risk = conditions.get('max_risk_score', 1.0)
                return risk_score > max_risk
                
            elif rule.rule_type == "performance":
                fitness_score = request.genome.fitness_score
                min_performance = conditions.get('min_performance_score', 0.0)
                return fitness_score < min_performance
                
            elif rule.rule_type == "operational":
                fitness_score = request.genome.fitness_score
                fitness_threshold = conditions.get('fitness_threshold', 0.0)
                return fitness_score >= fitness_threshold
                
            elif rule.rule_type == "compliance":
                compliance_required = conditions.get('compliance_required', False)
                return compliance_required
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking rule conditions: {e}")
            return False
            
    async def _apply_rule_action(self, rule: GovernanceRule, request: ApprovalRequest):
        """Apply rule action to approval request."""
        try:
            if rule.action == "approve":
                request.status = ApprovalStatus.APPROVED
                request.approver = "governance_rule"
                request.approval_timestamp = datetime.now()
                request.comments = f"Auto-approved by rule: {rule.name}"
                
            elif rule.action == "reject":
                request.status = ApprovalStatus.REJECTED
                request.approver = "governance_rule"
                request.approval_timestamp = datetime.now()
                request.rejection_reason = f"Rejected by rule: {rule.name}"
                
            elif rule.action == "escalate":
                # Escalate to higher approval level
                if request.approval_level == ApprovalLevel.SUPERVISOR:
                    request.approval_level = ApprovalLevel.MANAGER
                elif request.approval_level == ApprovalLevel.MANAGER:
                    request.approval_level = ApprovalLevel.DIRECTOR
                elif request.approval_level == ApprovalLevel.DIRECTOR:
                    request.approval_level = ApprovalLevel.EXECUTIVE
                    
            elif rule.action == "require_review":
                request.status = ApprovalStatus.UNDER_REVIEW
                request.comments = f"Under review by rule: {rule.name}"
                
        except Exception as e:
            logger.error(f"Error applying rule action: {e}")
            
    def add_governance_rule(self, rule: GovernanceRule):
        """Add a governance rule."""
        self.governance_rules.append(rule)
        logger.info(f"Added governance rule: {rule.name}")
        
    def remove_governance_rule(self, rule_id: str):
        """Remove a governance rule."""
        self.governance_rules = [r for r in self.governance_rules if r.rule_id != rule_id]
        logger.info(f"Removed governance rule: {rule_id}")
        
    def add_approver(self, level: ApprovalLevel, approver: str):
        """Add an approver for a specific level."""
        if level not in self.approvers:
            self.approvers[level] = []
        self.approvers[level].append(approver)
        logger.info(f"Added approver {approver} for level {level.value}")
        
    def remove_approver(self, level: ApprovalLevel, approver: str):
        """Remove an approver for a specific level."""
        if level in self.approvers:
            self.approvers[level] = [a for a in self.approvers[level] if a != approver]
        logger.info(f"Removed approver {approver} for level {level.value}")
        
    def get_pending_requests(self, approver: Optional[str] = None) -> List[ApprovalRequest]:
        """Get pending approval requests."""
        pending = [r for r in self.approval_requests.values() if r.status == ApprovalStatus.PENDING]
        
        if approver:
            # Filter by approver's approval level
            approver_levels = []
            for level, approvers in self.approvers.items():
                if approver in approvers:
                    approver_levels.append(level)
                    
            pending = [r for r in pending if r.approval_level in approver_levels]
            
        return pending
        
    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get approval request by ID."""
        return self.approval_requests.get(request_id)
        
    def set_approval_callback(self, request_id: str, callback: Callable):
        """Set callback function for when approval is granted."""
        self.approval_callbacks[request_id] = callback
        
    async def cleanup_expired_requests(self):
        """Clean up expired approval requests."""
        try:
            expired_requests = []
            
            for request_id, request in self.approval_requests.items():
                if (request.status == ApprovalStatus.PENDING and 
                    request.expiry_time and 
                    datetime.now() > request.expiry_time):
                    request.status = ApprovalStatus.EXPIRED
                    expired_requests.append(request_id)
                    
            if expired_requests:
                logger.info(f"Cleaned up {len(expired_requests)} expired requests")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired requests: {e}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get gateway summary."""
        status_counts = {}
        for status in ApprovalStatus:
            status_counts[status.value] = len([
                r for r in self.approval_requests.values() if r.status == status
            ])
            
        return {
            'total_requests': len(self.approval_requests),
            'status_counts': status_counts,
            'governance_rules': len(self.governance_rules),
            'approvers': {level.value: approvers for level, approvers in self.approvers.items()},
            'auto_approval_threshold': self.auto_approval_threshold,
            'approval_timeout_hours': self.approval_timeout_hours
        } 