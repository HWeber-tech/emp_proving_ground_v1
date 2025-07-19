"""
EMP Human Gateway v1.1

Human approval gateway for the governance layer.
Manages human oversight and approval workflows for trading strategies.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class ApprovalLevel(Enum):
    """Approval level enumeration."""
    AUTO = "auto"
    SUPERVISOR = "supervisor"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class ApprovalRequest:
    """Approval request for human review."""
    
    def __init__(self, request_id: str, strategy_id: str, genome_id: str,
                 request_type: str, urgency: str = "normal",
                 approver: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.request_id = request_id
        self.strategy_id = strategy_id
        self.genome_id = genome_id
        self.request_type = request_type
        self.urgency = urgency
        self.approver = approver
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.status = ApprovalStatus.PENDING
        self.approval_level = self._determine_approval_level()
        self.expires_at = self._calculate_expiry()
        
    def _determine_approval_level(self) -> ApprovalLevel:
        """Determine required approval level based on request type and urgency."""
        if self.request_type == "strategy_deployment" and self.urgency == "high":
            return ApprovalLevel.MANAGER
        elif self.request_type == "risk_limit_change":
            return ApprovalLevel.SUPERVISOR
        elif self.request_type == "large_position":
            return ApprovalLevel.DIRECTOR
        else:
            return ApprovalLevel.AUTO
            
    def _calculate_expiry(self) -> datetime:
        """Calculate expiry time based on urgency."""
        if self.urgency == "high":
            return self.created_at + timedelta(hours=1)
        elif self.urgency == "normal":
            return self.created_at + timedelta(hours=4)
        else:
            return self.created_at + timedelta(hours=24)


class HumanGateway:
    """Human approval gateway for governance decisions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.auto_approval_threshold = self.config.get('auto_approval_threshold', 0.8)
        self.escalation_threshold = self.config.get('escalation_threshold', 0.3)
        self.requests: Dict[str, ApprovalRequest] = {}
        self.approvers: Dict[str, ApprovalLevel] = {}
        self.notification_callbacks: List[Callable] = []
        
        # Load existing requests
        self._load_requests()
        
        logger.info("Human Gateway initialized")
        
    def create_approval_request(self, strategy_id: str, genome_id: str,
                              request_type: str, urgency: str = "normal",
                              approver: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new approval request."""
        try:
            request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{strategy_id}"
            
            request = ApprovalRequest(
                request_id=request_id,
                strategy_id=strategy_id,
                genome_id=genome_id,
                request_type=request_type,
                urgency=urgency,
                approver=approver,
                metadata=metadata
            )
            
            self.requests[request_id] = request
            
            # Save requests
            self._save_requests()
            
            # Send notifications
            self._send_notifications(request)
            
            logger.info(f"Approval request created: {request_id} for {strategy_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Error creating approval request: {e}")
            raise
            
    def approve_request(self, request_id: str, approver: str,
                       reason: Optional[str] = None) -> bool:
        """Approve an approval request."""
        try:
            if request_id not in self.requests:
                logger.warning(f"Approval request not found: {request_id}")
                return False
                
            request = self.requests[request_id]
            
            # Check if request is still valid
            if request.status != ApprovalStatus.PENDING:
                logger.warning(f"Request {request_id} is not pending")
                return False
                
            if datetime.now() > request.expires_at:
                request.status = ApprovalStatus.EXPIRED
                logger.warning(f"Request {request_id} has expired")
                return False
                
            # Check approval level
            if not self._can_approve(approver, request.approval_level):
                logger.warning(f"Approver {approver} cannot approve level {request.approval_level}")
                return False
                
            # Approve request
            request.status = ApprovalStatus.APPROVED
            request.approver = approver
            request.updated_at = datetime.now()
            request.metadata['approval_reason'] = reason
            request.metadata['approval_timestamp'] = datetime.now().isoformat()
            
            # Save requests
            self._save_requests()
            
            logger.info(f"Request {request_id} approved by {approver}")
            return True
            
        except Exception as e:
            logger.error(f"Error approving request {request_id}: {e}")
            return False
            
    def reject_request(self, request_id: str, approver: str,
                      reason: Optional[str] = None) -> bool:
        """Reject an approval request."""
        try:
            if request_id not in self.requests:
                logger.warning(f"Approval request not found: {request_id}")
                return False
                
            request = self.requests[request_id]
            
            # Check if request is still valid
            if request.status != ApprovalStatus.PENDING:
                logger.warning(f"Request {request_id} is not pending")
                return False
                
            if datetime.now() > request.expires_at:
                request.status = ApprovalStatus.EXPIRED
                logger.warning(f"Request {request_id} has expired")
                return False
                
            # Reject request
            request.status = ApprovalStatus.REJECTED
            request.approver = approver
            request.updated_at = datetime.now()
            request.metadata['rejection_reason'] = reason
            request.metadata['rejection_timestamp'] = datetime.now().isoformat()
            
            # Save requests
            self._save_requests()
            
            logger.info(f"Request {request_id} rejected by {approver}")
            return True
            
        except Exception as e:
            logger.error(f"Error rejecting request {request_id}: {e}")
            return False
            
    def escalate_request(self, request_id: str, approver: str,
                        reason: Optional[str] = None) -> bool:
        """Escalate an approval request to higher level."""
        try:
            if request_id not in self.requests:
                logger.warning(f"Approval request not found: {request_id}")
                return False
                
            request = self.requests[request_id]
            
            # Check if request is still valid
            if request.status != ApprovalStatus.PENDING:
                logger.warning(f"Request {request_id} is not pending")
                return False
                
            # Escalate approval level
            current_level = request.approval_level
            escalated_level = self._escalate_approval_level(current_level)
            
            if escalated_level == current_level:
                logger.warning(f"Request {request_id} cannot be escalated further")
                return False
                
            request.approval_level = escalated_level
            request.status = ApprovalStatus.ESCALATED
            request.updated_at = datetime.now()
            request.metadata['escalation_reason'] = reason
            request.metadata['escalation_timestamp'] = datetime.now().isoformat()
            request.metadata['previous_level'] = current_level.value
            request.metadata['new_level'] = escalated_level.value
            
            # Save requests
            self._save_requests()
            
            # Send notifications for escalation
            self._send_notifications(request)
            
            logger.info(f"Request {request_id} escalated to {escalated_level.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error escalating request {request_id}: {e}")
            return False
            
    def get_pending_requests(self, approver: Optional[str] = None) -> List[ApprovalRequest]:
        """Get pending approval requests."""
        pending_requests = []
        
        for request in self.requests.values():
            if request.status == ApprovalStatus.PENDING:
                # Check if request has expired
                if datetime.now() > request.expires_at:
                    request.status = ApprovalStatus.EXPIRED
                    continue
                    
                # Filter by approver if specified
                if approver is None or self._can_approve(approver, request.approval_level):
                    pending_requests.append(request)
                    
        # Sort by urgency and creation time
        pending_requests.sort(key=lambda r: (r.urgency != "high", r.created_at))
        
        return pending_requests
        
    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a specific approval request."""
        return self.requests.get(request_id)
        
    def add_approver(self, approver_id: str, level: ApprovalLevel):
        """Add an approver with specified level."""
        self.approvers[approver_id] = level
        logger.info(f"Approver added: {approver_id} with level {level.value}")
        
    def remove_approver(self, approver_id: str):
        """Remove an approver."""
        if approver_id in self.approvers:
            del self.approvers[approver_id]
            logger.info(f"Approver removed: {approver_id}")
            
    def add_notification_callback(self, callback: Callable):
        """Add a notification callback."""
        self.notification_callbacks.append(callback)
        
    def _can_approve(self, approver: str, required_level: ApprovalLevel) -> bool:
        """Check if approver can approve at the required level."""
        if approver not in self.approvers:
            return False
            
        approver_level = self.approvers[approver]
        
        # Higher levels can approve lower level requests
        level_hierarchy = {
            ApprovalLevel.AUTO: 0,
            ApprovalLevel.SUPERVISOR: 1,
            ApprovalLevel.MANAGER: 2,
            ApprovalLevel.DIRECTOR: 3,
            ApprovalLevel.EXECUTIVE: 4
        }
        
        return level_hierarchy[approver_level] >= level_hierarchy[required_level]
        
    def _escalate_approval_level(self, current_level: ApprovalLevel) -> ApprovalLevel:
        """Escalate to the next approval level."""
        escalation_map = {
            ApprovalLevel.AUTO: ApprovalLevel.SUPERVISOR,
            ApprovalLevel.SUPERVISOR: ApprovalLevel.MANAGER,
            ApprovalLevel.MANAGER: ApprovalLevel.DIRECTOR,
            ApprovalLevel.DIRECTOR: ApprovalLevel.EXECUTIVE,
            ApprovalLevel.EXECUTIVE: ApprovalLevel.EXECUTIVE  # Cannot escalate further
        }
        
        return escalation_map.get(current_level, ApprovalLevel.EXECUTIVE)
        
    def _send_notifications(self, request: ApprovalRequest):
        """Send notifications for approval requests."""
        for callback in self.notification_callbacks:
            try:
                callback(request)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
                
    def _load_requests(self):
        """Load approval requests from file."""
        try:
            requests_file = Path("data/approval_requests.json")
            if requests_file.exists():
                with open(requests_file, 'r') as f:
                    data = json.load(f)
                    
                for request_data in data.get('requests', []):
                    request = self._deserialize_request(request_data)
                    if request:
                        self.requests[request.request_id] = request
                        
                logger.info(f"Loaded {len(self.requests)} approval requests")
                
        except Exception as e:
            logger.error(f"Error loading approval requests: {e}")
            
    def _save_requests(self):
        """Save approval requests to file."""
        try:
            requests_file = Path("data/approval_requests.json")
            requests_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'saved_at': datetime.now().isoformat(),
                'requests': [self._serialize_request(req) for req in self.requests.values()]
            }
            
            with open(requests_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving approval requests: {e}")
            
    def _serialize_request(self, request: ApprovalRequest) -> Dict[str, Any]:
        """Serialize approval request to dictionary."""
        return {
            'request_id': request.request_id,
            'strategy_id': request.strategy_id,
            'genome_id': request.genome_id,
            'request_type': request.request_type,
            'urgency': request.urgency,
            'approver': request.approver,
            'metadata': request.metadata,
            'created_at': request.created_at.isoformat(),
            'updated_at': request.updated_at.isoformat(),
            'status': request.status.value,
            'approval_level': request.approval_level.value,
            'expires_at': request.expires_at.isoformat()
        }
        
    def _deserialize_request(self, data: Dict[str, Any]) -> Optional[ApprovalRequest]:
        """Deserialize approval request from dictionary."""
        try:
            request = ApprovalRequest(
                request_id=data['request_id'],
                strategy_id=data['strategy_id'],
                genome_id=data['genome_id'],
                request_type=data['request_type'],
                urgency=data['urgency'],
                approver=data.get('approver'),
                metadata=data.get('metadata', {})
            )
            
            # Restore state
            request.created_at = datetime.fromisoformat(data['created_at'])
            request.updated_at = datetime.fromisoformat(data['updated_at'])
            request.status = ApprovalStatus(data['status'])
            request.approval_level = ApprovalLevel(data['approval_level'])
            request.expires_at = datetime.fromisoformat(data['expires_at'])
            
            return request
            
        except Exception as e:
            logger.error(f"Error deserializing request: {e}")
            return None 