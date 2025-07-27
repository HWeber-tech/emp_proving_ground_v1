"""
EMP Core Exceptions v1.2

Defines system-wide exception types for the EMP system. These exceptions
provide consistent error handling across all layers with detailed context.
"""

from typing import Optional, Dict, Any
import traceback
from datetime import datetime


class EMPException(Exception):
    """Base exception for all EMP system errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        self.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp,
            'exception_type': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        """String representation with error code."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class SensoryException(EMPException):
    """Exception raised by sensory layer operations."""
    
    def __init__(self, message: str, sensor_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="SENSORY_ERROR", **kwargs)
        self.sensor_type = sensor_type


class ThinkingException(EMPException):
    """Exception raised by thinking layer operations."""
    
    def __init__(self, message: str, thought_process: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="THINKING_ERROR", **kwargs)
        self.thought_process = thought_process


class TradingException(EMPException):
    """Exception raised by trading layer operations."""
    
    def __init__(self, message: str, strategy_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="TRADING_ERROR", **kwargs)
        self.strategy_id = strategy_id


class EvolutionException(EMPException):
    """Exception raised by evolution layer operations."""
    
    def __init__(self, message: str, generation: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="EVOLUTION_ERROR", **kwargs)
        self.generation = generation


class GovernanceException(EMPException):
    """Exception raised by governance layer operations."""
    
    def __init__(self, message: str, governance_rule: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="GOVERNANCE_ERROR", **kwargs)
        self.governance_rule = governance_rule


class OperationalException(EMPException):
    """Exception raised by operational layer operations."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="OPERATIONAL_ERROR", **kwargs)
        self.operation = operation


class ConfigurationException(EMPException):
    """Exception raised by configuration operations."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class ValidationException(EMPException):
    """Exception raised by validation operations."""
    
    def __init__(self, message: str, validation_rule: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.validation_rule = validation_rule


class DataException(EMPException):
    """Exception raised by data operations."""
    
    def __init__(self, message: str, data_source: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DATA_ERROR", **kwargs)
        self.data_source = data_source


class NetworkException(EMPException):
    """Exception raised by network operations."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)
        self.endpoint = endpoint


class TimeoutException(EMPException):
    """Exception raised by timeout operations."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)
        self.timeout_seconds = timeout_seconds


class InsufficientDataException(EMPException):
    """Exception raised when insufficient data is available."""
    
    def __init__(self, message: str, required_data_points: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="INSUFFICIENT_DATA", **kwargs)
        self.required_data_points = required_data_points


class StrategyException(EMPException):
    """Exception raised by strategy operations."""
    
    def __init__(self, message: str, strategy_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="STRATEGY_ERROR", **kwargs)
        self.strategy_id = strategy_id


class RiskException(EMPException):
    """Exception raised by risk management operations."""
    
    def __init__(self, message: str, risk_metric: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="RISK_ERROR", **kwargs)
        self.risk_metric = risk_metric


class ExecutionException(EMPException):
    """Exception raised by execution operations."""
    
    def __init__(self, message: str, order_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="EXECUTION_ERROR", **kwargs)
        self.order_id = order_id


class SimulationException(EMPException):
    """Exception raised by simulation operations."""
    
    def __init__(self, message: str, simulation_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="SIMULATION_ERROR", **kwargs)
        self.simulation_id = simulation_id


class GenomeException(EMPException):
    """Exception raised by genome operations."""
    
    def __init__(self, message: str, genome_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="GENOME_ERROR", **kwargs)
        self.genome_id = genome_id


class FitnessException(EMPException):
    """Exception raised by fitness evaluation operations."""
    
    def __init__(self, message: str, fitness_function: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="FITNESS_ERROR", **kwargs)
        self.fitness_function = fitness_function


class EventBusException(EMPException):
    """Exception raised by event bus operations."""
    
    def __init__(self, message: str, event_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="EVENT_BUS_ERROR", **kwargs)
        self.event_type = event_type


class StateStoreException(EMPException):
    """Exception raised by state store operations."""
    
    def __init__(self, message: str, state_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="STATE_STORE_ERROR", **kwargs)
        self.state_key = state_key


class AuditException(EMPException):
    """Exception raised by audit operations."""
    
    def __init__(self, message: str, audit_event: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="AUDIT_ERROR", **kwargs)
        self.audit_event = audit_event


class HumanApprovalException(EMPException):
    """Exception raised by human approval operations."""
    
    def __init__(self, message: str, approval_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="HUMAN_APPROVAL_ERROR", **kwargs)
        self.approval_type = approval_type


class IntegrationException(EMPException):
    """Exception raised by component integration operations."""
    
    def __init__(self, message: str, component_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="INTEGRATION_ERROR", **kwargs)
        self.component_name = component_name
