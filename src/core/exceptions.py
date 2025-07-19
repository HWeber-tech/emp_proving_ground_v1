"""
EMP Core Exceptions v1.1

Defines system-wide exception types for the EMP system. These exceptions
provide consistent error handling across all layers.
"""

from typing import Optional, Dict, Any


class EMPException(Exception):
    """Base exception for all EMP system errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}


class SensoryException(EMPException):
    """Exception raised by sensory layer operations."""
    pass


class ThinkingException(EMPException):
    """Exception raised by thinking layer operations."""
    pass


class TradingException(EMPException):
    """Exception raised by trading layer operations."""
    pass


class EvolutionException(EMPException):
    """Exception raised by evolution layer operations."""
    pass


class GovernanceException(EMPException):
    """Exception raised by governance layer operations."""
    pass


class OperationalException(EMPException):
    """Exception raised by operational layer operations."""
    pass


class ConfigurationException(EMPException):
    """Exception raised by configuration operations."""
    pass


class ValidationException(EMPException):
    """Exception raised by validation operations."""
    pass


class DataException(EMPException):
    """Exception raised by data operations."""
    pass


class NetworkException(EMPException):
    """Exception raised by network operations."""
    pass


class TimeoutException(EMPException):
    """Exception raised by timeout operations."""
    pass


class InsufficientDataException(EMPException):
    """Exception raised when insufficient data is available."""
    pass


class StrategyException(EMPException):
    """Exception raised by strategy operations."""
    pass


class RiskException(EMPException):
    """Exception raised by risk management operations."""
    pass


class ExecutionException(EMPException):
    """Exception raised by execution operations."""
    pass


class SimulationException(EMPException):
    """Exception raised by simulation operations."""
    pass


class GenomeException(EMPException):
    """Exception raised by genome operations."""
    pass


class FitnessException(EMPException):
    """Exception raised by fitness evaluation operations."""
    pass


class EventBusException(EMPException):
    """Exception raised by event bus operations."""
    pass


class StateStoreException(EMPException):
    """Exception raised by state store operations."""
    pass


class AuditException(EMPException):
    """Exception raised by audit operations."""
    pass


class HumanApprovalException(EMPException):
    """Exception raised by human approval operations."""
    pass 