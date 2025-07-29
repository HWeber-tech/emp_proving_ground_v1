"""
World-Class Exception Handling Framework
Comprehensive exception hierarchy for production trading system.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime


class EMPException(Exception):
    """Base exception for EMP Proving Ground system."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        
        # Log the exception immediately
        logger = logging.getLogger(__name__)
        logger.error(f"[{self.error_code}] {message}", extra={
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        })


# Trading System Exceptions
class TradingException(EMPException):
    """Base exception for trading-related errors."""
    pass


class OrderExecutionException(TradingException):
    """Exception raised during order execution failures."""
    
    def __init__(self, message: str, order_id: str = None, symbol: str = None, **kwargs):
        context = {'order_id': order_id, 'symbol': symbol}
        context.update(kwargs)
        super().__init__(message, context=context)


class FIXAPIException(TradingException):
    """Exception raised for FIX API communication errors."""
    
    def __init__(self, message: str, session_type: str = None, broker_response: str = None, **kwargs):
        context = {'session_type': session_type, 'broker_response': broker_response}
        context.update(kwargs)
        super().__init__(message, context=context)


class RiskManagementException(TradingException):
    """Exception raised for risk management violations."""
    
    def __init__(self, message: str, risk_type: str = None, current_exposure: float = None, **kwargs):
        context = {'risk_type': risk_type, 'current_exposure': current_exposure}
        context.update(kwargs)
        super().__init__(message, context=context)


# Data Integration Exceptions
class DataException(EMPException):
    """Base exception for data-related errors."""
    pass


class DataIngestionException(DataException):
    """Exception raised during data ingestion failures."""
    
    def __init__(self, message: str, data_source: str = None, symbol: str = None, **kwargs):
        context = {'data_source': data_source, 'symbol': symbol}
        context.update(kwargs)
        super().__init__(message, context=context)


class DataValidationException(DataException):
    """Exception raised for data validation failures."""
    
    def __init__(self, message: str, validation_type: str = None, invalid_fields: list = None, **kwargs):
        context = {'validation_type': validation_type, 'invalid_fields': invalid_fields}
        context.update(kwargs)
        super().__init__(message, context=context)


class MarketDataException(DataException):
    """Exception raised for market data specific errors."""
    
    def __init__(self, message: str, provider: str = None, symbol: str = None, timeframe: str = None, **kwargs):
        context = {'provider': provider, 'symbol': symbol, 'timeframe': timeframe}
        context.update(kwargs)
        super().__init__(message, context=context)


# Evolution Engine Exceptions
class EvolutionException(EMPException):
    """Base exception for evolution engine errors."""
    pass


class PopulationException(EvolutionException):
    """Exception raised for population management errors."""
    
    def __init__(self, message: str, population_size: int = None, generation: int = None, **kwargs):
        context = {'population_size': population_size, 'generation': generation}
        context.update(kwargs)
        super().__init__(message, context=context)


class FitnessEvaluationException(EvolutionException):
    """Exception raised during fitness evaluation failures."""
    
    def __init__(self, message: str, genome_id: str = None, fitness_type: str = None, **kwargs):
        context = {'genome_id': genome_id, 'fitness_type': fitness_type}
        context.update(kwargs)
        super().__init__(message, context=context)


class GeneticOperationException(EvolutionException):
    """Exception raised during genetic operations (crossover, mutation)."""
    
    def __init__(self, message: str, operation: str = None, parent_ids: list = None, **kwargs):
        context = {'operation': operation, 'parent_ids': parent_ids}
        context.update(kwargs)
        super().__init__(message, context=context)


# System Infrastructure Exceptions
class SystemException(EMPException):
    """Base exception for system infrastructure errors."""
    pass


class ConfigurationException(SystemException):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str, config_key: str = None, config_value: Any = None, **kwargs):
        context = {'config_key': config_key, 'config_value': str(config_value)}
        context.update(kwargs)
        super().__init__(message, context=context)


class ResourceException(SystemException):
    """Exception raised for resource management errors."""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None, **kwargs):
        context = {'resource_type': resource_type, 'resource_id': resource_id}
        context.update(kwargs)
        super().__init__(message, context=context)


class ValidationException(SystemException):
    """Exception raised for system validation failures."""
    
    def __init__(self, message: str, component: str = None, validation_rule: str = None, **kwargs):
        context = {'component': component, 'validation_rule': validation_rule}
        context.update(kwargs)
        super().__init__(message, context=context)


# Exception Handler Utilities
class ExceptionHandler:
    """World-class exception handling utilities."""
    
    @staticmethod
    def handle_with_retry(func, max_retries: int = 3, exceptions: tuple = (Exception,)):
        """Handle function execution with retry logic."""
        for attempt in range(max_retries):
            try:
                return func()
            except exceptions as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
    
    @staticmethod
    def log_and_raise(exception_class, message: str, **kwargs):
        """Log exception details and raise the exception."""
        logger = logging.getLogger(__name__)
        logger.error(f"Raising {exception_class.__name__}: {message}")
        raise exception_class(message, **kwargs)
    
    @staticmethod
    def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
        """Create standardized error context."""
        context = {
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat(),
        }
        context.update(kwargs)
        return context


# Exception Decorators
def handle_exceptions(exception_class=EMPException, reraise=True):
    """Decorator for standardized exception handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, EMPException):
                    if reraise:
                        raise
                    return None
                
                # Convert generic exceptions to EMP exceptions
                context = ExceptionHandler.create_error_context(
                    operation=func.__name__,
                    args=str(args),
                    kwargs=str(kwargs)
                )
                
                if reraise:
                    raise exception_class(
                        f"Error in {func.__name__}: {str(e)}",
                        context=context
                    ) from e
                else:
                    logging.error(f"Handled exception in {func.__name__}: {e}")
                    return None
        return wrapper
    return decorator


def trading_operation(func):
    """Decorator for trading operations with specialized error handling."""
    @handle_exceptions(TradingException)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def data_operation(func):
    """Decorator for data operations with specialized error handling."""
    @handle_exceptions(DataException)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def evolution_operation(func):
    """Decorator for evolution operations with specialized error handling."""
    @handle_exceptions(EvolutionException)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

