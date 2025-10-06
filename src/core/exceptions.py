"""
World-Class Exception Handling Framework
Comprehensive exception hierarchy for production trading system.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, ParamSpec, TypedDict, TypeVar, cast

try:  # Python 3.10 compatibility for typing.Unpack
    from typing import Unpack  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback to typing_extensions
    try:
        from typing_extensions import Unpack  # type: ignore
    except ImportError as exc:  # pragma: no cover - minimal shim if extensions missing
        class _Unpack:  # pylint: disable=too-few-public-methods
            def __getitem__(self, item):  # type: ignore[override]
                return item

        Unpack = _Unpack()  # type: ignore[assignment]

from src.core.types import JSONArray, JSONObject, JSONValue

# Type variables for decorators/utilities
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class _ExceptionKwargs(TypedDict, total=False):
    error_code: str | None
    context: JSONObject | None


# Minimal, stable core exception anchors
class CoreError(Exception):
    """Base error for core subsystems.

    Framework-agnostic and safe to import from anywhere:
        from src.core.exceptions import CoreError
    """

    pass


class ConfigurationError(CoreError):
    """Raised for configuration-related problems (missing keys, invalid values)."""

    pass


class DependencyError(CoreError):
    """Raised when a required dependency/service is unavailable or misconfigured."""

    pass


class ValidationError(CoreError):
    """Raised for validation failures in inputs, state, or configuration."""

    pass


class EMPException(Exception):
    """Base exception for EMP Proving Ground system."""

    def __init__(
        self, message: str, error_code: str | None = None, context: JSONObject | None = None
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.error_code: str = error_code or self.__class__.__name__
        self.context: JSONObject = context or {}
        self.timestamp: datetime = datetime.utcnow()
        # Log the exception immediately
        logger = logging.getLogger(__name__)
        logger.error(
            f"[{self.error_code}] {message}",
            extra={
                "error_code": self.error_code,
                "context": self.context,
                "timestamp": self.timestamp.isoformat(),
            },
        )


# Trading System Exceptions
class TradingException(EMPException):
    """Base exception for trading-related errors."""

    pass


class OrderExecutionException(TradingException):
    """Exception raised during order execution failures."""

    def __init__(
        self,
        message: str,
        order_id: str | None = None,
        symbol: str | None = None,
        **kwargs: JSONValue,
    ) -> None:
        context: JSONObject = {"order_id": order_id, "symbol": symbol}
        context.update(kwargs)
        super().__init__(message, context=context)


class FIXAPIException(TradingException):
    """Exception raised for FIX API communication errors."""

    def __init__(
        self,
        message: str,
        session_type: str | None = None,
        broker_response: str | None = None,
        **kwargs: JSONValue,
    ) -> None:
        context: JSONObject = {"session_type": session_type, "broker_response": broker_response}
        context.update(kwargs)
        super().__init__(message, context=context)


class RiskManagementException(TradingException):
    """Exception raised for risk management violations."""

    def __init__(
        self,
        message: str,
        risk_type: str | None = None,
        current_exposure: float | None = None,
        **kwargs: JSONValue,
    ) -> None:
        context: JSONObject = {"risk_type": risk_type, "current_exposure": current_exposure}
        context.update(kwargs)
        super().__init__(message, context=context)


# Data Integration Exceptions
class DataException(EMPException):
    """Base exception for data-related errors."""

    pass


class DataIngestionException(DataException):
    """Exception raised during data ingestion failures."""

    def __init__(
        self,
        message: str,
        data_source: str | None = None,
        symbol: str | None = None,
        **kwargs: JSONValue,
    ) -> None:
        context: JSONObject = {"data_source": data_source, "symbol": symbol}
        context.update(kwargs)
        super().__init__(message, context=context)


class DataValidationException(DataException):
    """Exception raised for data validation failures."""

    def __init__(
        self,
        message: str,
        validation_type: str | None = None,
        invalid_fields: list[str] | None = None,
        **kwargs: JSONValue,
    ) -> None:
        inv_fields_json = cast(JSONArray | None, None)
        if invalid_fields is not None:
            inv_fields_json = cast(JSONArray, [str(x) for x in invalid_fields])
        context: JSONObject = {
            "validation_type": validation_type,
            "invalid_fields": inv_fields_json,
        }
        context.update(kwargs)
        super().__init__(message, context=context)


class MarketDataException(DataException):
    """Exception raised for market data specific errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        **kwargs: JSONValue,
    ) -> None:
        context: JSONObject = {"provider": provider, "symbol": symbol, "timeframe": timeframe}
        context.update(kwargs)
        super().__init__(message, context=context)


# Evolution Engine Exceptions
class EvolutionException(EMPException):
    """Base exception for evolution engine errors."""

    pass


class PopulationException(EvolutionException):
    """Exception raised for population management errors."""

    def __init__(
        self,
        message: str,
        population_size: int | None = None,
        generation: int | None = None,
        **kwargs: JSONValue,
    ) -> None:
        context: JSONObject = {"population_size": population_size, "generation": generation}
        context.update(kwargs)
        super().__init__(message, context=context)


class FitnessEvaluationException(EvolutionException):
    """Exception raised during fitness evaluation failures."""

    def __init__(
        self,
        message: str,
        genome_id: str | None = None,
        fitness_type: str | None = None,
        **kwargs: JSONValue,
    ) -> None:
        context: JSONObject = {"genome_id": genome_id, "fitness_type": fitness_type}
        context.update(kwargs)
        super().__init__(message, context=context)


class GeneticOperationException(EvolutionException):
    """Exception raised during genetic operations (crossover, mutation)."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        parent_ids: list[str] | None = None,
        **kwargs: JSONValue,
    ) -> None:
        parent_ids_json = cast(JSONArray | None, None)
        if parent_ids is not None:
            parent_ids_json = cast(JSONArray, [str(x) for x in parent_ids])
        context: JSONObject = {"operation": operation, "parent_ids": parent_ids_json}
        context.update(kwargs)
        super().__init__(message, context=context)


# System Infrastructure Exceptions
class SystemException(EMPException):
    """Base exception for system infrastructure errors."""

    pass


class ConfigurationException(SystemException):
    """Exception raised for configuration errors."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: JSONValue | None = None,
        **kwargs: JSONValue,
    ) -> None:
        context: JSONObject = {"config_key": config_key, "config_value": str(config_value)}
        context.update(kwargs)
        super().__init__(message, context=context)


class ResourceException(SystemException):
    """Exception raised for resource management errors."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs: JSONValue,
    ) -> None:
        context: JSONObject = cast(
            JSONObject, {"resource_type": resource_type, "resource_id": resource_id}
        )
        context.update(kwargs)
        super().__init__(message, context=context)


class ValidationException(SystemException):
    """Exception raised for system validation failures."""

    def __init__(
        self,
        message: str,
        component: str | None = None,
        validation_rule: str | None = None,
        **kwargs: JSONValue,
    ) -> None:
        context: JSONObject = cast(
            JSONObject, {"component": component, "validation_rule": validation_rule}
        )
        context.update(kwargs)
        super().__init__(message, context=context)


# Exception Handler Utilities
class ExceptionHandler:
    """World-class exception handling utilities."""

    @staticmethod
    def handle_with_retry(
        func: Callable[[], T],
        max_retries: int = 3,
        exceptions: tuple[type[BaseException], ...] = (Exception,),
    ) -> T:
        """Handle function execution with retry logic."""
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        for attempt in range(max_retries):
            try:
                return func()
            except exceptions as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
        # Should be unreachable because the loop returns or raises
        raise RuntimeError("Exhausted retries without success")  # pragma: no cover

    @staticmethod
    def log_and_raise(
        exception_class: type[EMPException], message: str, **kwargs: Unpack[_ExceptionKwargs]
    ) -> None:
        """Log exception details and raise the exception."""
        logger = logging.getLogger(__name__)
        logger.error(f"Raising {exception_class.__name__}: {message}")
        raise exception_class(message, **kwargs)

    @staticmethod
    def create_error_context(operation: str, **kwargs: JSONValue) -> JSONObject:
        """Create standardized error context."""
        context: JSONObject = {
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
        }
        context.update(kwargs)
        return context


# Exception Decorators
def handle_exceptions(
    exception_class: type[EMPException] = EMPException,
    reraise: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R | None]]:
    """Decorator for standardized exception handling."""

    def decorator(func: Callable[P, R]) -> Callable[P, R | None]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
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
                    kwargs=str(kwargs),
                )

                if reraise:
                    raise exception_class(
                        f"Error in {func.__name__}: {str(e)}",
                        context=context,
                    ) from e
                else:
                    logging.error(f"Handled exception in {func.__name__}: {e}")
                    return None

        return wrapper

    return decorator


def trading_operation(func: Callable[P, R]) -> Callable[P, R | None]:
    """Decorator for trading operations with specialized error handling."""

    @handle_exceptions(TradingException)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        return func(*args, **kwargs)

    return wrapper


def data_operation(func: Callable[P, R]) -> Callable[P, R | None]:
    """Decorator for data operations with specialized error handling."""

    @handle_exceptions(DataException)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        return func(*args, **kwargs)

    return wrapper


def evolution_operation(func: Callable[P, R]) -> Callable[P, R | None]:
    """Decorator for evolution operations with specialized error handling."""

    @handle_exceptions(EvolutionException)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        return func(*args, **kwargs)

    return wrapper


__all__ = [
    "CoreError",
    "ConfigurationError",
    "DependencyError",
    "ValidationError",
    "EMPException",
    "TradingException",
    "OrderExecutionException",
    "FIXAPIException",
    "RiskManagementException",
    "DataException",
    "DataIngestionException",
    "DataValidationException",
    "MarketDataException",
    "EvolutionException",
    "PopulationException",
    "FitnessEvaluationException",
    "GeneticOperationException",
    "SystemException",
    "ConfigurationException",
    "ResourceException",
    "ValidationException",
]
