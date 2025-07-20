"""
Enhanced Structured Logging with Correlation IDs and Metrics

Provides comprehensive logging with correlation IDs, performance metrics,
and audit trails for production-grade observability.
"""

import logging
import json
import uuid
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from contextvars import ContextVar
import traceback

from src.core.events import EventBus
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

@dataclass
class LogEntry:
    """Structured log entry with metadata."""
    timestamp: str
    level: str
    message: str
    correlation_id: str
    component: str
    operation: str
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

class StructuredLogger:
    """
    Enhanced structured logger with correlation IDs and metrics.
    
    Features:
    - Correlation ID tracking across requests
    - Performance metrics collection
    - Error tracking with stack traces
    - Audit trail for trading actions
    - Real-time log streaming
    """
    
    def __init__(self, event_bus: EventBus, state_store: StateStore):
        self.event_bus = event_bus
        self.state_store = state_store
        self.component = "EMP"
        self.log_buffer: List[LogEntry] = []
        self.max_buffer_size = 1000
        
    def set_correlation_id(self, cid: str) -> None:
        """Set correlation ID for current context."""
        correlation_id.set(cid)
        
    def get_correlation_id(self) -> str:
        """Get current correlation ID."""
        current = correlation_id.get()
        if not current:
            current = str(uuid.uuid4())
            self.set_correlation_id(current)
        return current
        
    def log(self, level: str, message: str, operation: str = None, 
            metadata: Dict[str, Any] = None, **kwargs) -> None:
        """Create structured log entry."""
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level=level.upper(),
            message=message,
            correlation_id=self.get_correlation_id(),
            component=self.component,
            operation=operation or "general",
            metadata=metadata or {},
            **kwargs
        )
        
        # Add to buffer
        self.log_buffer.append(entry)
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer.pop(0)
            
        # Publish to event bus for real-time monitoring
        asyncio.create_task(self._publish_log(entry))
        
        # Also log to standard logger
        getattr(logger, level.lower())(message, extra=asdict(entry))
        
    def info(self, message: str, operation: str = None, metadata: Dict[str, Any] = None):
        """Log info message."""
        self.log("info", message, operation, metadata)
        
    def warning(self, message: str, operation: str = None, metadata: Dict[str, Any] = None):
        """Log warning message."""
        self.log("warning", message, operation, metadata)
        
    def error(self, message: str, operation: str = None, metadata: Dict[str, Any] = None,
              exc_info: Exception = None):
        """Log error message with optional exception."""
        error_data = {}
        if exc_info:
            error_data['error'] = str(exc_info)
            error_data['stack_trace'] = traceback.format_exc()
            
        self.log("error", message, operation, {**(metadata or {}), **error_data})
        
    def critical(self, message: str, operation: str = None, metadata: Dict[str, Any] = None,
                 exc_info: Exception = None):
        """Log critical message."""
        error_data = {}
        if exc_info:
            error_data['error'] = str(exc_info)
            error_data['stack_trace'] = traceback.format_exc()
            
        self.log("critical", message, operation, {**(metadata or {}), **error_data})
        
    def debug(self, message: str, operation: str = None, metadata: Dict[str, Any] = None):
        """Log debug message."""
        self.log("debug", message, operation, metadata)
        
    def audit(self, action: str, details: Dict[str, Any], user_id: str = None):
        """Log audit trail for trading actions."""
        self.log(
            "info",
            f"Audit: {action}",
            operation="audit",
            metadata={
                "action": action,
                "details": details,
                "audit": True
            },
            user_id=user_id or "system"
        )
        
    def performance(self, operation: str, duration_ms: float, 
                    metadata: Dict[str, Any] = None):
        """Log performance metrics."""
        self.log(
            "info",
            f"Performance: {operation} completed in {duration_ms:.2f}ms",
            operation=operation,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
    def trade_action(self, action: str, symbol: str, volume: float, price: float,
                     trade_id: str = None, metadata: Dict[str, Any] = None):
        """Log trading actions with full audit trail."""
        self.audit(
            f"trade_{action}",
            {
                "symbol": symbol,
                "volume": volume,
                "price": price,
                "trade_id": trade_id or str(uuid.uuid4()),
                **(metadata or {})
            }
        )
        
    def market_data(self, symbol: str, price: float, source: str, 
                    latency_ms: float, metadata: Dict[str, Any] = None):
        """Log market data updates."""
        self.log(
            "info",
            f"Market data: {symbol} @ {price}",
            operation="market_data",
            duration_ms=latency_ms,
            metadata={
                "symbol": symbol,
                "price": price,
                "source": source,
                **(metadata or {})
            }
        )
        
    def strategy_execution(self, strategy_id: str, decision: str, 
                          confidence: float, metadata: Dict[str, Any] = None):
        """Log strategy execution decisions."""
        self.log(
            "info",
            f"Strategy {strategy_id}: {decision} (confidence: {confidence:.2f})",
            operation="strategy_execution",
            metadata={
                "strategy_id": strategy_id,
                "decision": decision,
                "confidence": confidence,
                **(metadata or {})
            }
        )
        
    def risk_check(self, check_type: str, passed: bool, details: Dict[str, Any]):
        """Log risk management checks."""
        self.log(
            "info" if passed else "warning",
            f"Risk check {check_type}: {'PASSED' if passed else 'FAILED'}",
            operation="risk_check",
            metadata={
                "check_type": check_type,
                "passed": passed,
                "details": details
            }
        )
        
    async def _publish_log(self, entry: LogEntry) -> None:
        """Publish log entry to event bus for real-time monitoring."""
        try:
            await self.event_bus.publish({
                'type': 'LogEntry',
                'data': asdict(entry)
            })
        except Exception as e:
            logger.error(f"Failed to publish log entry: {e}")
            
    async def get_logs(self, level: str = None, operation: str = None,
                      correlation_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve logs from buffer with filtering."""
        logs = []
        
        for entry in reversed(self.log_buffer):
            if level and entry.level != level.upper():
                continue
            if operation and entry.operation != operation:
                continue
            if correlation_id and entry.correlation_id != correlation_id:
                continue
                
            logs.append(asdict(entry))
            
            if len(logs) >= limit:
                break
                
        return logs
        
    async def get_performance_metrics(self, operation: str = None,
                                    time_window: int = 3600) -> Dict[str, Any]:
        """Get performance metrics for operations."""
        cutoff_time = datetime.utcnow().timestamp() - time_window
        
        relevant_logs = [
            entry for entry in self.log_buffer
            if entry.duration_ms is not None and
            entry.operation != "audit" and
            (operation is None or entry.operation == operation) and
            datetime.fromisoformat(entry.timestamp).timestamp() > cutoff_time
        ]
        
        if not relevant_logs:
            return {
                "operation": operation or "all",
                "count": 0,
                "avg_duration": 0,
                "min_duration": 0,
                "max_duration": 0,
                "p95_duration": 0,
                "p99_duration": 0
            }
            
        durations = [entry.duration_ms for entry in relevant_logs]
        durations.sort()
        
        return {
            "operation": operation or "all",
            "count": len(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "p95_duration": durations[int(len(durations) * 0.95)],
            "p99_duration": durations[int(len(durations) * 0.99)]
        }
        
    async def get_error_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get error summary for monitoring."""
        cutoff_time = datetime.utcnow().timestamp() - time_window
        
        error_logs = [
            entry for entry in self.log_buffer
            if entry.level in ['ERROR', 'CRITICAL'] and
            datetime.fromisoformat(entry.timestamp).timestamp() > cutoff_time
        ]
        
        error_counts = {}
        for entry in error_logs:
            error_type = entry.metadata.get('error_type', 'unknown')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
        return {
            "total_errors": len(error_logs),
            "error_types": error_counts,
            "recent_errors": [
                {
                    "timestamp": entry.timestamp,
                    "message": entry.message,
                    "operation": entry.operation,
                    "error": entry.error
                }
                for entry in error_logs[-10:]  # Last 10 errors
            ]
        }
        
    def clear_buffer(self):
        """Clear log buffer."""
        self.log_buffer.clear()
        
    def get_correlation_context(self) -> Dict[str, str]:
        """Get current correlation context."""
        return {
            "correlation_id": self.get_correlation_id(),
            "session_id": str(uuid.uuid4()),
            "request_id": str(uuid.uuid4())
        }


# Global logger instance
_logger: Optional[StructuredLogger] = None


async def get_structured_logger(event_bus: EventBus, state_store: StateStore) -> StructuredLogger:
    """Get or create global structured logger instance."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger(event_bus, state_store)
    return _logger


# Context manager for correlation IDs
class CorrelationContext:
    """Context manager for correlation ID scoping."""
    
    def __init__(self, cid: str = None):
        self.cid = cid or str(uuid.uuid4())
        self.token = None
        
    def __enter__(self):
        self.token = correlation_id.set(self.cid)
        return self.cid
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            correlation_id.reset(self.token)


# Decorator for automatic performance logging
def log_performance(operation_name: str):
    """Decorator to automatically log performance metrics."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = await get_structured_logger(None, None)  # Will be set at runtime
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                await logger.performance(operation_name, duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                await logger.error(
                    f"Error in {operation_name}",
                    operation=operation_name,
                    exc_info=e,
                    duration_ms=duration_ms
                )
                raise
                
        return wrapper
    return decorator
