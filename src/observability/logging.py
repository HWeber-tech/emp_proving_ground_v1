"""Structured logging helpers for EMP Professional Predator."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import IO, Any, Mapping

__all__ = [
    "StructuredLogConfig",
    "StructuredJsonFormatter",
    "configure_structured_logging",
]


_STRUCTURED_HANDLER_NAME = "emp-structured-logger"
# Canonical LogRecord attributes documented in the stdlib; anything outside this
# set is treated as an ``extra`` payload that should be serialised for
# downstream consumers.
_LOG_RECORD_RESERVED_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "process",
        "processName",
        "taskName",
        "message",
    }
)


@dataclass(frozen=True)
class StructuredLogConfig:
    """Configuration for the structured logging handler."""

    component: str
    level: int = logging.INFO
    static_fields: Mapping[str, Any] = field(default_factory=dict)
    stream: IO[str] | None = None


def _serialize(value: Any) -> Any:
    """Coerce Python objects into JSON-friendly representations."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, set):
        # Deterministic ordering keeps diffs stable when logs are captured.
        return sorted(_serialize(item) for item in value)
    return repr(value)


def _json_default(value: Any) -> str:
    """Fallback serialiser for ``json.dumps``."""

    return repr(value)


class StructuredJsonFormatter(logging.Formatter):
    """Format log records as JSON with UTC timestamps and contextual fields."""

    def __init__(self, *, static_fields: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self._static_fields = dict(static_fields or {})

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = dict(self._static_fields)
        payload["timestamp"] = (
            datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        payload["level"] = record.levelname
        payload["logger"] = record.name
        payload["message"] = record.getMessage()
        payload["module"] = record.module
        payload["function"] = record.funcName
        payload["line"] = record.lineno
        payload["process"] = record.processName
        payload["thread"] = record.threadName

        extra_keys = set(record.__dict__) - _LOG_RECORD_RESERVED_FIELDS
        for key in sorted(extra_keys):
            payload[key] = _serialize(record.__dict__[key])

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, default=_json_default)


def _resolve_log_level(level: int | str) -> int:
    """Normalise logging level representations."""

    if isinstance(level, int):
        return level
    candidate = logging.getLevelName(str(level).strip().upper())
    if isinstance(candidate, int):
        return candidate
    return logging.INFO


def configure_structured_logging(
    *,
    component: str,
    level: int | str = logging.INFO,
    static_fields: Mapping[str, Any] | None = None,
    stream: IO[str] | None = None,
) -> logging.Handler:
    """Configure the root logger with a JSON formatter."""

    resolved_level = _resolve_log_level(level)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.set_name(_STRUCTURED_HANDLER_NAME)
    formatter_fields = {"component": component}
    if static_fields:
        for key, value in static_fields.items():
            formatter_fields[str(key)] = value
    handler.setFormatter(StructuredJsonFormatter(static_fields=formatter_fields))

    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        if getattr(existing, "name", None) == _STRUCTURED_HANDLER_NAME:
            root_logger.removeHandler(existing)
    root_logger.addHandler(handler)
    root_logger.setLevel(resolved_level)
    return handler
