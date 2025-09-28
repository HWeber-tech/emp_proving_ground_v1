"""Structured logging helpers for EMP Professional Predator.

Beyond formatting JSON log lines, this module optionally mirrors the runtime
log stream into a local OpenTelemetry collector when explicitly configured.
The OpenTelemetry dependency remains optional—callers must request it via
settings, allowing lightweight environments to keep their logging stack
dependency free.
"""

from __future__ import annotations

import json
import logging
import threading
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Mapping

import yaml

__all__ = [
    "StructuredLogConfig",
    "StructuredJsonFormatter",
    "configure_structured_logging",
    "OpenTelemetryLoggingSettings",
    "load_opentelemetry_logging_settings",
]


_STRUCTURED_HANDLER_NAME = "emp-structured-logger"
_OTEL_HANDLER_NAME = "emp-otel-logger"


# OpenTelemetry logging support is optional; we import lazily and degrade
# gracefully when the dependency is not present.
try:  # pragma: no cover - exercised when OpenTelemetry dependencies installed
    from opentelemetry.exporter.otlp.proto.http._log_exporter import (
        OTLPLogExporter as _OTLPLogExporter,
    )
    from opentelemetry.sdk._logs import (  # type: ignore[attr-defined]
        LoggerProvider as _LoggerProvider,
        LoggingHandler as _LoggingHandler,
    )
    from opentelemetry.sdk._logs.export import (  # type: ignore[attr-defined]
        BatchLogRecordProcessor as _BatchLogRecordProcessor,
    )
    from opentelemetry.sdk.resources import Resource as _Resource
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    _OTEL_LOGGING_AVAILABLE = False
    _OTLPLogExporter = object  # type: ignore[assignment]
    _LoggerProvider = object  # type: ignore[assignment]
    _LoggingHandler = logging.Handler  # type: ignore[assignment]
    _BatchLogRecordProcessor = object  # type: ignore[assignment]
    _Resource = object  # type: ignore[assignment]
else:  # pragma: no cover - exercised when OpenTelemetry is installed
    _OTEL_LOGGING_AVAILABLE = True

ResourceAttributeValue = (
    str
    | bool
    | int
    | float
    | tuple[str, ...]
    | tuple[bool, ...]
    | tuple[int, ...]
    | tuple[float, ...]
)

logger = logging.getLogger(__name__)

_otel_lock = threading.Lock()
_otel_handler: logging.Handler | None = None

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


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    normalised = str(value).strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class OpenTelemetryLoggingSettings:
    """Configuration for mirroring structured logs into OpenTelemetry."""

    enabled: bool = False
    endpoint: str = "http://localhost:4318/v1/logs"
    timeout: float = 10.0
    insecure: bool = True
    compression: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    resource_attributes: Mapping[str, ResourceAttributeValue] = field(default_factory=dict)

    @staticmethod
    def _coerce_headers(headers: Mapping[str, object] | None) -> Mapping[str, str]:
        if not headers:
            return {}
        coerced: dict[str, str] = {}
        for key, value in headers.items():
            coerced[str(key)] = str(value)
        return coerced

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "OpenTelemetryLoggingSettings":
        enabled = _coerce_bool(data.get("enabled"), default=False)
        endpoint = str(data.get("endpoint", cls.endpoint))
        timeout_raw = data.get("timeout", cls.timeout)
        try:
            timeout = float(timeout_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            timeout = cls.timeout
        insecure = _coerce_bool(data.get("insecure"), default=cls.insecure)
        compression = data.get("compression")
        if compression is not None:
            compression = str(compression)
        headers = cls._coerce_headers(
            data.get("headers") if isinstance(data.get("headers"), Mapping) else None
        )

        resource_attributes: dict[str, ResourceAttributeValue] = {}
        raw_resource = data.get("resource", {})
        if isinstance(raw_resource, Mapping):
            for key, value in raw_resource.items():
                resource_attributes[str(key)] = _coerce_resource_value(value)

        return cls(
            enabled=enabled,
            endpoint=endpoint,
            timeout=timeout,
            insecure=insecure,
            compression=compression,
            headers=headers,
            resource_attributes=resource_attributes,
        )


def _coerce_resource_value(value: object) -> ResourceAttributeValue:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        coerced: list[str | bool | int | float] = []
        for item in value:
            if isinstance(item, (str, bool, int, float)):
                coerced.append(item)
            else:
                coerced.append(str(item))
        return tuple(coerced)  # type: ignore[return-value]
    return str(value)


def load_opentelemetry_logging_settings(path: str | Path) -> OpenTelemetryLoggingSettings:
    """Load OpenTelemetry logging settings from a YAML file."""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    if isinstance(data, Mapping) and "opentelemetry" in data and isinstance(
        data["opentelemetry"], Mapping
    ):
        data_mapping = data["opentelemetry"]
    elif isinstance(data, Mapping):
        data_mapping = data
    else:  # pragma: no cover - defensive guard against malformed YAML
        raise ValueError("OpenTelemetry logging configuration must be a mapping")

    assert isinstance(data_mapping, Mapping)
    return OpenTelemetryLoggingSettings.from_mapping(data_mapping)


def _create_otlp_exporter(settings: OpenTelemetryLoggingSettings) -> object:
    kwargs: dict[str, object] = {
        "endpoint": settings.endpoint,
        "timeout": settings.timeout,
    }
    if settings.headers:
        kwargs["headers"] = dict(settings.headers)
    if settings.compression:
        kwargs["compression"] = settings.compression
    if settings.insecure is not None:
        kwargs["insecure"] = settings.insecure
    return _OTLPLogExporter(**kwargs)  # type: ignore[operator]


def _initialise_otel_logging(
    settings: OpenTelemetryLoggingSettings,
    level: int,
    root_logger: logging.Logger,
) -> logging.Handler | None:
    if not settings.enabled:
        return None
    if not _OTEL_LOGGING_AVAILABLE:
        logger.warning(
            "OpenTelemetry logging requested but dependencies are unavailable;"
            " skipping OTLP export"
        )
        return None

    with _otel_lock:
        global _otel_handler
        if _otel_handler is None:
            resource = _Resource.create(dict(settings.resource_attributes))  # type: ignore[attr-defined]
            provider = _LoggerProvider(resource=resource)  # type: ignore[call-arg]
            exporter = _create_otlp_exporter(settings)
            processor = _BatchLogRecordProcessor(exporter)  # type: ignore[call-arg]
            provider.add_log_record_processor(processor)  # type: ignore[attr-defined]
            handler = _LoggingHandler(level=level, logger_provider=provider)  # type: ignore[call-arg]
            handler.set_name(_OTEL_HANDLER_NAME)
            _otel_handler = handler
        else:
            _otel_handler.setLevel(level)
    assert _otel_handler is not None

    for existing in list(root_logger.handlers):
        if getattr(existing, "name", None) == _OTEL_HANDLER_NAME:
            root_logger.removeHandler(existing)
    root_logger.addHandler(_otel_handler)
    return _otel_handler


def configure_structured_logging(
    *,
    component: str,
    level: int | str = logging.INFO,
    static_fields: Mapping[str, Any] | None = None,
    stream: IO[str] | None = None,
    otel_settings: OpenTelemetryLoggingSettings | None = None,
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

    if otel_settings:
        _initialise_otel_logging(otel_settings, resolved_level, root_logger)
    return handler
