"""Centralised structured logging helpers using :mod:`structlog`.

This module provides a single place to configure the logging stack so the
trading runtime produces JSON log entries enriched with contextual metadata.
It also exposes helpers that bind order-level correlation identifiers, making
it trivial to trace FIX events across asynchronous callbacks and background
tasks.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import sys
import threading
from typing import Any, Iterable, Iterator, Mapping

import structlog

from src.observability.tracing import OpenTelemetrySettings
from src.observability.logging import load_opentelemetry_logging_settings

try:  # pragma: no cover - optional dependency in minimal environments
    from opentelemetry._logs import get_logger_provider, set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import (
        BatchLogRecordProcessor,
        ConsoleLogExporter,
    )
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.sdk.resources import Resource
except ModuleNotFoundError:  # pragma: no cover - optional dependency in minimal envs
    get_logger_provider = set_logger_provider = None  # type: ignore[assignment]
    LoggerProvider = LoggingHandler = None  # type: ignore[assignment]
    BatchLogRecordProcessor = ConsoleLogExporter = OTLPLogExporter = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]


BoundLogger = structlog.stdlib.BoundLogger

__all__ = [
    "configure_structlog",
    "load_structlog_otel_settings",
    "get_logger",
    "order_logging_context",
    "bind_context",
    "unbind_context",
]


_CONFIGURED = False
_OTEL_WARNING_EMITTED = False
_OTEL_LOCK = threading.Lock()

_OTEL_LOGGING_AVAILABLE = bool(
    LoggerProvider
    and LoggingHandler
    and BatchLogRecordProcessor
    and OTLPLogExporter
    and get_logger_provider
    and set_logger_provider
    and Resource
)

logger = logging.getLogger(__name__)


def _configure_otel_logging_handler(
    *, level: int, settings: OpenTelemetrySettings
) -> LoggingHandler | None:
    """Configure an OpenTelemetry log handler when instrumentation is enabled."""

    if not settings.enabled:
        return None

    global _OTEL_WARNING_EMITTED

    if not _OTEL_LOGGING_AVAILABLE:
        if not _OTEL_WARNING_EMITTED:
            logger.warning(
                "OpenTelemetry logging requested but dependencies are unavailable; "
                "skipping structured log exporter",
            )
            _OTEL_WARNING_EMITTED = True
        return None

    assert LoggerProvider is not None
    assert LoggingHandler is not None
    assert BatchLogRecordProcessor is not None
    assert OTLPLogExporter is not None
    assert get_logger_provider is not None
    assert set_logger_provider is not None
    assert Resource is not None

    resource_attributes: dict[str, str] = {"service.name": settings.service_name}
    if settings.environment:
        resource_attributes["deployment.environment"] = settings.environment

    with _OTEL_LOCK:
        provider = get_logger_provider()
        if not isinstance(provider, LoggerProvider) or not getattr(
            provider, "_emp_logging_configured", False
        ):
            provider = LoggerProvider(resource=Resource.create(resource_attributes))

            endpoint = settings.logs_endpoint or settings.endpoint
            headers = settings.logs_headers or settings.headers
            timeout = (
                settings.logs_timeout
                if settings.logs_timeout is not None
                else settings.timeout
            )

            if endpoint:
                exporter = OTLPLogExporter(
                    endpoint=endpoint,
                    headers=dict(headers) if headers is not None else None,
                    timeout=timeout,
                )
                provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

            if settings.console_exporter:
                provider.add_log_record_processor(
                    BatchLogRecordProcessor(ConsoleLogExporter())
                )

            set_logger_provider(provider)
            setattr(provider, "_emp_logging_configured", True)
            logger.info(
                "OpenTelemetry logging enabled for service %s", settings.service_name
            )

        handler = LoggingHandler(level=level, logger_provider=provider)
        handler.set_name("opentelemetry")
        return handler


def _fallback_keyvalue_renderer() -> Any:
    """Return a lightweight key=value renderer compatible with structlog."""

    def _renderer(_logger: Any, _name: str, event_dict: Mapping[str, Any]) -> str:
        parts: list[str] = []
        event_value = event_dict.get("event")
        if event_value is not None:
            parts.append(f"event={event_value!r}")
        for key in sorted(k for k in event_dict if k != "event"):
            parts.append(f"{key}={event_dict[key]!r}")
        return " ".join(parts)

    return _renderer


def _select_renderer(output_format: str | None) -> Any:
    """Return a structlog renderer for the requested output format."""

    if not output_format:
        return structlog.processors.JSONRenderer()

    normalized = output_format.strip().lower()
    if not normalized or normalized in {"json", "structured"}:
        return structlog.processors.JSONRenderer()
    if normalized in {"keyvalue", "kv", "text"}:
        try:
            from structlog.processors import KeyValueRenderer  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            return _fallback_keyvalue_renderer()
        return KeyValueRenderer(key_order=["event"], sort_keys=True)
    if normalized in {"console", "pretty"}:
        try:
            from structlog.dev import ConsoleRenderer
        except ImportError:  # pragma: no cover - structlog without dev extras
            logger.warning(
                "structlog.dev unavailable; falling back to JSON renderer",
                extra={"structlog.output_format": output_format},
            )
            return structlog.processors.JSONRenderer()
        return ConsoleRenderer(colors=False)

    logger.warning(
        "Unsupported structlog output format %r; defaulting to JSON",
        output_format,
    )
    return structlog.processors.JSONRenderer()


def _build_structlog_handler(
    *, stream: Any | None, destination: str | Path | None
) -> logging.Handler:
    """Create a logging handler for structlog output."""

    if stream is not None:
        handler = logging.StreamHandler(stream)
        handler.set_name("structlog")
        return handler

    if destination is None:
        handler = logging.StreamHandler()
        handler.set_name("structlog")
        return handler

    dest_text = str(destination).strip()
    if not dest_text:
        handler = logging.StreamHandler()
        handler.set_name("structlog")
        return handler

    lowered = dest_text.lower()
    if lowered in {"stdout", "standard_output", "sys.stdout"}:
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name("structlog")
        return handler
    if lowered in {"stderr", "standard_error", "sys.stderr", "default"}:
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name("structlog")
        return handler

    path = Path(dest_text).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning(
            "Failed to create structlog destination directory; defaulting to stderr",
            extra={"structlog.destination": str(path)},
            exc_info=True,
        )
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name("structlog")
        return handler

    try:
        file_handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "Failed to open structlog destination %s: %s; defaulting to stderr",
            path,
            exc,
        )
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name("structlog")
        return handler

    file_handler.set_name(f"structlog:{path}")
    return file_handler


def configure_structlog(
    *,
    level: int = logging.INFO,
    stream: Any | None = None,
    output_format: str | None = None,
    destination: str | Path | None = None,
    otel_settings: OpenTelemetrySettings | None = None,
) -> None:
    """Configure ``structlog`` to emit JSON log lines with context variables.

    Args:
        level: Minimum logging level. Defaults to :data:`logging.INFO`.
        stream: Optional IO stream for the root handler, primarily used in tests.
        output_format: Format of the emitted log records. Supported values are
            ``"json"`` (default), ``"keyvalue"`` or ``"kv"``, and ``"console"``.
        destination: Optional log destination. Accepts ``"stdout"``, ``"stderr"``
            (or ``"default"``), or a filesystem path. When omitted, logs are
            written to ``stderr``.
        otel_settings: Optional OpenTelemetry configuration. When provided and
            instrumentation is enabled, structured log records are forwarded to
            the configured collector alongside the standard stream handler.
    """

    global _CONFIGURED

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    pre_chain: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=_select_renderer(output_format),
        foreign_pre_chain=pre_chain,
    )

    handler = _build_structlog_handler(stream=stream, destination=destination)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    if otel_settings is not None:
        otel_handler = _configure_otel_logging_handler(
            level=level, settings=otel_settings
        )
        if otel_handler is not None:
            root_logger.addHandler(otel_handler)

    structlog.configure(
        processors=[
            *pre_chain,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def load_structlog_otel_settings(
    path: str | Path,
    *,
    default_service_name: str = "emp-professional-runtime",
    default_environment: str | None = None,
) -> OpenTelemetrySettings:
    """Load OpenTelemetry settings for structlog exporters from a YAML profile.

    The helper bridges the observability logging profiles stored under
    :mod:`config/observability/` with the :func:`configure_structlog`
    instrumentation by translating the ``logging.yaml`` schema into the
    :class:`~src.observability.tracing.OpenTelemetrySettings` structure used by
    the runtime entrypoint.

    Args:
        path: Path to a YAML configuration compatible with
            :func:`src.observability.logging.load_opentelemetry_logging_settings`.
        default_service_name: Service name applied when the profile does not
            define a ``service.name`` resource attribute.
        default_environment: Deployment environment applied when the profile
            omits ``deployment.environment``.

    Returns:
        An :class:`OpenTelemetrySettings` instance ready to be passed to
        :func:`configure_structlog`.
    """

    logging_settings = load_opentelemetry_logging_settings(path)

    resource = logging_settings.resource_attributes
    service_name = str(resource.get("service.name", default_service_name))
    environment_value = resource.get("deployment.environment")
    if environment_value is None:
        environment = default_environment
    else:
        environment = str(environment_value)

    logs_headers = dict(logging_settings.headers)
    if not logs_headers:
        headers_mapping: dict[str, str] | None = None
    else:
        headers_mapping = logs_headers

    return OpenTelemetrySettings(
        enabled=logging_settings.enabled,
        service_name=service_name,
        environment=environment,
        endpoint=None,
        headers=None,
        timeout=None,
        console_exporter=False,
        logs_endpoint=logging_settings.endpoint,
        logs_headers=headers_mapping,
        logs_timeout=logging_settings.timeout,
    )


def _ensure_configured() -> None:
    global _CONFIGURED
    if not _CONFIGURED:
        configure_structlog()


def get_logger(name: str | None = None) -> BoundLogger:
    """Return a structlog bound logger, configuring the stack on first use."""

    _ensure_configured()
    return structlog.get_logger(name)


def bind_context(**values: Any) -> None:
    """Bind key/value pairs to the current context for subsequent log records."""

    if values:
        structlog.contextvars.bind_contextvars(**values)


def unbind_context(*keys: str) -> None:
    """Remove keys from the current logging context if they were previously bound."""

    if keys:
        structlog.contextvars.unbind_contextvars(*keys)


@contextmanager
def order_logging_context(
    order_id: str | None = None,
    *,
    correlation_id: str | None = None,
    extra_keys: Iterable[str] | None = None,
    **values: Any,
) -> Iterator[BoundLogger]:
    """Context manager that binds order correlation metadata for log records.

    Args:
        order_id: Order identifier to bind into the logging context.
        correlation_id: Optional correlation identifier. Defaults to ``order_id``
            when not provided.
        extra_keys: Explicit keys to unbind when the context exits. This is
            primarily useful when values are added by downstream helpers.
        **values: Additional metadata to bind alongside the identifiers.
    """

    _ensure_configured()

    context: dict[str, Any] = {}
    keys_to_unbind: list[str] = []

    if order_id is not None:
        context["order_id"] = order_id
        keys_to_unbind.append("order_id")

    corr_id = correlation_id or order_id
    if corr_id is not None:
        context["correlation_id"] = corr_id
        keys_to_unbind.append("correlation_id")

    for key, value in values.items():
        context[key] = value
        keys_to_unbind.append(key)

    if extra_keys:
        keys_to_unbind.extend(extra_keys)

    bind_context(**context)
    try:
        yield get_logger().bind(**context)
    finally:
        if keys_to_unbind:
            unbind_context(*keys_to_unbind)

