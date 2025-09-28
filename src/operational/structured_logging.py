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
from typing import Any, Iterable, Iterator

import structlog


BoundLogger = structlog.stdlib.BoundLogger

__all__ = [
    "configure_structlog",
    "get_logger",
    "order_logging_context",
    "bind_context",
    "unbind_context",
]


_CONFIGURED = False


def configure_structlog(*, level: int = logging.INFO, stream: Any | None = None) -> None:
    """Configure ``structlog`` to emit JSON log lines with context variables.

    Args:
        level: Minimum logging level. Defaults to :data:`logging.INFO`.
        stream: Optional IO stream for the root handler, primarily used in tests.
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
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=pre_chain,
    )

    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

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

