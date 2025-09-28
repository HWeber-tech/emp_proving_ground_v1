from __future__ import annotations

import io
import json
import logging

import pytest

from src.observability.tracing import OpenTelemetrySettings
from src.operational.structured_logging import (
    configure_structlog,
    get_logger,
    order_logging_context,
)

try:  # pragma: no cover - optional dependency in minimal environments
    from opentelemetry._logs import get_logger_provider, set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk.resources import Resource
except ModuleNotFoundError:  # pragma: no cover - optional dependency in minimal envs
    get_logger_provider = set_logger_provider = None  # type: ignore[assignment]
    LoggerProvider = LoggingHandler = Resource = None  # type: ignore[assignment]


_OTEL_LOGGING_AVAILABLE = bool(
    LoggerProvider and LoggingHandler and get_logger_provider and set_logger_provider and Resource
)


def _read_single_record(buffer: io.StringIO) -> dict[str, object]:
    contents = buffer.getvalue().strip().splitlines()
    assert contents, "expected at least one log record"
    return json.loads(contents[-1])


def test_order_logging_context_binds_and_unbinds_order_metadata() -> None:
    buffer = io.StringIO()
    configure_structlog(level=logging.INFO, stream=buffer)

    with order_logging_context("ORD-123") as log:
        log.info("acknowledged", status="ACK")

    record = _read_single_record(buffer)
    assert record["event"] == "acknowledged"
    assert record["order_id"] == "ORD-123"
    assert record["correlation_id"] == "ORD-123"
    assert record["status"] == "ACK"

    buffer.seek(0)
    buffer.truncate(0)

    logger = get_logger("test_structured_logging")
    logger.info("no_context")

    record = _read_single_record(buffer)
    assert record["event"] == "no_context"
    assert "order_id" not in record
    assert "correlation_id" not in record


def test_order_logging_context_supports_custom_correlation_id() -> None:
    buffer = io.StringIO()
    configure_structlog(level=logging.INFO, stream=buffer)

    with order_logging_context("ORD-456", correlation_id="chain-1") as log:
        log.info("fill")

    record = _read_single_record(buffer)
    assert record["event"] == "fill"
    assert record["order_id"] == "ORD-456"
    assert record["correlation_id"] == "chain-1"


@pytest.mark.skipif(not _OTEL_LOGGING_AVAILABLE, reason="OpenTelemetry logging SDK not installed")
def test_configure_structlog_wires_opentelemetry_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.operational.structured_logging._CONFIGURED", False)
    monkeypatch.setattr("src.operational.structured_logging._OTEL_WARNING_EMITTED", False)

    buffer = io.StringIO()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    settings = OpenTelemetrySettings(enabled=True, service_name="test-service", environment="test-env")
    configure_structlog(level=logging.INFO, stream=buffer, otel_settings=settings)

    handlers = [handler for handler in root_logger.handlers if isinstance(handler, LoggingHandler)]
    assert handlers, "expected OpenTelemetry logging handler to be attached"
    assert handlers[0].level == logging.INFO

    provider = get_logger_provider()
    assert isinstance(provider, LoggerProvider)
    assert provider.resource.attributes.get("service.name") == "test-service"
    assert provider.resource.attributes.get("deployment.environment") == "test-env"

    # Cleanup to avoid leaking configuration into other tests.
    root_logger.handlers.clear()
    monkeypatch.setattr("src.operational.structured_logging._CONFIGURED", False)
    if set_logger_provider is not None:
        set_logger_provider(LoggerProvider(resource=Resource.create({})))
