from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from src.observability.tracing import OpenTelemetrySettings
from src.operational.structured_logging import (
    configure_structlog,
    get_logger,
    load_structlog_otel_settings,
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


def test_configure_structlog_supports_keyvalue_output() -> None:
    buffer = io.StringIO()
    configure_structlog(level=logging.INFO, stream=buffer, output_format="keyvalue")

    logger = get_logger("structlog-keyvalue")
    logger.info("kv_log", foo="bar")

    contents = buffer.getvalue().strip().splitlines()
    assert contents, "expected log output"
    last = contents[-1]
    assert "event='kv_log'" in last
    assert "foo='bar'" in last


def test_configure_structlog_writes_to_file_destination(tmp_path: Path) -> None:
    destination = tmp_path / "logs" / "runtime.log"

    configure_structlog(
        level=logging.INFO,
        destination=destination,
        output_format="keyvalue",
    )

    logger = get_logger("structlog-file")
    logger.info("file_log", foo="baz")

    for handler in logging.getLogger().handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            flush()

    contents = destination.read_text(encoding="utf-8").strip().splitlines()
    assert contents, "expected file log output"
    last = contents[-1]
    assert "event='file_log'" in last
    assert "foo='baz'" in last


def test_load_structlog_otel_settings_translates_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "logging.yaml"
    config_path.write_text(
        """
opentelemetry:
  enabled: true
  endpoint: "http://collector:4318/v1/logs"
  timeout: 3.5
  headers:
    Authorization: Bearer token
  resource:
    service.name: emp-local-runtime
    deployment.environment: local-dev
""",
        encoding="utf-8",
    )

    settings = load_structlog_otel_settings(config_path)

    assert settings.enabled is True
    assert settings.logs_endpoint == "http://collector:4318/v1/logs"
    assert settings.logs_timeout == 3.5
    assert settings.logs_headers == {"Authorization": "Bearer token"}
    assert settings.service_name == "emp-local-runtime"
    assert settings.environment == "local-dev"


def test_load_structlog_otel_settings_applies_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "logging.yaml"
    config_path.write_text(
        """
enabled: true
endpoint: "http://localhost:4318/v1/logs"
""",
        encoding="utf-8",
    )

    settings = load_structlog_otel_settings(
        config_path,
        default_service_name="default-service",
        default_environment="demo-env",
    )

    assert settings.enabled is True
    assert settings.service_name == "default-service"
    assert settings.environment == "demo-env"
    assert settings.logs_endpoint == "http://localhost:4318/v1/logs"
    assert settings.logs_headers is None


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
