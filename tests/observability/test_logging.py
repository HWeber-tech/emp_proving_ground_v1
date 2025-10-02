from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Iterator

import pytest

from src.observability.logging import (
    OpenTelemetryLoggingSettings,
    configure_structured_logging,
    load_opentelemetry_logging_settings,
)


pytestmark = pytest.mark.guardrail


@pytest.fixture()
def reset_logging() -> Iterator[None]:
    root = logging.getLogger()
    previous_handlers = root.handlers[:]
    previous_level = root.level
    previous_propagate = root.propagate
    try:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        yield
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)
        root.propagate = previous_propagate


def _last_record(stream: io.StringIO) -> dict[str, object]:
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert lines, "expected at least one log line"
    return json.loads(lines[-1])


def test_configure_structured_logging_emits_json(reset_logging: Iterator[None]) -> None:
    stream = io.StringIO()
    configure_structured_logging(
        component="runtime",
        level="INFO",
        static_fields={"deployment": "staging"},
        stream=stream,
    )

    logger = logging.getLogger("test.runtime")
    logger.info("ingest started", extra={"run_id": "abc123", "count": 3})

    record = _last_record(stream)
    assert record["component"] == "runtime"
    assert record["deployment"] == "staging"
    assert record["message"] == "ingest started"
    assert record["run_id"] == "abc123"
    assert record["count"] == 3
    assert record["level"] == "INFO"
    assert record["logger"] == "test.runtime"
    assert "timestamp" in record


def test_structured_logging_includes_exception(reset_logging: Iterator[None]) -> None:
    stream = io.StringIO()
    configure_structured_logging(component="runtime", stream=stream)

    logger = logging.getLogger("test.runtime.exc")
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("ingest failed")

    record = _last_record(stream)
    assert record["message"] == "ingest failed"
    assert "ValueError" in record["exc_info"]


def test_configure_structured_logging_idempotent(reset_logging: Iterator[None]) -> None:
    first_stream = io.StringIO()
    first_handler = configure_structured_logging(component="runtime", stream=first_stream)
    second_stream = io.StringIO()
    second_handler = configure_structured_logging(component="runtime", stream=second_stream)

    root = logging.getLogger()
    assert first_handler not in root.handlers
    assert second_handler in root.handlers
    assert (
        sum(
            1
            for handler in root.handlers
            if getattr(handler, "name", "") == "emp-structured-logger"
        )
        == 1
    )


def test_configure_structured_logging_invokes_otel(reset_logging: Iterator[None], monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    settings = OpenTelemetryLoggingSettings(enabled=True, endpoint="http://collector")
    calls: dict[str, object] = {}

    def _fake_initialise(
        provided_settings: OpenTelemetryLoggingSettings,
        level: int,
        root_logger: logging.Logger,
    ) -> logging.Handler | None:
        calls["called"] = (provided_settings, level, root_logger)
        handler = logging.StreamHandler(stream)
        handler.set_name("emp-otel-logger")
        return handler

    monkeypatch.setattr("src.observability.logging._initialise_otel_logging", _fake_initialise)

    configure_structured_logging(component="runtime", stream=stream, otel_settings=settings)

    assert "called" in calls
    called_settings, called_level, _ = calls["called"]  # type: ignore[misc]
    assert called_settings is settings
    assert called_level == logging.INFO


def test_configure_structured_logging_skips_otel(reset_logging: Iterator[None], monkeypatch: pytest.MonkeyPatch) -> None:
    invoked = False

    def _unexpected(*args: object, **kwargs: object) -> None:
        nonlocal invoked
        invoked = True

    monkeypatch.setattr("src.observability.logging._initialise_otel_logging", _unexpected)

    configure_structured_logging(component="runtime", stream=io.StringIO())

    assert not invoked


def test_load_opentelemetry_logging_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "logging.yaml"
    config_path.write_text(
        """
opentelemetry:
  enabled: false
  endpoint: http://example
  timeout: 15
  insecure: false
  compression: gzip
  headers:
    Authorization: secret
  resource:
    service.name: emp-test
    deployment.environment: test
""".strip(),
        encoding="utf-8",
    )

    settings = load_opentelemetry_logging_settings(config_path)
    assert settings.enabled is False
    assert settings.endpoint == "http://example"
    assert settings.timeout == 15
    assert settings.insecure is False
    assert settings.compression == "gzip"
    assert settings.headers == {"Authorization": "secret"}
    assert settings.resource_attributes["service.name"] == "emp-test"
