from __future__ import annotations

import asyncio
from contextlib import nullcontext

import pytest

try:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # type: ignore[import-not-found]
        InMemorySpanExporter,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency missing
    Resource = TracerProvider = SimpleSpanProcessor = InMemorySpanExporter = None  # type: ignore[assignment]
    _OTEL_SDK_AVAILABLE = False
else:  # pragma: no cover - executed when OpenTelemetry is installed
    _OTEL_SDK_AVAILABLE = True

from src.core.event_bus import AsyncEventBus, Event
from src.observability.tracing import (
    OpenTelemetryEventBusTracer,
    OpenTelemetrySettings,
    configure_event_bus_tracer,
    parse_opentelemetry_settings,
)


class RecordingTracer:
    def __init__(self) -> None:
        self.publish_calls: list[tuple[str, str | None, dict[str, object]]] = []
        self.handler_calls: list[tuple[str, str, dict[str, object]]] = []

    def publish_span(
        self,
        *,
        event_type: str,
        event_source: str | None,
        metadata: dict[str, object] | None = None,
    ) -> nullcontext:
        self.publish_calls.append((event_type, event_source, dict(metadata or {})))
        return nullcontext()

    def handler_span(
        self,
        *,
        event_type: str,
        handler_name: str,
        metadata: dict[str, object] | None = None,
    ) -> nullcontext:
        self.handler_calls.append((event_type, handler_name, dict(metadata or {})))
        return nullcontext()


@pytest.mark.asyncio()
async def test_async_event_bus_preserves_event_order() -> None:
    bus = AsyncEventBus()
    received: list[int] = []
    processed = asyncio.Event()
    total_events = 8

    async def _handler(event: Event) -> None:
        received.append(event.payload)
        if len(received) == total_events:
            processed.set()

    bus.subscribe("sequence.test", _handler)
    await bus.start()
    try:
        for value in range(total_events):
            await bus.publish(Event(type="sequence.test", payload=value))
        await asyncio.wait_for(processed.wait(), timeout=1)
    finally:
        await bus.stop()

    assert received == list(range(total_events))


@pytest.mark.asyncio()
async def test_async_event_bus_tracing_records_publish_and_handler_metadata() -> None:
    tracer = RecordingTracer()
    bus = AsyncEventBus(tracer=tracer)
    processed = asyncio.Event()

    async def _handler(event: Event) -> None:
        processed.set()

    bus.subscribe("telemetry.test", _handler)
    await bus.start()
    try:
        await bus.publish(Event(type="telemetry.test", payload={"foo": "bar"}, source="unit"))
        await asyncio.wait_for(processed.wait(), timeout=1)
    finally:
        await bus.stop()

    assert tracer.publish_calls, "publish span should be recorded"
    event_type, source, metadata = tracer.publish_calls[0]
    assert event_type == "telemetry.test"
    assert source == "unit"
    assert metadata.get("mode") == "async"
    assert "queue_size" in metadata
    assert "subscriber_count" in metadata

    assert tracer.handler_calls, "handler span should be recorded"
    handler_event_type, handler_name, handler_metadata = tracer.handler_calls[0]
    assert handler_event_type == "telemetry.test"
    assert handler_name.startswith("<function")
    assert handler_metadata.get("dispatch_lag_ms") is not None


@pytest.mark.skipif(not _OTEL_SDK_AVAILABLE, reason="OpenTelemetry SDK not installed")
def test_open_telemetry_event_bus_tracer_records_span_attributes() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test-service"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-instrumentation")

    otel_tracer = OpenTelemetryEventBusTracer(tracer=tracer)
    with otel_tracer.publish_span(
        event_type="telemetry.test", event_source="unit", metadata={"queue_size": 3}
    ):
        pass
    with otel_tracer.handler_span(
        event_type="telemetry.test",
        handler_name="handler",
        metadata={"dispatch_lag_ms": 5.5},
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    publish_span, handler_span = spans
    assert publish_span.name == "event_bus.publish"
    assert publish_span.attributes["event.bus.event_type"] == "telemetry.test"
    assert publish_span.attributes["event.bus.queue_size"] == 3
    assert handler_span.name == "event_bus.handle"
    assert handler_span.attributes["event.bus.handler"] == "handler"
    assert handler_span.attributes["event.bus.dispatch_lag_ms"] == 5.5


def test_parse_opentelemetry_settings_parses_headers_and_timeout() -> None:
    extras = {
        "OTEL_ENABLED": "true",
        "OTEL_SERVICE_NAME": "emp-runtime",
        "OTEL_ENVIRONMENT": "staging",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example/v1/traces",
        "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer token, X-Team=ops",
        "OTEL_EXPORTER_OTLP_TIMEOUT": "7.5",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": "https://otel.example/v1/logs",
        "OTEL_EXPORTER_OTLP_LOGS_HEADERS": "X-Logs=enabled",
        "OTEL_EXPORTER_OTLP_LOGS_TIMEOUT": "5.25",
        "OTEL_CONSOLE_EXPORTER": "false",
    }
    settings = parse_opentelemetry_settings(extras)
    assert settings.enabled is True
    assert settings.service_name == "emp-runtime"
    assert settings.environment == "staging"
    assert settings.endpoint == "https://otel.example/v1/traces"
    assert settings.headers == {"Authorization": "Bearer token", "X-Team": "ops"}
    assert settings.timeout == pytest.approx(7.5)
    assert settings.logs_endpoint == "https://otel.example/v1/logs"
    assert settings.logs_headers == {"X-Logs": "enabled"}
    assert settings.logs_timeout == pytest.approx(5.25)
    assert settings.console_exporter is False


@pytest.mark.skipif(not _OTEL_SDK_AVAILABLE, reason="OpenTelemetry SDK not installed")
def test_configure_event_bus_tracer_configures_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, TracerProvider] = {}

    def _fake_get_provider() -> TracerProvider | None:
        return captured.get("provider")

    def _fake_set_provider(provider: TracerProvider) -> None:
        captured["provider"] = provider

    monkeypatch.setattr("src.observability.tracing.trace.get_tracer_provider", _fake_get_provider)
    monkeypatch.setattr("src.observability.tracing.trace.set_tracer_provider", _fake_set_provider)

    settings = OpenTelemetrySettings(
        enabled=True,
        service_name="emp-runtime",
        environment="demo",
        endpoint=None,
        headers=None,
        timeout=None,
        console_exporter=True,
    )

    tracer = configure_event_bus_tracer(settings)
    assert tracer is not None
    provider = captured.get("provider")
    assert isinstance(provider, TracerProvider)
    assert getattr(provider, "_emp_configured", False)


def test_configure_event_bus_tracer_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.observability.tracing._OPENTELEMETRY_AVAILABLE", False)
    monkeypatch.setattr("src.observability.tracing.trace", None, raising=False)
    monkeypatch.setattr(
        "src.observability.tracing._MISSING_DEPENDENCY_LOGGED", False, raising=False
    )

    settings = OpenTelemetrySettings(enabled=True)
    assert configure_event_bus_tracer(settings) is None
