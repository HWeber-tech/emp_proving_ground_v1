import asyncio
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import pytest

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # type: ignore[import-not-found]
        InMemorySpanExporter,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency missing
    TracerProvider = SimpleSpanProcessor = InMemorySpanExporter = None  # type: ignore[assignment]
    _OTEL_SDK_AVAILABLE = False
else:  # pragma: no cover - executed in environments with OpenTelemetry installed
    _OTEL_SDK_AVAILABLE = True

from src.observability.tracing import (
    OpenTelemetryRuntimeTracer,
    OpenTelemetrySettings,
    RuntimeTracer,
    configure_runtime_tracer,
)
from src.runtime.runtime_builder import RuntimeApplication, RuntimeWorkload


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: Dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class RecordingRuntimeTracer(RuntimeTracer):
    def __init__(self) -> None:
        self.workload_spans: List[Tuple[str, Dict[str, Any], RecordingSpan]] = []
        self.operation_spans: List[Tuple[str, Dict[str, Any], RecordingSpan]] = []

    @contextmanager
    def workload_span(
        self,
        *,
        workload: str,
        metadata: Dict[str, Any] | None = None,
    ):
        span = RecordingSpan()
        info = dict(metadata or {})
        self.workload_spans.append((workload, info, span))
        yield span

    @contextmanager
    def operation_span(
        self,
        *,
        name: str,
        metadata: Dict[str, Any] | None = None,
    ):
        span = RecordingSpan()
        info = dict(metadata or {})
        self.operation_spans.append((name, info, span))
        yield span


@pytest.mark.asyncio()
async def test_runtime_application_records_workload_and_operation_spans() -> None:
    tracer = RecordingRuntimeTracer()

    async def _ingest_workload() -> None:
        await asyncio.sleep(0)

    async def _startup() -> None:
        await asyncio.sleep(0)

    async def _shutdown() -> None:
        await asyncio.sleep(0)

    app = RuntimeApplication(
        ingestion=RuntimeWorkload(
            name="timescale-ingest",
            factory=_ingest_workload,
            description="Timescale ingest orchestrator",
            metadata={"mode": "institutional"},
        ),
        tracer=tracer,
    )
    app.add_startup_callback(lambda: _startup())
    app.add_shutdown_callback(lambda: _shutdown())

    await app.run()

    assert tracer.workload_spans, "workload span should be recorded"
    workload_name, workload_metadata, workload_span = tracer.workload_spans[0]
    assert workload_name == "timescale-ingest"
    assert workload_metadata["workload.description"] == "Timescale ingest orchestrator"
    assert workload_metadata["workload.metadata.mode"] == "institutional"
    assert workload_span.attributes["runtime.workload.status"] == "completed"

    operation_names = [entry[0] for entry in tracer.operation_spans]
    assert "runtime.startup" in operation_names
    assert "runtime.shutdown" in operation_names
    for _, _, span in tracer.operation_spans:
        assert span.attributes["runtime.operation.status"] == "completed"


@pytest.mark.skipif(not _OTEL_SDK_AVAILABLE, reason="OpenTelemetry SDK not installed")
def test_open_telemetry_runtime_tracer_records_attributes() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("runtime-test")

    runtime_tracer = OpenTelemetryRuntimeTracer(tracer=tracer)
    with runtime_tracer.workload_span(
        workload="ingest", metadata={"workload.description": "desc"}
    ) as span:
        span.set_attribute("runtime.workload.status", "completed")
    with runtime_tracer.operation_span(
        name="runtime.startup", metadata={"callback": "bootstrap"}
    ) as span:
        span.set_attribute("runtime.operation.status", "completed")

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    workload_span, operation_span = spans
    assert workload_span.name == "runtime.workload"
    assert workload_span.attributes["runtime.workload.name"] == "ingest"
    assert workload_span.attributes["runtime.workload.description"] == "desc"
    assert operation_span.name == "runtime.operation"
    assert operation_span.attributes["runtime.operation.name"] == "runtime.startup"
    assert operation_span.attributes["runtime.callback"] == "bootstrap"


@pytest.mark.skipif(not _OTEL_SDK_AVAILABLE, reason="OpenTelemetry SDK not installed")
def test_configure_runtime_tracer_configures_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, TracerProvider] = {}

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
        console_exporter=False,
    )

    tracer = configure_runtime_tracer(settings)
    assert tracer is not None
    provider = captured.get("provider")
    assert isinstance(provider, TracerProvider)
    assert getattr(provider, "_emp_configured", False)


def test_configure_runtime_tracer_disabled_returns_none() -> None:
    settings = OpenTelemetrySettings(enabled=False)
    assert configure_runtime_tracer(settings) is None


def test_configure_runtime_tracer_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.observability.tracing._OPENTELEMETRY_AVAILABLE", False)
    monkeypatch.setattr("src.observability.tracing.trace", None, raising=False)
    monkeypatch.setattr(
        "src.observability.tracing._MISSING_DEPENDENCY_LOGGED", False, raising=False
    )

    settings = OpenTelemetrySettings(enabled=True)
    assert configure_runtime_tracer(settings) is None
