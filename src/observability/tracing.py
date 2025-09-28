from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    ContextManager,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    cast,
)

trace: Any
SpanKind: Any
OTLPSpanExporter: Any
Resource: Any
TracerProvider: Any
BatchSpanProcessor: Any
ConsoleSpanExporter: Any
SimpleSpanProcessor: Any

try:  # pragma: no cover - exercised via runtime import
    from opentelemetry import trace as _trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as _OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource as _Resource
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor as _BatchSpanProcessor,
        ConsoleSpanExporter as _ConsoleSpanExporter,
        SimpleSpanProcessor as _SimpleSpanProcessor,
    )
    from opentelemetry.trace import SpanKind as _SpanKind
except ModuleNotFoundError:  # pragma: no cover - dependency optional in minimal environments
    trace = cast(Any, None)

    class _FallbackSpanKind:
        INTERNAL = None

    SpanKind = cast(Any, _FallbackSpanKind)
    OTLPSpanExporter = cast(Any, None)
    Resource = cast(Any, None)
    TracerProvider = cast(Any, object)
    BatchSpanProcessor = ConsoleSpanExporter = SimpleSpanProcessor = cast(Any, object)
    _OPENTELEMETRY_AVAILABLE = False
else:  # pragma: no cover - exercised in environments with OpenTelemetry installed
    trace = _trace
    SpanKind = _SpanKind
    OTLPSpanExporter = _OTLPSpanExporter
    Resource = _Resource
    TracerProvider = _TracerProvider
    BatchSpanProcessor = _BatchSpanProcessor
    ConsoleSpanExporter = _ConsoleSpanExporter
    SimpleSpanProcessor = _SimpleSpanProcessor
    _OPENTELEMETRY_AVAILABLE = True

DEFAULT_SPAN_KIND = getattr(SpanKind, "INTERNAL", None)

_MISSING_DEPENDENCY_LOGGED = False

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from opentelemetry.trace import Tracer
else:  # pragma: no cover - runtime fallback when dependency missing
    Tracer = Any

ResourceAttributeValue = (
    str
    | bool
    | int
    | float
    | Sequence[str]
    | Sequence[bool]
    | Sequence[int]
    | Sequence[float]
)

logger = logging.getLogger(__name__)


class EventBusTracer(Protocol):
    """Interface for tracing event bus operations."""

    def publish_span(
        self,
        *,
        event_type: str,
        event_source: str | None,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]: ...

    def handler_span(
        self,
        *,
        event_type: str,
        handler_name: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]: ...


class RuntimeTracer(Protocol):
    """Interface for tracing runtime workloads and operations."""

    def workload_span(
        self,
        *,
        workload: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]: ...

    def operation_span(
        self,
        *,
        name: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]: ...


class NullEventBusTracer:
    """No-op tracer used when tracing is disabled."""

    def publish_span(
        self,
        *,
        event_type: str,
        event_source: str | None,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]:
        return nullcontext()

    def handler_span(
        self,
        *,
        event_type: str,
        handler_name: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]:
        return nullcontext()


class NullRuntimeTracer:
    """No-op runtime tracer used when instrumentation is disabled."""

    def workload_span(
        self,
        *,
        workload: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]:
        return nullcontext()

    def operation_span(
        self,
        *,
        name: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]:
        return nullcontext()


def _normalise_metadata(metadata: Mapping[str, object] | None, *, prefix: str) -> dict[str, object]:
    if not metadata:
        return {}
    attributes: dict[str, object] = {}
    for raw_key, value in metadata.items():
        if value is None:
            continue
        key = raw_key if raw_key.startswith(prefix) else f"{prefix}{raw_key}"
        if isinstance(value, (str, bool, int, float)):
            attributes[key] = value
        else:
            attributes[key] = str(value)
    return attributes


@dataclass
class OpenTelemetryEventBusTracer:
    """OpenTelemetry-backed tracer for event bus activity."""

    tracer: Any
    publish_span_name: str = "event_bus.publish"
    handler_span_name: str = "event_bus.handle"
    span_kind: Any = DEFAULT_SPAN_KIND

    def publish_span(
        self,
        *,
        event_type: str,
        event_source: str | None,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]:
        attributes: MutableMapping[str, object] = {
            "event.bus.event_type": event_type,
        }
        if event_source:
            attributes["event.bus.source"] = event_source
        attributes.update(_normalise_metadata(metadata, prefix="event.bus."))
        return self.tracer.start_as_current_span(
            self.publish_span_name,
            kind=self.span_kind,
            attributes=dict(attributes),
        )

    def handler_span(
        self,
        *,
        event_type: str,
        handler_name: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]:
        attributes: MutableMapping[str, object] = {
            "event.bus.event_type": event_type,
            "event.bus.handler": handler_name,
        }
        attributes.update(_normalise_metadata(metadata, prefix="event.bus."))
        return self.tracer.start_as_current_span(
            self.handler_span_name,
            kind=self.span_kind,
            attributes=dict(attributes),
        )


@dataclass
class OpenTelemetryRuntimeTracer:
    """OpenTelemetry-backed tracer for runtime workloads."""

    tracer: Any
    workload_span_name: str = "runtime.workload"
    operation_span_name: str = "runtime.operation"
    span_kind: Any = DEFAULT_SPAN_KIND

    def workload_span(
        self,
        *,
        workload: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]:
        attributes: MutableMapping[str, object] = {
            "runtime.workload.name": workload,
        }
        attributes.update(_normalise_metadata(metadata, prefix="runtime."))
        return self.tracer.start_as_current_span(
            self.workload_span_name,
            kind=self.span_kind,
            attributes=dict(attributes),
        )

    def operation_span(
        self,
        *,
        name: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextManager[object]:
        attributes: MutableMapping[str, object] = {
            "runtime.operation.name": name,
        }
        attributes.update(_normalise_metadata(metadata, prefix="runtime."))
        return self.tracer.start_as_current_span(
            self.operation_span_name,
            kind=self.span_kind,
            attributes=dict(attributes),
        )


@dataclass(frozen=True)
class OpenTelemetrySettings:
    enabled: bool = False
    service_name: str = "emp-professional-runtime"
    environment: str | None = None
    endpoint: str | None = None
    headers: Mapping[str, str] | None = None
    timeout: float | None = None
    console_exporter: bool = False
    logs_endpoint: str | None = None
    logs_headers: Mapping[str, str] | None = None
    logs_timeout: float | None = None


def _coerce_bool(value: str | bool | None, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_headers(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    headers: dict[str, str] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            headers[key] = value
    return headers or None


def parse_opentelemetry_settings(
    extras: Mapping[str, str] | None,
) -> OpenTelemetrySettings:
    mapping = extras or {}
    enabled = _coerce_bool(mapping.get("OTEL_ENABLED"), default=False)
    service_name = mapping.get("OTEL_SERVICE_NAME", "emp-professional-runtime").strip()
    environment = mapping.get("OTEL_ENVIRONMENT") or None
    endpoint = (mapping.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip() or None
    headers = _parse_headers(mapping.get("OTEL_EXPORTER_OTLP_HEADERS"))
    timeout_raw = mapping.get("OTEL_EXPORTER_OTLP_TIMEOUT")
    timeout = None
    if timeout_raw:
        try:
            timeout = float(timeout_raw)
        except (TypeError, ValueError):
            timeout = None
    console_exporter = _coerce_bool(mapping.get("OTEL_CONSOLE_EXPORTER"), default=False)
    logs_endpoint = (mapping.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT") or "").strip() or None
    logs_headers = _parse_headers(mapping.get("OTEL_EXPORTER_OTLP_LOGS_HEADERS"))
    logs_timeout_raw = mapping.get("OTEL_EXPORTER_OTLP_LOGS_TIMEOUT")
    logs_timeout = None
    if logs_timeout_raw:
        try:
            logs_timeout = float(logs_timeout_raw)
        except (TypeError, ValueError):
            logs_timeout = None
    return OpenTelemetrySettings(
        enabled=enabled,
        service_name=service_name or "emp-professional-runtime",
        environment=environment,
        endpoint=endpoint,
        headers=headers,
        timeout=timeout,
        console_exporter=console_exporter,
        logs_endpoint=logs_endpoint,
        logs_headers=logs_headers,
        logs_timeout=logs_timeout,
    )


_PROVIDER_LOCK = threading.Lock()


def _configure_tracer_provider(settings: OpenTelemetrySettings) -> Tracer | None:
    if not settings.enabled:
        return None

    if not _OPENTELEMETRY_AVAILABLE or trace is None:
        global _MISSING_DEPENDENCY_LOGGED
        if not _MISSING_DEPENDENCY_LOGGED:
            logger.warning(
                "OpenTelemetry requested but dependencies are unavailable; disabling instrumentation",
            )
            _MISSING_DEPENDENCY_LOGGED = True
        return None

    resource_attributes: dict[str, ResourceAttributeValue] = {
        "service.name": settings.service_name
    }
    if settings.environment:
        resource_attributes["deployment.environment"] = settings.environment

    with _PROVIDER_LOCK:
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider) and getattr(provider, "_emp_configured", False):
            return provider.get_tracer(settings.service_name)

        new_provider = TracerProvider(resource=Resource.create(resource_attributes))

        if settings.endpoint:
            exporter = OTLPSpanExporter(
                endpoint=settings.endpoint,
                headers=dict(settings.headers) if settings.headers is not None else None,
                timeout=settings.timeout,
            )
            new_provider.add_span_processor(BatchSpanProcessor(exporter))
        if settings.console_exporter:
            new_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        try:
            trace.set_tracer_provider(new_provider)
            provider = new_provider
        except Exception:  # pragma: no cover - defensive guard for reconfiguration
            logger.debug("OpenTelemetry tracer provider already configured", exc_info=True)
            provider = trace.get_tracer_provider()
        else:
            setattr(provider, "_emp_configured", True)
            logger.info("OpenTelemetry tracing enabled for service %s", settings.service_name)

        return provider.get_tracer(settings.service_name)


def configure_event_bus_tracer(
    settings: OpenTelemetrySettings,
) -> EventBusTracer | None:
    """Configure and return an OpenTelemetry tracer for the event bus."""

    tracer = _configure_tracer_provider(settings)
    if tracer is None:
        return None
    return OpenTelemetryEventBusTracer(tracer=tracer)


def configure_runtime_tracer(
    settings: OpenTelemetrySettings,
) -> RuntimeTracer | None:
    """Configure and return an OpenTelemetry tracer for runtime workloads."""

    tracer = _configure_tracer_provider(settings)
    if tracer is None:
        return None
    return OpenTelemetryRuntimeTracer(tracer=tracer)
