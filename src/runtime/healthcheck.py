"""Runtime health evaluation and lightweight HTTP healthcheck server."""

from __future__ import annotations

import logging
import ssl
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from aiohttp import web

from src.governance.system_config import ConnectionProtocol, DataBackboneMode
from src.operations.data_backbone import DataBackboneReadinessSnapshot
from src.runtime.predator_app import ProfessionalPredatorApp
from src.security.auth_tokens import (
    AuthTokenError,
    ExpiredTokenError,
    decode_access_token,
)


logger = logging.getLogger(__name__)


class RuntimeHealthStatus(str):
    """String enum representing the aggregate runtime health state."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class RuntimeHealthCheck:
    """Individual health check entry surfaced by the runtime health snapshot."""

    name: str
    status: str
    summary: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeHealthSnapshot:
    """Aggregate runtime health state captured at a point in time."""

    status: str
    generated_at: datetime
    checks: tuple[RuntimeHealthCheck, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "generated_at": self.generated_at.isoformat(),
            "checks": [check.as_dict() for check in self.checks],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def _combine_status(current: str, new: str) -> str:
    if current == RuntimeHealthStatus.FAIL or new == RuntimeHealthStatus.FAIL:
        return RuntimeHealthStatus.FAIL
    if current == RuntimeHealthStatus.WARN or new == RuntimeHealthStatus.WARN:
        return RuntimeHealthStatus.WARN
    return RuntimeHealthStatus.OK


def _parse_iso_timestamp(raw: object) -> datetime | None:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw.astimezone(UTC)
    try:
        parsed = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


def _ingest_component(
    snapshot: DataBackboneReadinessSnapshot | None, name: str
) -> Mapping[str, object] | None:
    if snapshot is None:
        return None
    for component in snapshot.components:
        if component.name == name:
            return component.metadata
    return None


def _coerce_str_tuple(value: object | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return (str(value),)


def evaluate_runtime_health(
    app: ProfessionalPredatorApp,
    *,
    ingest_warn_after: float = 900.0,
    ingest_fail_after: float = 1800.0,
    decision_warn_after: float = 180.0,
    decision_fail_after: float = 600.0,
) -> RuntimeHealthSnapshot:
    """Evaluate FIX connectivity, market-data freshness, and telemetry exporters."""

    now = datetime.now(tz=UTC)
    checks: list[RuntimeHealthCheck] = []
    overall = RuntimeHealthStatus.OK

    # FIX connectivity
    if app.config.connection_protocol is ConnectionProtocol.fix:
        manager = getattr(app, "fix_connection_manager", None)
        broker = getattr(app, "broker_interface", None)
        broker_running = bool(getattr(broker, "running", False))
        initiator = None
        if manager is not None:
            try:
                initiator = manager.get_initiator("trade")
            except Exception:  # pragma: no cover - defensive diagnostics
                logger.debug("FIX initiator lookup failed", exc_info=True)
                initiator = None
        status = RuntimeHealthStatus.OK
        summary = "FIX sessions active"
        if manager is None or initiator is None:
            status = RuntimeHealthStatus.FAIL
            summary = "FIX sessions unavailable"
        elif not broker_running:
            status = RuntimeHealthStatus.WARN
            summary = "FIX broker interface idle"
        checks.append(
            RuntimeHealthCheck(
                name="fix_connectivity",
                status=status,
                summary=summary,
                metadata={
                    "manager_present": manager is not None,
                    "broker_running": broker_running,
                    "initiator_present": initiator is not None,
                },
            )
        )
        overall = _combine_status(overall, status)
    else:
        checks.append(
            RuntimeHealthCheck(
                name="fix_connectivity",
                status=RuntimeHealthStatus.NOT_APPLICABLE,
                summary=f"FIX not required for {app.config.connection_protocol.value}",
                metadata={},
            )
        )

    # Market-data freshness
    snapshot = None
    if hasattr(app, "get_last_data_backbone_snapshot"):
        try:
            snapshot = app.get_last_data_backbone_snapshot()
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to fetch data backbone snapshot", exc_info=True)
            snapshot = None

    market_status = RuntimeHealthStatus.OK
    market_summary = "market data fresh"
    market_metadata: dict[str, object] = {}
    if app.config.data_backbone_mode is DataBackboneMode.institutional:
        ingest_metadata = _ingest_component(snapshot, "ingest_health")
        generated_at = (
            _parse_iso_timestamp(ingest_metadata.get("generated_at")) if ingest_metadata else None
        )
        if generated_at is None:
            market_status = RuntimeHealthStatus.WARN
            market_summary = "no ingest telemetry captured"
        else:
            age = (now - generated_at).total_seconds()
            market_metadata["age_seconds"] = age
            market_metadata["generated_at"] = generated_at.isoformat()
            if age > ingest_fail_after:
                market_status = RuntimeHealthStatus.FAIL
                market_summary = "ingest telemetry stale"
            elif age > ingest_warn_after:
                market_status = RuntimeHealthStatus.WARN
                market_summary = "ingest telemetry aging"
    else:
        sensory = getattr(app, "sensory_organ", None)
        latest_decision_ts: datetime | None = None
        if sensory is not None and hasattr(sensory, "status"):
            try:
                status_payload = sensory.status()
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Bootstrap runtime status call failed", exc_info=True)
            else:
                telemetry = (
                    status_payload.get("telemetry") if isinstance(status_payload, Mapping) else None
                )
                last_decision = (
                    telemetry.get("last_decision") if isinstance(telemetry, Mapping) else None
                )
                if isinstance(last_decision, Mapping):
                    latest_decision_ts = _parse_iso_timestamp(last_decision.get("generated_at"))
        if latest_decision_ts is None:
            market_status = RuntimeHealthStatus.WARN
            market_summary = "no recent decision telemetry"
        else:
            age = (now - latest_decision_ts).total_seconds()
            market_metadata["age_seconds"] = age
            market_metadata["generated_at"] = latest_decision_ts.isoformat()
            if age > decision_fail_after:
                market_status = RuntimeHealthStatus.FAIL
                market_summary = "decision telemetry stale"
            elif age > decision_warn_after:
                market_status = RuntimeHealthStatus.WARN
                market_summary = "decision telemetry aging"

    checks.append(
        RuntimeHealthCheck(
            name="market_data",
            status=market_status,
            summary=market_summary,
            metadata=market_metadata,
        )
    )
    overall = _combine_status(overall, market_status)

    # Telemetry exporters
    event_bus = getattr(app, "event_bus", None)
    bus_running = bool(getattr(event_bus, "is_running", lambda: False)()) if event_bus else False
    kafka_metadata = _ingest_component(snapshot, "kafka_streaming")
    kafka_expected = kafka_metadata is not None
    publishers = (
        _coerce_str_tuple(kafka_metadata.get("publishers")) if kafka_metadata else tuple()
    )
    topics = _coerce_str_tuple(kafka_metadata.get("topics")) if kafka_metadata else tuple()
    telemetry_status = RuntimeHealthStatus.OK
    telemetry_summary = "telemetry exporters online"
    if not bus_running:
        telemetry_status = RuntimeHealthStatus.WARN
        telemetry_summary = "runtime event bus inactive"
    if kafka_expected and not publishers:
        telemetry_status = RuntimeHealthStatus.FAIL
        telemetry_summary = "Kafka publishers unavailable"

    checks.append(
        RuntimeHealthCheck(
            name="telemetry_exporters",
            status=telemetry_status,
            summary=telemetry_summary,
            metadata={
                "event_bus_running": bus_running,
                "kafka_expected": kafka_expected,
                "kafka_topics": list(topics),
                "kafka_publishers": list(publishers),
            },
        )
    )
    overall = _combine_status(overall, telemetry_status)

    return RuntimeHealthSnapshot(
        status=overall,
        generated_at=now,
        checks=tuple(checks),
        metadata={
            "protocol": app.config.connection_protocol.value,
            "backbone_mode": app.config.data_backbone_mode.value,
        },
    )


MetricSample = tuple[float, Mapping[str, str]]
MetricValue = float | MetricSample | Sequence[MetricSample]


def _escape_label_value(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\"", '\\"')
    )


def _format_prometheus_metrics(metrics: Mapping[str, MetricValue]) -> str:
    def _iter_samples(value: MetricValue) -> Iterable[MetricSample]:
        if isinstance(value, (int, float)):
            yield float(value), {}
            return
        if isinstance(value, tuple) and len(value) == 2:
            numeric, labels = value
            if isinstance(numeric, (int, float)) and isinstance(labels, Mapping):
                yield float(numeric), labels
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if not (isinstance(item, tuple) and len(item) == 2):
                    continue
                numeric, labels = item
                if not isinstance(labels, Mapping):
                    continue
                if isinstance(numeric, (int, float)):
                    yield float(numeric), labels

    lines: list[str] = []
    for name, value in metrics.items():
        lines.append(f"# TYPE {name} gauge")
        samples = list(_iter_samples(value))
        if not samples:
            samples = [(0.0, {})]
        for numeric, labels in samples:
            label_text = ""
            if labels:
                rendered = ",".join(
                    f"{str(key)}=\"{_escape_label_value(val)}\""
                    for key, val in sorted(labels.items(), key=lambda item: str(item[0]))
                )
                label_text = f"{{{rendered}}}"
            lines.append(f"{name}{label_text} {format(numeric, 'g')}")
    return "\n".join(lines) + "\n"


def _status_to_value(status: str) -> float:
    if status == RuntimeHealthStatus.FAIL:
        return 2.0
    if status == RuntimeHealthStatus.WARN:
        return 1.0
    return 0.0


def _collect_runtime_metrics(
    app: ProfessionalPredatorApp,
    *,
    ingest_warn_after: float = 900.0,
    ingest_fail_after: float = 1800.0,
    decision_warn_after: float = 180.0,
    decision_fail_after: float = 600.0,
) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {
        "event_lag_ms": 0.0,
        "queue_depth": 0.0,
        "p50_infer_ms": 0.0,
        "p90_infer_ms": 0.0,
        "p99_infer_ms": 0.0,
        "drops": 0.0,
        "risk_halted": 0.0,
        "runtime_latency_p99_seconds": 0.0,
        "event_handler_exception_rate_per_minute": 0.0,
        "event_handler_exceptions_total": 0.0,
        "runtime_exception_rate_per_minute": 0.0,
        "runtime_exceptions_total": 0.0,
        "runtime_health_status": 0.0,
    }

    event_bus = getattr(app, "event_bus", None)
    if event_bus is not None and hasattr(event_bus, "get_statistics"):
        try:
            stats = event_bus.get_statistics()
        except Exception:  # pragma: no cover - defensive metrics guard
            logger.debug("Failed to collect event bus statistics", exc_info=True)
        else:
            queue_size = getattr(stats, "queue_size", None)
            if isinstance(queue_size, int):
                metrics["queue_depth"] = float(queue_size)
            dropped = getattr(stats, "dropped_events", None)
            if isinstance(dropped, (int, float)):
                metrics["drops"] = float(dropped)
            last_event = getattr(stats, "last_event_timestamp", None)
            if isinstance(last_event, (int, float)):
                lag_ms = max(0.0, (time.time() - float(last_event)) * 1000.0)
                metrics["event_lag_ms"] = lag_ms
            handler_errors = getattr(stats, "handler_errors", None)
            uptime_seconds = getattr(stats, "uptime_seconds", None)
            if isinstance(handler_errors, (int, float)):
                metrics["event_handler_exceptions_total"] = float(handler_errors)
                metrics["runtime_exceptions_total"] = float(handler_errors)
                if isinstance(uptime_seconds, (int, float)) and uptime_seconds > 0:
                    rate = float(handler_errors) / (float(uptime_seconds) / 60.0)
                    metrics["event_handler_exception_rate_per_minute"] = rate
                    metrics["runtime_exception_rate_per_minute"] = rate

    latency_snapshot: Mapping[str, object] | None = None
    broker = getattr(app, "broker_interface", None)
    if broker is not None:
        describe_metrics = getattr(broker, "describe_metrics", None)
        if callable(describe_metrics):
            try:
                candidate = describe_metrics()
            except Exception:  # pragma: no cover - defensive metrics guard
                logger.debug("Failed to describe broker metrics", exc_info=True)
            else:
                if isinstance(candidate, Mapping):
                    latency_snapshot = candidate

    if latency_snapshot is None:
        trading_manager = getattr(app, "trading_manager", None)
        if trading_manager is not None:
            describe_exec = getattr(trading_manager, "get_execution_stats", None)
            if callable(describe_exec):
                try:
                    candidate = describe_exec()
                except Exception:  # pragma: no cover - defensive metrics guard
                    logger.debug("Failed to collect execution stats for metrics", exc_info=True)
                else:
                    if isinstance(candidate, Mapping):
                        latency_snapshot = candidate

    if latency_snapshot is not None:
        p99_seconds: float | None = None
        for source_key, metric_name in (
            ("p50_latency_s", "p50_infer_ms"),
            ("p90_latency_s", "p90_infer_ms"),
            ("p99_latency_s", "p99_infer_ms"),
            ("p50_latency_ms", "p50_infer_ms"),
            ("p90_latency_ms", "p90_infer_ms"),
            ("p99_latency_ms", "p99_infer_ms"),
        ):
            if metric_name not in metrics:
                continue
            value = latency_snapshot.get(source_key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if source_key.endswith("_s"):
                seconds_value = max(numeric, 0.0)
                numeric = seconds_value * 1000.0
                if metric_name == "p99_infer_ms" and p99_seconds is None:
                    p99_seconds = seconds_value
            elif metric_name == "p99_infer_ms" and p99_seconds is None:
                p99_seconds = max(numeric / 1000.0, 0.0)
            metrics[metric_name] = max(numeric, 0.0)
        if p99_seconds is not None:
            metrics["runtime_latency_p99_seconds"] = p99_seconds

    trading_manager = getattr(app, "trading_manager", None)
    if trading_manager is not None:
        describe_exec = getattr(trading_manager, "get_execution_stats", None)
        if callable(describe_exec):
            try:
                exec_snapshot = describe_exec()
            except Exception:  # pragma: no cover - defensive metrics guard
                logger.debug("Failed to evaluate risk posture from execution stats", exc_info=True)
            else:
                if isinstance(exec_snapshot, Mapping):
                    guardrail = exec_snapshot.get("guardrail_force")
                    if isinstance(guardrail, Mapping) and guardrail.get("force_paper"):
                        metrics["risk_halted"] = 1.0
                    else:
                        expires_raw = exec_snapshot.get("guardrail_force_paper_until")
                        expires_at = _parse_iso_datetime(expires_raw)
                        if expires_at is not None and expires_at > datetime.now(tz=UTC):
                            metrics["risk_halted"] = 1.0

    snapshot = evaluate_runtime_health(
        app,
        ingest_warn_after=ingest_warn_after,
        ingest_fail_after=ingest_fail_after,
        decision_warn_after=decision_warn_after,
        decision_fail_after=decision_fail_after,
    )
    metrics["runtime_health_status"] = _status_to_value(snapshot.status)
    check_samples: list[MetricSample] = []
    for check in snapshot.checks:
        check_samples.append((_status_to_value(check.status), {"check": str(check.name)}))
    if check_samples:
        metrics["runtime_health_check_status"] = tuple(check_samples)

    return metrics


def _parse_iso_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


StartupCallback = Callable[[], Awaitable[None] | None]


class RuntimeHealthServer:
    """Minimal aiohttp server exposing ``/health`` with runtime health telemetry."""

    def __init__(
        self,
        app: ProfessionalPredatorApp,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        path: str = "/health",
        metrics_path: str = "/metrics",
        auth_secret: str,
        health_roles: Sequence[str] | None = None,
        metrics_roles: Sequence[str] | None = None,
        token_audience: str | None = None,
        ingest_warn_after: float = 900.0,
        ingest_fail_after: float = 1800.0,
        decision_warn_after: float = 180.0,
        decision_fail_after: float = 600.0,
        ssl_context: ssl.SSLContext | None = None,
        cert_path: str | None = None,
        key_path: str | None = None,
    ) -> None:
        self._app = app
        self._host = host
        self._port = int(port)
        self._path = path if path.startswith("/") else f"/{path}"
        self._metrics_path = (
            metrics_path if metrics_path.startswith("/") else f"/{metrics_path}"
        )
        if not auth_secret:
            raise ValueError("auth_secret must be provided for RuntimeHealthServer")
        self._auth_secret = auth_secret
        self._token_audience = token_audience
        self._auth_realm = "runtime-health"
        self._health_roles = self._normalise_roles(health_roles, ("runtime.health:read",))
        self._metrics_roles = self._normalise_roles(
            metrics_roles, ("runtime.health:read", "runtime.metrics:read")
        )
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._resolved_port: int | None = None
        self._ingest_warn_after = float(ingest_warn_after)
        self._ingest_fail_after = float(ingest_fail_after)
        self._decision_warn_after = float(decision_warn_after)
        self._decision_fail_after = float(decision_fail_after)
        self._ssl_context = self._prepare_ssl_context(ssl_context, cert_path, key_path)

    @staticmethod
    def _prepare_ssl_context(
        ssl_context: ssl.SSLContext | None,
        cert_path: str | None,
        key_path: str | None,
    ) -> ssl.SSLContext | None:
        if ssl_context is not None:
            return ssl_context

        cert = Path(cert_path).expanduser() if cert_path else None
        key = Path(key_path).expanduser() if key_path else None

        if cert is None and key is None:
            return None
        if cert is None or key is None:
            raise ValueError(
                "RuntimeHealthServer requires both certificate and key paths when TLS is enabled"
            )

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(cert), keyfile=str(key))
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        return context

    @staticmethod
    def _normalise_roles(
        roles: Sequence[str] | None, default: tuple[str, ...]
    ) -> tuple[str, ...]:
        if roles is None:
            return default
        normalised: list[str] = []
        for role in roles:
            text = str(role).strip()
            if not text:
                continue
            if text not in normalised:
                normalised.append(text)
        return tuple(normalised)

    def _require_roles(
        self, request: web.Request, required_roles: tuple[str, ...]
    ) -> Mapping[str, object]:
        if not required_roles:
            return {}

        header = request.headers.get("Authorization", "").strip()
        if not header.lower().startswith("bearer "):
            raise web.HTTPUnauthorized(
                reason="missing bearer token",
                headers={"WWW-Authenticate": f'Bearer realm="{self._auth_realm}"'},
            )
        token = header[7:].strip()
        if not token:
            raise web.HTTPUnauthorized(
                reason="missing bearer token",
                headers={"WWW-Authenticate": f'Bearer realm="{self._auth_realm}"'},
            )

        try:
            payload = decode_access_token(
                token,
                secret=self._auth_secret,
                expected_audience=self._token_audience,
            )
        except ExpiredTokenError as exc:
            raise web.HTTPUnauthorized(
                reason="token expired",
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="{self._auth_realm}", error="invalid_token", '
                        "error_description=\"token expired\""
                    )
                },
            ) from exc
        except AuthTokenError as exc:
            raise web.HTTPUnauthorized(
                reason="invalid token",
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="{self._auth_realm}", error="invalid_token", '
                        "error_description=\"signature mismatch\""
                    )
                },
            ) from exc

        roles_claim = payload.get("roles")
        roles: set[str]
        if isinstance(roles_claim, Sequence) and not isinstance(roles_claim, (str, bytes, bytearray)):
            roles = {str(role) for role in roles_claim}
        elif roles_claim is None:
            roles = set()
        else:
            roles = {str(roles_claim)}

        if not set(required_roles).issubset(roles):
            raise web.HTTPForbidden(reason="insufficient role")

        return payload

    async def start(self) -> None:
        if self._runner is not None:
            return

        async def _handle_health(_request: web.Request) -> web.Response:
            self._require_roles(_request, self._health_roles)
            snapshot = evaluate_runtime_health(
                self._app,
                ingest_warn_after=self._ingest_warn_after,
                ingest_fail_after=self._ingest_fail_after,
                decision_warn_after=self._decision_warn_after,
                decision_fail_after=self._decision_fail_after,
            )
            return web.json_response(snapshot.as_dict())

        async def _handle_metrics(_request: web.Request) -> web.Response:
            self._require_roles(_request, self._metrics_roles)
            metrics = _collect_runtime_metrics(
                self._app,
                ingest_warn_after=self._ingest_warn_after,
                ingest_fail_after=self._ingest_fail_after,
                decision_warn_after=self._decision_warn_after,
                decision_fail_after=self._decision_fail_after,
            )
            body = _format_prometheus_metrics(metrics)
            return web.Response(
                text=body,
                content_type="text/plain; version=0.0.4",
            )

        web_app = web.Application()
        web_app.add_routes(
            [
                web.get(self._path, _handle_health),
                web.get(self._metrics_path, _handle_metrics),
            ]
        )

        self._runner = web.AppRunner(web_app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port, ssl_context=self._ssl_context)
        await self._site.start()

        sockets = getattr(self._site, "_server", None)
        if sockets and getattr(sockets, "sockets", None):
            sock = sockets.sockets[0]
            self._resolved_port = int(sock.getsockname()[1])
        else:
            self._resolved_port = self._port

        logger.info(
            "🩺 Runtime health endpoint available at %s (metrics at %s)",
            self.url,
            self.metrics_url,
        )

    async def stop(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._resolved_port = None

    @property
    def port(self) -> int:
        return self._resolved_port if self._resolved_port is not None else self._port

    @property
    def url(self) -> str:
        return f"https://{self._host}:{self.port}{self._path}"

    @property
    def metrics_url(self) -> str:
        return f"https://{self._host}:{self.port}{self._metrics_path}"

    def summary(self) -> Mapping[str, object]:
        return {
            "host": self._host,
            "port": self.port,
            "path": self._path,
            "url": self.url,
            "metrics_path": self._metrics_path,
            "metrics_url": self.metrics_url,
        }


__all__ = [
    "RuntimeHealthCheck",
    "RuntimeHealthServer",
    "RuntimeHealthSnapshot",
    "RuntimeHealthStatus",
    "evaluate_runtime_health",
]
