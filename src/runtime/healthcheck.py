"""Runtime health evaluation and lightweight HTTP healthcheck server."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from collections.abc import Iterable
from typing import Awaitable, Callable, Mapping

from aiohttp import web

from src.governance.system_config import ConnectionProtocol, DataBackboneMode
from src.operations.data_backbone import DataBackboneReadinessSnapshot
from src.runtime.predator_app import ProfessionalPredatorApp


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
        ingest_warn_after: float = 900.0,
        ingest_fail_after: float = 1800.0,
        decision_warn_after: float = 180.0,
        decision_fail_after: float = 600.0,
    ) -> None:
        self._app = app
        self._host = host
        self._port = int(port)
        self._path = path if path.startswith("/") else f"/{path}"
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._resolved_port: int | None = None
        self._ingest_warn_after = float(ingest_warn_after)
        self._ingest_fail_after = float(ingest_fail_after)
        self._decision_warn_after = float(decision_warn_after)
        self._decision_fail_after = float(decision_fail_after)

    async def start(self) -> None:
        if self._runner is not None:
            return

        async def _handle_health(_request: web.Request) -> web.Response:
            snapshot = evaluate_runtime_health(
                self._app,
                ingest_warn_after=self._ingest_warn_after,
                ingest_fail_after=self._ingest_fail_after,
                decision_warn_after=self._decision_warn_after,
                decision_fail_after=self._decision_fail_after,
            )
            return web.json_response(snapshot.as_dict())

        web_app = web.Application()
        web_app.add_routes([web.get(self._path, _handle_health)])

        self._runner = web.AppRunner(web_app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        sockets = getattr(self._site, "_server", None)
        if sockets and getattr(sockets, "sockets", None):
            sock = sockets.sockets[0]
            self._resolved_port = int(sock.getsockname()[1])
        else:
            self._resolved_port = self._port

        logger.info("🩺 Runtime health endpoint available at %s", self.url)

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
        return f"http://{self._host}:{self.port}{self._path}"

    def summary(self) -> Mapping[str, object]:
        return {
            "host": self._host,
            "port": self.port,
            "path": self._path,
            "url": self.url,
        }


__all__ = [
    "RuntimeHealthCheck",
    "RuntimeHealthServer",
    "RuntimeHealthSnapshot",
    "RuntimeHealthStatus",
    "evaluate_runtime_health",
]
