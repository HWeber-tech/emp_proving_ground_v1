#!/usr/bin/env python3
"""Professional Predator runtime entrypoint leveraging the runtime builder."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys
from types import TracebackType
from typing import Mapping, cast

from src.data_foundation.duckdb_security import (
    resolve_encrypted_duckdb_path,
    verify_encrypted_duckdb_path,
)
from src.governance.system_config import (
    ConnectionProtocol,
    EmpTier,
    RunMode,
    SystemConfig,
    render_config_diff,
)
from src.observability.tracing import parse_opentelemetry_settings
from src.operations.configuration_audit import (
    evaluate_configuration_audit,
    persist_configuration_snapshot,
)
from src.operational.metrics import start_metrics_server
from src.operational.structured_logging import (
    configure_structlog,
    get_logger,
    load_structlog_otel_settings,
)
from src.runtime.determinism import resolve_seed, seed_runtime
from src.runtime.predator_app import build_professional_predator_app
from src.runtime.runtime_builder import (
    _execute_timescale_ingest,
    build_professional_runtime_application,
)
from src.runtime.runtime_runner import run_runtime_application


logger = get_logger(__name__)

__all__ = ["_execute_timescale_ingest", "main"]


DEFAULT_CONFIG_SNAPSHOT_PATH = Path("artifacts/runtime/config_snapshot.json")
ExcInfo = tuple[type[BaseException], BaseException, TracebackType | None]


def _resolve_config_snapshot_path(extras: Mapping[str, str] | None) -> Path | None:
    """Resolve the destination path for the configuration snapshot."""

    if not extras:
        return DEFAULT_CONFIG_SNAPSHOT_PATH

    raw_path = extras.get("CONFIG_SNAPSHOT_PATH")
    if raw_path is None:
        return DEFAULT_CONFIG_SNAPSHOT_PATH

    text = str(raw_path).strip()
    if not text:
        return DEFAULT_CONFIG_SNAPSHOT_PATH

    lowered = text.lower()
    if lowered in {"disabled", "none", "off"}:
        return None

    return Path(text).expanduser()


def _capture_configuration_snapshot(
    config: SystemConfig,
    extras: Mapping[str, str] | None,
    rng_seed: int | None,
) -> tuple[Path | None, ExcInfo | None, Path | None]:
    """Evaluate and persist the configuration snapshot if enabled."""

    target = _resolve_config_snapshot_path(extras)
    if target is None:
        return None, None, None

    metadata = {"source": "runtime_boot"}
    if rng_seed is not None:
        metadata["rng_seed"] = rng_seed

    try:
        snapshot = evaluate_configuration_audit(config, metadata=metadata)
        persisted = persist_configuration_snapshot(snapshot, target)
        return persisted, None, target
    except Exception:
        exc_info = cast(ExcInfo, sys.exc_info())
        return None, exc_info, target


def _maybe_start_metrics_exporter(extras: Mapping[str, str] | None) -> None:
    """Start the Prometheus exporter when enabled via configuration."""

    if extras is None:
        extras = {}

    raw_enabled = extras.get("METRICS_EXPORTER_ENABLED")
    if raw_enabled is not None:
        text = str(raw_enabled).strip().lower()
        if text in {"0", "false", "no", "off", "disabled"}:
            logger.info("Prometheus metrics exporter disabled via extras")
            return
        if text not in {"", "1", "true", "yes", "on", "enabled"}:
            logger.warning(
                "Unrecognised METRICS_EXPORTER_ENABLED value %r; defaulting to enabled",
                raw_enabled,
            )

    port: int | None = None
    raw_port = extras.get("METRICS_EXPORTER_PORT")
    if raw_port is not None:
        try:
            port = int(str(raw_port).strip())
        except (TypeError, ValueError):
            logger.warning(
                "Invalid METRICS_EXPORTER_PORT value %r; defaulting to EMP_METRICS_PORT",
                raw_port,
            )

    cert_path = extras.get("METRICS_EXPORTER_TLS_CERT_PATH")
    if cert_path is not None:
        cert_path = str(cert_path).strip() or None

    key_path = extras.get("METRICS_EXPORTER_TLS_KEY_PATH")
    if key_path is not None:
        key_path = str(key_path).strip() or None

    try:
        start_metrics_server(port=port, cert_path=cert_path, key_path=key_path)
    except ValueError as exc:
        logger.warning("Prometheus metrics exporter not started: %s", exc)
    except Exception:  # pragma: no cover - defensive guard around exporter init
        logger.exception("Failed to start Prometheus metrics exporter")


def _normalise_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_structlog_level(raw: object | None) -> int:
    text = _normalise_optional_str(raw)
    if text is None:
        return logging.INFO
    candidate = logging.getLevelName(text.upper())
    if isinstance(candidate, int):
        return candidate
    logger.warning(
        "Unrecognised STRUCTLOG_LEVEL %r; defaulting to INFO",
        raw,
        extra={"structlog.level_invalid": text},
    )
    return logging.INFO


def _parse_structlog_destination(raw: object | None) -> str | Path | None:
    text = _normalise_optional_str(raw)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"stdout", "stderr", "default"}:
        return lowered
    return Path(text).expanduser()


async def main() -> None:
    """Main entry point for Professional Predator."""

    config = SystemConfig.from_env()
    extras: Mapping[str, str] = dict(config.extras) if config.extras else {}

    rng_seed, invalid_seed_entries = resolve_seed(extras)
    if rng_seed is not None:
        seed_runtime(rng_seed)

    snapshot_path, snapshot_exc_info, snapshot_target = _capture_configuration_snapshot(
        config, extras, rng_seed
    )

    otel_settings = parse_opentelemetry_settings(extras)
    if not otel_settings.enabled:
        structlog_profile = extras.get("STRUCTLOG_OTEL_CONFIG")
        if structlog_profile:
            profile_hint = structlog_profile.strip()
            if profile_hint.lower() in {"default", "local", "local-dev"}:
                profile_path = Path("config/observability/logging.yaml")
            else:
                profile_path = Path(profile_hint)
            try:
                otel_settings = load_structlog_otel_settings(
                    profile_path,
                    default_service_name=otel_settings.service_name,
                    default_environment=otel_settings.environment,
                )
            except FileNotFoundError:
                logger.warning(
                    "Structured logging OpenTelemetry profile not found",  # pragma: no cover - exercised via integration
                    extra={"structlog.otel_config": str(profile_path)},
                )
            except ValueError as exc:
                logger.warning(
                    "Failed to load structlog OpenTelemetry profile: %s",  # pragma: no cover - defensive guard
                    exc,
                    extra={"structlog.otel_config": str(profile_path)},
                )
    structlog_level = _parse_structlog_level(extras.get("STRUCTLOG_LEVEL"))
    raw_format = extras.get("STRUCTLOG_OUTPUT_FORMAT") or extras.get("STRUCTLOG_FORMAT")
    output_format = _normalise_optional_str(raw_format)
    destination = _parse_structlog_destination(extras.get("STRUCTLOG_DESTINATION"))

    configure_structlog(
        level=structlog_level,
        output_format=output_format,
        destination=destination,
        otel_settings=otel_settings,
    )
    _maybe_start_metrics_exporter(extras)

    if rng_seed is not None:
        logger.info("🔐 Deterministic RNG seed initialised", extra={"rng_seed": rng_seed})
    else:
        logger.warning("No deterministic RNG seed provided; runtime seeding skipped")

    if invalid_seed_entries:
        logger.warning(
            "Ignoring invalid RNG seed values",
            extra={"rng_seed.invalid": invalid_seed_entries},
        )

    if snapshot_exc_info is not None and snapshot_target is not None:
        logger.warning(
            "Failed to persist configuration snapshot",
            extra={"config_snapshot_path": str(snapshot_target)},
            exc_info=snapshot_exc_info,
        )
    elif snapshot_path is not None:
        logger.info(
            "📸 Configuration snapshot persisted",
            extra={"config_snapshot_path": str(snapshot_path)},
        )
    else:
        logger.debug("Configuration snapshot persistence disabled")

    from src.system.requirements_check import assert_scientific_stack

    assert_scientific_stack()

    parser = argparse.ArgumentParser(description="EMP Professional Predator")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip Tier-0 data ingestion at startup",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--paper-mode",
        action="store_true",
        help="Force paper trading configuration overrides",
    )
    mode_group.add_argument(
        "--live-mode",
        action="store_true",
        help="Force live trading configuration overrides",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="EURUSD,GBPUSD",
        help="Comma-separated symbols for Tier-0 ingest",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/tier0.duckdb",
        help="DuckDB path for Tier-0 ingest",
    )
    args, _ = parser.parse_known_args()

    if args.paper_mode:
        config = config.with_updated(
            run_mode=RunMode.paper,
            confirm_live=False,
            connection_protocol=ConnectionProtocol.paper,
        )
    elif args.live_mode:
        config = config.with_updated(
            run_mode=RunMode.live,
            confirm_live=True,
            connection_protocol=ConnectionProtocol.fix,
        )

    logger.info(render_config_diff(config))

    try:
        app = await build_professional_predator_app(config=config)
    except Exception:
        logger.exception("❌ Error initializing Professional Predator")
        raise

    try:
        async with app:
            summary = app.summary()
            logger.info("📊 System Summary:")
            for key, value in summary.items():
                if key == "components" and isinstance(value, dict):
                    logger.info("  components:")
                    for comp_key, comp_val in value.items():
                        logger.info("    %s: %s", comp_key, comp_val)
                else:
                    logger.info("  %s: %s", key, value)

            tier = app.config.tier
            if tier is EmpTier.tier_2:
                raise NotImplementedError("Tier-2 evolutionary mode is not yet supported")

            duckdb_destination = resolve_encrypted_duckdb_path(args.db)
            verify_encrypted_duckdb_path(duckdb_destination)
            runtime_app = build_professional_runtime_application(
                app,
                skip_ingest=args.skip_ingest,
                symbols_csv=args.symbols,
                duckdb_path=str(duckdb_destination),
            )

            plan_summary = runtime_app.summary()
            ingestion_plan = plan_summary.get("ingestion")
            if ingestion_plan:
                logger.info(
                    "🧭 Ingestion workload: %s (%s)",
                    ingestion_plan.get("name"),
                    ingestion_plan.get("description"),
                )
                metadata = ingestion_plan.get("metadata")
                if metadata:
                    logger.info("    metadata: %s", metadata)

            trading_plan = plan_summary.get("trading")
            if trading_plan:
                logger.info(
                    "🧭 Trading workload: %s (%s)",
                    trading_plan.get("name"),
                    trading_plan.get("description"),
                )

            await run_runtime_application(
                runtime_app,
                logger=logger,
                namespace="runtime.main",
            )
    except asyncio.CancelledError:
        logger.info("⏹️ Received cancellation signal")
        raise
    except Exception:
        logger.exception("❌ Professional Predator failed")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        sys.exit(1)
