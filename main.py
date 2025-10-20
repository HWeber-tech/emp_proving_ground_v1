#!/usr/bin/env python3
"""Professional Predator runtime entrypoint leveraging the runtime builder."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys

from src.governance.system_config import (
    ConnectionProtocol,
    EmpTier,
    RunMode,
    SystemConfig,
)
from src.observability.tracing import parse_opentelemetry_settings
from src.runtime.predator_app import build_professional_predator_app
from src.runtime.runtime_builder import (
    _execute_timescale_ingest,
    build_professional_runtime_application,
)
from src.runtime.runtime_runner import run_runtime_application

from src.operational.structured_logging import (
    configure_structlog,
    get_logger,
    load_structlog_otel_settings,
)


logger = get_logger(__name__)

__all__ = ["_execute_timescale_ingest", "main"]


async def main() -> None:
    """Main entry point for Professional Predator."""

    config = SystemConfig.from_env()
    otel_settings = parse_opentelemetry_settings(config.extras)
    if not otel_settings.enabled:
        structlog_profile = config.extras.get("STRUCTLOG_OTEL_CONFIG")
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
    configure_structlog(level=logging.INFO, otel_settings=otel_settings)

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

            runtime_app = build_professional_runtime_application(
                app,
                skip_ingest=args.skip_ingest,
                symbols_csv=args.symbols,
                duckdb_path=args.db,
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
