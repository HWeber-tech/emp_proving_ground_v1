#!/usr/bin/env python3
"""Professional Predator runtime entrypoint leveraging the runtime builder."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from src.governance.system_config import EmpTier, SystemConfig
from src.runtime.predator_app import build_professional_predator_app
from src.runtime.runtime_builder import (
    _execute_timescale_ingest,
    build_professional_runtime_application,
)


logger = logging.getLogger(__name__)

__all__ = ["_execute_timescale_ingest", "main"]


async def main() -> None:
    """Main entry point for Professional Predator."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from src.system.requirements_check import assert_scientific_stack

    assert_scientific_stack()

    parser = argparse.ArgumentParser(description="EMP Professional Predator")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip Tier-0 data ingestion at startup",
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

    try:
        app = await build_professional_predator_app(config=SystemConfig.from_env())
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

            await runtime_app.run()
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
