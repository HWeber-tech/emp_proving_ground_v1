#!/usr/bin/env python3
"""Professional Predator runtime entrypoint with explicit lifecycle management."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.governance.system_config import EmpTier, SystemConfig
from src.runtime.predator_app import ProfessionalPredatorApp, build_professional_predator_app

logger = logging.getLogger(__name__)


async def _run_tier0_ingest(
    app: ProfessionalPredatorApp,
    *,
    symbols_csv: str,
    db_path: str,
) -> None:
    """Execute Tier-0 ingest and fan out data through the configured sensors."""

    from src.data_foundation.ingest.yahoo_ingest import fetch_daily_bars, store_duckdb

    symbols = [s.strip() for s in symbols_csv.split(",") if s.strip()]
    if not symbols:
        logger.info("No symbols supplied for Tier-0 ingest; skipping")
        return

    logger.info("📥 Tier-0 ingest for %s", symbols)
    sensor_items = list(app.sensors.items())
    destination = Path(db_path)

    def _ingest() -> tuple[int, int]:
        df = fetch_daily_bars(symbols)
        if df.empty:
            return 0, 0

        store_duckdb(df, destination)

        total_signals = 0
        for name, sensor in sensor_items:
            try:
                signals = sensor.process(df)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Sensor %s failed during Tier-0 ingest: %s", name, exc)
            else:
                total_signals += len(signals)

        return len(df), total_signals

    try:
        rows, signal_count = await asyncio.to_thread(_ingest)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Tier-0 ingest failed (continuing): %s", exc)
        return

    if rows:
        logger.info("✅ Stored %s rows to %s", rows, db_path)
    logger.info("🧠 Signals produced: count=%s", signal_count)


async def main() -> None:
    """Main entry point for Professional Predator."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from src.system.requirements_check import assert_scientific_stack

    assert_scientific_stack()

    parser = argparse.ArgumentParser(description="EMP Professional Predator")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip Tier-0 data ingestion at startup")
    parser.add_argument("--symbols", type=str, default="EURUSD,GBPUSD", help="Comma-separated symbols for Tier-0 ingest")
    parser.add_argument("--db", type=str, default="data/tier0.duckdb", help="DuckDB path for Tier-0 ingest")
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
            if tier is EmpTier.tier_0 and not args.skip_ingest:
                await _run_tier0_ingest(app, symbols_csv=args.symbols, db_path=args.db)
            elif tier is EmpTier.tier_1:
                logger.info("🧩 Tier-1 (Timescale/Redis) not implemented yet")
            elif tier is EmpTier.tier_2:
                raise NotImplementedError("Tier-2 evolutionary mode is not yet supported")

            await app.run_forever()
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
