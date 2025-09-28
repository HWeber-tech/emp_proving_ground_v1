"""Generate scenario backtest artifacts for the high-impact roadmap."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading.strategies.catalog_loader import load_strategy_catalog
from src.trading.strategies.scenario_backtests import (
    DEFAULT_SCENARIOS,
    MarketScenario,
    run_catalog_backtests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic strategy scenarios and store artifacts."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("config/trading/strategy_catalog.yaml"),
        help="Path to the strategy catalog configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/strategies/default_scenarios.json"),
        help="Path to the JSON artifact output.",
    )
    return parser.parse_args()


async def _run(catalog_path: Path, scenarios: Sequence[MarketScenario], output_path: Path) -> None:
    catalog = load_strategy_catalog(catalog_path)
    results = await run_catalog_backtests(catalog, scenarios)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "catalog_version": catalog.version,
        "scenario_count": len(scenarios),
        "strategies": [definition.as_dict() for definition in catalog.enabled_strategies()],
        "results": [result.as_dict() for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    asyncio.run(_run(args.catalog, DEFAULT_SCENARIOS, args.output))
    print(f"Wrote scenario backtests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
