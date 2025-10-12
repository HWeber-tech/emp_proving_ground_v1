#!/usr/bin/env python3
"""Ingest a real-market CSV slice into Timescale and emit a belief snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_integration.real_data_slice import RealDataSliceConfig, run_real_data_slice


def _default_symbol(csv_path: Path) -> str:
    frame = pd.read_csv(csv_path)
    if "symbol" not in frame.columns or frame.empty:
        raise ValueError("CSV file must contain a 'symbol' column with at least one row")
    return str(frame["symbol"].iloc[0])


def _ensure_sqlite_parent(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "sqlite":
        return
    path = parsed.path
    if path.startswith("//"):
        candidate = Path(path.lstrip("/"))
    else:
        candidate = Path(path)
    parent = candidate.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        help="CSV containing daily market data (defaults to project fixture unless a provider is used)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Symbol to ingest; defaults to the first symbol present in the CSV",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Optional market data provider to download from (e.g. yahoo); bypasses CSV ingest",
    )
    parser.add_argument(
        "--timescale-url",
        type=str,
        default=f"sqlite:///{(Path('artifacts') / 'timescale_real_slice.db').resolve()}",
        help="SQLAlchemy URL for Timescale/SQLite storage (default writes to artifacts/)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Optional override for the number of days to ingest",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Source label recorded in Timescale (defaults to provider or fixture)",
    )
    parser.add_argument(
        "--belief-id",
        type=str,
        default="cli-real-data-slice",
        help="Belief identifier used when emitting the belief state",
    )
    args = parser.parse_args()

    provider = args.provider.strip() if args.provider else None

    csv_path: Path | None
    if args.csv_path is not None:
        csv_path = args.csv_path.resolve()
    elif provider is None:
        csv_path = Path("tests/data/eurusd_daily_slice.csv").resolve()
    else:
        csv_path = None

    if csv_path is not None and not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if args.symbol:
        symbol = args.symbol.strip()
    elif csv_path is not None:
        symbol = _default_symbol(csv_path)
    else:
        raise ValueError("Symbol must be provided when ingesting from a provider without a CSV")
    if not symbol:
        raise ValueError("Symbol must not be empty")

    _ensure_sqlite_parent(args.timescale_url)
    settings = TimescaleConnectionSettings(
        url=args.timescale_url,
        application_name="real-data-slice-cli",
    )

    source = (args.source.strip() if args.source else "") or provider or "fixture"

    config = RealDataSliceConfig(
        symbol=symbol,
        csv_path=csv_path,
        provider=provider,
        source=source,
        lookback_days=args.lookback_days,
        belief_id=args.belief_id,
    )

    outcome = run_real_data_slice(config=config, settings=settings)

    snapshot = outcome.sensory_snapshot
    integrated = snapshot["integrated_signal"]
    belief_state = outcome.belief_state

    print("Timescale URL:", settings.url)
    print("Rows ingested:", outcome.ingest_result.rows_written)
    print("Market frame rows:", len(outcome.market_data))
    print(
        "Sensory snapshot:",
        f"symbol={snapshot['symbol']} generated_at={snapshot['generated_at']}",
    )
    print(
        "Integrated signal:",
        f"strength={integrated.strength:.4f} confidence={integrated.confidence:.4f}",
    )
    print(
        "Belief posterior:",
        f"strength={belief_state.posterior.strength:.4f} confidence={belief_state.posterior.confidence:.4f}",
    )
    print("Features:", ", ".join(belief_state.features))


if __name__ == "__main__":
    main()
