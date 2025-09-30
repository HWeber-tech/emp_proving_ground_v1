"""Hydrate canonical pricing datasets for sensors, strategies, and tests.

This CLI delivers the roadmap requirement for a repeatable "data bootstrap"
step that seeds local development and CI environments with normalised OHLCV
fixtures.  It wraps :class:`src.data_foundation.pipelines.pricing_pipeline.PricingPipeline`
so all vendor implementations share a single validation and formatting path.

Key features:
* Supports live vendor fetches (default Yahoo) as well as deterministic file
  based sources for offline runs.
* Persists canonical Parquet/CSV artefacts via ``PricingCache`` with
  retention-aware pruning so local and CI environments remain tidy.
* Surfaces data-quality findings returned by the pricing pipeline and can
  optionally fail the run when critical issues are detected.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.data_foundation.pipelines.pricing_pipeline import (
    PricingPipeline,
    PricingPipelineConfig,
    PricingPipelineResult,
    PricingQualityIssue,
    PricingVendor,
)
from src.data_foundation.cache.pricing_cache import PricingCache


class _FilePricingVendor:
    """Minimal vendor adapter that reads pre-downloaded datasets from disk."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def fetch(self, config: PricingPipelineConfig) -> pd.DataFrame:  # noqa: D401 - protocol match
        if not self._path.exists():
            raise FileNotFoundError(f"Source dataset not found: {self._path}")

        suffix = self._path.suffix.lower()
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(self._path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(self._path)
        if suffix in {".json", ".jsonl"}:
            lines = [json.loads(line) for line in self._path.read_text().splitlines() if line.strip()]
            return pd.DataFrame.from_records(lines)

        raise ValueError(
            "Unsupported file extension for pricing vendor: "
            f"{self._path.suffix}. Expected csv, parquet, or json"
        )


def _parse_symbols(symbols: str | None) -> list[str]:
    if not symbols:
        return []
    return [sym.strip() for sym in symbols.split(",") if sym.strip()]


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive for CLI misuse
        raise argparse.ArgumentTypeError(f"Invalid datetime format: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _issues_payload(issues: Iterable[PricingQualityIssue]) -> list[Mapping[str, object]]:
    payload: list[Mapping[str, object]] = []
    for issue in issues:
        record = {
            "code": issue.code,
            "severity": issue.severity,
            "message": issue.message,
            "symbol": issue.symbol,
            "context": dict(issue.context),
        }
        payload.append(record)
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _slug_component(value: str) -> str:
    """Return a filesystem-friendly representation of ``value``."""

    cleaned = value.strip().lower()
    safe_chars = [char if char.isalnum() else "_" for char in cleaned]
    return "".join(safe_chars) or "value"


def _ensure_source_column(frame: pd.DataFrame, vendor: str) -> pd.DataFrame:
    """Return a copy of ``frame`` with a populated ``source`` column."""

    if "source" in frame.columns:
        enriched = frame.copy()
        enriched["source"] = enriched["source"].fillna(vendor)
        return enriched
    enriched = frame.copy()
    enriched["source"] = vendor
    return enriched


def _write_dataset(frame: pd.DataFrame, path: Path, fmt: str) -> None:
    """Persist ``frame`` using the requested file ``fmt``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        frame.to_csv(path, index=False)
    elif fmt == "parquet":
        frame.to_parquet(path, index=False)
    else:  # pragma: no cover - parser guards against other formats
        raise ValueError(f"Unsupported dataset format: {fmt}")


def _run_pipeline(args: argparse.Namespace) -> tuple[PricingPipelineConfig, PricingPipelineResult]:
    symbols = _parse_symbols(args.symbols)
    config = PricingPipelineConfig(
        symbols=symbols,
        vendor=args.vendor,
        interval=args.interval,
        lookback_days=args.lookback_days,
        start=_parse_datetime(args.start),
        end=_parse_datetime(args.end),
        minimum_coverage_ratio=args.minimum_coverage,
    )

    vendor_registry: dict[str, PricingVendor] = {}
    if args.source_path is not None:
        vendor_registry[args.vendor] = _FilePricingVendor(Path(args.source_path))

    pipeline = PricingPipeline(vendor_registry=vendor_registry)
    return config, pipeline.run(config)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        type=str,
        default="EURUSD=X,GBPUSD=X,USDJPY=X",
        help="Comma-separated list of symbols to fetch",
    )
    parser.add_argument(
        "--vendor",
        type=str,
        default="yahoo",
        help="Registered pricing vendor (default: yahoo)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Bar interval requested from the vendor",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Historical window (in days) to request when start/end not provided",
    )
    parser.add_argument("--start", type=str, default=None, help="ISO timestamp for start window")
    parser.add_argument("--end", type=str, default=None, help="ISO timestamp for end window")
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.6,
        help="Minimum ratio of expected candles required before surfacing a warning",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_foundation/cache/pricing"),
        help="Directory where cache artefacts will be written",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help="Optional pre-downloaded dataset used instead of fetching from a vendor",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional explicit path for metadata JSON (defaults to <output>/pricing_metadata.json)",
    )
    parser.add_argument(
        "--issues-path",
        type=Path,
        default=None,
        help="Optional explicit path for issues JSON (defaults to <output>/pricing_issues.json)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=("parquet", "csv"),
        help="Dataset format written to the output directory (default: parquet)",
    )
    parser.add_argument(
        "--cache-retention-days",
        type=int,
        default=14,
        help="Number of days of cached datasets to keep before pruning",
    )
    parser.add_argument(
        "--cache-max-entries",
        type=int,
        default=None,
        help="Optional hard cap on the number of cached datasets to retain",
    )
    parser.add_argument(
        "--fail-on-quality",
        action="store_true",
        help="Exit with status 2 if the pipeline reports error-severity issues",
    )

    args = parser.parse_args(argv)

    try:
        config, result = _run_pipeline(args)
    except Exception as exc:  # pragma: no cover - CLI level guard
        print(f"[ERROR] Failed to run pricing pipeline: {exc}", file=sys.stderr)
        return 1

    cache = PricingCache(args.output)
    entry = cache.store(
        config,
        result,
        retention_days=args.cache_retention_days,
        max_entries=args.cache_max_entries,
    )

    issues_payload = list(entry.issues_payload)

    dataset_format = args.format.lower()
    friendly_dataset = args.output / (
        f"pricing_{_slug_component(args.vendor)}_{_slug_component(args.interval)}.{dataset_format}"
    )

    enriched_frame = _ensure_source_column(result.data, args.vendor)
    _write_dataset(enriched_frame, friendly_dataset, dataset_format)

    default_metadata_path = args.metadata_path or args.output / "pricing_metadata.json"
    default_issues_path = args.issues_path or args.output / "pricing_issues.json"

    summary_metadata = {
        "dataset_path": str(friendly_dataset),
        "row_count": int(len(enriched_frame)),
        "symbols": config.normalised_symbols(),
        "vendor": args.vendor,
        "interval": args.interval,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    _write_json(default_metadata_path, summary_metadata)
    _write_json(default_issues_path, {"issues": issues_payload})

    quality_errors = [issue for issue in issues_payload if issue["severity"].lower() == "error"]

    summary = {
        "dataset": str(friendly_dataset),
        "rows": summary_metadata["row_count"],
        "symbols": summary_metadata["symbols"],
        "issues": issues_payload,
    }
    print(json.dumps(summary, indent=2))

    if quality_errors and args.fail_on_quality:
        print(
            f"[ERROR] {len(quality_errors)} quality issues reported; failing per --fail-on-quality",
            file=sys.stderr,
        )
        return 2

    return 0


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":  # pragma: no cover - exercised via CLI execution
    raise SystemExit(main())
