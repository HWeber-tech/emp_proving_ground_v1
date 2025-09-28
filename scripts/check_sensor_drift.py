"""CLI for evaluating sensory drift using the monitoring harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.sensory.monitoring import evaluate_sensor_drift


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file {path} does not exist")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - dependency optional
            raise RuntimeError(f"Failed to read parquet file {path}: {exc}")
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path)
    raise ValueError(f"Unsupported file extension: {suffix}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input data file (csv, parquet, json)")
    parser.add_argument(
        "--sensors",
        help="Comma-separated list of sensor columns to analyse (defaults to numeric columns)",
    )
    parser.add_argument("--baseline", type=int, default=240, help="Baseline window size")
    parser.add_argument("--evaluation", type=int, default=60, help="Evaluation window size")
    parser.add_argument(
        "--min-observations",
        type=int,
        default=20,
        help="Minimum observations required per window",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Absolute z-score threshold before flagging drift",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON summary",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with non-zero status if any sensor breaches the threshold",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = _read_frame(args.input)
    sensor_columns = None
    if args.sensors:
        sensor_columns = [column.strip() for column in args.sensors.split(",") if column.strip()]

    summary = evaluate_sensor_drift(
        frame,
        sensor_columns=sensor_columns,
        baseline_window=args.baseline,
        evaluation_window=args.evaluation,
        min_observations=args.min_observations,
        z_threshold=args.z_threshold,
    )

    print("Sensor drift summary:")
    for result in summary.results:
        z_repr = f"{result.z_score:.2f}" if result.z_score is not None else "n/a"
        status = "DRIFT" if result.exceeded else "OK"
        print(
            f"  {result.sensor}: mean={result.evaluation_mean:.4f} "
            f"baseline={result.baseline.mean:.4f} z={z_repr} status={status}"
        )

    if args.output:
        payload = summary.as_dict()
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote summary to {args.output}")

    if summary.exceeded and args.fail_on_drift:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
