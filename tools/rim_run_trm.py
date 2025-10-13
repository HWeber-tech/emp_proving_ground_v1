#!/usr/bin/env python3
"""Production runner CLI for the Reflection Intelligence Module (TRM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.reflection.trm import TRMRunner, load_runtime_config
from src.reflection.trm.model import TRMModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the production TRM pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to rim.config.yml (falls back to example if omitted)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional override for model weights JSON",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Emit runtime diagnostics to stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        bundle = load_runtime_config(args.config)
    except (FileNotFoundError, TypeError) as exc:
        print(f"[TRM] Failed to load config: {exc}", file=sys.stderr)
        return 1

    config = bundle.config
    if args.model is not None:
        config.model.path = args.model

    try:
        model = TRMModel.load(config.model.path, temperature=config.model.temperature)
    except (OSError, ValueError) as exc:
        print(f"[TRM] Failed to load model weights: {exc}", file=sys.stderr)
        return 1

    runner = TRMRunner(config, model, config_hash=bundle.config_hash)
    result = runner.run()

    if args.debug:
        if result.skipped_reason:
            print(
                f"[TRM] Skipped execution reason={result.skipped_reason} runtime={result.runtime_seconds:.3f}s",
                file=sys.stdout,
            )
        else:
            print(
                f"[TRM] Emitted {result.suggestions_count} suggestions in {result.runtime_seconds:.3f}s"
                f" -> {result.suggestions_path}",
                file=sys.stdout,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
