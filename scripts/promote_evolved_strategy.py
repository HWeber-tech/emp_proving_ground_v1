#!/usr/bin/env python3
"""Promote GA champions into the strategy registry when feature flags allow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.governance.promotion import (
    PromotionFeatureFlags,
    PromotionResult,
    promote_manifest_to_registry,
)
from src.governance.strategy_registry import StrategyRegistry, StrategyStatus


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the GA manifest JSON file produced by Evolution Lab runs.",
    )
    parser.add_argument(
        "--registry-db",
        type=Path,
        default=Path("artifacts/governance/strategy_registry.db"),
        help="Location of the SQLite registry database.",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Force-enable promotion regardless of environment feature flags.",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Automatically approve the promoted strategy (overrides feature flag).",
    )
    parser.add_argument(
        "--target-status",
        type=str,
        default=None,
        help="Target status when approving (evolved, approved, active, inactive).",
    )
    parser.add_argument(
        "--min-fitness",
        type=float,
        default=None,
        help="Override the minimum fitness threshold for promotion.",
    )
    parser.add_argument(
        "--gate-results",
        type=Path,
        default=None,
        help=(
            "Path to ablation gate results JSON; when provided, promotion will "
            "skip or roll back if any gates fail."
        ),
    )
    return parser.parse_args()


def _resolve_flags(args: argparse.Namespace) -> PromotionFeatureFlags:
    flags = PromotionFeatureFlags.from_env()
    overrides: dict[str, object] = {}
    if args.enable:
        overrides["register_enabled"] = True
    if args.approve:
        overrides["auto_approve"] = True
        overrides.setdefault("target_status", StrategyStatus.APPROVED)
    if args.target_status:
        overrides["target_status"] = _parse_status(args.target_status, flags.target_status)
    if args.min_fitness is not None:
        overrides["min_fitness"] = float(args.min_fitness)
    return flags.with_overrides(**overrides) if overrides else flags


def _parse_status(raw: str, fallback: StrategyStatus) -> StrategyStatus:
    normalised = raw.strip().lower()
    for status in StrategyStatus:
        if status.value == normalised or status.name.lower() == normalised:
            return status
    return fallback


def _ensure_registry(path: Path) -> StrategyRegistry:
    path.parent.mkdir(parents=True, exist_ok=True)
    return StrategyRegistry(str(path))


def _print_result(result: PromotionResult) -> None:
    if result.skipped and result.reason == "feature_flag_disabled":
        print("⚠️ Promotion skipped: feature flag disabled.")
        return
    if result.skipped:
        reason = result.reason or "unknown"
        print(f"⚠️ Promotion skipped for {result.genome_id or 'n/a'} ({reason}).")
        return
    status = "updated" if result.status_updated else "recorded"
    print(
        f"✅ Registered {result.genome_id} with fitness {result.fitness:.3f} and {status} status."
    )


def main() -> int:
    args = _parse_args()
    flags = _resolve_flags(args)
    registry = _ensure_registry(args.registry_db)
    result = promote_manifest_to_registry(
        args.manifest,
        registry,
        flags=flags,
        gate_results_path=args.gate_results,
    )
    _print_result(result)
    return 0 if result.registered or result.skipped else 1


if __name__ == "__main__":
    sys.exit(main())
