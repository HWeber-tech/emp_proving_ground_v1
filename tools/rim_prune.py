#!/usr/bin/env python3
"""Prune RIM suggestion artifacts older than the retention window."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Tuple, Dict, Any

from rim_shadow_run import (
    CONFIG_PATH,
    DEFAULT_CONFIG,
    SUGGESTIONS_DIR,
    load_yaml_config,
)

RETENTION_DAYS_DEFAULT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune aged RIM suggestion artifacts")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH if CONFIG_PATH.exists() else DEFAULT_CONFIG,
        help="Path to the RIM config file.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=RETENTION_DAYS_DEFAULT,
        help="Retention period in days (default: 30).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without removing them.",
    )
    return parser.parse_args()


def resolve_publish_dir(config: Dict[str, Any]) -> Path:
    publish_channel = str(config.get("publish_channel", "file://artifacts/rim_suggestions"))
    if publish_channel.startswith("file://"):
        return Path(publish_channel[len("file://") :])
    return SUGGESTIONS_DIR


def prune(target_dir: Path, cutoff: dt.datetime, dry_run: bool) -> Tuple[int, int]:
    removed = 0
    scanned = 0
    if not target_dir.exists():
        return removed, scanned

    for path in sorted(target_dir.glob("*.jsonl")):
        scanned += 1
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        if mtime < cutoff:
            removed += 1
            if dry_run:
                print(f"[RIM] Would remove {path}")
            else:
                try:
                    path.unlink()
                    if os.getenv("RIM_DEBUG"):
                        print(f"[RIM] Removed {path}")
                except OSError as exc:
                    print(f"[RIM] Failed to remove {path}: {exc}")
    return removed, scanned


def main() -> int:
    args = parse_args()
    try:
        config, _ = load_yaml_config(args.config)
    except FileNotFoundError as exc:
        print(f"[RIM] {exc}")
        return 1

    target_dir = resolve_publish_dir(config)
    cutoff = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=max(args.days, 0))
    removed, scanned = prune(target_dir, cutoff, args.dry_run)

    print(
        f"[RIM] Prune complete: scanned={scanned} removed={removed} "
        f"cutoff={cutoff.isoformat()} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
