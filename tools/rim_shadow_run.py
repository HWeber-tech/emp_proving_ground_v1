#!/usr/bin/env python3
"""Shadow-mode runner for the Reflection Intelligence Module (RIM).

Reads Decision Diary JSONL files, aggregates heuristics, and emits placeholder
suggestions that comply with interfaces/rim_types.json. No trading logic lives
here; this is scaffolding for documentation and integration.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

CONFIG_PATH = Path("config/reflection/rim.config.yml")
DEFAULT_CONFIG = Path("config/reflection/rim.config.example.yml")
DIARIES_DIR = Path("artifacts/diaries")
SUGGESTIONS_DIR = Path("artifacts/rim_suggestions")
LOG_DIR_DEFAULT = Path("artifacts/rim_logs")
MODEL_HASH = "stub-trm-v0"
SCHEMA_VERSION = "rim.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RIM shadow runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH if CONFIG_PATH.exists() else DEFAULT_CONFIG,
        help="Path to RIM config (YAML).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Emit additional diagnostics to stdout.",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Tuple[Dict[str, Any], Path]:
    resolved = path if path.exists() else DEFAULT_CONFIG
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = resolved.read_text()
    return _parse_simple_yaml(text), resolved


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse a minimal subset of YAML (key: value with indentation)."""

    def coerce(value: str) -> Any:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(0, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, rest = raw_line.strip().partition(":")
        if not key:
            continue
        value = rest.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value:
            node: Dict[str, Any] = {}
            parent[key] = node
            stack.append((indent + 2, node))
        else:
            parent[key] = coerce(value)
    return root


def latest_diary_file(diaries_dir: Path) -> Path | None:
    if not diaries_dir.exists():
        return None
    candidates = sorted(
        (p for p in diaries_dir.iterdir() if p.is_file() and p.suffix == ".jsonl"),
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_diary_entries(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def aggregate_by_strategy(entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        strategy = entry.get("strategy_id", "unknown")
        pnl = float(entry.get("pnl", 0.0))
        bucket = buckets.setdefault(strategy, {"count": 0, "pnl": 0.0, "risk_flags": 0})
        bucket["count"] += 1
        bucket["pnl"] += pnl
        bucket["risk_flags"] += len(entry.get("risk_flags", []))
    return buckets


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_suggestions(
    aggregates: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    input_hash: str,
    config_hash: str,
    run_timestamp: dt.datetime,
) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    confidence_floor = float(config.get("confidence_floor", 0.5))
    suggestion_cap = int(config.get("suggestion_cap", 10))
    minutes = int(config.get("window_minutes", 1440))

    for idx, (strategy, stats) in enumerate(sorted(aggregates.items(), key=lambda kv: kv[1]["pnl"])):
        if len(suggestions) >= suggestion_cap:
            break
        if stats["count"] == 0:
            continue
        avg_pnl = stats["pnl"] / max(stats["count"], 1)
        risk_pressure = stats["risk_flags"] / max(stats["count"], 1)
        confidence = min(0.95, max(confidence_floor, 0.5 + min(0.2, abs(avg_pnl) / 10000.0) + min(0.15, risk_pressure * 0.05)))
        suggestion_type = "WEIGHT_ADJUST" if avg_pnl < 0 else "EXPERIMENT_PROPOSAL"
        rationale = (
            f"Average pnl {avg_pnl:.2f} with {risk_pressure:.2f} risk flags per trade across {stats['count']} entries"
        )
        payload: Dict[str, Any]
        if suggestion_type == "WEIGHT_ADJUST":
            payload = {
                "strategy_id": strategy,
                "proposed_weight_delta": round(max(-0.1, min(-0.01, avg_pnl / 100000.0)), 4),
                "window_minutes": minutes,
            }
        else:
            payload = {
                "hypothesis": f"Investigate positive pnl regime for {strategy}",
                "strategy_candidates": [strategy],
                "duration_minutes": minutes // 2,
            }
        suggestion = {
            "schema_version": SCHEMA_VERSION,
            "input_hash": input_hash,
            "model_hash": MODEL_HASH,
            "config_hash": config_hash,
            "suggestion_id": f"rim-{run_timestamp:%Y%m%d}-{idx:04d}",
            "type": suggestion_type,
            "payload": payload,
            "confidence": round(confidence, 2),
            "rationale": rationale,
            "audit_ids": [f"diary-{run_timestamp:%Y%m%d}-{idx:04d}"],
            "created_at": run_timestamp.replace(microsecond=0).isoformat() + "Z",
        }
        if suggestion["confidence"] >= confidence_floor:
            suggestions.append(suggestion)
    return suggestions


def write_suggestions(suggestions: List[Dict[str, Any]], publish_channel: str, timestamp: dt.datetime) -> Path | None:
    if not suggestions:
        return None
    if publish_channel.startswith("file://"):
        target_dir = Path(publish_channel[len("file://") :])
    else:
        target_dir = SUGGESTIONS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"rim-suggestions-UTC-{timestamp.strftime('%Y%m%dT%H%M%S')}.jsonl"
    with output_path.open("w") as fh:
        for suggestion in suggestions:
            fh.write(json.dumps(suggestion) + "\n")
    return output_path


def log_metrics(log_dir: Path, runtime_s: float, suggestions_count: int, timestamp: dt.datetime) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    runtime_ms = runtime_s * 1000.0
    percentiles = [runtime_ms]  # Single measurement for placeholder runner
    p50 = statistics.median(percentiles)
    p95 = max(percentiles)
    log_path = log_dir / f"rim-{timestamp:%Y%m%d}.log"
    line = (
        f"{timestamp.isoformat()}Z runtime_ms={runtime_ms:.2f} p50_ms={p50:.2f} p95_ms={p95:.2f} "
        f"suggestions={suggestions_count} acceptance_rate=0.0"
    )
    with log_path.open("a") as fh:
        fh.write(line + "\n")


def main() -> int:
    args = parse_args()
    try:
        config, resolved_config_path = load_yaml_config(args.config)
    except FileNotFoundError as exc:
        print(f"[RIM] {exc}", file=sys.stderr)
        return 1

    if config.get("kill_switch") is True:
        if args.debug or os.getenv("RIM_DEBUG"):
            print("[RIM] Kill switch enabled; exiting without suggestions.")
        return 0

    publish_channel = str(config.get("publish_channel", "file://artifacts/rim_suggestions"))
    log_dir = Path(config.get("telemetry", {}).get("log_dir", LOG_DIR_DEFAULT))

    start = time.perf_counter()
    run_timestamp = dt.datetime.utcnow()

    diary_path = latest_diary_file(DIARIES_DIR)
    if not diary_path:
        if args.debug or os.getenv("RIM_DEBUG"):
            print(f"[RIM] No diary files found in {DIARIES_DIR}; emitting empty suggestion set.")
        suggestions: List[Dict[str, Any]] = []
        input_hash = compute_hash("no-diaries")
    else:
        entries = load_diary_entries(diary_path)
        if args.debug or os.getenv("RIM_DEBUG"):
            print(f"[RIM] Loaded {len(entries)} entries from {diary_path}")
        aggregates = aggregate_by_strategy(entries)
        input_hash = compute_hash(json.dumps(aggregates, sort_keys=True))
        config_hash = compute_hash(resolved_config_path.read_text())
        suggestions = build_suggestions(aggregates, config, input_hash, config_hash, run_timestamp)
        if args.debug or os.getenv("RIM_DEBUG"):
            print(f"[RIM] Generated {len(suggestions)} suggestions")
        output_path = write_suggestions(suggestions, publish_channel, run_timestamp)
        if output_path and (args.debug or os.getenv("RIM_DEBUG")):
            print(f"[RIM] Wrote suggestions to {output_path}")

    runtime_s = time.perf_counter() - start
    config_hash = compute_hash(resolved_config_path.read_text())
    log_metrics(log_dir, runtime_s, len(suggestions), run_timestamp)

    if not suggestions:
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
