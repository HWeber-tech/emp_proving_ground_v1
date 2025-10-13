#!/usr/bin/env python3
"""Shadow-mode runner for the Reflection Intelligence Module (RIM).

Executes the TRM pipeline against the latest decision diaries, emitting nightly
suggestions plus governance digests while leaving live promotion gates untouched.
Legacy helpers remain for operations drills still wired to the lightweight
scaffolding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.reflection.trm import TRMRunner, load_runtime_config
from src.reflection.trm.adapter import RIMInputAdapter
from src.reflection.trm.config import RIMRuntimeConfig
from src.reflection.trm.encoder import RIMEncoder
from src.reflection.trm.model import TRMModel
from src.reflection.trm.runner import TRMRunResult

CONFIG_PATH = Path("config/reflection/rim.config.yml")
DEFAULT_CONFIG = Path("config/reflection/rim.config.example.yml")
DEFAULT_DIARIES_DIR = Path("artifacts/diaries")
DEFAULT_DIARY_GLOB = "diaries-*.jsonl"
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
        "--min-entries",
        type=int,
        default=None,
        help="Override the minimum entry guard for shadow runs (default: config value).",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=None,
        help="Custom lock path for the shadow job (defaults to config lock with '-shadow').",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Emit additional diagnostics to stdout.",
    )
    return parser.parse_args()


def _debug_enabled(flag: bool) -> bool:
    return flag or bool(os.getenv("RIM_DEBUG"))


def _shadow_lock_path(config: RIMRuntimeConfig, override: Path | None) -> Path:
    if override is not None:
        return override
    base = config.lock_path
    suffix = base.suffix
    stem = base.stem or "rim"
    return base.parent / f"{stem}_shadow{suffix}"


def _preview_batch(config: RIMRuntimeConfig) -> None:
    adapter = RIMInputAdapter(config.diaries_dir, config.diary_glob, config.window_minutes)
    batch = adapter.load_batch()
    if batch is None:
        print(
            f"[RIM] No diary files matching '{config.diary_glob}' in {config.diaries_dir};"
            " unable to preview batch.",
            file=sys.stdout,
        )
        return

    entries = batch.entries
    print(
        f"[RIM] Preview loaded {len(entries)} entries spanning {batch.window.minutes} minutes"
        f" across {len({entry.strategy_id for entry in entries})} strategies.",
        file=sys.stdout,
    )
    encoder = RIMEncoder()
    encodings = encoder.encode(entries)
    print(
        f"[RIM] Encoder produced {len(encodings)} strategy vectors; feature_dim="
        f"{len(encodings[0].features) if encodings else 0}.",
        file=sys.stdout,
    )
    if encodings:
        sample = encodings[0]
        sample_features = json.dumps(sample.features, sort_keys=True)
        print(
            f"[RIM] Sample features for {sample.strategy_id}: {sample_features}",
            file=sys.stdout,
        )


def write_skip_governance_artifacts(
    config: RIMRuntimeConfig,
    *,
    run_id: str,
    timestamp: dt.datetime,
    reason: str,
    config_hash: str,
    model_hash: str,
) -> None:
    timestamp_iso = timestamp.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    digest = {
        "run_id": run_id,
        "generated_at": timestamp_iso,
        "suggestion_count": 0,
        "input_hash": None,
        "model_hash": model_hash,
        "config_hash": config_hash,
        "window": None,
        "queue_path": None,
        "artifact_path": None,
        "by_type": {},
        "confidence": {"min": None, "avg": None, "max": None},
        "targets": [],
        "top_suggestions": [],
        "skip_reason": reason,
    }
    digest_path = config.governance_digest_path
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(json.dumps(digest, indent=2, sort_keys=True), encoding="utf-8")

    markdown_lines = [
        "# TRM Governance Reflection Summary",
        "",
        f"- Run ID: {run_id}",
        f"- Generated: {timestamp_iso}",
        "- Suggestions: 0",
        "- Queue: n/a",
        "- Artifact: n/a",
        f"- Status: skipped ({reason})",
        "",
        "No suggestions were generated because the shadow run was skipped.",
    ]
    markdown_path = config.governance_markdown_path
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


def run_shadow_job(
    config_path: Path | None,
    *,
    min_entries: int | None = None,
    lock_path: Path | None = None,
    debug: bool = False,
) -> TRMRunResult:
    resolved_path = config_path if config_path and config_path.exists() else None
    bundle = load_runtime_config(resolved_path)
    config = bundle.config

    config.kill_switch = False
    config.enable_governance_gate = True
    if min_entries is not None:
        config.min_entries = max(1, min_entries)
    else:
        config.min_entries = max(1, config.min_entries)
    config.lock_path = _shadow_lock_path(config, lock_path)

    model = TRMModel.load(config.model.path, temperature=config.model.temperature)

    if debug:
        print(f"[RIM] Using config {bundle.source_path}", file=sys.stdout)
        _preview_batch(config)

    runner = TRMRunner(config, model, config_hash=bundle.config_hash)
    result = runner.run()

    timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    if result.run_id is None:
        result.run_id = f"skip-{timestamp.strftime('%Y%m%dT%H%M%S')}"

    if result.skipped_reason:
        if config.enable_governance_gate:
            write_skip_governance_artifacts(
                config,
                run_id=result.run_id,
                timestamp=timestamp,
                reason=result.skipped_reason,
                config_hash=bundle.config_hash,
                model_hash=model.model_hash,
            )
        if debug:
            print(
                f"[RIM] Shadow run skipped: reason={result.skipped_reason} run_id={result.run_id} "
                f"runtime={result.runtime_seconds:.3f}s",
                file=sys.stdout,
            )
            print(f"[RIM] Governance digest: {config.governance_digest_path}", file=sys.stdout)
        return result

    if debug:
        suggestions_path = result.suggestions_path or "<none>"
        print(
            f"[RIM] Shadow run emitted {result.suggestions_count} suggestions in "
            f"{result.runtime_seconds:.3f}s -> {suggestions_path}",
            file=sys.stdout,
        )
        print(f"[RIM] Governance digest: {config.governance_digest_path}", file=sys.stdout)
        print(f"[RIM] Governance markdown: {config.governance_markdown_path}", file=sys.stdout)
        print(f"[RIM] Governance queue: {config.governance_queue_path}", file=sys.stdout)
    return result


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


def latest_diary_file(diaries_dir: Path, diary_glob: str) -> Path | None:
    if not diaries_dir.exists():
        return None
    candidates = sorted(
        (
            p
            for p in diaries_dir.glob(diary_glob)
            if p.is_file()
        ),
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
    debug = _debug_enabled(args.debug)
    try:
        run_shadow_job(
            args.config,
            min_entries=args.min_entries,
            lock_path=args.lock_path,
            debug=debug,
        )
    except FileNotFoundError as exc:
        print(f"[RIM] {exc}", file=sys.stderr)
        return 1
    except (TypeError, ValueError) as exc:
        print(f"[RIM] Failed to initialise shadow runner: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[RIM] Unexpected failure: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
