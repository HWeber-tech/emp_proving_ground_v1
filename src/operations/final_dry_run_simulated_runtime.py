"""Simulated runtime used by the final dry run smoke harness.

This module emits structured log lines, maintains a rolling decision diary,
and refreshes a lightweight performance report so the dry run harness can be
exercised in short integration tests without connecting to external market
feeds or broker endpoints.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, MutableSequence


@dataclass(slots=True)
class _RuntimeSettings:
    duration: timedelta
    tick_interval: timedelta
    diary_path: Path | None
    performance_path: Path | None
    sharpe_ratio: float


class _ShutdownFlag:
    """Capture termination signals so the loop can exit gracefully."""

    __slots__ = ("_stop",)

    def __init__(self) -> None:
        self._stop = False

    @property
    def should_stop(self) -> bool:
        return self._stop

    def request(self, _signum: int, _frame: Any) -> None:  # pragma: no cover - signal
        self._stop = True


def _parse_args(argv: list[str] | None = None) -> _RuntimeSettings:
    parser = argparse.ArgumentParser(description="Simulated runtime for dry run smoke tests")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=60.0,
        help="How long the simulated runtime should execute (default: 60 seconds).",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=2.0,
        help="Interval (seconds) between heartbeat logs and evidence refreshes.",
    )
    parser.add_argument(
        "--diary",
        type=Path,
        help="Optional path where decision diary entries should be appended.",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        help="Optional path where performance telemetry should be written as JSON.",
    )
    parser.add_argument(
        "--sharpe",
        type=float,
        default=1.2,
        help="Sharpe ratio reported in the performance telemetry (default: 1.2).",
    )
    args = parser.parse_args(argv)

    if args.duration_seconds <= 0:
        raise SystemExit("--duration-seconds must be positive")
    if args.tick_interval <= 0:
        raise SystemExit("--tick-interval must be positive")

    duration = timedelta(seconds=float(args.duration_seconds))
    tick_interval = timedelta(seconds=float(args.tick_interval))
    diary_path = Path(args.diary) if args.diary else None
    performance_path = Path(args.performance) if args.performance else None

    return _RuntimeSettings(
        duration=duration,
        tick_interval=tick_interval,
        diary_path=diary_path,
        performance_path=performance_path,
        sharpe_ratio=float(args.sharpe),
    )


def _update_diary(
    path: Path,
    entries: MutableSequence[dict[str, Any]],
    sequence: int,
    now: datetime,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "entry_id": f"smoke-{sequence:05d}",
        "recorded_at": now.astimezone(UTC).isoformat(),
        "policy_id": "smoke-policy",
        "decision": {
            "tactic_id": "smoke-tactic",
            "rationale": "Simulated heartbeat",
            "parameters": {"sequence": sequence},
        },
        "regime_state": {
            "regime": "smoke",
            "confidence": 0.9,
            "features": {"sequence": sequence},
        },
        "outcomes": {
            "status": "pending",
        },
        "metadata": {
            "source": "final_dry_run_simulated_runtime",
        },
    }
    entries.append(entry)
    payload = {
        "generated_at": now.astimezone(UTC).isoformat(),
        "entries": list(entries),
        "probe_registry": {
            "generated_at": now.astimezone(UTC).isoformat(),
            "probes": {},
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_performance_payload(path: Path, sequence: int, now: datetime, sharpe: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    period_start = now - timedelta(minutes=5)
    payload = {
        "generated_at": now.astimezone(UTC).isoformat(),
        "period_start": period_start.astimezone(UTC).isoformat(),
        "trades": sequence,
        "roi": 0.015 * sequence,
        "win_rate": 0.58,
        "sharpe_ratio": sharpe,
        "window_duration_seconds": 300.0,
        "metadata": {
            "paper_pnl": round(sequence * 12.5, 2),
            "max_drawdown": -0.04,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _log_event(event: str, sequence: int, severity: str, now: datetime) -> None:
    record = {
        "timestamp": now.astimezone(UTC).isoformat(),
        "level": severity,
        "event": event,
        "message": f"{event} #{sequence}",
        "seq": sequence,
    }
    print(json.dumps(record), flush=True)


def main(argv: list[str] | None = None) -> int:
    settings = _parse_args(argv)
    shutdown = _ShutdownFlag()

    for signame in ("SIGTERM", "SIGINT"):
        if hasattr(signal, signame):
            signal.signal(getattr(signal, signame), shutdown.request)

    start = time.monotonic()
    deadline = start + settings.duration.total_seconds()
    sequence = 0

    next_tick = start
    diary_entries: list[dict[str, Any]] = []
    try:
        while not shutdown.should_stop:
            now = time.monotonic()
            if now >= deadline:
                break

            sequence += 1
            emitted_at = datetime.now(tz=UTC)
            _log_event("heartbeat", sequence, "info", emitted_at)
            if settings.diary_path is not None:
                _update_diary(settings.diary_path, diary_entries, sequence, emitted_at)
            if settings.performance_path is not None:
                _write_performance_payload(
                    settings.performance_path,
                    sequence,
                    emitted_at,
                    settings.sharpe_ratio,
                )

            next_tick += settings.tick_interval.total_seconds()
            sleep_for = max(0.0, next_tick - time.monotonic())
            time.sleep(sleep_for)

        # Emit a final summary line so the harness captures completion.
        completion_time = datetime.now(tz=UTC)
        _log_event("shutdown", sequence, "info", completion_time)
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    sys.exit(main())
