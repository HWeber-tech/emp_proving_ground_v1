from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

DEFAULT_RELATIVE_PATH = Path("tests/.telemetry/flake_runs.json")
MAX_LONGREPR_LENGTH = 4000


def _default_metadata() -> dict[str, Any]:
    """Build the default metadata block for telemetry payloads."""

    return {
        "session_start": time.time(),
        "python": sys.version,
        "platform": sys.platform,
        "ci": bool(os.environ.get("CI")),
    }


@dataclass
class FlakeTelemetrySink:
    """Accumulates flake telemetry and persists it as JSON."""

    output_path: Path
    metadata: dict[str, Any] = field(default_factory=_default_metadata)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_path = self.output_path.expanduser()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def record_event(
        self,
        *,
        nodeid: str,
        outcome: str,
        duration: float,
        was_xfail: bool,
        longrepr: str | None,
    ) -> None:
        self.events.append(
            {
                "nodeid": nodeid,
                "outcome": outcome,
                "duration": float(duration),
                "was_xfail": was_xfail,
                "longrepr": clip_longrepr(longrepr),
            }
        )

    def flush(self, exit_status: int) -> dict[str, Any]:
        payload = {
            "meta": {
                **self.metadata,
                "session_end": time.time(),
                "exit_status": exit_status,
                "event_count": len(self.events),
            },
            "events": self.events,
        }
        self.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return payload


def resolve_output_path(
    root: Path,
    explicit: str | PathLike[str] | None,
    ini_value: str | PathLike[str] | None,
    env_value: str | PathLike[str] | None,
) -> Path:
    """Resolve the telemetry output path to an absolute location and ensure parent exists."""

    candidate: str | None = None
    for raw_value in (explicit, env_value, ini_value):
        if raw_value is None:
            continue
        try:
            normalized_value = os.fspath(raw_value)
        except TypeError:
            continue
        if isinstance(normalized_value, bytes):
            try:
                normalized_value = normalized_value.decode()
            except UnicodeDecodeError:
                continue
        stripped = normalized_value.strip()
        if stripped:
            candidate = stripped
            break

    if candidate is None:
        candidate = str(DEFAULT_RELATIVE_PATH)

    expanded = os.path.expandvars(candidate)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def should_record_event(outcome: object, was_xfail: bool) -> bool:
    """Return True when an outcome warrants telemetry storage."""

    if was_xfail:
        return True
    if not isinstance(outcome, str):
        return False
    normalized = outcome.strip().lower()
    if not normalized:
        return False
    return normalized in {"failed", "error"}


def clip_longrepr(
    text: str | None,
    limit: int | None = MAX_LONGREPR_LENGTH,
) -> str | None:
    """Trim long representations so the telemetry payload stays small."""

    if text is None:
        return None

    normalized = text.rstrip("\r\n")
    if limit is None:
        return normalized
    if limit < 0:
        limit = 0

    total_length = len(normalized)
    if total_length <= limit:
        return normalized
    if limit <= 0:
        return f"… [truncated {total_length} chars]"

    head_limit = min(limit, total_length)
    while head_limit > 0:
        omitted = total_length - head_limit
        suffix = f"… [truncated {omitted} chars]"
        if head_limit + len(suffix) <= limit:
            truncated = normalized[:head_limit]
            return f"{truncated}{suffix}"
        if limit < len(suffix):
            truncated = normalized[:head_limit]
            return f"{truncated}{suffix}" if truncated else suffix
        head_limit -= 1

    return f"… [truncated {total_length} chars]"


__all__ = [
    "FlakeTelemetrySink",
    "clip_longrepr",
    "resolve_output_path",
    "should_record_event",
]
