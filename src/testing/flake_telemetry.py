from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
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
                "longrepr": longrepr,
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
    explicit: str | None,
    ini_value: str | None,
    env_value: str | None,
) -> Path:
    """Resolve the telemetry output path to an absolute location and ensure parent exists."""

    candidate = explicit or env_value or ini_value or str(DEFAULT_RELATIVE_PATH)
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
    return outcome.lower() in {"failed", "error"}


def clip_longrepr(
    text: str | None,
    limit: int | None = MAX_LONGREPR_LENGTH,
) -> str | None:
    """Trim long representations so the telemetry payload stays small."""

    if text is None:
        return None

    normalized = text.rstrip("\n")
    if limit is None:
        return normalized
    if limit < 0:
        limit = 0
    if len(normalized) <= limit:
        return normalized

    truncated = normalized[:limit]
    omitted = len(normalized) - limit
    return f"{truncated}… [truncated {omitted} chars]"


__all__ = [
    "FlakeTelemetrySink",
    "clip_longrepr",
    "resolve_output_path",
    "should_record_event",
]
