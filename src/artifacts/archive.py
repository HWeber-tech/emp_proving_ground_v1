"""Helpers for mirroring artifacts into dated evidence directories."""

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for older runtimes
    UTC = timezone.utc  # type: ignore[assignment]


_KIND_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9_.-]+")
_RUN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9_.-]+")


def _default_root() -> Path:
    override = os.getenv("ALPHATRADE_ARTIFACT_ROOT")
    if override:
        return Path(override).expanduser()
    return Path("artifacts")


def _normalise_kind(kind: str) -> str:
    normalised = _KIND_PATTERN.sub("-", kind.strip().lower())
    normalised = re.sub("-+", "-", normalised).strip("-.")
    return normalised or "misc"


def _normalise_run_id(run_id: str | None, timestamp: datetime) -> str:
    if not run_id:
        return timestamp.strftime("run-%H%M%S")
    cleaned = _RUN_PATTERN.sub("-", run_id.strip())
    cleaned = re.sub("-+", "-", cleaned)
    cleaned = cleaned.strip("-._")
    return cleaned or timestamp.strftime("run-%H%M%S")


def archive_artifact(
    kind: str,
    source: str | Path,
    *,
    root: str | Path | None = None,
    timestamp: datetime | None = None,
    run_id: str | None = None,
    target_name: str | None = None,
) -> Path | None:
    """Copy ``source`` into a dated ``artifacts/<kind>/`` folder.

    Returns the destination path when the copy succeeds. Missing sources are
    treated as a no-op so producers can call this helper opportunistically.
    """

    src_path = Path(source)
    if not src_path.exists():
        logger.debug("archive_artifact skipped missing source: %s", src_path)
        return None

    ts = (timestamp or datetime.now(tz=UTC)).astimezone(UTC)
    root_path = Path(root).expanduser() if root else _default_root().expanduser()
    kind_segment = _normalise_kind(kind)

    dest_dir = (
        root_path
        / kind_segment
        / f"{ts.year:04d}"
        / f"{ts.month:02d}"
        / f"{ts.day:02d}"
        / _normalise_run_id(run_id, ts)
    )

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - filesystem edge cases
        logger.exception("Failed to prepare artifact directory: %s", dest_dir)
        return None

    dest_name = Path(target_name).name if target_name else src_path.name
    if not dest_name:
        dest_name = src_path.name
    destination = dest_dir / dest_name
    try:
        if src_path.resolve() == destination.resolve():
            logger.debug(
                "archive_artifact skipped copy for identical path",
                extra={"source": str(src_path), "destination": str(destination)},
            )
            return destination
    except (OSError, RuntimeError):
        pass
    try:
        shutil.copy2(src_path, destination)
    except Exception:  # pragma: no cover - filesystem edge cases
        logger.exception("Failed to archive artifact %s -> %s", src_path, destination)
        return None

    logger.debug("Archived artifact", extra={"source": str(src_path), "destination": str(destination)})
    return destination


__all__ = ["archive_artifact"]
