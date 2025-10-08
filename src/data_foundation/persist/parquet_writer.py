"""Helpers for writing ingest events into partitioned Parquet datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Exposed for tests so pandas can be swapped with a lightweight stub.
_pd = pd

_DEFAULT_FILENAME = "events.parquet"


def _coerce_event_rows(events: Iterable[Mapping[str, object]] | None) -> list[dict[str, object]]:
    """Normalise raw ingest events into dictionaries."""

    if events is None:
        return []
    if isinstance(events, Sequence):
        return [dict(event) for event in events]
    return [dict(event) for event in list(events)]


def _resolve_partition_directory(out_dir: Path | str, partition: str | None) -> Path:
    """Create and return the directory that will host the Parquet file."""

    target = Path(out_dir)
    partition_value = (partition or "").strip()
    if partition_value:
        target = target / f"partition={partition_value}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_events_parquet(
    *,
    events: Iterable[Mapping[str, object]] | None,
    out_dir: Path | str,
    partition: str | None = None,
    filename: str | None = None,
) -> str:
    """Persist ingest events to a Parquet file and return its path.

    Returns an empty string when the payload cannot be serialised.  Errors are
    logged so CI guardrails can flag regressions without raising hard failures
    during ingest drills.
    """

    rows = _coerce_event_rows(events)
    if not rows:
        return ""
    row_count = len(rows)

    try:
        frame = _pd.DataFrame(rows)
    except Exception:  # pragma: no cover - defensive logging hook
        logger.exception("Failed to convert ingest events into a pandas DataFrame")
        return ""

    destination_dir = _resolve_partition_directory(out_dir, partition)
    candidate = (filename or _DEFAULT_FILENAME).strip() or _DEFAULT_FILENAME
    destination = destination_dir / candidate

    try:
        frame.to_parquet(str(destination), index=False)
    except Exception:  # pragma: no cover - defensive logging hook
        logger.exception(
            "Failed to persist ingest events to Parquet",
            extra={
                "rows": row_count,
                "partition": partition,
                "destination": str(destination),
            },
        )
        return ""

    return str(destination)


__all__ = ["write_events_parquet"]
