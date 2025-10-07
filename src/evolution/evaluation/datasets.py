"""Utilities for persisting and rehydrating recorded sensory snapshots.

The modernization roadmap calls for proving adaptive strategies against
recorded sensory data.  The replay evaluator and adversarial selector already
consume :class:`RecordedSensorySnapshot` instances, but existing tooling lacked a
compact way to capture live sensory observations and feed them back into those
pipelines.  This module provides JSONL helpers that serialise canonical sensory
snapshots (including lineage metadata) and load them back into the replay
engine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.evolution.evaluation.recorded_replay import RecordedSensorySnapshot
from src.sensory.lineage import SensorLineageRecord
from src.sensory.monitoring.sensor_drift import SensorDriftSummary
from src.sensory.signals import IntegratedSignal

__all__ = [
    "dump_recorded_snapshots",
    "load_recorded_snapshots",
]

logger = logging.getLogger(__name__)


def _ensure_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        raise IsADirectoryError(f"expected file path, received directory: {candidate!s}")
    return candidate


def _as_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _serialise_value(value):  # type: ignore[no-untyped-def]
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return _as_utc_timestamp(value)
    if isinstance(value, SensorLineageRecord):
        return value.as_dict()
    if isinstance(value, IntegratedSignal):
        payload: dict[str, object] = {
            "direction": value.direction,
            "strength": value.strength,
            "confidence": value.confidence,
            "contributing": list(value.contributing),
        }
        try:
            payload["timestamp"] = _as_utc_timestamp(value.timestamp)
        except Exception:
            pass
        return payload
    if isinstance(value, SensorDriftSummary):
        return value.as_dict()
    if is_dataclass(value):
        try:
            return _serialise_value(asdict(value))
        except Exception:
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # pandas.Timestamp compatibility
        except Exception:
            pass
    if isinstance(value, Mapping):
        serialised: MutableMapping[str, object] = {}
        for key, item in value.items():
            serialised[str(key)] = _serialise_value(item)
        return dict(serialised)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_serialise_value(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _serialise_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _serialise_value(val) for key, val in snapshot.items()}


def dump_recorded_snapshots(
    snapshots: Iterable[Mapping[str, object]],
    path: str | Path,
    *,
    append: bool = False,
) -> int:
    """Serialise sensory snapshots to a JSONL file.

    Args:
        snapshots: Iterable of raw sensory snapshots as returned by
            :meth:`RealSensoryOrgan.observe`.
        path: Destination JSONL file.
        append: When ``True`` appends to ``path`` instead of overwriting.

    Returns:
        Number of snapshots persisted.
    """

    destination = _ensure_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with destination.open(mode, encoding="utf-8") as fh:
        for snapshot in snapshots:
            try:
                payload = _serialise_snapshot(snapshot)
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.warning("Failed to serialise sensory snapshot: %s", exc, exc_info=exc)
                continue
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_recorded_snapshots(
    path: str | Path,
    *,
    limit: int | None = None,
    strict: bool = False,
) -> list[RecordedSensorySnapshot]:
    """Load recorded sensory snapshots from a JSONL file.

    Invalid JSON lines or payloads that fail snapshot coercion are skipped unless
    ``strict`` is ``True``.
    """

    source = _ensure_path(path)
    if not source.exists():
        return []
    snapshots: list[RecordedSensorySnapshot] = []
    with source.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if limit is not None and len(snapshots) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                message = f"Invalid JSON on line {line_number}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                logger.warning(message)
                continue
            try:
                snapshot = RecordedSensorySnapshot.from_snapshot(payload)
            except Exception as exc:
                message = f"Failed to coerce snapshot on line {line_number}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                logger.warning(message)
                continue
            snapshots.append(snapshot)
    return snapshots
