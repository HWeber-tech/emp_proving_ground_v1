"""Lineage helpers for sensory telemetry.

The roadmap calls for executable sensory organs that surface lineage telemetry
alongside their primary readings so downstream systems can trace how signals
were produced.  This module provides a small, serialization-safe container that
the HOW/ANOMALY sensors can embed inside their metadata payloads.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from threading import Lock
from typing import Any, MutableMapping

__all__ = [
    "SensorLineageRecord",
    "SensorLineageRecorder",
    "build_lineage_record",
]


def _coerce_value(value: Any) -> Any:
    """Convert *value* into a JSON-serialisable primitive."""

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _coerce_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_value(item) for item in value]

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _sanitise_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    cleaned: MutableMapping[str, Any] = {}
    if not mapping:
        return {}
    for key, value in mapping.items():
        try:
            cleaned[str(key)] = _coerce_value(value)
        except Exception:
            cleaned[str(key)] = "<unserialisable>"
    return dict(cleaned)


@dataclass(slots=True, frozen=True)
class SensorLineageRecord:
    """Immutable snapshot describing how a sensory signal was produced."""

    dimension: str
    source: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "source": self.source,
            "generated_at": self.generated_at.isoformat(),
            "inputs": _sanitise_mapping(self.inputs),
            "outputs": _sanitise_mapping(self.outputs),
            "telemetry": _sanitise_mapping(self.telemetry),
            "metadata": _sanitise_mapping(self.metadata),
        }


def build_lineage_record(
    dimension: str,
    source: str,
    *,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    telemetry: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SensorLineageRecord:
    """Create a :class:`SensorLineageRecord` with sanitised payloads."""

    return SensorLineageRecord(
        dimension=dimension,
        source=source,
        inputs=dict(inputs or {}),
        outputs=dict(outputs or {}),
        telemetry=dict(telemetry or {}),
        metadata=dict(metadata or {}),
    )


class SensorLineageRecorder:
    """Bounded recorder that preserves recent sensory lineage payloads."""

    def __init__(self, *, max_records: int = 256) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self._records: deque[SensorLineageRecord] = deque(maxlen=max_records)
        self._lock = Lock()

    def record(self, record: SensorLineageRecord) -> None:
        """Store a lineage record, keeping only the most recent entries."""

        if not isinstance(record, SensorLineageRecord):  # defensive guard for callers
            raise TypeError("record must be a SensorLineageRecord instance")
        with self._lock:
            self._records.append(record)

    def history(
        self,
        limit: int | None = None,
        *,
        serialise: bool = True,
    ) -> list[dict[str, Any]] | list[SensorLineageRecord]:
        """Return recorded lineage records, newest first.

        When ``serialise`` is ``True`` (the default), the payloads are returned as
        JSON-safe dictionaries via :meth:`SensorLineageRecord.as_dict`.
        """

        with self._lock:
            items = list(self._records)

        if limit is not None:
            if limit <= 0:
                return []
            items = items[-limit:]

        items = list(reversed(items))

        if not serialise:
            return items

        return [record.as_dict() for record in items]

    def latest(self) -> dict[str, Any] | None:
        """Return the most recent lineage payload, if available."""

        with self._lock:
            if not self._records:
                return None
            record = self._records[-1]
        return record.as_dict()

    def clear(self) -> None:
        """Remove all stored lineage records."""

        with self._lock:
            self._records.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._records)
