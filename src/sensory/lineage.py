"""Lineage helpers for sensory telemetry.

The roadmap calls for executable sensory organs that surface lineage telemetry
alongside their primary readings so downstream systems can trace how signals
were produced.  This module provides a small, serialization-safe container that
the HOW/ANOMALY sensors can embed inside their metadata payloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, MutableMapping

__all__ = ["SensorLineageRecord", "build_lineage_record"]


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

