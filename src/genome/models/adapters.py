from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from .genome import DecisionGenome

logger = logging.getLogger(__name__)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Fetch key from dict-like or attr from object-like."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def from_legacy(obj: Any) -> DecisionGenome:
    """Adapt a legacy dict/object into a canonical DecisionGenome.

    Accepts dict-like or object-like inputs. Coerces/normalizes fields and
    provides safe defaults when missing. On any failure, returns a minimal
    valid genome with a generated ID.
    """
    try:
        # Build a mapping compatible with DecisionGenome.from_dict
        data: dict[str, Any] = {
            "id": _get(obj, "id"),
            "parameters": _get(obj, "parameters", {}) or {},
            "fitness": _get(obj, "fitness", None),
            "generation": _get(obj, "generation", 0),
            "species_type": _get(obj, "species_type"),
            "parent_ids": _get(obj, "parent_ids", []),
            "mutation_history": _get(obj, "mutation_history", []),
            "performance_metrics": _get(obj, "performance_metrics", {}) or {},
            "created_at": _get(obj, "created_at", time.time()),
            "metadata": _get(obj, "metadata", {}) or {},
        }
        # Pass through canonical coercion/validation
        return DecisionGenome.from_dict(data)
    except Exception as e:
        logger.error("from_legacy failed, returning minimal genome: %s", e)
        return DecisionGenome(
            id=str(uuid.uuid4()),
            parameters={},
            fitness=None,
            generation=0,
            species_type=None,
            parent_ids=[],
            mutation_history=[],
            performance_metrics={},
            created_at=time.time(),
            metadata={},
        )


def to_legacy_view(genome: DecisionGenome) -> dict[str, Any]:
    """Return a plain dict view using canonical keys for legacy consumers."""
    d = genome.to_dict()
    # Ensure types are JSON-serializable/simple
    return {
        "id": d["id"],
        "parameters": d["parameters"],
        "fitness": d["fitness"],
        "generation": d["generation"],
        "species_type": d["species_type"],
        "parent_ids": d["parent_ids"],
        "mutation_history": d["mutation_history"],
        "performance_metrics": d["performance_metrics"],
        "created_at": d["created_at"],
        "metadata": d.get("metadata", {}),
    }


__all__ = ["from_legacy", "to_legacy_view"]
