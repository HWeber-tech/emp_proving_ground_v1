"""Canonical species management helpers for ecosystem evolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

__all__ = ["EvolvedPredatorProfile", "SpeciesManager"]


@dataclass(slots=True)
class EvolvedPredatorProfile:
    """Lightweight description of an evolved predator specialist."""

    predator_id: str
    predator_type: str
    performance_metrics: Mapping[str, float]


class SpeciesManager:
    """Canonical facade for tracking and evolving predator specialists.

    The implementation deliberately stays lightweight so downstream callers can
    depend on a stable surface while richer behaviour continues to incubate in
    higher-layer experiments.
    """

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []

    async def evolve_specialist(
        self,
        *,
        niche: Any,
        base_population: Sequence[object] | None = None,
        specialization_pressure: float = 0.5,
    ) -> EvolvedPredatorProfile:
        """Produce a deterministic specialist profile for the requested niche."""

        pressure = max(0.0, min(1.0, float(specialization_pressure)))
        predator_type = str(
            getattr(niche, "regime_type", getattr(niche, "niche_type", "generalist"))
        )
        predator_id = f"predator-{len(self._history) + 1}"
        metrics = {
            "adaptation_score": round(0.55 + pressure * 0.35, 3),
            "fitness": round(0.5 + pressure * 0.4, 3),
            "resilience": round(0.6 + pressure * 0.25, 3),
        }
        profile = EvolvedPredatorProfile(
            predator_id=predator_id,
            predator_type=predator_type,
            performance_metrics=metrics,
        )

        self._history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "predator_id": predator_id,
                "predator_type": predator_type,
                "pressure": pressure,
                "population_size": len(base_population or ()),
            }
        )
        return profile

    def get_population_metrics(self) -> dict[str, Any]:
        """Expose summary statistics for recently evolved specialists."""

        if not self._history:
            return {
                "total_evolved": 0,
                "last_evolved": None,
                "average_pressure": 0.0,
            }

        total_pressure = sum(entry["pressure"] for entry in self._history)
        return {
            "total_evolved": len(self._history),
            "last_evolved": self._history[-1],
            "average_pressure": round(total_pressure / len(self._history), 3),
        }

    def list_history(self) -> list[Mapping[str, Any]]:
        """Return a shallow copy of the evolution history for inspection."""

        return [dict(record) for record in self._history]

