from __future__ import annotations

"""
Genome Provider Adapter
=======================

Concrete adapter that conforms to core's GenomeProvider Protocol by delegating to
the concrete genome implementation in this package. All imports are guarded and
methods return safe fallbacks on error to satisfy the non-raising contract.

This module does NOT import core modules at runtime; it only depends on local
genome implementations and provides local fallbacks to avoid introducing cycles.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Guarded imports of concrete genome functions/classes
# Pre-declare dynamic bindings with permissive types to satisfy mypy in fallback branches
_DecisionGenome: Any = None
_mutate: Any = None
_new_genome: Any = None
_from_legacy: Any = None
_to_legacy_view: Any = None
if TYPE_CHECKING:
    from src.genome.models.genome import DecisionGenome  # type-only
try:
    from src.genome.models.genome import DecisionGenome as __DecisionGenome
    from src.genome.models.genome import mutate as __mutate
    from src.genome.models.genome import new_genome as __new_genome
    _DecisionGenome = __DecisionGenome  # type: ignore[assignment]
    _mutate = __mutate
    _new_genome = __new_genome
except Exception:
    _DecisionGenome = None  # type: ignore[assignment]
    _mutate = None
    _new_genome = None

try:
    from src.genome.models.adapters import from_legacy as _from_legacy
    from src.genome.models.adapters import to_legacy_view as _to_legacy_view
except Exception:
    _from_legacy = None
    _to_legacy_view = None


def _coerce_float_map(mapping: Any) -> Dict[str, float]:
    """Best-effort coercion of mapping/object to Dict[str, float]."""
    result: Dict[str, float] = {}
    try:
        items = mapping.items() if isinstance(mapping, dict) else vars(mapping).items()
    except Exception:
        return result
    for k, v in items:
        try:
            key = str(k)
        except Exception:
            continue
        try:
            if v is None:
                continue
            result[key] = float(v)
        except Exception:
            continue
    return result


@dataclass
class _FallbackGenome:
    """
    Local minimal fallback genome with attribute shape compatible with core usage.

    Provides:
      - fields: id, parameters, fitness, generation, species_type, parent_ids,
                mutation_history, performance_metrics, created_at
      - methods: with_updated(), to_dict()
    """

    id: str
    parameters: Dict[str, float] = field(default_factory=dict)
    fitness: Optional[float] = None
    generation: int = 0
    species_type: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def with_updated(self, **kwargs: Any) -> "_FallbackGenome":
        data = {
            "id": self.id,
            "parameters": dict(self.parameters),
            "fitness": self.fitness,
            "generation": int(self.generation),
            "species_type": self.species_type,
            "parent_ids": list(self.parent_ids),
            "mutation_history": list(self.mutation_history),
            "performance_metrics": dict(self.performance_metrics),
            "created_at": float(self.created_at),
        }
        for k, v in kwargs.items():
            if k in ("parameters", "performance_metrics") and isinstance(v, dict):
                data[k] = dict(v)
            elif k in ("parent_ids", "mutation_history") and isinstance(v, list):
                data[k] = list(v)
            elif k in data:
                data[k] = v
        return _FallbackGenome(**data)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parameters": dict(self.parameters),
            "fitness": self.fitness,
            "generation": int(self.generation),
            "species_type": self.species_type,
            "parent_ids": list(self.parent_ids),
            "mutation_history": list(self.mutation_history),
            "performance_metrics": dict(self.performance_metrics),
            "created_at": float(self.created_at),
        }


class GenomeProviderAdapter:
    """
    Adapter conforming to core GenomeProvider Protocol.

    Delegates to src.genome.models.genome and src.genome.models.adapters when available.
    Swallows exceptions and returns safe fallbacks.
    """

    def new_genome(
        self,
        id: str,
        parameters: Dict[str, float],
        generation: int = 0,
        species_type: Optional[str] = None,
    ) -> Any:
        try:
            if callable(_new_genome):
                return _new_genome(
                    id=id,
                    parameters=parameters,
                    generation=generation,
                    species_type=species_type,
                )
        except Exception as e:
            logger.error("GenomeProviderAdapter.new_genome failed: %s", e)
        # Fallback
        try:
            return _FallbackGenome(
                id=str(id),
                parameters=_coerce_float_map(parameters or {}),
                fitness=None,
                generation=int(generation) if isinstance(generation, int) else 0,
                species_type=str(species_type) if species_type is not None else None,
                parent_ids=[],
                mutation_history=[],
                performance_metrics={},
                created_at=time.time(),
            )
        except Exception:
            return _FallbackGenome(id=str(id))

    def mutate(self, genome: Any, mutation: str, new_params: Dict[str, float]) -> Any:
        try:
            if (
                callable(_mutate)
                and _DecisionGenome is not None
                and isinstance(genome, _DecisionGenome)
            ):
                return _mutate(genome, mutation, new_params)
        except Exception as e:
            logger.error("GenomeProviderAdapter.mutate failed via concrete impl: %s", e)
        # Fallback: shallow param update + mutation tag
        try:
            params = getattr(genome, "parameters", {}) or {}
            updated = dict(params) if isinstance(params, dict) else {}
            updates = _coerce_float_map(new_params or {})
            updated.update(updates)

            mh = list(getattr(genome, "mutation_history", []) or [])
            gen = getattr(genome, "generation", 0)
            try:
                gen_i = int(gen)
            except Exception:
                gen_i = 0
            mh.append(f"g{max(0, gen_i)}:mutation:{mutation}")

            if hasattr(genome, "with_updated") and callable(getattr(genome, "with_updated")):
                return genome.with_updated(parameters=updated, mutation_history=mh)
            # Construct fallback copy preserving basic metadata if possible
            gid = getattr(genome, "id", f"fallback_{int(time.time()*1000)}")
            species = getattr(genome, "species_type", None)
            return _FallbackGenome(
                id=str(gid),
                parameters=updated,
                fitness=getattr(genome, "fitness", None),
                generation=max(0, int(gen)) if isinstance(gen, int) else 0,
                species_type=str(species) if species is not None else None,
                parent_ids=list(getattr(genome, "parent_ids", []) or []),
                mutation_history=mh,
                performance_metrics=_coerce_float_map(
                    getattr(genome, "performance_metrics", {}) or {}
                ),
                created_at=float(getattr(genome, "created_at", time.time()) or time.time()),
            )
        except Exception:
            return genome

    def from_legacy(self, obj: Any) -> Any:
        try:
            if callable(_from_legacy):
                return _from_legacy(obj)
        except Exception as e:
            logger.error("GenomeProviderAdapter.from_legacy failed via concrete impl: %s", e)
        # Fallback adaptation
        try:
            get = (
                (lambda o, k, d=None: o.get(k, d))
                if isinstance(obj, dict)
                else (lambda o, k, d=None: getattr(o, k, d))
            )
            return _FallbackGenome(
                id=str(get(obj, "id", f"fallback_{int(time.time()*1000)}")),
                parameters=_coerce_float_map(get(obj, "parameters", {}) or {}),
                fitness=None,
                generation=int(get(obj, "generation", 0) or 0),
                species_type=(
                    str(get(obj, "species_type", None))
                    if get(obj, "species_type", None) is not None
                    else None
                ),
                parent_ids=list(get(obj, "parent_ids", []) or []),
                mutation_history=list(get(obj, "mutation_history", []) or []),
                performance_metrics=_coerce_float_map(get(obj, "performance_metrics", {}) or {}),
                created_at=float(get(obj, "created_at", time.time()) or time.time()),
            )
        except Exception:
            return _FallbackGenome(id=f"fallback_{int(time.time()*1000)}")

    def to_legacy_view(self, genome: Any) -> Dict[str, Any] | Any:
        try:
            if (
                callable(_to_legacy_view)
                and _DecisionGenome is not None
                and isinstance(genome, _DecisionGenome)
            ):
                return _to_legacy_view(genome)
        except Exception as e:
            logger.error("GenomeProviderAdapter.to_legacy_view failed via concrete impl: %s", e)
        # Fallbacks
        try:
            to_dict = getattr(genome, "to_dict", None)
            if callable(to_dict):
                d = to_dict()
                return d if isinstance(d, dict) else {}
            if isinstance(genome, dict):
                return dict(genome)
        except Exception:
            pass
        return {}


__all__ = ["GenomeProviderAdapter"]
