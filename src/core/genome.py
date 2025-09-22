"""
Core Genome Port
================

Defines the GenomeProvider Protocol used by core services (e.g., population manager)
to remain decoupled from the concrete genome implementation in the genome package.

Also provides:
- NoOpGenomeProvider: safe, non-raising stub implementation
- Simple registry: register_genome_provider / get_genome_provider
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, cast, runtime_checkable


@runtime_checkable
class GenomeProvider(Protocol):
    """
    Protocol for genome operations required by core.

    The concrete implementation resides in the genome package (via an adapter)
    and must swallow exceptions internally to align with core's safety goals.
    """

    def new_genome(
        self,
        id: str,
        parameters: Dict[str, float],
        generation: int = 0,
        species_type: Optional[str] = None,
    ) -> object:
        """
        Create a new genome-like object with the given parameters.
        Returns Any to allow decoupling from concrete model classes.
        """

    def mutate(self, genome: object, mutation: str, new_params: Dict[str, float]) -> object:
        """
        Apply a mutation to the provided genome-like object and return a new instance/object.
        """

    def from_legacy(self, obj: object) -> object:
        """
        Adapt a legacy dict/object to a genome-like object suitable for core usage.
        """

    def to_legacy_view(self, genome: object) -> Dict[str, Any] | Any:
        """
        Return a plain dict or legacy-compatible view for downstream consumers.
        """


@dataclass
class _CoreGenomeStub:
    """
    Minimal, in-core genome-shaped stub used by NoOpGenomeProvider.

    This class exists purely to provide a non-raising, attribute-compatible object
    that resembles the expected genome interface (fields + with_updated method).
    It is NOT a domain model and should not be used as a replacement for the
    concrete genome implementation in src.genome.* packages.
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

    def with_updated(self, **kwargs: object) -> "_CoreGenomeStub":
        """Return a shallow-copied updated instance. Only known fields are applied."""
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
            if k in data:
                # Basic defensive copies for mutables
                if k in ("parameters", "performance_metrics") and isinstance(v, dict):
                    data[k] = dict(v)
                elif k in ("parent_ids", "mutation_history") and isinstance(v, list):
                    data[k] = list(v)
                else:
                    data[k] = cast(Any, v)
        return _CoreGenomeStub(**data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict view of this stub."""
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


def _coerce_numeric_mapping(mapping: object) -> Dict[str, float]:
    """Best-effort coercion of a mapping/object to Dict[str, float]."""
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
            fv = float(v)
            result[key] = float(fv)
        except Exception:
            continue
    return result


def _genome_models(sym: str) -> Any:
    return getattr(importlib.import_module("src.genome.models.genome_adapter"), sym)


class NoOpGenomeProvider:
    """
    NoOp implementation of GenomeProvider.

    Behavior:
    - Never raises; swallows exceptions and returns safe placeholders.
    - Returns a lightweight in-core stub object with the expected attributes to
      minimize breakage in consumers that access fields like .fitness, .parameters,
      .with_updated(), etc.

    NOTE: This is a stub. For real operations, register a concrete provider adapter
    from the genome package via register_genome_provider().
    """

    def new_genome(
        self,
        id: str,
        parameters: Dict[str, float],
        generation: int = 0,
        species_type: Optional[str] = None,
    ) -> object:
        try:
            return _CoreGenomeStub(
                id=str(id),
                parameters=_coerce_numeric_mapping(parameters or {}),
                fitness=None,
                generation=int(generation) if isinstance(generation, int) else 0,
                species_type=str(species_type) if species_type is not None else None,
                parent_ids=[],
                mutation_history=[],
                performance_metrics={},
                created_at=time.time(),
            )
        except Exception:
            return _CoreGenomeStub(id=str(id))

    def mutate(self, genome: object, mutation: str, new_params: Dict[str, float]) -> object:
        try:
            # Best-effort: if stub-like, update parameters and append a simple tag.
            if hasattr(genome, "parameters") and hasattr(genome, "with_updated"):
                params = getattr(genome, "parameters", {}) or {}
                if isinstance(params, dict):
                    updated = dict(params)
                    updates = _coerce_numeric_mapping(new_params or {})
                    updated.update(updates)
                    mh = list(getattr(genome, "mutation_history", []) or [])
                    gen = getattr(genome, "generation", 0)
                    if isinstance(gen, int):
                        mh.append(f"g{gen}:mutation:{mutation}")
                    return genome.with_updated(parameters=updated, mutation_history=mh)
            # Fallback: return input genome unchanged
            return genome
        except Exception:
            return genome

    def from_legacy(self, obj: object) -> object:
        try:
            # If it's already stub-like (has id/parameters), return as is
            if hasattr(obj, "id") and hasattr(obj, "parameters"):
                return obj
            # Adapt dict/object into stub
            get = (
                (lambda o, k, d=None: o.get(k, d))
                if isinstance(obj, dict)
                else (lambda o, k, d=None: getattr(o, k, d))
            )
            return _CoreGenomeStub(
                id=str(get(obj, "id", f"noop_{int(time.time() * 1000)}")),
                parameters=_coerce_numeric_mapping(get(obj, "parameters", {}) or {}),
                fitness=None,
                generation=int(get(obj, "generation", 0) or 0),
                species_type=(
                    str(get(obj, "species_type", None))
                    if get(obj, "species_type", None) is not None
                    else None
                ),
                parent_ids=list(get(obj, "parent_ids", []) or []),
                mutation_history=list(get(obj, "mutation_history", []) or []),
                performance_metrics=_coerce_numeric_mapping(
                    get(obj, "performance_metrics", {}) or {}
                ),
                created_at=float(get(obj, "created_at", time.time()) or time.time()),
            )
        except Exception:
            return _CoreGenomeStub(id=f"noop_{int(time.time() * 1000)}")

    def to_legacy_view(self, genome: object) -> Dict[str, Any] | Any:
        try:
            # Prefer an existing to_dict() if present
            to_dict = getattr(genome, "to_dict", None)
            if callable(to_dict):
                d = to_dict()
                return d if isinstance(d, dict) else {}
            # If it's a mapping already, return it as-is
            if isinstance(genome, dict):
                return dict(genome)
        except Exception:
            pass
        # Safe default
        return {}


# Registry
_GENOME_PROVIDER: Optional[GenomeProvider] = None
_NOOP_SINGLETON: NoOpGenomeProvider = NoOpGenomeProvider()


def register_genome_provider(provider: GenomeProvider) -> None:
    """
    Register a concrete GenomeProvider implementation.

    This does not import or couple core to any concrete domain package;
    the caller is responsible for constructing the provider (e.g., via orchestration).
    """
    global _GENOME_PROVIDER
    _GENOME_PROVIDER = provider


def get_genome_provider() -> GenomeProvider:
    """
    Return the registered GenomeProvider or the NoOp fallback if none is set.

    If no provider has been registered, attempt to use the concrete genome adapter
    from src.genome.models.genome_adapter to provide real genome instances.
    """
    global _GENOME_PROVIDER
    if _GENOME_PROVIDER is not None:
        return _GENOME_PROVIDER
    try:
        # Local dynamic import to avoid static cross-layer dependency
        GenomeProviderAdapter = _genome_models("GenomeProviderAdapter")
        _GENOME_PROVIDER = GenomeProviderAdapter()
        return _GENOME_PROVIDER
    except Exception:
        return _NOOP_SINGLETON


__all__ = [
    "GenomeProvider",
    "NoOpGenomeProvider",
    "register_genome_provider",
    "get_genome_provider",
]
