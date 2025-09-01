"""
Canonical DecisionGenome model and builders (M5)
- Canonical dataclass with coercion helpers
- Immutable-style update utilities
- Builders for creation and mutation
"""

from __future__ import annotations

import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _to_float(value: Any) -> Optional[float]:
    try:
        # Avoid converting None to 0.0 implicitly
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _coerce_numeric_mapping(mapping: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(mapping, dict):
        items = mapping.items()
    else:
        # Try to get object-like attributes; fallback to empty
        try:
            from typing import cast as _cast
            items = vars(_cast(Any, mapping)).items()
        except Exception:
            return result

    for k, v in items:
        try:
            key = str(k)
        except Exception:
            continue
        fv = _to_float(v)
        if fv is not None:
            result[key] = float(fv)
    return result


def _force_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        out: list[str] = []
        from typing import cast as _cast
        for item in list(_cast(Any, value)):
            out.append(str(item))
        return out
    except Exception:
        return []


def _to_mutation_tag(entry: Any) -> Optional[str]:
    # Accept dict-like legacy entries and convert to canonical string tag
    try:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            gen = entry.get("generation", 0)
            param = entry.get("parameter", "")
            old = entry.get("old_value", "")
            new = entry.get("new_value", "")
            try:
                gen_i = int(gen)
            except Exception:
                gen_i = 0
            return f"g{max(0, gen_i)}:{param}:{old}->{new}"
        # Object-like with attributes
        gen = getattr(entry, "generation", 0)
        param = getattr(entry, "parameter", "")
        old = getattr(entry, "old_value", "")
        new = getattr(entry, "new_value", "")
        try:
            gen_i = int(gen)
        except Exception:
            gen_i = 0
        return f"g{max(0, gen_i)}:{param}:{old}->{new}"
    except Exception:
        return None


@dataclass
class DecisionGenome:
    """Canonical DecisionGenome for evolutionary optimization."""

    id: str
    parameters: dict[str, float]
    fitness: Optional[float] = None
    generation: int = 0
    species_type: Optional[str] = None
    parent_ids: list[str] = field(default_factory=list)
    mutation_history: list[str] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict with deep copies for mutables."""
        return {
            "id": self.id,
            "parameters": copy.deepcopy(self.parameters),
            "fitness": self.fitness,
            "generation": int(self.generation),
            "species_type": self.species_type,
            "parent_ids": list(self.parent_ids),
            "mutation_history": list(self.mutation_history),
            "performance_metrics": copy.deepcopy(self.performance_metrics),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionGenome":
        """Deserialize with coercion rules and safe defaults.

        Rules:
        - generation coerced to int, clamped to >= 0
        - numeric coercion of parameters/performance_metrics to floats (drop non-numeric)
        - parent_ids/mutation_history forced to list[str]; mutation_history entries normalized to tags
        """
        try:
            if not isinstance(data, dict):
                # Try object-like attr access
                try:
                    from typing import cast as _cast
                    data = dict(vars(_cast(Any, data)))
                except Exception:
                    data = {}

            id_val = data.get("id")
            if not isinstance(id_val, str) or not id_val:
                id_val = str(uuid.uuid4())

            # generation
            gen_raw = data.get("generation", 0)
            try:
                gen = int(gen_raw)
            except Exception:
                gen = 0
            gen = max(0, gen)

            # fitness
            fit_raw = data.get("fitness", None)
            fit: Optional[float]
            if fit_raw is None:
                fit = None
            else:
                fit = _to_float(fit_raw)
                if fit is None:
                    fit = None

            species_type = data.get("species_type")
            if species_type is not None:
                try:
                    species_type = str(species_type)
                except Exception:
                    species_type = None

            params = _coerce_numeric_mapping(data.get("parameters", {}))
            perf = _coerce_numeric_mapping(data.get("performance_metrics", {}))

            parent_ids = _force_str_list(data.get("parent_ids", []))

            mh_raw = data.get("mutation_history", [])
            mh_list: list[str] = []
            for entry in _force_str_list(mh_raw) if isinstance(mh_raw, str) else (mh_raw or []):
                tag = _to_mutation_tag(entry)
                if tag:
                    mh_list.append(tag)

            created_at = data.get("created_at")
            cat = _to_float(created_at)
            if cat is None:
                cat = time.time()

            return cls(
                id=id_val,
                parameters=params,
                fitness=fit,
                generation=gen,
                species_type=species_type,
                parent_ids=parent_ids,
                mutation_history=mh_list,
                performance_metrics=perf,
                created_at=cat,
            )
        except Exception as e:
            logger.error("DecisionGenome.from_dict failed: %s", e)
            # Minimal valid fallback
            return cls(
                id=str(uuid.uuid4()),
                parameters={},
                fitness=None,
                generation=0,
                species_type=None,
                parent_ids=[],
                mutation_history=[],
                performance_metrics={},
                created_at=time.time(),
            )

    def with_updated(self, **kwargs: Any) -> "DecisionGenome":
        """Return a new instance with updated fields and deep-copied mutables."""
        data = {
            "id": self.id,
            "parameters": copy.deepcopy(self.parameters),
            "fitness": self.fitness,
            "generation": int(self.generation),
            "species_type": self.species_type,
            "parent_ids": list(self.parent_ids),
            "mutation_history": list(self.mutation_history),
            "performance_metrics": copy.deepcopy(self.performance_metrics),
            "created_at": float(self.created_at),
        }

        # If caller passes mutables, ensure we copy them too for safety
        if "parameters" in kwargs and isinstance(kwargs["parameters"], dict):
            kwargs["parameters"] = copy.deepcopy(kwargs["parameters"])
        if "parent_ids" in kwargs and isinstance(kwargs["parent_ids"], list):
            kwargs["parent_ids"] = list(kwargs["parent_ids"])
        if "mutation_history" in kwargs and isinstance(kwargs["mutation_history"], list):
            kwargs["mutation_history"] = list(kwargs["mutation_history"])
        if "performance_metrics" in kwargs and isinstance(kwargs["performance_metrics"], dict):
            kwargs["performance_metrics"] = copy.deepcopy(kwargs["performance_metrics"])

        data.update(kwargs)
        # Cast to a permissive mapping before kwargs expansion to satisfy typing
        from typing import cast as _cast

        return DecisionGenome(**_cast(dict[str, Any], data))


def new_genome(
    id: str,
    parameters: dict[str, float],
    generation: int = 0,
    species_type: Optional[str] = None,
) -> DecisionGenome:
    """Builder: create a new canonical genome with coercions applied."""
    gen = max(0, int(generation)) if isinstance(generation, int) else 0
    params = _coerce_numeric_mapping(parameters or {})
    return DecisionGenome(
        id=id,
        parameters=params,
        fitness=None,
        generation=gen,
        species_type=species_type,
        parent_ids=[],
        mutation_history=[],
        performance_metrics={},
        created_at=time.time(),
    )


def mutate(genome: DecisionGenome, mutation: str, new_params: dict[str, float]) -> DecisionGenome:
    """Builder: return a new genome updated with new_params and appended mutation tag(s).

    For each changed param, append a tag: "g{gen}:{key}:{old}->{new}"
    """
    # Start from a deep copy of parameters
    coerced_updates = _coerce_numeric_mapping(new_params or {})
    updated_params = copy.deepcopy(genome.parameters)

    tags: list[str] = []
    for key, new_val in coerced_updates.items():
        old_val = updated_params.get(key)
        if old_val is None or float(old_val) != float(new_val):
            tags.append(f"g{genome.generation}:{key}:{old_val}->{new_val}")
            updated_params[key] = float(new_val)

    if not tags:
        # Nothing changed; still return a new instance with safe copies
        return genome.with_updated(parameters=updated_params)

    new_history = list(genome.mutation_history)
    # Prepend mutation type for traceability if provided (optional context)
    if mutation:
        # Optional: include a header tag for mutation type at generation boundary
        new_history.append(f"g{genome.generation}:mutation:{mutation}")

    new_history.extend(tags)

    return genome.with_updated(parameters=updated_params, mutation_history=new_history)


__all__ = ["DecisionGenome", "new_genome", "mutate"]
