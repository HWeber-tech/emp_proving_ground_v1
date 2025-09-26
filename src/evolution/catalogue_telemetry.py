"""Telemetry helpers for reporting genome catalogue usage.

These utilities translate the population statistics exposed by
``EvolutionEngine``/``PopulationManager`` into sanitized payloads for the
runtime event bus and documentation surfaces.  They keep context-engineering
considerations front and center by quoting catalogue metadata alongside the
population breakdown so operators can immediately see which institutional
genomes seeded a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, TypeGuard


def _is_numeric_like(value: object) -> TypeGuard[int | float | str]:
    """Return True for values that can be losslessly coerced via float()."""

    return isinstance(value, (int, float, str)) and value is not True and value is not False


def _coerce_int_value(value: object, *, default: int = 0) -> int:
    """Best-effort conversion to ``int`` handling numeric strings and floats."""

    if _is_numeric_like(value):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _coerce_float_value(value: object, *, default: float | None = 0.0) -> float | None:
    """Best-effort conversion to ``float`` handling numeric strings."""

    if _is_numeric_like(value):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_int_mapping(payload: Mapping[str, object] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(payload, Mapping):
        return result
    for key, value in payload.items():
        if _is_numeric_like(value):
            try:
                result[str(key)] = int(float(value))
            except ValueError:
                continue
        else:
            continue
    return result


def _coerce_float_mapping(payload: Mapping[str, object] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(payload, Mapping):
        return result
    for key, value in payload.items():
        if _is_numeric_like(value):
            try:
                result[str(key)] = float(value)
            except ValueError:
                continue
        else:
            continue
    return result


def _coerce_tags(payload: Iterable[object] | None) -> tuple[str, ...]:
    if payload is None:
        return ()
    tags: list[str] = []
    for item in payload:
        text = str(item).strip()
        if text:
            tags.append(text)
    return tuple(tags)


@dataclass(slots=True, frozen=True)
class EvolutionCatalogueEntrySnapshot:
    """Lightweight snapshot of a catalogue entry exposed via telemetry."""

    identifier: str
    name: str
    species: str
    generation: int
    tags: tuple[str, ...] = field(default_factory=tuple)
    performance_metrics: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "name": self.name,
            "species": self.species,
            "generation": self.generation,
            "tags": list(self.tags),
            "performance_metrics": dict(self.performance_metrics),
        }


@dataclass(slots=True, frozen=True)
class EvolutionCatalogueSnapshot:
    """Telemetry-friendly catalogue snapshot derived from population stats."""

    seed_source: str
    generation: int
    population_size: int
    species_distribution: Mapping[str, int]
    catalogue_name: str
    catalogue_version: str
    catalogue_size: int
    catalogue_species: Mapping[str, int]
    source_notes: Sequence[str]
    seeded_at: float | None = None
    entries: Sequence[EvolutionCatalogueEntrySnapshot] = field(default_factory=tuple)

    def as_dict(self, *, max_entries: int | None = None) -> dict[str, object]:
        if max_entries is None or max_entries < 0:
            max_entries = len(self.entries)

        entries_payload = [entry.as_dict() for entry in self.entries[:max_entries]]

        return {
            "seed_source": self.seed_source,
            "generation": self.generation,
            "population_size": self.population_size,
            "species_distribution": dict(self.species_distribution),
            "catalogue": {
                "name": self.catalogue_name,
                "version": self.catalogue_version,
                "size": self.catalogue_size,
                "species": dict(self.catalogue_species),
                "source_notes": list(self.source_notes),
                "seeded_at": self.seeded_at,
                "entries": entries_payload,
            },
        }

    def fingerprint(self) -> tuple[object, ...]:
        """Return a tuple suitable for deduplicating published telemetry."""

        species_tuple = tuple(sorted(self.species_distribution.items()))
        catalogue_tuple = (
            self.catalogue_name,
            self.catalogue_version,
            tuple(sorted(self.catalogue_species.items())),
        )
        return (
            self.seed_source,
            self.generation,
            self.population_size,
            species_tuple,
            catalogue_tuple,
        )


def build_catalogue_snapshot(
    population_stats: Mapping[str, object],
    *,
    max_entries: int = 12,
) -> EvolutionCatalogueSnapshot | None:
    """Construct a snapshot from evolution population statistics.

    Parameters
    ----------
    population_stats:
        Mapping produced by :meth:`EvolutionEngine.get_population_statistics`.
    max_entries:
        Upper bound on catalogue entries to retain in the telemetry payload.
    """

    catalogue_raw = population_stats.get("catalogue")
    if not isinstance(catalogue_raw, Mapping):
        return None

    seed_source = str(population_stats.get("seed_source", "unknown"))
    generation = _coerce_int_value(population_stats.get("generation"))
    population_size = _coerce_int_value(population_stats.get("population_size"))
    species_raw = population_stats.get("species_distribution")
    species_distribution = _coerce_int_mapping(species_raw if isinstance(species_raw, Mapping) else None)

    catalogue_name = str(catalogue_raw.get("name", "unknown"))
    catalogue_version = str(catalogue_raw.get("version", "unknown"))
    catalogue_size = _coerce_int_value(catalogue_raw.get("size"))
    species_payload = catalogue_raw.get("species")
    catalogue_species = _coerce_int_mapping(species_payload if isinstance(species_payload, Mapping) else None)
    source_notes_iter = catalogue_raw.get("source_notes")
    source_notes = (
        tuple(str(note) for note in source_notes_iter)
        if isinstance(source_notes_iter, Sequence)
        else ()
    )
    seeded_at = None
    raw_seeded = catalogue_raw.get("seeded_at")
    seeded_at = _coerce_float_value(raw_seeded, default=None) if raw_seeded is not None else None

    entries_raw = catalogue_raw.get("entries")
    entries: list[EvolutionCatalogueEntrySnapshot] = []
    if isinstance(entries_raw, Sequence):
        for entry_obj in entries_raw:
            if not isinstance(entry_obj, Mapping):
                continue
            identifier = str(entry_obj.get("id", ""))
            name = str(entry_obj.get("name", identifier))
            species = str(entry_obj.get("species", ""))
            generation_val = _coerce_int_value(entry_obj.get("generation"))
            tags = _coerce_tags(entry_obj.get("tags"))
            metrics = _coerce_float_mapping(entry_obj.get("performance_metrics"))
            entries.append(
                EvolutionCatalogueEntrySnapshot(
                    identifier=identifier,
                    name=name,
                    species=species,
                    generation=generation_val,
                    tags=tags,
                    performance_metrics=metrics,
                )
            )

    if max_entries >= 0:
        entries = entries[: max_entries or len(entries)]

    return EvolutionCatalogueSnapshot(
        seed_source=seed_source,
        generation=generation,
        population_size=population_size,
        species_distribution=species_distribution,
        catalogue_name=catalogue_name,
        catalogue_version=catalogue_version,
        catalogue_size=catalogue_size,
        catalogue_species=catalogue_species,
        source_notes=source_notes,
        seeded_at=seeded_at,
        entries=tuple(entries),
    )


__all__ = [
    "EvolutionCatalogueEntrySnapshot",
    "EvolutionCatalogueSnapshot",
    "build_catalogue_snapshot",
]
