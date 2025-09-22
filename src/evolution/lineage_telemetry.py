"""Telemetry helpers that capture evolution lineage snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.core.evolution.engine import EvolutionSummary
    from src.orchestration.evolution_cycle import ChampionRecord


def _normalize_sequence(values: Sequence[Any] | None, *, max_items: int) -> tuple[Any, ...]:
    if not values:
        return tuple()
    items = list(values)[: max_items if max_items >= 0 else None]
    return tuple(items)


def _ensure_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    return {}


def _summary_to_mapping(summary: "EvolutionSummary | Mapping[str, Any] | None") -> dict[str, Any]:
    if summary is None:
        return {}
    if isinstance(summary, Mapping):
        return {str(key): value for key, value in summary.items()}
    try:  # dataclasses and pydantic models expose ``model_dump``/``dict`` helpers
        from dataclasses import asdict

        return asdict(summary)  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        to_dict = getattr(summary, "dict", None)
        if callable(to_dict):
            return {str(key): value for key, value in to_dict().items()}
    except Exception:
        pass
    try:
        to_dump = getattr(summary, "model_dump", None)
        if callable(to_dump):
            return {str(key): value for key, value in to_dump().items()}
    except Exception:
        pass
    return {}


@dataclass(slots=True)
class EvolutionLineageSnapshot:
    """Immutable snapshot describing the latest evolution lineage state."""

    generation: int
    champion_id: str
    fitness: float
    registered: bool
    species: str | None = None
    parent_ids: tuple[str, ...] = field(default_factory=tuple)
    mutation_history: tuple[str, ...] = field(default_factory=tuple)
    seed_source: str | None = None
    species_distribution: Mapping[str, int] = field(default_factory=dict)
    evaluation_metadata: Mapping[str, Any] = field(default_factory=dict)
    catalogue: Mapping[str, Any] | None = None
    summary: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(
        self,
        *,
        max_parents: int = 5,
        max_mutations: int = 8,
    ) -> dict[str, Any]:
        """Serialise the snapshot into an event-friendly payload."""

        parents = list(self.parent_ids[: max_parents if max_parents >= 0 else None])
        mutations = list(self.mutation_history[: max_mutations if max_mutations >= 0 else None])

        payload: dict[str, Any] = {
            "generation": self.generation,
            "champion": {
                "id": self.champion_id,
                "fitness": float(self.fitness),
                "registered": bool(self.registered),
                "species": self.species,
                "parent_ids": parents,
                "mutation_history": mutations,
            },
            "population": {
                "seed_source": self.seed_source,
                "species_distribution": dict(self.species_distribution),
            },
        }
        if self.evaluation_metadata:
            payload["champion"]["metadata"] = dict(self.evaluation_metadata)
        if self.catalogue:
            payload["catalogue"] = dict(self.catalogue)
        if self.summary:
            payload["summary"] = dict(self.summary)
        return payload

    def fingerprint(self) -> tuple[Any, ...]:
        """Return a stable fingerprint to deduplicate telemetry events."""

        species_items: Tuple[Tuple[str, int], ...] = tuple(
            sorted((str(k), int(v)) for k, v in self.species_distribution.items())
        )
        catalogue_name = None
        if isinstance(self.catalogue, Mapping):
            catalogue_name = self.catalogue.get("name")
        return (
            int(self.generation),
            self.champion_id,
            float(self.fitness),
            bool(self.registered),
            self.species,
            self.parent_ids,
            self.mutation_history,
            self.seed_source,
            species_items,
            catalogue_name,
        )

    def to_markdown(self) -> str:
        """Render a compact markdown summary for logs or dashboards."""

        lines = [
            f"### Evolution lineage – generation {self.generation}",
            f"- **Champion** `{self.champion_id}` ({self.species or 'unlabelled'})",
            f"- **Fitness** `{self.fitness:.4f}` (registered: {'yes' if self.registered else 'no'})",
        ]
        if self.parent_ids:
            parents = ", ".join(self.parent_ids[:5])
            if len(self.parent_ids) > 5:
                parents += ", …"
            lines.append(f"- **Parents** {parents}")
        if self.mutation_history:
            mutations = ", ".join(self.mutation_history[:5])
            if len(self.mutation_history) > 5:
                mutations += ", …"
            lines.append(f"- **Mutations** {mutations}")
        if self.seed_source:
            lines.append(f"- **Seed source** {self.seed_source}")
        if self.species_distribution:
            species = ", ".join(
                f"{name}: {count}" for name, count in sorted(self.species_distribution.items())
            )
            lines.append(f"- **Species distribution** {species}")
        if self.catalogue and "name" in self.catalogue:
            lines.append(f"- **Catalogue** {self.catalogue['name']}")
        return "\n".join(lines)


def build_lineage_snapshot(
    stats: Mapping[str, object] | None,
    champion: "ChampionRecord | None",
    *,
    summary: "EvolutionSummary | Mapping[str, Any] | None" = None,
    max_parent_items: int = 16,
    max_mutation_items: int = 24,
) -> EvolutionLineageSnapshot | None:
    """Fuse population statistics and champion metadata into a lineage snapshot."""

    if champion is None:
        return None
    stats = stats or {}
    seed_source = str(stats.get("seed_source")) if stats.get("seed_source") is not None else None
    distribution = _ensure_mapping(stats.get("species_distribution"))
    catalogue = None
    raw_catalogue = stats.get("catalogue")
    if isinstance(raw_catalogue, Mapping):
        catalogue = {str(key): value for key, value in raw_catalogue.items()}

    generation = int(stats.get("generation", 0))
    if summary:
        summary_mapping = _summary_to_mapping(summary)
        generation = int(summary_mapping.get("generation", generation))
    else:
        summary_mapping = {}

    metadata = _ensure_mapping(champion.report.metadata)

    return EvolutionLineageSnapshot(
        generation=generation,
        champion_id=champion.genome_id,
        fitness=champion.fitness,
        registered=champion.registered,
        species=champion.species,
        parent_ids=_normalize_sequence(champion.parent_ids, max_items=max_parent_items),
        mutation_history=_normalize_sequence(
            champion.mutation_history, max_items=max_mutation_items
        ),
        seed_source=seed_source,
        species_distribution=distribution,
        evaluation_metadata=metadata,
        catalogue=catalogue,
        summary=summary_mapping,
    )
