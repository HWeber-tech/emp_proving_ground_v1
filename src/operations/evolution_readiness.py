"""Evolution readiness evaluation aligned with the modernization roadmap.

This module inspects the evolution feature-flag posture, seed provenance
metadata, and lineage telemetry to produce an actionable readiness snapshot.
It exists so governance reviewers can confirm that realistic genome seeding,
lineage wiring, and adaptive-run gating behave as promised in the roadmap
without mutating the roadmap artefact itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping, Sequence

from src.evolution.lineage_telemetry import EvolutionLineageSnapshot

__all__ = [
    "EvolutionReadinessStatus",
    "EvolutionReadinessSnapshot",
    "evaluate_evolution_readiness",
]


class EvolutionReadinessStatus(StrEnum):
    """Severity levels for the evolution readiness snapshot."""

    blocked = "blocked"
    review = "review"
    ready = "ready"


def _normalise_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return False


def _normalise_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sequence_of_strings(value: object | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        items = [part.strip() for part in value.replace(";", ",").split(",")]
        return tuple(item for item in items if item)
    if isinstance(value, Mapping):
        candidates = value.values()
    else:
        candidates = value  # type: ignore[assignment]

    result: list[str] = []
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        for entry in candidates:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                result.append(text)
    return tuple(result)


def _seed_template_names(seed_metadata: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(seed_metadata, Mapping):
        return tuple()

    templates = seed_metadata.get("seed_templates")
    names: list[str] = []

    if isinstance(templates, Mapping):
        templates = templates.values()

    if isinstance(templates, Sequence) and not isinstance(templates, (str, bytes)):
        for entry in templates:
            if isinstance(entry, Mapping):
                name = _normalise_str(entry.get("name"))
                if name:
                    names.append(name)
            else:
                text = _normalise_str(entry)
                if text:
                    names.append(text)

    if not names:
        names.extend(_sequence_of_strings(seed_metadata.get("seed_names")))

    return tuple(dict.fromkeys(names))


def _lineage_as_dict(
    snapshot: EvolutionLineageSnapshot | Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if snapshot is None:
        return None
    if isinstance(snapshot, EvolutionLineageSnapshot):
        return snapshot.as_dict(max_parents=16, max_mutations=24)
    if isinstance(snapshot, Mapping):
        return snapshot
    return None


@dataclass(slots=True)
class EvolutionReadinessSnapshot:
    """Aggregated readiness view for evolution governance."""

    generated_at: datetime
    status: EvolutionReadinessStatus
    adaptive_runs_enabled: bool
    seed_source: str | None
    seed_templates: tuple[str, ...]
    lineage_generation: int | None
    champion_id: str | None
    champion_registered: bool | None
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "adaptive_runs_enabled": self.adaptive_runs_enabled,
            "seed_source": self.seed_source,
            "seed_templates": list(self.seed_templates),
            "lineage_generation": self.lineage_generation,
            "champion_id": self.champion_id,
            "champion_registered": self.champion_registered,
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }
        return payload

    def to_markdown(self) -> str:
        lines = [
            "### Evolution readiness",
            f"- Status: {self.status.value}",
            f"- Adaptive runs enabled: {'yes' if self.adaptive_runs_enabled else 'no'}",
        ]
        if self.seed_source:
            lines.append(f"- Seed source: {self.seed_source}")
        if self.seed_templates:
            lines.append(
                "- Seed templates: " + ", ".join(sorted(self.seed_templates))
            )
        if self.lineage_generation is not None:
            lines.append(f"- Lineage generation: {self.lineage_generation}")
        if self.champion_id:
            lines.append(
                "- Champion: "
                f"{self.champion_id} (registered: {'yes' if self.champion_registered else 'no'})"
            )
        if self.issues:
            lines.append("")
            lines.append("**Issues**")
            for issue in self.issues:
                lines.append(f"- {issue}")
        return "\n".join(lines)


def evaluate_evolution_readiness(
    *,
    adaptive_runs_enabled: bool,
    population_stats: Mapping[str, Any] | None,
    lineage_snapshot: EvolutionLineageSnapshot | Mapping[str, Any] | None,
    now: datetime | None = None,
) -> EvolutionReadinessSnapshot:
    """Fuse feature-flag, seed, and lineage posture into a readiness snapshot."""

    stats = population_stats or {}
    seed_source = _normalise_str(stats.get("seed_source"))
    seed_metadata = stats.get("seed_metadata") if isinstance(stats, Mapping) else None
    seed_templates = _seed_template_names(seed_metadata) if seed_metadata else tuple()
    lineage = _lineage_as_dict(lineage_snapshot)

    issues: list[str] = []
    status = EvolutionReadinessStatus.ready if adaptive_runs_enabled else EvolutionReadinessStatus.review

    if not seed_source:
        status = EvolutionReadinessStatus.blocked
        issues.append("Seed source missing from population statistics")
    if not seed_templates:
        status = EvolutionReadinessStatus.blocked
        issues.append("Seed metadata missing realistic templates")

    lineage_generation: int | None = None
    champion_id: str | None = None
    champion_registered: bool | None = None

    if lineage is None:
        status = EvolutionReadinessStatus.blocked
        issues.append("Lineage snapshot unavailable")
    else:
        champion = lineage.get("champion") if isinstance(lineage, Mapping) else None
        if isinstance(champion, Mapping):
            champion_id = _normalise_str(champion.get("id"))
            if champion_id is None:
                issues.append("Champion identifier missing from lineage telemetry")
                status = EvolutionReadinessStatus.blocked
            champion_registered = _normalise_bool(champion.get("registered"))
        else:
            issues.append("Champion metadata missing from lineage snapshot")
            status = EvolutionReadinessStatus.blocked

        generation = lineage.get("generation") if isinstance(lineage, Mapping) else None
        if isinstance(generation, (int, float)):
            lineage_generation = int(generation)
        else:
            try:
                lineage_generation = int(float(str(generation)))
            except (TypeError, ValueError):
                lineage_generation = None

        if lineage_generation is None:
            issues.append("Lineage generation missing from telemetry")
            status = EvolutionReadinessStatus.blocked

    if not adaptive_runs_enabled:
        issues.append(
            "Adaptive runs disabled via EVOLUTION_ENABLE_ADAPTIVE_RUNS feature flag"
        )

    metadata: dict[str, Any] = {}
    if isinstance(seed_metadata, Mapping):
        metadata["seed_metadata"] = dict(seed_metadata)
    if lineage is not None:
        metadata["lineage"] = dict(lineage)

    snapshot = EvolutionReadinessSnapshot(
        generated_at=now or datetime.now(tz=UTC),
        status=status,
        adaptive_runs_enabled=adaptive_runs_enabled,
        seed_source=seed_source,
        seed_templates=seed_templates,
        lineage_generation=lineage_generation,
        champion_id=champion_id,
        champion_registered=champion_registered,
        issues=tuple(issues),
        metadata=metadata,
    )
    return snapshot
