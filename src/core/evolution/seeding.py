"""Realistic genome seeding helpers for the evolution engine.

The roadmap calls for replacing the naive random genome initialisation with
seeds that reflect the institutional strategy catalogue described in the EMP
Encyclopedia.  This module exposes a lightweight sampler that produces
catalogue-inspired genomes with jittered parameters, lineage metadata, and
performance fingerprints so lineage telemetry can trace their provenance even
before adaptive runs are enabled.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from src.genome.catalogue import CatalogueEntry, GenomeCatalogue, load_default_catalogue


logger = logging.getLogger(__name__)


def _log_seed_warning(genome: object, action: str, exc: Exception) -> None:
    """Record failures when injecting seed metadata into genomes."""

    genome_id = getattr(genome, "id", "<unknown>")
    logger.warning(
        "Seed hardening: %s for genome %s failed: %s",
        action,
        genome_id,
        exc,
        exc_info=exc,
    )


def _safe_seed_setattr(genome: object, attribute: str, value: object) -> None:
    try:
        setattr(genome, attribute, value)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        _log_seed_warning(genome, f"setting attribute '{attribute}'", exc)


def _ensure_float(value: float) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
def _jitter(
    base: float,
    *,
    rng: random.Random,
    pct: float,
    minimum: float | None = None,
    clamp_min: float | None = None,
) -> float:
    scale = abs(base) if abs(base) > 1e-6 else 1.0
    perturb = rng.gauss(0.0, pct * scale)
    value = base + perturb
    if clamp_min is not None:
        value = max(clamp_min, value)
    if minimum is not None:
        value = max(minimum, value)
    return float(value)


@dataclass(frozen=True)
class GenomeSeedTemplate:
    """Template describing a realistic institutional genome seed."""

    name: str
    species: str
    base_parameters: Mapping[str, float]
    parameter_jitter: Mapping[str, float] = field(default_factory=dict)
    parent_ids: Sequence[str] = field(default_factory=tuple)
    mutation_history: Sequence[str] = field(default_factory=tuple)
    performance_metrics: Mapping[str, float] = field(default_factory=dict)
    tags: Sequence[str] = field(default_factory=tuple)
    catalogue_entry_id: str | None = None

    def spawn(self, rng: random.Random) -> "GenomeSeed":
        parameters: dict[str, float] = {}
        for key, value in self.base_parameters.items():
            jitter = float(self.parameter_jitter.get(key, 0.06))
            jitter = max(0.0, min(jitter, 0.5))
            minimum = 0.0
            clamp_min = None
            if key.endswith("_window"):
                clamp_min = 1.0
                minimum = 1.0
            if "risk" in key or "tolerance" in key or key.endswith("_factor"):
                minimum = 0.0
            parameters[key] = _jitter(
                _ensure_float(value), rng=rng, pct=jitter, minimum=minimum, clamp_min=clamp_min
            )

        metrics = {str(k): _ensure_float(v) for k, v in self.performance_metrics.items()}
        history = tuple(str(item) for item in self.mutation_history if item)
        parents = tuple(str(item) for item in self.parent_ids if item)
        tags = tuple(str(item) for item in self.tags if item)

        return GenomeSeed(
            name=self.name,
            species=self.species,
            parameters=parameters,
            parent_ids=parents,
            mutation_history=history,
            performance_metrics=metrics,
            tags=tags,
            catalogue_entry_id=self.catalogue_entry_id,
        )


@dataclass(slots=True)
class GenomeSeed:
    """Materialised genome seed with metadata for lineage telemetry."""

    name: str
    species: str
    parameters: Mapping[str, float]
    parent_ids: Sequence[str]
    mutation_history: Sequence[str]
    performance_metrics: Mapping[str, float]
    tags: Sequence[str]
    catalogue_entry_id: str | None = None

    def metadata(self) -> dict[str, object]:
        payload: MutableMapping[str, object] = {
            "seed_name": self.name,
            "seed_tags": list(self.tags),
            "seed_species": self.species,
        }
        if self.catalogue_entry_id:
            payload["seed_catalogue_id"] = self.catalogue_entry_id
        if self.parent_ids:
            payload["seed_parent_ids"] = list(self.parent_ids)
        if self.mutation_history:
            payload["seed_mutation_history"] = list(self.mutation_history)
        if self.performance_metrics:
            payload["seed_performance_metrics"] = dict(self.performance_metrics)
        return dict(payload)


def apply_seed_to_genome(genome: Any, seed: GenomeSeed) -> Any:
    """Apply lineage metadata from a seed to a realised genome."""

    updates: dict[str, object] = {}
    if seed.parent_ids:
        updates["parent_ids"] = list(seed.parent_ids)
    if seed.mutation_history:
        updates["mutation_history"] = list(seed.mutation_history)
    if seed.performance_metrics:
        existing = getattr(genome, "performance_metrics", {}) or {}
        merged: MutableMapping[str, float]
        if isinstance(existing, Mapping):
            merged = dict(existing)
        else:
            merged = {}
        merged.update({str(k): float(v) for k, v in seed.performance_metrics.items()})
        updates["performance_metrics"] = merged

    applied_via_update = False
    if updates and hasattr(genome, "with_updated"):
        try:
            genome = genome.with_updated(**updates)
            applied_via_update = True
        except Exception as exc:  # pragma: no cover - defensive guardrail
            _log_seed_warning(genome, "applying seed updates via with_updated", exc)

    if not applied_via_update:
        for key, value in updates.items():
            _safe_seed_setattr(genome, key, value)

    metadata = seed.metadata()
    if metadata:
        try:
            existing_meta = getattr(genome, "metadata", {}) or {}
            if isinstance(existing_meta, Mapping):
                merged_meta = dict(existing_meta)
            else:
                merged_meta = {}
            merged_meta.update(metadata)
            setattr(genome, "metadata", merged_meta)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            _log_seed_warning(genome, "merging seed metadata", exc)

    return genome


def _extract_seed_metadata(candidate: Any) -> Mapping[str, object] | None:
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, Mapping):
        if any(str(key).startswith("seed_") for key in metadata.keys()):
            return metadata
    direct_seed = {
        key: getattr(candidate, key)
        for key in ("seed_name", "seed_tags", "seed_species")
        if hasattr(candidate, key)
    }
    return direct_seed or None


def _normalise_seed_tags(tags: object) -> tuple[str, ...]:
    if tags is None:
        return tuple()
    if isinstance(tags, str):
        text = tags.strip()
        return (text,) if text else tuple()
    if isinstance(tags, Mapping):
        values = tags.values()
    else:
        values = tags

    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            result.append(text)
    return tuple(result)


def _normalise_seed_sequence(values: object) -> tuple[str, ...]:
    """Normalise sequence-like metadata fields into comparable tuples."""

    if values is None:
        return tuple()
    if isinstance(values, str):
        text = values.strip()
        return (text,) if text else tuple()
    if isinstance(values, Mapping):
        values = values.values()

    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        normalised: list[str] = []
        for entry in values:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                normalised.append(text)
        return tuple(normalised)
    return tuple()


def _ordered_counter(counter: Counter[str]) -> dict[str, int]:
    ordered_keys = sorted(counter.keys(), key=lambda item: (-counter[item], item))
    return {key: counter[key] for key in ordered_keys}


def summarize_seed_metadata(population: Sequence[Any]) -> dict[str, object] | None:
    """Return aggregated seed provenance metadata for a population."""

    name_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    seed_species_counts: Counter[str] = Counter()
    catalogue_id_counts: Counter[str] = Counter()
    parent_counts: Counter[str] = Counter()
    mutation_counts: Counter[str] = Counter()
    total_with_metadata = 0

    for candidate in population:
        metadata = _extract_seed_metadata(candidate)
        if metadata is None:
            continue

        seed_name = metadata.get("seed_name")
        if isinstance(seed_name, str) and seed_name:
            name_counts[seed_name] += 1

        tags = metadata.get("seed_tags")
        for tag in _normalise_seed_tags(tags):
            tag_counts[tag] += 1

        seed_species = metadata.get("seed_species")
        if isinstance(seed_species, str) and seed_species:
            seed_species_counts[seed_species] += 1

        catalogue_id = metadata.get("seed_catalogue_id")
        if isinstance(catalogue_id, str) and catalogue_id:
            catalogue_id_counts[catalogue_id] += 1

        for parent in _normalise_seed_sequence(metadata.get("seed_parent_ids")):
            parent_counts[parent] += 1

        mutations = metadata.get("seed_mutation_history")
        if not mutations:
            mutations = metadata.get("seed_mutations")
        for mutation in _normalise_seed_sequence(mutations):
            mutation_counts[mutation] += 1

        total_with_metadata += 1

    if (
        not name_counts
        and not tag_counts
        and not seed_species_counts
        and not catalogue_id_counts
        and not parent_counts
        and not mutation_counts
    ):
        return None

    summary: dict[str, object] = {}
    if total_with_metadata:
        summary["total_seeded"] = total_with_metadata

    if name_counts:
        ordered_names = _ordered_counter(name_counts)
        summary["seed_names"] = ordered_names
        total_named = sum(name_counts.values())
        summary["seed_templates"] = [
            {
                "name": name,
                "count": count,
                "share": count / float(total_named),
            }
            for name, count in ordered_names.items()
        ]

    if tag_counts:
        summary["seed_tags"] = _ordered_counter(tag_counts)

    if seed_species_counts:
        summary["seed_species"] = _ordered_counter(seed_species_counts)

    if catalogue_id_counts:
        summary["seed_catalogue_ids"] = _ordered_counter(catalogue_id_counts)

    if parent_counts:
        summary["seed_parent_ids"] = _ordered_counter(parent_counts)

    if mutation_counts:
        summary["seed_mutations"] = _ordered_counter(mutation_counts)

    return summary


_CATALOGUE_JITTER_OVERRIDES: dict[str, Mapping[str, float]] = {
    "catalogue/trend-alpha": {
        "momentum_window": 0.08,
        "trend_sensitivity": 0.04,
        "risk_tolerance": 0.05,
    },
    "catalogue/mean-reversion-sigma": {
        "momentum_window": 0.12,
        "mean_reversion_factor": 0.06,
    },
    "catalogue/carry-overlay": {
        "carry_tilt": 0.07,
        "macro_overlay_weight": 0.07,
    },
    "catalogue/vol-breakout": {
        "vol_breakout_zscore": 0.05,
        "volatility_threshold": 0.08,
    },
    "catalogue/liquidity-harvest": {
        "microstructure_edge": 0.09,
        "liquidity_bias": 0.08,
    },
    "catalogue/macro-fusion": {
        "macro_overlay_weight": 0.07,
        "event_risk_budget": 0.08,
    },
}


def _template_from_entry(entry: CatalogueEntry) -> GenomeSeedTemplate:
    parameter_jitter = dict(
        _CATALOGUE_JITTER_OVERRIDES.get(entry.identifier)
        or _CATALOGUE_JITTER_OVERRIDES.get(entry.name, {})
    )
    return GenomeSeedTemplate(
        name=entry.name,
        species=entry.species,
        base_parameters=dict(entry.parameters),
        parameter_jitter=parameter_jitter,
        parent_ids=tuple(entry.parent_ids),
        mutation_history=tuple(entry.mutation_history),
        performance_metrics=dict(entry.performance_metrics),
        tags=tuple(entry.tags),
        catalogue_entry_id=entry.identifier,
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _coerce_float_mapping(payload: Mapping[str, Any] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(payload, Mapping):
        return result
    for key, value in payload.items():
        try:
            result[str(key)] = float(value)
        except Exception:
            if isinstance(value, bool):
                result[str(key)] = 1.0 if value else 0.0
            else:
                continue
    return result


def _derive_jitter_from_bounds(
    key: str,
    base_value: float,
    bounds: Mapping[str, Any] | None,
) -> float | None:
    if not isinstance(bounds, Mapping):
        return None
    raw = bounds.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 2:
        return None
    try:
        low = float(raw[0])
        high = float(raw[1])
    except Exception:
        return None
    span = abs(high - low)
    if span <= 0.0:
        return None
    reference = abs(base_value)
    if reference <= 1e-6:
        reference = max(abs(low), abs(high), 1.0)
    jitter = span / max(reference, 1.0)
    # Clamp to a sensible percentage band to avoid explosive variance
    jitter = min(0.5, max(0.02, jitter * 0.15))
    return jitter


def _extract_manifest_tags(
    experiment: str,
    seed_value: Any,
    dataset_info: Mapping[str, Any] | None,
    genome_payload: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    tags: list[str] = [f"experiment:{experiment}"]
    if seed_value is not None:
        tags.append(f"seed:{seed_value}")
    if isinstance(dataset_info, Mapping):
        dataset_name = dataset_info.get("name")
        if dataset_name:
            tags.append(f"dataset:{dataset_name}")
    if isinstance(genome_payload, Mapping):
        for key, value in genome_payload.items():
            if isinstance(value, bool):
                tags.append(f"{key}:{'on' if value else 'off'}")
    return tuple(tags)


def _extract_manifest_mutations(
    experiment: str,
    seed_value: Any,
    dataset_info: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    mutation_history: list[str] = [f"artifact:{experiment}"]
    if seed_value is not None:
        mutation_history.append(f"seed:{seed_value}")
    if isinstance(dataset_info, Mapping):
        dataset_name = dataset_info.get("name")
        if dataset_name:
            mutation_history.append(f"dataset:{dataset_name}")
    return tuple(mutation_history)


def _extract_manifest_parents(
    experiment: str,
    leaderboard: Sequence[Mapping[str, Any]] | None,
) -> tuple[str, ...]:
    parents: list[str] = []
    if leaderboard:
        for entry in leaderboard[:2]:
            if not isinstance(entry, Mapping):
                continue
            generation = entry.get("generation")
            if generation is None:
                continue
            parents.append(f"{experiment}-gen-{generation}")
    return tuple(parents)


@lru_cache(maxsize=1)
def load_experiment_seed_templates(
    artifacts_dir: Path | None = None,
) -> tuple[GenomeSeedTemplate, ...]:
    """Load genome seed templates from recorded evolution experiment manifests."""

    if artifacts_dir is None:
        artifacts_dir = _project_root() / "artifacts" / "evolution"

    templates: list[GenomeSeedTemplate] = []
    if not artifacts_dir.exists():
        return tuple()

    for manifest_path in sorted(artifacts_dir.glob("*/manifest.json")):
        try:
            content = manifest_path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            manifest = json.loads(content)
        except json.JSONDecodeError:
            continue

        best_genome = manifest.get("best_genome")
        if not isinstance(best_genome, Mapping):
            continue

        parameters = _coerce_float_mapping(best_genome)
        if not parameters:
            continue

        bounds = None
        config = manifest.get("config")
        if isinstance(config, Mapping):
            maybe_bounds = config.get("bounds")
            if isinstance(maybe_bounds, Mapping):
                bounds = maybe_bounds

        parameter_jitter: dict[str, float] = {}
        for key, value in parameters.items():
            jitter = _derive_jitter_from_bounds(key, value, bounds)
            if jitter is not None:
                parameter_jitter[key] = jitter

        best_metrics = manifest.get("best_metrics") if isinstance(manifest, Mapping) else None
        metrics = _coerce_float_mapping(best_metrics if isinstance(best_metrics, Mapping) else None)

        experiment = str(manifest.get("experiment") or manifest_path.parent.name)
        seed_value = manifest.get("seed")
        dataset_info = manifest.get("dataset") if isinstance(manifest, Mapping) else None
        leaderboard = manifest.get("leaderboard")
        leaderboard_seq: Sequence[Mapping[str, Any]] | None
        if isinstance(leaderboard, Sequence) and not isinstance(leaderboard, (str, bytes)):
            leaderboard_seq = leaderboard
        else:
            leaderboard_seq = None

        parents = _extract_manifest_parents(experiment, leaderboard_seq)
        mutation_history = _extract_manifest_mutations(experiment, seed_value, dataset_info)
        tags = _extract_manifest_tags(experiment, seed_value, dataset_info, best_genome)

        catalogue_entry_id = f"artifact/{experiment}"
        if seed_value is not None:
            catalogue_entry_id = f"{catalogue_entry_id}/seed-{seed_value}"

        template = GenomeSeedTemplate(
            name=f"{experiment} seed {seed_value}" if seed_value is not None else experiment,
            species=str(manifest.get("species") or experiment),
            base_parameters=parameters,
            parameter_jitter=parameter_jitter,
            parent_ids=parents,
            mutation_history=mutation_history,
            performance_metrics=metrics,
            tags=tags,
            catalogue_entry_id=catalogue_entry_id,
        )
        templates.append(template)

    return tuple(templates)


@lru_cache(maxsize=1)
def _load_catalogue_templates() -> tuple[GenomeSeedTemplate, ...]:
    templates: list[GenomeSeedTemplate] = []
    try:
        catalogue: GenomeCatalogue = load_default_catalogue()
    except Exception:
        catalogue_entries: tuple[GenomeSeedTemplate, ...] = tuple()
    else:
        catalogue_entries = tuple(_template_from_entry(entry) for entry in catalogue.entries)
        templates.extend(catalogue_entries)

    experiment_templates = load_experiment_seed_templates()
    if experiment_templates:
        templates.extend(experiment_templates)

    if templates:
        return tuple(templates)
    return _FALLBACK_TEMPLATES


_FALLBACK_TEMPLATES: tuple[GenomeSeedTemplate, ...] = (
    GenomeSeedTemplate(
        name="Trend Surfer Alpha",
        species="trend_rider",
        base_parameters={
            "risk_tolerance": 0.45,
            "position_size_factor": 0.036,
            "stop_loss_factor": 0.018,
            "take_profit_factor": 0.072,
            "trend_sensitivity": 0.82,
            "volatility_threshold": 0.006,
            "correlation_threshold": 0.42,
            "momentum_window": 48.0,
            "mean_reversion_factor": 0.18,
            "market_regime_sensitivity": 0.62,
            "liquidity_bias": 0.31,
        },
        parameter_jitter={
            "momentum_window": 0.08,
            "trend_sensitivity": 0.04,
            "risk_tolerance": 0.05,
        },
        parent_ids=("desk-trend-2019",),
        mutation_history=("g0:seed:trend-alpha", "g0:mutation:macro-overlay"),
        performance_metrics={
            "sharpe_ratio": 1.34,
            "cagr": 0.19,
            "max_drawdown": -0.11,
            "hit_rate": 0.57,
        },
        tags=("fx", "trend", "institutional"),
        catalogue_entry_id="catalogue/trend-alpha",
    ),
    GenomeSeedTemplate(
        name="Sigma Reversion",
        species="mean_reversion",
        base_parameters={
            "risk_tolerance": 0.38,
            "position_size_factor": 0.028,
            "stop_loss_factor": 0.012,
            "take_profit_factor": 0.041,
            "trend_sensitivity": 0.34,
            "volatility_threshold": 0.0045,
            "correlation_threshold": 0.55,
            "momentum_window": 14.0,
            "mean_reversion_factor": 0.74,
            "market_regime_sensitivity": 0.47,
            "liquidity_bias": 0.24,
            "macro_overlay_weight": 0.36,
        },
        parameter_jitter={
            "momentum_window": 0.12,
            "mean_reversion_factor": 0.06,
        },
        parent_ids=("desk-meanrev-2021",),
        mutation_history=("g0:seed:mean-reversion", "g1:mutation:drawdown-tuning"),
        performance_metrics={
            "sharpe_ratio": 1.12,
            "cagr": 0.14,
            "max_drawdown": -0.08,
            "hit_rate": 0.63,
        },
        tags=("fx", "mean-reversion", "institutional"),
        catalogue_entry_id="catalogue/mean-reversion-sigma",
    ),
    GenomeSeedTemplate(
        name="Carry Overlay",
        species="carry_overlay",
        base_parameters={
            "risk_tolerance": 0.41,
            "position_size_factor": 0.033,
            "stop_loss_factor": 0.02,
            "take_profit_factor": 0.066,
            "trend_sensitivity": 0.58,
            "volatility_threshold": 0.0052,
            "correlation_threshold": 0.47,
            "momentum_window": 32.0,
            "mean_reversion_factor": 0.29,
            "market_regime_sensitivity": 0.54,
            "carry_tilt": 0.61,
            "macro_overlay_weight": 0.42,
        },
        parameter_jitter={
            "carry_tilt": 0.07,
            "macro_overlay_weight": 0.07,
        },
        parent_ids=("desk-carry-2020", "desk-carry-2022"),
        mutation_history=("g0:seed:carry", "g2:mutation:regime-split"),
        performance_metrics={
            "sharpe_ratio": 1.27,
            "cagr": 0.17,
            "max_drawdown": -0.09,
            "hit_rate": 0.59,
        },
        tags=("carry", "macro", "multi-asset"),
        catalogue_entry_id="catalogue/carry-overlay",
    ),
    GenomeSeedTemplate(
        name="Volatility Breakout",
        species="volatility_arb",
        base_parameters={
            "risk_tolerance": 0.52,
            "position_size_factor": 0.027,
            "stop_loss_factor": 0.015,
            "take_profit_factor": 0.05,
            "trend_sensitivity": 0.49,
            "volatility_threshold": 0.0078,
            "correlation_threshold": 0.38,
            "momentum_window": 21.0,
            "mean_reversion_factor": 0.33,
            "market_regime_sensitivity": 0.66,
            "vol_breakout_zscore": 2.4,
            "liquidity_bias": 0.27,
        },
        parameter_jitter={
            "vol_breakout_zscore": 0.05,
            "volatility_threshold": 0.08,
        },
        parent_ids=("desk-vol-2018",),
        mutation_history=("g0:seed:vol-breakout", "g3:mutation:vol-threshold"),
        performance_metrics={
            "sharpe_ratio": 1.41,
            "cagr": 0.21,
            "max_drawdown": -0.12,
            "hit_rate": 0.53,
        },
        tags=("volatility", "swing", "macro"),
        catalogue_entry_id="catalogue/vol-breakout",
    ),
    GenomeSeedTemplate(
        name="Liquidity Harvest",
        species="liquidity_scalper",
        base_parameters={
            "risk_tolerance": 0.33,
            "position_size_factor": 0.022,
            "stop_loss_factor": 0.009,
            "take_profit_factor": 0.027,
            "trend_sensitivity": 0.27,
            "volatility_threshold": 0.0035,
            "correlation_threshold": 0.62,
            "momentum_window": 9.0,
            "mean_reversion_factor": 0.69,
            "market_regime_sensitivity": 0.39,
            "liquidity_bias": 0.48,
            "microstructure_edge": 0.73,
        },
        parameter_jitter={
            "microstructure_edge": 0.09,
            "liquidity_bias": 0.08,
        },
        parent_ids=("desk-liq-2022",),
        mutation_history=("g0:seed:liquidity", "g1:mutation:microstructure"),
        performance_metrics={
            "sharpe_ratio": 1.18,
            "cagr": 0.12,
            "max_drawdown": -0.07,
            "hit_rate": 0.66,
        },
        tags=("liquidity", "high-frequency", "fx"),
        catalogue_entry_id="catalogue/liquidity-harvest",
    ),
    GenomeSeedTemplate(
        name="Macro Fusion",
        species="macro_overlay",
        base_parameters={
            "risk_tolerance": 0.47,
            "position_size_factor": 0.031,
            "stop_loss_factor": 0.019,
            "take_profit_factor": 0.058,
            "trend_sensitivity": 0.52,
            "volatility_threshold": 0.0063,
            "correlation_threshold": 0.49,
            "momentum_window": 26.0,
            "mean_reversion_factor": 0.41,
            "market_regime_sensitivity": 0.71,
            "macro_overlay_weight": 0.55,
            "event_risk_budget": 0.44,
        },
        parameter_jitter={
            "macro_overlay_weight": 0.07,
            "event_risk_budget": 0.08,
        },
        parent_ids=("desk-macro-2020", "desk-macro-2023"),
        mutation_history=("g0:seed:macro", "g2:mutation:event-hedge"),
        performance_metrics={
            "sharpe_ratio": 1.25,
            "cagr": 0.18,
            "max_drawdown": -0.1,
            "hit_rate": 0.58,
        },
        tags=("macro", "regime", "institutional"),
        catalogue_entry_id="catalogue/macro-fusion",
    ),
)


class RealisticGenomeSeeder:
    """Sampler producing catalogue-inspired genome seeds with jitter."""

    def __init__(
        self,
        templates: Sequence[GenomeSeedTemplate] | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        if templates is not None:
            seeds = tuple(templates)
        else:
            seeds = _load_catalogue_templates()
        if not seeds:
            raise ValueError("At least one genome seed template is required")
        self._templates = seeds
        self._rng = rng if rng is not None else random.Random()
        self._cursor = 0

    @property
    def templates(self) -> tuple[GenomeSeedTemplate, ...]:
        return self._templates

    def sample(self) -> GenomeSeed:
        template = self._templates[self._cursor % len(self._templates)]
        self._cursor += 1
        return template.spawn(self._rng)

    def reset(self) -> None:
        self._cursor = 0


__all__ = [
    "apply_seed_to_genome",
    "summarize_seed_metadata",
    "GenomeSeed",
    "GenomeSeedTemplate",
    "RealisticGenomeSeeder",
    "load_experiment_seed_templates",
]
