"""Curated genome catalogue derived from historical professional strategies.

This module exposes a small but representative catalogue of institutional
genomes so evolution runs can seed a meaningful population instead of relying on
fully random parameter draws.  The catalogue is intentionally lightweight—it
captures the parameter clusters and performance fingerprints surfaced in the
concept blueprint and roadmap briefs—yet it behaves like the future production
catalogue surface (metadata, sampling, telemetry hooks).

Context:
    * EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md outlines calibrated trend, carry,
      liquidity, macro, and volatility playbooks for the tier‑1 evolution loop.
    * The roadmap’s “Next” outcomes call for replacing the stub genome provider
      with a seeded catalogue so population management reflects real desks.

The helpers here keep the data and behaviour together so orchestration layers
can request catalogue-backed seeds, inspect metadata for telemetry, and fall
back to random seeds when the feature flag is disabled.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .models.genome import DecisionGenome

CataloguePayload = Mapping[str, object]


def _coerce_parameters(parameters: Mapping[str, float]) -> dict[str, float]:
    coerced: dict[str, float] = {}
    for key, value in parameters.items():
        try:
            coerced[str(key)] = float(value)
        except Exception:
            continue
    return coerced


def _coerce_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            snapshot[str(key)] = float(value)
        except Exception:
            continue
    return snapshot


@dataclass(frozen=True)
class CatalogueEntry:
    """Immutable template describing a calibrated genome."""

    identifier: str
    name: str
    species: str
    parameters: Mapping[str, float]
    performance_metrics: Mapping[str, float] = field(default_factory=dict)
    mutation_history: Sequence[str] = field(default_factory=tuple)
    parent_ids: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)
    generation: int = 0
    created_at: float | None = None

    def instantiate(self, *, instance_suffix: str | None = None) -> DecisionGenome:
        """Materialise the entry as a ``DecisionGenome`` instance."""

        performance_metrics = _coerce_metrics(self.performance_metrics)

        metadata_payload: dict[str, object] = {
            "seed_name": self.name,
            "seed_species": self.species,
            "seed_catalogue_id": self.identifier,
            "seed_source": "catalogue",
        }

        tags = [str(tag).strip() for tag in self.tags if str(tag).strip()]
        if tags:
            metadata_payload["seed_tags"] = tags

        parent_ids = [str(pid).strip() for pid in self.parent_ids if str(pid).strip()]
        if parent_ids:
            metadata_payload["seed_parent_ids"] = parent_ids

        mutation_history = [
            str(entry).strip() for entry in self.mutation_history if str(entry).strip()
        ]
        if mutation_history:
            metadata_payload["seed_mutation_history"] = mutation_history

        if performance_metrics:
            metadata_payload["seed_performance_metrics"] = dict(performance_metrics)

        payload: dict[str, object] = {
            "id": self.identifier
            if instance_suffix is None
            else f"{self.identifier}::{instance_suffix}",
            "parameters": _coerce_parameters(self.parameters),
            "fitness": None,
            "generation": max(0, int(self.generation)),
            "species_type": self.species,
            "parent_ids": list(self.parent_ids),
            "mutation_history": list(self.mutation_history),
            "performance_metrics": performance_metrics,
            "created_at": self.created_at or time.time(),
            "metadata": metadata_payload,
        }
        return DecisionGenome.from_dict(payload)

    def metadata(self) -> CataloguePayload:
        return {
            "id": self.identifier,
            "name": self.name,
            "species": self.species,
            "tags": list(self.tags),
            "performance_metrics": _coerce_metrics(self.performance_metrics),
            "generation": max(0, int(self.generation)),
        }


class GenomeCatalogue:
    """Container exposing metadata and sampling helpers for catalogue entries."""

    def __init__(
        self,
        entries: Iterable[CatalogueEntry],
        *,
        name: str,
        version: str,
        source_notes: Sequence[str] | None = None,
    ) -> None:
        self._entries: tuple[CatalogueEntry, ...] = tuple(entries)
        self._name = name
        self._version = version
        self._source_notes = tuple(source_notes or ())
        self._species_index: dict[str, list[CatalogueEntry]] = {}
        for entry in self._entries:
            self._species_index.setdefault(entry.species, []).append(entry)

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def entries(self) -> tuple[CatalogueEntry, ...]:
        return self._entries

    def species_counts(self) -> dict[str, int]:
        return {species: len(items) for species, items in self._species_index.items()}

    def metadata(self) -> CataloguePayload:
        species_counts = self.species_counts()
        return {
            "name": self._name,
            "version": self._version,
            "size": len(self._entries),
            "species": species_counts,
            "source_notes": list(self._source_notes),
        }

    def describe_entries(self) -> list[CataloguePayload]:
        return [entry.metadata() for entry in self._entries]

    def as_genomes(self) -> list[DecisionGenome]:
        return [entry.instantiate() for entry in self._entries]

    def sample(
        self,
        count: int,
        *,
        rng: random.Random | None = None,
        shuffle: bool = True,
    ) -> list[DecisionGenome]:
        if count <= 0 or not self._entries:
            return []

        generator = rng or random.Random()
        indices = list(range(len(self._entries)))
        if shuffle:
            generator.shuffle(indices)

        selections: list[DecisionGenome] = []
        cursor = 0
        iteration = 0
        while len(selections) < count:
            if cursor >= len(indices):
                cursor = 0
                iteration += 1
                if shuffle:
                    generator.shuffle(indices)
            entry = self._entries[indices[cursor]]
            cursor += 1
            suffix = f"{iteration}-{len(selections):03d}" if iteration else str(len(selections))
            selections.append(entry.instantiate(instance_suffix=suffix))
        return selections


def load_default_catalogue() -> GenomeCatalogue:
    """Return the curated institutional genome catalogue."""

    entries = (
        CatalogueEntry(
            identifier="catalogue/trend-alpha",
            name="Trend Surfer Alpha",
            species="trend_rider",
            parameters={
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
            performance_metrics={
                "sharpe_ratio": 1.34,
                "cagr": 0.19,
                "max_drawdown": -0.11,
                "hit_rate": 0.57,
            },
            mutation_history=("g0:seed:trend-alpha", "g0:mutation:macro-overlay"),
            parent_ids=("desk-trend-2019",),
            tags=("fx", "swing", "institutional"),
            generation=4,
        ),
        CatalogueEntry(
            identifier="catalogue/mean-reversion-sigma",
            name="Sigma Reversion",
            species="mean_reversion",
            parameters={
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
            performance_metrics={
                "sharpe_ratio": 1.12,
                "cagr": 0.14,
                "max_drawdown": -0.08,
                "hit_rate": 0.63,
            },
            mutation_history=("g0:seed:mean-reversion", "g1:mutation:drawdown-tuning"),
            parent_ids=("desk-meanrev-2021",),
            tags=("fx", "mean-reversion", "institutional"),
            generation=5,
        ),
        CatalogueEntry(
            identifier="catalogue/carry-overlay",
            name="Carry Overlay",
            species="carry_overlay",
            parameters={
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
            performance_metrics={
                "sharpe_ratio": 1.27,
                "cagr": 0.17,
                "max_drawdown": -0.09,
                "hit_rate": 0.59,
            },
            mutation_history=("g0:seed:carry", "g2:mutation:regime-split"),
            parent_ids=("desk-carry-2020", "desk-carry-2022"),
            tags=("carry", "macro", "multi-asset"),
            generation=6,
        ),
        CatalogueEntry(
            identifier="catalogue/vol-breakout",
            name="Volatility Breakout",
            species="volatility_arb",
            parameters={
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
            performance_metrics={
                "sharpe_ratio": 1.41,
                "cagr": 0.21,
                "max_drawdown": -0.12,
                "hit_rate": 0.53,
            },
            mutation_history=("g0:seed:vol-breakout", "g3:mutation:vol-threshold"),
            parent_ids=("desk-vol-2018",),
            tags=("volatility", "swing", "macro"),
            generation=7,
        ),
        CatalogueEntry(
            identifier="catalogue/liquidity-harvest",
            name="Liquidity Harvest",
            species="liquidity_scalper",
            parameters={
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
            performance_metrics={
                "sharpe_ratio": 1.18,
                "cagr": 0.12,
                "max_drawdown": -0.07,
                "hit_rate": 0.66,
            },
            mutation_history=("g0:seed:liquidity", "g1:mutation:microstructure"),
            parent_ids=("desk-liq-2022",),
            tags=("liquidity", "high-frequency", "fx"),
            generation=5,
        ),
        CatalogueEntry(
            identifier="catalogue/macro-fusion",
            name="Macro Fusion",
            species="macro_overlay",
            parameters={
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
            performance_metrics={
                "sharpe_ratio": 1.25,
                "cagr": 0.18,
                "max_drawdown": -0.1,
                "hit_rate": 0.58,
            },
            mutation_history=("g0:seed:macro", "g2:mutation:event-hedge"),
            parent_ids=("desk-macro-2020", "desk-macro-2023"),
            tags=("macro", "regime", "institutional"),
            generation=6,
        ),
    )

    notes = (
        "Blend of 2018-2024 FX desk calibrations across trend, carry, macro, and liquidity playbooks.",
        "Performance metrics scaled to paper-risk budgets for tier-1 institutional deployments.",
    )

    return GenomeCatalogue(
        entries, name="institutional_default", version="2025.09", source_notes=notes
    )


__all__ = [
    "CatalogueEntry",
    "GenomeCatalogue",
    "load_default_catalogue",
]
