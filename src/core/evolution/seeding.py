"""Realistic genome seeding helpers for the evolution engine.

The roadmap calls for replacing the naive random genome initialisation with
seeds that reflect the institutional strategy catalogue described in the EMP
Encyclopedia.  This module exposes a lightweight sampler that produces
catalogue-inspired genomes with jittered parameters, lineage metadata, and
performance fingerprints so lineage telemetry can trace their provenance even
before adaptive runs are enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Mapping, MutableMapping, Sequence


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

    def metadata(self) -> dict[str, object]:
        payload: MutableMapping[str, object] = {
            "seed_name": self.name,
            "seed_tags": list(self.tags),
            "seed_species": self.species,
        }
        return dict(payload)


_DEFAULT_TEMPLATES: tuple[GenomeSeedTemplate, ...] = (
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
        seeds = tuple(templates) if templates is not None else _DEFAULT_TEMPLATES
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
    "GenomeSeed",
    "GenomeSeedTemplate",
    "RealisticGenomeSeeder",
]

