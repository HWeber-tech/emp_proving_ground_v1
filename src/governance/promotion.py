"""Feature-flagged promotion utilities for evolution experiment champions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from src.governance.strategy_registry import (
    StrategyRegistry,
    StrategyRegistryError,
    StrategyStatus,
)

try:  # Lazy import so the module remains light for callers that do not need GA
    from src.evolution.experiments.ma_crossover_ga import MovingAverageGenome
except Exception:  # pragma: no cover - imported lazily for optional dependency
    MovingAverageGenome = None  # type: ignore[assignment]

__all__ = [
    "PromotionFeatureFlags",
    "PromotionResult",
    "promote_manifest_to_registry",
]


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, *, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_status(raw: str | None, fallback: StrategyStatus) -> StrategyStatus:
    if raw is None:
        return fallback
    normalised = raw.strip().lower()
    for status in StrategyStatus:
        if status.value == normalised or status.name.lower() == normalised:
            return status
    return fallback


def _to_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


@dataclass(slots=True, frozen=True)
class PromotionFeatureFlags:
    """Feature switches controlling how GA champions are promoted."""

    register_enabled: bool = False
    auto_approve: bool = False
    target_status: StrategyStatus = StrategyStatus.EVOLVED
    min_fitness: float = 0.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "PromotionFeatureFlags":
        env = env or os.environ
        register_enabled = _parse_bool(env.get("EVOLUTION_PROMOTE_STRATEGIES"))
        auto_approve = _parse_bool(env.get("EVOLUTION_PROMOTE_AUTO_APPROVE"))
        default_status = StrategyStatus.APPROVED if auto_approve else StrategyStatus.EVOLVED
        target_status = _normalise_status(env.get("EVOLUTION_PROMOTE_TARGET_STATUS"), default_status)
        min_fitness = _parse_float(env.get("EVOLUTION_PROMOTE_MIN_FITNESS"), default=0.0)
        return cls(
            register_enabled=register_enabled,
            auto_approve=auto_approve,
            target_status=target_status,
            min_fitness=min_fitness,
        )

    def with_overrides(
        self,
        *,
        register_enabled: bool | None = None,
        auto_approve: bool | None = None,
        target_status: StrategyStatus | None = None,
        min_fitness: float | None = None,
    ) -> "PromotionFeatureFlags":
        """Return a copy with selective overrides for CLI tooling."""

        return replace(
            self,
            register_enabled=self.register_enabled if register_enabled is None else register_enabled,
            auto_approve=self.auto_approve if auto_approve is None else auto_approve,
            target_status=self.target_status if target_status is None else target_status,
            min_fitness=self.min_fitness if min_fitness is None else min_fitness,
        )


@dataclass(slots=True)
class PromotionResult:
    """Outcome of a promotion attempt."""

    genome_id: str | None
    registered: bool
    status_updated: bool
    skipped: bool
    reason: str | None = None
    fitness: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "genome_id": self.genome_id,
            "registered": self.registered,
            "status_updated": self.status_updated,
            "skipped": self.skipped,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.fitness is not None:
            payload["fitness"] = self.fitness
        return payload


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manifest not found at {path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Manifest at {path} is not valid JSON") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Manifest payload must be a mapping")
    return data


def _build_ma_genome(manifest: Mapping[str, Any]) -> tuple[Any, dict[str, Any], Mapping[str, Any]]:
    if MovingAverageGenome is None:  # pragma: no cover - import guard
        raise RuntimeError("MovingAverageGenome unavailable; install optional dependencies")

    genome_payload = manifest.get("best_genome")
    if not isinstance(genome_payload, Mapping):
        raise ValueError("Manifest missing 'best_genome' mapping")

    metrics = manifest.get("best_metrics") or {}
    if not isinstance(metrics, Mapping):
        raise ValueError("Manifest 'best_metrics' must be a mapping")

    try:
        genome = MovingAverageGenome(
            short_window=int(genome_payload.get("short_window", 0)),
            long_window=int(genome_payload.get("long_window", 0)),
            risk_fraction=float(genome_payload.get("risk_fraction", 0.0)),
            use_var_guard=bool(genome_payload.get("use_var_guard", True)),
            use_drawdown_guard=bool(genome_payload.get("use_drawdown_guard", True)),
        )
    except Exception as exc:
        raise ValueError(f"Invalid best_genome payload: {exc}") from exc

    identifier_parts = [
        str(manifest.get("experiment", "ma_crossover_ga")),
        str(manifest.get("seed", "seed")),
        f"s{genome.short_window}",
        f"l{genome.long_window}",
        f"r{int(round(genome.risk_fraction * 1000))}",
    ]
    identifier = "::".join(part for part in identifier_parts if part)
    decision_genome = genome.to_decision_genome(identifier=identifier)

    generation = manifest.get("generations")
    if generation is None:
        config = manifest.get("config")
        if isinstance(config, Mapping):
            generation = config.get("generations")
    try:
        decision_genome.generation = int(generation or 0)
    except Exception:  # pragma: no cover - guard for unexpected payloads
        decision_genome.generation = 0

    performance_metrics = {
        "fitness": float(metrics.get("fitness", 0.0)),
        "sharpe_ratio": float(metrics.get("sharpe", 0.0)),
        "sortino_ratio": float(metrics.get("sortino", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "total_return": float(metrics.get("total_return", 0.0)),
    }
    try:
        decision_genome.performance_metrics.update(performance_metrics)
    except Exception:  # pragma: no cover - defensive guard
        decision_genome.performance_metrics = performance_metrics  # type: ignore[assignment]

    metadata: MutableMapping[str, Any] = {
        "source": "evolution_lab",
        "experiment": manifest.get("experiment"),
        "seed": manifest.get("seed"),
        "dataset": manifest.get("dataset"),
    }
    try:
        setattr(decision_genome, "metadata", metadata)
    except Exception:  # pragma: no cover - optional attribute in dataclass
        pass

    return decision_genome, performance_metrics, genome_payload


def _build_fitness_report(
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[dict[str, Any], float]:
    fitness = float(metrics.get("fitness", 0.0))
    report = {
        "fitness_score": fitness,
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "sharpe_ratio": float(metrics.get("sharpe", 0.0)),
        "total_return": float(metrics.get("total_return", 0.0)),
        "volatility": float(metrics.get("volatility", 0.0)),
    }
    metadata: dict[str, Any] = {
        "experiment": manifest.get("experiment"),
        "generated_at": manifest.get("generated_at"),
        "seed": manifest.get("seed"),
        "manifest_path": str(manifest_path),
    }
    dataset = manifest.get("dataset")
    if dataset:
        metadata["dataset"] = dataset
    config = manifest.get("config")
    if config:
        metadata["config"] = config
    report["metadata"] = metadata
    return report, fitness


def _build_provenance(
    manifest: Mapping[str, Any],
    genome_id: str,
    genome_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    seeded_at = _to_timestamp(str(manifest.get("generated_at")))
    catalogue_metadata = {
        "experiment": manifest.get("experiment"),
        "seed": manifest.get("seed"),
        "config": manifest.get("config"),
    }
    provenance: dict[str, Any] = {
        "seed_source": "evolution_lab",
        "catalogue": {
            "name": str(manifest.get("experiment", "evolution_lab")),
            "version": str((manifest.get("config") or {}).get("generations", "")),
            "seeded_at": seeded_at,
            "metadata": catalogue_metadata,
        },
        "entry": {
            "id": genome_id,
            "name": f"{manifest.get('experiment', 'experiment')} champion",
            "metadata": {
                "best_genome": dict(genome_payload),
                "best_metrics": dict(manifest.get("best_metrics") or {}),
            },
        },
    }
    return provenance


def _load_gate_results(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Gate results not found at {path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Gate results at {path} are not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Gate results payload must be a mapping")
    return payload


def _summarise_gate_failures(payload: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    gates = payload.get("gates")
    if not isinstance(gates, Sequence):
        raise ValueError("Gate results payload missing 'gates' sequence")

    failing: list[str] = []
    for entry in gates:
        if not isinstance(entry, Mapping):
            continue
        passed = entry.get("passed")
        if passed is True:
            continue
        gate_id = entry.get("gate_id") or entry.get("id") or "gate"
        failing.append(str(gate_id))
    return (len(failing) == 0, tuple(failing))


def _rollback_on_gate_failure(registry: StrategyRegistry, genome_id: str) -> bool:
    existing = registry.get_strategy(genome_id)
    if existing is None:
        return False

    status_value = str(existing.get("status", "")).strip().lower()
    try:
        current_status = StrategyStatus(status_value)
    except ValueError:
        return False

    rollback_target: StrategyStatus | None = None
    if current_status is StrategyStatus.APPROVED_DEFAULT:
        rollback_target = StrategyStatus.APPROVED_FALLBACK
    elif current_status is StrategyStatus.APPROVED:
        rollback_target = StrategyStatus.EVOLVED
    elif current_status is StrategyStatus.ACTIVE:
        rollback_target = StrategyStatus.APPROVED
    else:
        return False

    try:
        return registry.update_strategy_status(genome_id, rollback_target)
    except StrategyRegistryError:
        return False


def promote_manifest_to_registry(
    manifest_path: str | Path,
    registry: StrategyRegistry,
    *,
    flags: PromotionFeatureFlags | None = None,
    gate_results_path: str | Path | None = None,
) -> PromotionResult:
    """Promote a GA champion described by ``manifest_path`` into the registry."""

    resolved_flags = flags or PromotionFeatureFlags.from_env()
    if not resolved_flags.register_enabled:
        return PromotionResult(
            genome_id=None,
            registered=False,
            status_updated=False,
            skipped=True,
            reason="feature_flag_disabled",
        )

    path = Path(manifest_path)
    try:
        manifest = _load_manifest(path)
    except Exception as exc:
        return PromotionResult(
            genome_id=None,
            registered=False,
            status_updated=False,
            skipped=True,
            reason=str(exc),
        )

    experiment = str(manifest.get("experiment", "")).strip()
    if experiment and experiment != "ma_crossover_ga":
        return PromotionResult(
            genome_id=None,
            registered=False,
            status_updated=False,
            skipped=True,
            reason=f"unsupported_experiment:{experiment}",
        )

    try:
        decision_genome, metrics, genome_payload = _build_ma_genome(manifest)
    except Exception as exc:
        return PromotionResult(
            genome_id=None,
            registered=False,
            status_updated=False,
            skipped=True,
            reason=str(exc),
        )

    report, fitness = _build_fitness_report(metrics, manifest, path)
    if fitness < resolved_flags.min_fitness:
        return PromotionResult(
            genome_id=decision_genome.id,
            registered=False,
            status_updated=False,
            skipped=True,
            reason="fitness_below_threshold",
            fitness=fitness,
        )

    provenance = _build_provenance(manifest, decision_genome.id, genome_payload)

    if gate_results_path is not None:
        gate_path = Path(gate_results_path)
        try:
            gate_payload = _load_gate_results(gate_path)
            gates_passed, failing_gates = _summarise_gate_failures(gate_payload)
        except (FileNotFoundError, ValueError) as exc:
            return PromotionResult(
                genome_id=decision_genome.id,
                registered=False,
                status_updated=False,
                skipped=True,
                reason=f"gate_results_error:{exc}",
                fitness=fitness,
            )
        if not gates_passed:
            failure_suffix = ",".join(failing_gates)
            gate_reason = "gate_regression"
            if failure_suffix:
                gate_reason = f"{gate_reason}:{failure_suffix}"
            rollback_applied = _rollback_on_gate_failure(registry, decision_genome.id)
            return PromotionResult(
                genome_id=decision_genome.id,
                registered=False,
                status_updated=rollback_applied,
                skipped=True,
                reason=gate_reason,
                fitness=fitness,
            )

    registered = registry.register_champion(decision_genome, dict(report), provenance=provenance)
    status_updated = False

    if registered and resolved_flags.auto_approve:
        target = resolved_flags.target_status.value
        if registry.update_strategy_status(decision_genome.id, target):
            status_updated = True

    return PromotionResult(
        genome_id=decision_genome.id,
        registered=registered,
        status_updated=status_updated,
        skipped=not registered,
        fitness=fitness,
        reason=None if registered else "registration_failed",
    )
