"""Utilities to promote GA experiment champions into the strategy registry.

This module fulfils the High-Impact Roadmap item that calls for wiring
offline GA experiment outputs into the governance registry behind a
feature flag. The helpers are intentionally conservative: they only act
when the flag is enabled, degrade gracefully when manifests are
incomplete, and default to *not* touching the registry to protect
paper-trading runs from accidental churn.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from src.evolution.experiments.ma_crossover_ga import MovingAverageGenome
from src.governance.strategy_registry import StrategyRegistry

FEATURE_FLAG_ENV = "EVOLUTION_PROMOTE_FROM_LAB"
REGISTRY_PATH_ENV = "EVOLUTION_STRATEGY_REGISTRY_PATH"

logger = logging.getLogger(__name__)

__all__ = ["maybe_promote_best_genome", "FEATURE_FLAG_ENV", "REGISTRY_PATH_ENV"]


def _flag_enabled(flag_env: str = FEATURE_FLAG_ENV) -> bool:
    raw = os.environ.get(flag_env)
    if raw is None:
        return False
    normalised = raw.strip().lower()
    return normalised in {"1", "true", "yes", "on", "enable", "enabled"}


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "on"}:
            return True
        if normalised in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalise_dataset(dataset: Any) -> dict[str, Any] | None:
    if not isinstance(dataset, Mapping):
        return None
    name = dataset.get("name")
    metadata = dataset.get("metadata")
    payload: dict[str, Any] = {}
    if isinstance(name, str) and name:
        payload["name"] = name
    if isinstance(metadata, Mapping):
        payload["metadata"] = dict(metadata)
    return payload or None


def _resolve_registry_path(path: str | os.PathLike[str] | None) -> Path:
    candidate = path or os.environ.get(REGISTRY_PATH_ENV)
    resolved = Path(candidate) if candidate else Path.cwd() / "governance.db"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _build_fitness_report(
    metrics: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    report = {
        "fitness_score": _coerce_float(
            metrics.get("fitness"),
            default=_coerce_float(metrics.get("fitness_score"), 0.0),
        ),
        "sharpe_ratio": _coerce_float(
            metrics.get("sharpe"),
            default=_coerce_float(metrics.get("sharpe_ratio"), 0.0),
        ),
        "sortino_ratio": _coerce_float(
            metrics.get("sortino"),
            default=_coerce_float(metrics.get("sortino_ratio"), 0.0),
        ),
        "max_drawdown": _coerce_float(metrics.get("max_drawdown"), 0.0),
        "total_return": _coerce_float(metrics.get("total_return"), 0.0),
    }

    dataset_payload = _normalise_dataset(manifest.get("dataset"))
    lab_metadata: dict[str, Any] = {
        "experiment": manifest.get("experiment"),
        "generated_at": manifest.get("generated_at"),
        "seed": manifest.get("seed"),
    }
    if dataset_payload:
        lab_metadata["dataset"] = dataset_payload
    config_section = manifest.get("config")
    if isinstance(config_section, Mapping):
        lab_metadata["config"] = asdict(config_section) if hasattr(config_section, "__dataclass_fields__") else dict(config_section)
    elif config_section is not None:
        lab_metadata["config"] = config_section
    replay = manifest.get("replay")
    if isinstance(replay, Mapping):
        lab_metadata["replay"] = dict(replay)
    code_version = manifest.get("code_version")
    if code_version:
        lab_metadata["code_version"] = code_version
    notes = manifest.get("notes") or manifest.get("manifest_notes")
    if notes:
        lab_metadata["notes"] = notes

    report["metadata"] = {"evolution_lab": lab_metadata}
    return report


def _build_provenance(
    manifest: Mapping[str, Any], genome_id: str, fitness_report: Mapping[str, Any]
) -> Mapping[str, Any]:
    dataset_payload = _normalise_dataset(manifest.get("dataset"))
    metrics = fitness_report.copy()
    metrics.pop("metadata", None)

    entry_metadata: dict[str, Any] = {
        "best_metrics": metrics,
    }
    if dataset_payload:
        entry_metadata["dataset"] = dataset_payload

    return {
        "seed_source": "evolution_lab",
        "catalogue": {
            "name": str(manifest.get("experiment", "evolution_lab")),
            "version": str(manifest.get("code_version") or "unversioned"),
            "seeded_at": time.time(),
            "metadata": {
                "seed": manifest.get("seed"),
                "generated_at": manifest.get("generated_at"),
            },
        },
        "entry": {
            "id": genome_id,
            "name": f"{manifest.get('experiment', 'experiment')} champion",
            "metadata": entry_metadata,
        },
    }


def maybe_promote_best_genome(
    manifest: Mapping[str, Any],
    *,
    registry: StrategyRegistry | None = None,
    flag_env: str = FEATURE_FLAG_ENV,
    registry_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Conditionally promote the manifest's best genome into the registry."""

    if not _flag_enabled(flag_env):
        return False

    best_genome_payload = manifest.get("best_genome")
    best_metrics_payload = manifest.get("best_metrics")

    if not isinstance(best_genome_payload, Mapping) or not isinstance(
        best_metrics_payload, Mapping
    ):
        logger.warning("Promotion flag enabled but manifest lacks best genome/metrics")
        return False

    try:
        genome_model = MovingAverageGenome(
            short_window=int(best_genome_payload["short_window"]),
            long_window=int(best_genome_payload["long_window"]),
            risk_fraction=float(best_genome_payload["risk_fraction"]),
            use_var_guard=_coerce_bool(best_genome_payload.get("use_var_guard", True)),
            use_drawdown_guard=_coerce_bool(
                best_genome_payload.get("use_drawdown_guard", True)
            ),
        )
    except Exception as exc:
        logger.error("Failed to build MovingAverageGenome from manifest: %s", exc)
        return False

    identifier = best_genome_payload.get("id")
    genome = genome_model.to_decision_genome(
        identifier=str(identifier) if isinstance(identifier, str) and identifier else None
    )

    fitness_report = _build_fitness_report(best_metrics_payload, manifest)
    provenance = _build_provenance(manifest, genome.id, fitness_report)

    created_registry = False
    registry_obj = registry
    if registry_obj is None:
        registry_path_resolved = _resolve_registry_path(registry_path)
        registry_obj = StrategyRegistry(str(registry_path_resolved))
        created_registry = True

    try:
        registered = registry_obj.register_champion(
            genome,
            dict(fitness_report),
            provenance=provenance,
        )
    except Exception as exc:
        logger.error("Failed to register champion genome: %s", exc)
        registered = False
    finally:
        if created_registry:
            with suppress(Exception):
                registry_obj.conn.close()

    return registered
