"""Promotion helpers for wiring GA champions into the strategy registry."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from src.evolution.experiments.ma_crossover_ga import MovingAverageGenome
from src.governance.strategy_registry import StrategyRegistry, StrategyStatus

__all__ = ["PromotionResult", "promote_ma_crossover_champion"]


@dataclass(slots=True)
class PromotionResult:
    """Outcome metadata for GA promotion attempts."""

    genome_id: str | None
    registered: bool
    feature_flag_name: str
    feature_flag_value: str | None
    target_status: StrategyStatus | None
    status_applied: bool


_STATUS_ALIASES: Mapping[str, StrategyStatus] = {
    "approved": StrategyStatus.APPROVED,
    "candidate": StrategyStatus.APPROVED,
    "paper": StrategyStatus.APPROVED,
    "paper_trading": StrategyStatus.APPROVED,
    "on": StrategyStatus.APPROVED,
    "true": StrategyStatus.APPROVED,
    "yes": StrategyStatus.APPROVED,
    "active": StrategyStatus.ACTIVE,
    "live": StrategyStatus.ACTIVE,
    "inactive": StrategyStatus.INACTIVE,
    "disabled": StrategyStatus.INACTIVE,
    "evolved": StrategyStatus.EVOLVED,
}

_FALSE_FLAG_VALUES = {"", "0", "false", "no", "off", "disable", "disabled"}


def _normalise_flag(flag_value: str | None) -> tuple[str | None, StrategyStatus | None]:
    if flag_value is None:
        return None, None
    normalized = flag_value.strip().lower()
    if normalized in _FALSE_FLAG_VALUES:
        return flag_value, None
    status = _STATUS_ALIASES.get(normalized)
    if status is not None:
        return flag_value, status
    return flag_value, None


def _load_manifest(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive guard
        raise ValueError("GA manifest must decode to a mapping")
    return payload


def _extract_genome(manifest: Mapping[str, Any]) -> MovingAverageGenome:
    genome_payload = manifest.get("best_genome")
    if not isinstance(genome_payload, Mapping):
        raise ValueError("Manifest missing 'best_genome' mapping")
    try:
        short_window = int(genome_payload.get("short_window"))
        long_window = int(genome_payload.get("long_window"))
        risk_fraction = float(genome_payload.get("risk_fraction"))
    except (TypeError, ValueError) as exc:  # pragma: no cover - validation
        raise ValueError("Manifest best_genome missing core parameters") from exc
    use_var_guard = bool(genome_payload.get("use_var_guard", True))
    use_drawdown_guard = bool(genome_payload.get("use_drawdown_guard", True))
    return MovingAverageGenome(
        short_window=short_window,
        long_window=long_window,
        risk_fraction=risk_fraction,
        use_var_guard=use_var_guard,
        use_drawdown_guard=use_drawdown_guard,
    )


def _build_genome_identifier(
    manifest: Mapping[str, Any], genome: MovingAverageGenome
) -> str:
    experiment = str(manifest.get("experiment") or "ma_crossover_ga")
    seed = manifest.get("seed")
    if seed is not None:
        seed_part = str(seed)
    else:
        seed_part = "seedless"
    window_part = f"{genome.short_window}-{genome.long_window}-{genome.risk_fraction:.3f}"
    return "::".join((experiment, seed_part, window_part))


def _build_fitness_report(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    feature_flag_name: str,
    feature_flag_value: str | None,
    target_status: StrategyStatus | None,
) -> dict[str, Any]:
    metrics = manifest.get("best_metrics") or {}
    dataset = manifest.get("dataset")
    config_section = manifest.get("config")
    replay = manifest.get("replay")
    leaderboard = manifest.get("leaderboard")

    metadata: MutableMapping[str, Any] = {
        "source": "ga_manifest",
        "manifest_path": str(manifest_path),
        "experiment": manifest.get("experiment"),
        "seed": manifest.get("seed"),
        "feature_flag": {
            "name": feature_flag_name,
            "value": feature_flag_value,
            "target_status": target_status.value if target_status else None,
        },
    }

    if isinstance(dataset, Mapping):
        metadata["dataset"] = {
            "name": dataset.get("name"),
            "metadata": dict(dataset.get("metadata") or {}),
        }
    if isinstance(config_section, Mapping):
        metadata["config"] = {
            "population_size": config_section.get("population_size"),
            "generations": config_section.get("generations"),
            "elite_count": config_section.get("elite_count"),
            "crossover_rate": config_section.get("crossover_rate"),
            "mutation_rate": config_section.get("mutation_rate"),
        }
    if isinstance(replay, Mapping):
        metadata["replay"] = {
            "command": replay.get("command"),
            "seed": replay.get("seed"),
        }
    if isinstance(leaderboard, list):
        metadata["leaderboard_generations"] = len(leaderboard)

    notes = manifest.get("notes")
    if notes:
        metadata["notes"] = notes

    metrics_payload = {
        "fitness_score": float(metrics.get("fitness", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "sharpe_ratio": float(metrics.get("sharpe", 0.0)),
        "total_return": float(metrics.get("total_return", 0.0)),
    }
    metrics_payload["metadata"] = dict(metadata)
    return metrics_payload


def promote_ma_crossover_champion(
    manifest_path: str | Path,
    registry: StrategyRegistry,
    *,
    feature_flag_name: str = "PAPER_TRADE_GA_MA_CROSSOVER",
    flag_override: str | None = None,
) -> PromotionResult:
    """Register the GA champion genome and apply feature-flag promotion."""

    path = Path(manifest_path)
    manifest = _load_manifest(path)
    genome = _extract_genome(manifest)
    genome_id = _build_genome_identifier(manifest, genome)

    flag_value_env = flag_override if flag_override is not None else os.getenv(feature_flag_name)
    flag_value, target_status = _normalise_flag(flag_value_env)

    decision_genome = genome.to_decision_genome(identifier=genome_id)
    champion_name = f"{manifest.get('experiment', 'GA Experiment')} champion"
    try:
        setattr(decision_genome, "name", champion_name)
    except Exception:  # pragma: no cover - legacy genome compatibility
        pass

    fitness_report = _build_fitness_report(
        manifest, path, feature_flag_name, flag_value, target_status
    )

    registered = registry.register_champion(
        decision_genome,
        fitness_report,
        provenance=None,
        status=StrategyStatus.EVOLVED,
    )

    status_applied = False
    if registered and target_status is not None and target_status != StrategyStatus.EVOLVED:
        status_applied = registry.update_strategy_status(genome_id, target_status.value)

    return PromotionResult(
        genome_id=genome_id,
        registered=registered,
        feature_flag_name=feature_flag_name,
        feature_flag_value=flag_value,
        target_status=target_status,
        status_applied=status_applied,
    )
