"""Suggestion post-processing utilities for TRM outputs."""

from __future__ import annotations

import datetime as dt
import hashlib
from typing import Iterable, Mapping, Sequence

from .config import RIMRuntimeConfig
from .types import DecisionDiaryEntry, RIMInputBatch, StrategyEncoding, StrategyInference

_SCHEMA_VERSION = "rim.v1"
_FLAG_THRESHOLD = 0.65
_EXPERIMENT_THRESHOLD = 0.55
_WEIGHT_THRESHOLD = 0.03


def build_suggestions(
    batch: RIMInputBatch,
    encodings: Sequence[StrategyEncoding],
    inferences: Iterable[StrategyInference],
    config: RIMRuntimeConfig,
    *,
    model_hash: str,
    config_hash: str,
) -> list[dict[str, object]]:
    now = dt.datetime.utcnow().replace(microsecond=0, tzinfo=dt.timezone.utc)
    lookup = {encoding.strategy_id: encoding for encoding in encodings}
    regime_lookup = _collect_regimes(batch.entries)
    suggestions: list[dict[str, object]] = []
    for inference in sorted(inferences, key=lambda item: -item.confidence):
        if len(suggestions) >= config.suggestion_cap:
            break
        encoding = lookup.get(inference.strategy_id)
        if encoding is None:
            continue
        suggestion = _materialise_suggestion(
            batch=batch,
            encoding=encoding,
            inference=inference,
            config=config,
            model_hash=model_hash,
            config_hash=config_hash,
            created_at=now,
            regime_lookup=regime_lookup,
        )
        if suggestion is not None:
            suggestions.append(suggestion)
    return suggestions


def _materialise_suggestion(
    *,
    batch: RIMInputBatch,
    encoding: StrategyEncoding,
    inference: StrategyInference,
    config: RIMRuntimeConfig,
    model_hash: str,
    config_hash: str,
    created_at: dt.datetime,
    regime_lookup: Mapping[str, tuple[str, ...]],
) -> dict[str, object] | None:
    confidence = round(float(inference.confidence), 2)
    if confidence < config.confidence_floor:
        return None

    suggestion_type: str
    payload: dict[str, object]
    stats = encoding.stats

    if inference.flag_probability >= _FLAG_THRESHOLD:
        suggestion_type = "STRATEGY_FLAG"
        payload = {
            "strategy_id": encoding.strategy_id,
            "reason": "elevated_risk",
            "risk_rate": round(stats.risk_rate, 2),
            "drawdown_ratio": round(stats.drawdown_ratio, 2),
            "recommended_action": "Pause strategy pending governance review",
        }
        rationale = (
            f"{encoding.strategy_id} recorded risk rate {stats.risk_rate:.2f} and drawdown ratio"
            f" {stats.drawdown_ratio:.2f}; flag raised for governance gate."
        )
    elif abs(inference.weight_delta) >= _WEIGHT_THRESHOLD:
        suggestion_type = "WEIGHT_ADJUST"
        payload = {
            "strategy_id": encoding.strategy_id,
            "proposed_weight_delta": round(inference.weight_delta, 3),
            "window_minutes": batch.window.minutes,
            "confidence": confidence,
        }
        direction = "increase" if inference.weight_delta > 0 else "reduce"
        rationale = (
            f"Model recommends {direction} weight by {abs(inference.weight_delta):.3f} based on"
            f" mean pnl {stats.mean_pnl:.2f} and win rate {stats.win_rate:.2%}."
        )
    elif inference.experiment_probability >= _EXPERIMENT_THRESHOLD:
        suggestion_type = "EXPERIMENT_PROPOSAL"
        experiment_name = _select_experiment(encoding)
        payload = {
            "hypothesis": f"Evaluate {experiment_name} adjustments",
            "strategy_candidates": [encoding.strategy_id],
            "duration_minutes": max(60, batch.window.minutes // 2),
            "experiment_id": experiment_name,
        }
        rationale = (
            f"Volatility {stats.volatility_mean:.2f} and pnl trend {stats.pnl_trend:.2f} support"
            f" experiment '{experiment_name}'."
        )
    else:
        return None

    audit_ids = encoding.audit_entry_hashes or (batch.input_hash,)
    suggestion_id = _build_suggestion_id(batch.input_hash, encoding.strategy_id, suggestion_type)
    target_strategy_ids = _resolve_target_strategy_ids(suggestion_type, payload, encoding.strategy_id)
    affected_regimes = _resolve_affected_regimes(regime_lookup, target_strategy_ids)
    evidence_payload = _build_evidence_payload(
        batch=batch,
        audit_ids=audit_ids,
        target_strategy_ids=target_strategy_ids,
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "input_hash": batch.input_hash,
        "model_hash": model_hash,
        "config_hash": config_hash,
        "suggestion_id": suggestion_id,
        "type": suggestion_type,
        "payload": payload,
        "confidence": confidence,
        "rationale": rationale,
        "audit_ids": list(audit_ids),
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "affected_regimes": list(affected_regimes),
        "evidence": evidence_payload,
    }


def _collect_regimes(entries: Iterable[DecisionDiaryEntry]) -> Mapping[str, tuple[str, ...]]:
    regime_map: dict[str, list[str]] = {}
    for entry in entries:
        strategy_id = entry.strategy_id
        regime_map.setdefault(strategy_id, [])
        regime_value = entry.regime
        if regime_value and regime_value not in regime_map[strategy_id]:
            regime_map[strategy_id].append(regime_value)
    return {key: tuple(values) for key, values in regime_map.items()}


def _resolve_target_strategy_ids(
    suggestion_type: str,
    payload: Mapping[str, object],
    default_strategy_id: str,
) -> tuple[str, ...]:
    if suggestion_type == "EXPERIMENT_PROPOSAL":
        candidates = payload.get("strategy_candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            deduped = _dedupe_preserve_order(str(candidate) for candidate in candidates if candidate)
            if deduped:
                return deduped
    strategy_id = payload.get("strategy_id")
    if isinstance(strategy_id, str) and strategy_id:
        return (strategy_id,)
    return (default_strategy_id,)


def _resolve_affected_regimes(
    regime_lookup: Mapping[str, tuple[str, ...]],
    strategy_ids: Sequence[str],
) -> tuple[str, ...]:
    regimes: list[str] = []
    for strategy_id in strategy_ids:
        for regime in regime_lookup.get(strategy_id, ()):  # preserve discovery order
            if regime not in regimes:
                regimes.append(regime)
    if regimes:
        return tuple(regimes)
    return ("unknown",)


def _build_evidence_payload(
    *,
    batch: RIMInputBatch,
    audit_ids: Sequence[str],
    target_strategy_ids: Sequence[str],
) -> dict[str, object]:
    window = batch.window
    evidence: dict[str, object] = {
        "input_hash": batch.input_hash,
        "audit_ids": list(audit_ids),
        "target_strategy_ids": list(_dedupe_preserve_order(target_strategy_ids)),
        "window": {
            "start": _format_timestamp(window.start),
            "end": _format_timestamp(window.end),
            "minutes": window.minutes,
        },
    }
    if batch.source_path is not None:
        evidence["diary_source"] = str(batch.source_path)
    return evidence


def _format_timestamp(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _dedupe_preserve_order(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _select_experiment(encoding: StrategyEncoding) -> str:
    stats = encoding.stats
    if stats.volatility_mean > 0.3:
        return "volatility_guardrail"
    if stats.risk_rate > 0.5:
        return "risk_pressure_backtest"
    if stats.win_rate < 0.45:
        return "alpha_refresh"
    return "stability_probe"


def _build_suggestion_id(input_hash: str, strategy_id: str, suggestion_type: str) -> str:
    digest = hashlib.sha1(f"{input_hash}:{strategy_id}:{suggestion_type}".encode("utf-8")).hexdigest()
    return f"rim-{digest[:16]}"


__all__ = ["build_suggestions"]
