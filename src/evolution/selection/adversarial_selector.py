"""Adversarial selector that validates strategies on recorded sensory data."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Mapping, Sequence

from src.evolution.evaluation.recorded_replay import (
    RecordedEvaluationResult,
    RecordedSensoryEvaluator,
    RecordedSensorySnapshot,
)
from src.evolution.evaluation.telemetry import (
    RecordedReplayTelemetrySnapshot,
    summarise_recorded_replay,
)
from src.sensory.lineage import SensorLineageRecorder

__all__ = ["AdversarialSelector"]

logger = logging.getLogger(__name__)


def _coerce_float_mapping(payload: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    cleaned: dict[str, float] = {}
    for key, value in payload.items():
        try:
            cleaned[str(key)] = float(value)  # type: ignore[arg-type]
        except Exception:
            continue
    return cleaned


def _extract_parameters(genome: object) -> dict[str, float]:
    if isinstance(genome, Mapping):
        return _coerce_float_mapping(genome)
    params = getattr(genome, "parameters", None)
    if isinstance(params, Mapping):
        return _coerce_float_mapping(params)
    return {}


def _extract_genome_id(genome: object) -> str:
    candidates = ("id", "identifier", "name", "genome_id")
    for attr in candidates:
        value = getattr(genome, attr, None)
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
    if isinstance(genome, Mapping):
        for attr in candidates:
            value = genome.get(attr)
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float)):
                return str(value)
    return f"genome-{id(genome)}"


class AdversarialSelector:
    """Select genomes by replaying recorded sensory data and scoring outcomes."""

    def __init__(
        self,
        *,
        snapshots: Sequence[RecordedSensorySnapshot] | None = None,
        evaluator: RecordedSensoryEvaluator | None = None,
        dataset_id: str | None = None,
        evaluation_id: str | None = None,
        lineage_recorder: SensorLineageRecorder | None = None,
        history_limit: int = 32,
    ) -> None:
        if evaluator is not None:
            self._evaluator = evaluator
        elif snapshots is not None:
            self._evaluator = RecordedSensoryEvaluator(snapshots)
        else:
            self._evaluator = None

        self._dataset_id = dataset_id
        self._evaluation_id = evaluation_id
        self._lineage_recorder = lineage_recorder
        self._history: deque[RecordedReplayTelemetrySnapshot] = deque(maxlen=max(1, history_limit))
        self._last_scores: dict[int, float] = {}
        self._last_snapshots: dict[int, RecordedReplayTelemetrySnapshot] = {}
        self._last_ranked: list[object] = []

    def select(self, population: Sequence[object] | None, k: int = 1) -> list[object]:
        if not population or k <= 0:
            self._last_ranked = []
            return []

        candidates = list(population)
        top_n = max(0, k)

        if self._evaluator is None:
            selection = candidates[:top_n]
            self._last_ranked = selection
            return selection

        evaluations: list[tuple[float, object]] = []

        for candidate in candidates:
            candidate_key = id(candidate)
            try:
                result = self._evaluator.evaluate(candidate)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Adversarial selection failed for %s: %s", candidate, exc)
                self._last_scores[candidate_key] = float("-inf")
                continue

            score = self._score(result)
            telemetry = self._build_telemetry(candidate, result)
            self._store_candidate_metrics(candidate_key, score, telemetry)
            evaluations.append((score, candidate))

        if not evaluations:
            selection = candidates[:top_n]
            self._last_ranked = selection
            return selection

        evaluations.sort(key=lambda item: item[0], reverse=True)
        ranked = [candidate for _, candidate in evaluations]
        selection = ranked[:top_n]
        if not selection:
            selection = candidates[:top_n]
        self._last_ranked = selection
        return selection

    def last_ranked(self) -> list[object]:
        return list(self._last_ranked)

    def score_for(self, genome: object) -> float | None:
        return self._last_scores.get(id(genome))

    def telemetry_for(
        self,
        genome: object,
        *,
        serialise: bool = False,
    ) -> RecordedReplayTelemetrySnapshot | dict[str, Any] | None:
        snapshot = self._last_snapshots.get(id(genome))
        if snapshot is None:
            return None
        if serialise:
            return snapshot.as_dict()
        return snapshot

    def evaluation_history(
        self,
        *,
        serialise: bool = False,
        limit: int | None = None,
    ) -> list[RecordedReplayTelemetrySnapshot] | list[dict[str, Any]]:
        if limit is not None and limit > 0:
            items = list(self._history)[-limit:]
        else:
            items = list(self._history)
        if serialise:
            return [snapshot.as_dict() for snapshot in items]
        return items

    def _build_telemetry(
        self,
        genome: object,
        result: RecordedEvaluationResult,
    ) -> RecordedReplayTelemetrySnapshot:
        genome_id = _extract_genome_id(genome)
        parameters = _extract_parameters(genome)
        metadata = {"selector": "adversarial"}
        return summarise_recorded_replay(
            result,
            genome_id=genome_id,
            dataset_id=self._dataset_id,
            evaluation_id=self._evaluation_id,
            parameters=parameters,
            metadata=metadata,
        )

    def _store_candidate_metrics(
        self,
        candidate_key: int,
        score: float,
        telemetry: RecordedReplayTelemetrySnapshot,
    ) -> None:
        self._last_scores[candidate_key] = float(score)
        self._last_snapshots[candidate_key] = telemetry
        self._history.append(telemetry)
        if self._lineage_recorder is not None:
            try:
                self._lineage_recorder.record(telemetry.lineage)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to record lineage for candidate %s", candidate_key)

    def _score(self, result: RecordedEvaluationResult) -> float:
        total_return = float(result.total_return)
        drawdown = float(result.max_drawdown)
        volatility = float(result.volatility)
        win_rate = float(result.win_rate)
        trades = int(result.trades)

        reward = total_return
        reward += max(0.0, win_rate - 0.5) * 0.4
        reward += min(trades, 25) * 0.01

        penalty = drawdown * 0.6 + volatility * 0.15
        if trades == 0:
            penalty += 0.2

        return reward - penalty
