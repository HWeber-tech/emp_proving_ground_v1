"""Mutation ledger for tracking evolutionary adaptation events.

The ledger persists JSON Lines entries capturing parameter mutation events,
fitness improvements during evolution cycles, and exploitability observations
emitted by the mini-league.  The intent is to provide an auditable timeline of
how adaptive components tweak strategy parameters, which champions improved the
fitness frontier, and how exploitability signals evolved over time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path("artifacts/evolution/mutation_ledger.jsonl")


@dataclass(slots=True)
class _LedgerRecord:
    """In-memory representation of a mutation ledger event."""

    event: str
    recorded_at: datetime
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "event": self.event,
            "recorded_at": self.recorded_at.isoformat(),
        }
        data.update(dict(self.payload))
        return data

    def to_json_entry(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "recorded_at": self.recorded_at.isoformat(),
            "payload": dict(self.payload),
        }


class MutationLedger:
    """Append-only JSONL ledger for mutation-related events."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_LEDGER_PATH
        self._lock = Lock()
        self._parameter_mutations: list[_LedgerRecord] = []
        self._fitness_improvements: list[_LedgerRecord] = []
        self._exploitability_results: list[_LedgerRecord] = []

    @property
    def path(self) -> Path:
        return self._path

    @property
    def parameter_mutations(self) -> tuple[_LedgerRecord, ...]:
        with self._lock:
            return tuple(self._parameter_mutations)

    @property
    def fitness_improvements(self) -> tuple[_LedgerRecord, ...]:
        with self._lock:
            return tuple(self._fitness_improvements)

    @property
    def exploitability_results(self) -> tuple[_LedgerRecord, ...]:
        with self._lock:
            return tuple(self._exploitability_results)

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the recorded ledger events."""

        with self._lock:
            return {
                "path": str(self._path),
                "parameter_mutations": [record.as_dict() for record in self._parameter_mutations],
                "fitness_improvements": [record.as_dict() for record in self._fitness_improvements],
                "exploitability_results": [record.as_dict() for record in self._exploitability_results],
            }

    def record_parameter_mutation(
        self,
        *,
        base_tactic_id: str,
        variant_id: str,
        parameter: str,
        original_value: Any,
        mutated_value: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a parameter mutation describing before/after values."""

        original = self._to_float(original_value)
        mutated = self._to_float(mutated_value)
        delta = None
        if original is not None and mutated is not None:
            delta = mutated - original

        payload: dict[str, Any] = {
            "base_tactic_id": str(base_tactic_id),
            "variant_id": str(variant_id),
            "parameter": str(parameter),
            "original_value": original,
            "mutated_value": mutated,
        }
        if delta is not None:
            payload["delta"] = delta
        if metadata:
            payload["metadata"] = self._sanitize(metadata)

        record = self._create_record("parameter_mutation", payload)
        self._store_record(self._parameter_mutations, record)

    def record_fitness_improvement(
        self,
        *,
        genome_id: str,
        previous_fitness: Any,
        new_fitness: Any,
        generation: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a record when the champion fitness frontier moves upward."""

        prev = self._to_float(previous_fitness)
        new = self._to_float(new_fitness)
        improvement = None
        if prev is not None and new is not None:
            improvement = new - prev

        generation_value = self._to_int(generation)

        payload: dict[str, Any] = {
            "genome_id": str(genome_id),
            "previous_fitness": prev,
            "new_fitness": new,
        }
        if improvement is not None:
            payload["improvement"] = improvement
        if generation_value is not None:
            payload["generation"] = generation_value
        if metadata:
            payload["metadata"] = self._sanitize(metadata)

        record = self._create_record("fitness_improvement", payload)
        self._store_record(self._fitness_improvements, record)

    def record_exploitability_result(
        self,
        observation: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist an exploitability observation emitted by the mini-league."""

        if observation is None:
            return

        selected_slot = self._normalise_slot(getattr(observation, "selected_slot", None))
        selected_agent_id = getattr(observation, "selected_agent_id", None)
        selected_gap = self._to_float(getattr(observation, "selected_gap", None))
        selected_penalty = self._to_float(getattr(observation, "selected_penalty", None))
        wow_delta = self._to_float(getattr(observation, "wow_delta", None))

        payload: dict[str, Any] = {
            "metric": getattr(observation, "metric", None),
            "tolerance_pct": self._to_float(getattr(observation, "tolerance_pct", None)),
            "current": {
                "agent_id": getattr(observation, "current_agent_id", None),
                "metric": self._to_float(getattr(observation, "current_metric", None)),
                "turnover": self._to_float(getattr(observation, "current_turnover", None)),
            },
            "selected_agent_id": selected_agent_id,
            "selected_slot": selected_slot,
            "selected_gap": selected_gap,
            "selected_penalty": selected_penalty,
            "wow_delta": wow_delta,
            "selected": {
                "gap": selected_gap,
                "agent_id": selected_agent_id,
                "penalty": selected_penalty,
                "slot": selected_slot,
            },
            "comparisons": [
                self._serialise_comparison(comparison)
                for comparison in self._ensure_sequence(getattr(observation, "comparisons", ()))
            ],
        }
        if metadata:
            payload["metadata"] = self._sanitize(metadata)

        record = self._create_record("exploitability_result", payload)
        self._store_record(self._exploitability_results, record)

    def _serialise_comparison(self, comparison: Any) -> Mapping[str, Any]:
        return {
            "slot": self._normalise_slot(getattr(comparison, "slot", None)),
            "agent_id": getattr(comparison, "agent_id", None),
            "metric": self._to_float(getattr(comparison, "metric", None)),
            "turnover": self._to_float(getattr(comparison, "turnover", None)),
            "turnover_diff_pct": self._to_float(getattr(comparison, "turnover_diff_pct", None)),
            "gap": self._to_float(getattr(comparison, "gap", None)),
            "turnover_variance": self._to_float(getattr(comparison, "turnover_variance", None)),
            "inventory_variance": self._to_float(getattr(comparison, "inventory_variance", None)),
            "lagrangian_penalty": self._to_float(getattr(comparison, "lagrangian_penalty", None)),
            "lagrangian_adjusted_gap": self._to_float(getattr(comparison, "lagrangian_adjusted_gap", None)),
        }

    def _create_record(self, event: str, payload: Mapping[str, Any]) -> _LedgerRecord:
        sanitized = self._sanitize(payload)
        return _LedgerRecord(
            event=str(event),
            recorded_at=datetime.now(timezone.utc),
            payload=sanitized,
        )

    def _store_record(self, bucket: list[_LedgerRecord], record: _LedgerRecord) -> None:
        entry = record.to_json_entry()
        try:
            encoded = json.dumps(entry, sort_keys=True)
        except Exception:  # pragma: no cover - best effort serialisation
            logger.debug(
                "Mutation ledger failed to serialise %s event", record.event, exc_info=True
            )
            encoded = None

        with self._lock:
            bucket.append(record)
            if encoded is None:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.write("\n")
            except Exception:  # pragma: no cover - ledger persistence is best-effort
                logger.debug(
                    "Mutation ledger failed to persist %s event", record.event, exc_info=True
                )

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None

    def _normalise_slot(self, value: Any) -> str | None:
        if isinstance(value, Enum):
            return str(value.value)
        if value is None:
            return None
        text = str(value)
        return text or None

    def _ensure_sequence(self, value: Any) -> Sequence[Any]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return value
        if value is None:
            return ()
        return (value,)

    def _sanitize(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {str(key): self._sanitize(val) for key, val in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._sanitize(item) for item in value]
        if hasattr(value, "__dict__"):
            return {
                str(key): self._sanitize(val)
                for key, val in vars(value).items()
                if not key.startswith("_")
            }
        return str(value)


_default_ledger: MutationLedger | None = None


def get_mutation_ledger() -> MutationLedger:
    """Return the process-wide mutation ledger singleton."""

    global _default_ledger
    if _default_ledger is None:
        _default_ledger = MutationLedger()
    return _default_ledger


def set_mutation_ledger(ledger: MutationLedger | None) -> None:
    """Override the process-wide mutation ledger (primarily for tests)."""

    global _default_ledger
    _default_ledger = ledger


__all__ = ["MutationLedger", "get_mutation_ledger", "set_mutation_ledger", "DEFAULT_LEDGER_PATH"]
