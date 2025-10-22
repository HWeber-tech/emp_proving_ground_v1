from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Iterable, Optional

import numpy as np

from . import faiss_pattern_memory as memory_module
from .faiss_pattern_memory import FAISSPatternMemory

logger = logging.getLogger(__name__)


ClockCallable = Callable[[], datetime]


@dataclass(slots=True)
class PendingExperience:
    """Container for deferred memory updates."""

    vector: np.ndarray
    metadata: dict[str, Any]


class MemoryAutoRetrainer:
    """Automates periodic FAISS memory retraining and compaction.

    The retrainer batches incremental updates, periodically rebuilds the FAISS
    index from persisted metadata, and prunes duplicate or stale entries. When
    the optional ``faiss`` dependency is unavailable, the behaviour gracefully
    falls back to working with the in-memory implementation.
    """

    def __init__(
        self,
        memory: FAISSPatternMemory,
        config: Optional[dict[str, Any]] = None,
        *,
        clock: Optional[ClockCallable] = None,
    ) -> None:
        self._memory = memory
        self._config = config or {}
        self._clock: ClockCallable = clock or datetime.utcnow
        interval_seconds = float(self._config.get("retrain_interval_seconds", 86400.0))
        if interval_seconds <= 0:
            logger.warning(
                "Non-positive retrain interval provided (%s); defaulting to 1 hour",
                interval_seconds,
            )
            interval_seconds = 3600.0
        self._interval = timedelta(seconds=interval_seconds)
        self._batch_size = int(self._config.get("incremental_batch_size", 128))
        if self._batch_size <= 0:
            self._batch_size = 128
        self._state_path = self._resolve_state_path(self._config.get("state_path"))
        self._pending: Deque[PendingExperience] = deque()
        self._last_retrain_at = self._load_last_retrain_at()

    @property
    def last_retrain_at(self) -> datetime:
        """Timestamp of the most recent successful retraining."""

        return self._last_retrain_at

    @property
    def next_due_at(self) -> datetime:
        """When the next retraining cycle should occur."""

        return self._last_retrain_at + self._interval

    @property
    def pending_updates(self) -> int:
        """Number of pending experiences awaiting ingestion."""

        return len(self._pending)

    def register_experience(self, vector: Iterable[float], metadata: dict[str, Any]) -> None:
        """Queue an experience for inclusion during the next retraining cycle."""

        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != self._memory.dimension:
            raise ValueError(
                f"Expected vector of shape ({self._memory.dimension},), "
                f"got {array.shape}"
            )

        payload = dict(metadata)
        self._pending.append(PendingExperience(vector=array, metadata=payload))
        if len(self._pending) >= self._batch_size:
            self._flush_pending_updates()

    def should_retrain(self) -> bool:
        """Return ``True`` if a retraining run is currently due."""

        now = self._clock()
        return now >= self.next_due_at

    def run_cycle(self, *, force: bool = False) -> dict[str, Any]:
        """Run a retraining cycle if due and return a summary payload."""

        now = self._clock()
        flushed = self._flush_pending_updates()
        summary: dict[str, Any] = {
            "run_requested_at": now.isoformat(),
            "flushed": flushed,
            "retrained": False,
        }

        if not (force or self.should_retrain()):
            summary["next_due_at"] = self.next_due_at.isoformat()
            return summary

        decay_summary: dict[str, Any] | None = None
        try:
            decay_summary = self._memory.apply_decay_protocol()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Failed to apply decay protocol: %s", exc)

        rebuild_summary = self._rebuild_index()
        self._last_retrain_at = now
        self._save_last_retrain_at(now)

        summary.update(
            {
                "retrained": True,
                "decay": decay_summary,
                "rebuild": rebuild_summary,
                "completed_at": now.isoformat(),
                "next_due_at": self.next_due_at.isoformat(),
            }
        )
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    def _resolve_state_path(self, override: Any) -> Path:
        if override:
            path = Path(str(override))
        else:
            base_path = getattr(self._memory, "metadata_path", Path("data/memory"))
            path = Path(base_path).with_name("auto_retraining_state.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_last_retrain_at(self) -> datetime:
        if not self._state_path.exists():
            return datetime.fromtimestamp(0, tz=None)
        try:
            with self._state_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:  # pragma: no cover - defensive guard
            logger.warning("Failed to load auto retraining state; assuming never run")
            return datetime.fromtimestamp(0, tz=None)

        timestamp = payload.get("last_retrain_at")
        if not isinstance(timestamp, str):
            return datetime.fromtimestamp(0, tz=None)
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            return datetime.fromtimestamp(0, tz=None)

    def _save_last_retrain_at(self, timestamp: datetime) -> None:
        payload = {"last_retrain_at": timestamp.isoformat()}
        with self._state_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=str)

    def _flush_pending_updates(self) -> int:
        flushed = 0
        while self._pending:
            experience = self._pending.popleft()
            try:
                self._memory.add_experience(experience.vector, experience.metadata)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Failed to add experience to memory: %s", exc)
                continue
            flushed += 1
        return flushed

    def _rebuild_index(self) -> dict[str, Any]:
        if hasattr(self._memory, "index") and getattr(memory_module, "faiss", None) is not None:
            return self._rebuild_faiss_index()
        return self._rebuild_fallback_memory()

    def _rebuild_faiss_index(self) -> dict[str, Any]:
        faiss = memory_module.faiss
        assert faiss is not None  # for type-checkers

        entries: list[tuple[datetime, str, np.ndarray, dict[str, Any]] | None] = []
        pruned_ids: list[str] = []
        seen_vectors: set[tuple[float, ...]] = set()

        for memory_id, record in list(self._memory.metadata.items()):
            if not isinstance(record, dict):
                pruned_ids.append(memory_id)
                self._memory.metadata.pop(memory_id, None)
                continue

            vector_raw = record.get("vector")
            try:
                vector = np.asarray(vector_raw, dtype=np.float32)
            except Exception:
                vector = np.empty(0, dtype=np.float32)
            if vector.ndim != 1 or vector.shape[0] != self._memory.dimension:
                pruned_ids.append(memory_id)
                self._memory.metadata.pop(memory_id, None)
                continue

            signature = tuple(np.round(vector, 6))
            if signature in seen_vectors:
                pruned_ids.append(memory_id)
                self._memory.metadata.pop(memory_id, None)
                continue
            seen_vectors.add(signature)

            timestamp_raw = record.get("timestamp")
            timestamp = self._parse_timestamp(timestamp_raw)
            cleaned_record = dict(record)
            cleaned_record.setdefault("metadata", {})
            entries.append((timestamp, memory_id, vector, cleaned_record))

        if not entries:
            # No entries left; reset index and metadata entirely.
            self._memory.index = faiss.IndexFlatL2(self._memory.dimension)
            self._memory.metadata = {}
            faiss.write_index(self._memory.index, str(self._memory.index_path))
            self._memory._save_metadata()
            return {
                "compacted": 0,
                "pruned": len(pruned_ids),
                "reindexed": 0,
                "pruned_ids": pruned_ids,
            }

        # Sort by recency so the newest entries are retained if pruning is required.
        entries.sort(key=lambda item: item[0], reverse=True)

        max_memories = getattr(self._memory, "max_memories", 0)
        if max_memories > 0 and len(entries) > max_memories:
            overflow = entries[max_memories:]
            entries = entries[:max_memories]
            pruned_ids.extend(memory_id for _, memory_id, _, _ in overflow)

        rebuilt_index = faiss.IndexFlatL2(self._memory.dimension)
        rebuilt_metadata: dict[str, Any] = {}

        # Oldest first so index positions match chronological order.
        entries.sort(key=lambda item: item[0])
        for position, (timestamp, memory_id, vector, record) in enumerate(entries):
            normalised = self._memory._normalise(vector)
            rebuilt_index.add(normalised.reshape(1, -1))
            record["vector"] = normalised.tolist()
            record["timestamp"] = timestamp.isoformat()
            record["index_position"] = position
            rebuilt_metadata[memory_id] = record

        self._memory.metadata = rebuilt_metadata
        self._memory.index = rebuilt_index
        self._memory._save_metadata()
        faiss.write_index(rebuilt_index, str(self._memory.index_path))

        return {
            "compacted": len(entries),
            "pruned": len(pruned_ids),
            "reindexed": len(entries),
            "pruned_ids": pruned_ids,
        }

    def _rebuild_fallback_memory(self) -> dict[str, Any]:
        if not hasattr(self._memory, "_memories"):
            return {"compacted": 0, "pruned": 0, "reindexed": 0, "pruned_ids": []}

        entries = list(self._memory._memories.items())
        if not entries:
            self._memory._save_metadata()
            return {"compacted": 0, "pruned": 0, "reindexed": 0, "pruned_ids": []}

        # Deduplicate memories that share an identical vector signature.
        seen_vectors: set[tuple[float, ...]] = set()
        pruned_ids: list[str] = []
        cleaned_entries: list[tuple[str, memory_module.MemoryEntry]] = []

        for memory_id, entry in entries:
            signature = tuple(np.round(entry.vector, 6))
            if signature in seen_vectors:
                pruned_ids.append(memory_id)
                continue
            seen_vectors.add(signature)
            cleaned_entries.append((memory_id, entry))

        max_memories = getattr(self._memory, "max_memories", 0)
        if max_memories > 0 and len(cleaned_entries) > max_memories:
            cleaned_entries.sort(key=lambda item: item[1].timestamp, reverse=True)
            overflow = cleaned_entries[max_memories:]
            cleaned_entries = cleaned_entries[:max_memories]
            pruned_ids.extend(memory_id for memory_id, _ in overflow)

        # Normalise vectors and rebuild storage ordered by timestamp.
        cleaned_entries.sort(key=lambda item: item[1].timestamp)
        self._memory._memories = {
            memory_id: memory_module.MemoryEntry(
                vector=self._memory._normalise(entry.vector),
                metadata=dict(entry.metadata),
                timestamp=entry.timestamp,
                learning_signal_id=entry.learning_signal_id,
            )
            for memory_id, entry in cleaned_entries
        }
        self._memory._save_metadata()

        for memory_id in pruned_ids:
            self._memory._memories.pop(memory_id, None)

        return {
            "compacted": len(cleaned_entries),
            "pruned": len(pruned_ids),
            "reindexed": len(cleaned_entries),
            "pruned_ids": pruned_ids,
        }

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.utcnow()

