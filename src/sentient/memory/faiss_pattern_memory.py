#!/usr/bin/env python3
"""
FAISSPatternMemory - Epic 1: The Predator's Instinct
Upgraded memory system for storing and recalling trading experiences.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

try:  # pragma: no cover - exercised via fallback behaviour in tests
    import faiss  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry in the FAISS index."""

    vector: np.ndarray
    metadata: dict[str, Any]
    timestamp: datetime
    learning_signal_id: str

    @property
    def features(self) -> dict[str, float]:
        """Get the features from metadata."""
        return cast(dict[str, float], self.metadata.get("features", {}))

    @property
    def outcome(self) -> dict[str, float]:
        """Get the outcome from metadata."""
        return cast(dict[str, float], self.metadata.get("outcome", {}))


class _BasePatternMemory:
    """Utility helpers shared between FAISS and fallback implementations."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dimension = int(config.get("vector_dimension", 64))
        self.index_path = Path(config.get("index_path", "data/memory/faiss_index"))
        self.metadata_path = Path(config.get("metadata_path", "data/memory/metadata.json"))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_counter = 0
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr


if faiss is not None:

    class FAISSPatternMemory(_BasePatternMemory):
        """FAISS-backed implementation when the optional dependency is installed."""

        def __init__(self, config: dict[str, Any]):
            super().__init__(config)
            self.index: Optional[faiss.Index] = None
            self._initialize_index()
            self._load_metadata()

        def _initialize_index(self) -> None:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(
                    "Loaded FAISS index with %s entries", int(self.index.ntotal)
                )
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info("Created new FAISS index")

        def _load_metadata(self) -> None:
            if not self.metadata_path.exists():
                return
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
            self.memory_counter = int(self.metadata.get("__counter__", 0))

        def _save_metadata(self) -> None:
            payload = dict(self.metadata)
            payload["__counter__"] = self.memory_counter
            with self.metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, default=str)

        def add_experience(self, vector: np.ndarray, metadata: dict[str, Any]) -> str:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} != {self.dimension}"
                )

            normalised = self._normalise(vector)
            if self.index is None:
                self._initialize_index()
            assert self.index is not None
            self.index.add(normalised.reshape(1, -1))

            memory_id = f"memory_{self.memory_counter}"
            self.memory_counter += 1
            self.metadata[memory_id] = {
                "vector": normalised.tolist(),
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat(),
                "index_position": int(self.index.ntotal) - 1,
            }
            self._save_metadata()
            faiss.write_index(self.index, str(self.index_path))
            logger.info("Added experience to memory: %s", memory_id)
            return memory_id

        def search_similar(
            self, query_vector: np.ndarray, k: int = 10
        ) -> list[dict[str, Any]]:
            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"Query vector dimension {len(query_vector)} != {self.dimension}"
                )

            if self.index is None:
                return []
            normalised = self._normalise(query_vector)
            distances, indices = self.index.search(normalised.reshape(1, -1), k)
            distances = cast(np.ndarray, distances)
            indices = cast(np.ndarray, indices)

            results: list[dict[str, Any]] = []
            for distance, idx in zip(distances[0], indices[0]):
                target = next(
                    (
                        mid
                        for mid, data in self.metadata.items()
                        if isinstance(data, dict)
                        and data.get("index_position") == int(idx)
                    ),
                    None,
                )
                if target is None:
                    continue
                stored = self.metadata[target]
                results.append(
                    {
                        "memory_id": target,
                        "distance": float(distance),
                        "metadata": stored["metadata"],
                        "timestamp": stored["timestamp"],
                    }
                )
            return results

        def get_memory_stats(self) -> dict[str, Any]:
            total = int(self.index.ntotal) if self.index is not None else 0
            return {
                "total_memories": total,
                "dimension": self.dimension,
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "memory_counter": self.memory_counter,
            }

        def clear_memory(self) -> None:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}
            self.memory_counter = 0
            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            logger.info("Cleared all memories")

        def get_recent_memories(self, count: int = 100) -> list[dict[str, Any]]:
            items = [
                (mid, data)
                for mid, data in self.metadata.items()
                if isinstance(data, dict)
            ]
            sorted_memories = sorted(
                items,
                key=lambda item: item[1].get("timestamp", ""),
                reverse=True,
            )
            return [
                {
                    "memory_id": mid,
                    "metadata": data.get("metadata", {}),
                    "timestamp": data.get("timestamp"),
                }
                for mid, data in sorted_memories[:count]
            ]


else:

    class FAISSPatternMemory(_BasePatternMemory):
        """In-memory fallback when the FAISS dependency is unavailable."""

        def __init__(self, config: dict[str, Any]):
            super().__init__(config)
            self._memories: dict[str, MemoryEntry] = {}
            self._load_metadata()
            logger.warning(
                "faiss module not available; using in-memory pattern memory fallback"
            )

        def _load_metadata(self) -> None:
            if not self.metadata_path.exists():
                return
            try:
                with self.metadata_path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            except Exception:
                logger.warning("Failed to load fallback memory metadata; starting fresh")
                return

            self.memory_counter = int(raw.get("__counter__", 0))
            entries = raw.get("entries", {})
            for memory_id, payload in entries.items():
                try:
                    entry = MemoryEntry(
                        vector=np.asarray(payload["vector"], dtype=np.float32),
                        metadata=cast(dict[str, Any], payload.get("metadata", {})),
                        timestamp=datetime.fromisoformat(payload["timestamp"]),
                        learning_signal_id=memory_id,
                    )
                except Exception:
                    continue
                self._memories[memory_id] = entry

        def _save_metadata(self) -> None:
            serialised = {
                memory_id: {
                    "vector": entry.vector.tolist(),
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp.isoformat(),
                }
                for memory_id, entry in self._memories.items()
            }
            payload = {"__counter__": self.memory_counter, "entries": serialised}
            with self.metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, default=str)

        def add_experience(self, vector: np.ndarray, metadata: dict[str, Any]) -> str:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} != {self.dimension}"
                )

            normalised = self._normalise(vector)
            memory_id = f"memory_{self.memory_counter}"
            self.memory_counter += 1
            entry = MemoryEntry(
                vector=normalised,
                metadata=dict(metadata),
                timestamp=datetime.utcnow(),
                learning_signal_id=memory_id,
            )
            self._memories[memory_id] = entry
            self._save_metadata()
            return memory_id

        def search_similar(
            self, query_vector: np.ndarray, k: int = 10
        ) -> list[dict[str, Any]]:
            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"Query vector dimension {len(query_vector)} != {self.dimension}"
                )

            if not self._memories:
                return []

            normalised = self._normalise(query_vector)
            scored: list[tuple[float, MemoryEntry]] = []
            for entry in self._memories.values():
                distance = float(np.linalg.norm(entry.vector - normalised))
                scored.append((distance, entry))

            scored.sort(key=lambda item: item[0])
            results: list[dict[str, Any]] = []
            for distance, entry in scored[:k]:
                results.append(
                    {
                        "memory_id": entry.learning_signal_id,
                        "distance": distance,
                        "metadata": entry.metadata,
                        "timestamp": entry.timestamp.isoformat(),
                    }
                )
            return results

        def get_memory_stats(self) -> dict[str, Any]:
            return {
                "total_memories": len(self._memories),
                "dimension": self.dimension,
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "memory_counter": self.memory_counter,
            }

        def clear_memory(self) -> None:
            self._memories.clear()
            self.memory_counter = 0
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            logger.info("Cleared all fallback memories")

        def get_recent_memories(self, count: int = 100) -> list[dict[str, Any]]:
            entries = sorted(
                self._memories.values(),
                key=lambda entry: entry.timestamp,
                reverse=True,
            )
            return [
                {
                    "memory_id": entry.learning_signal_id,
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp.isoformat(),
                }
                for entry in entries[:count]
            ]
