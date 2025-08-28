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
from typing import Any, Dict, List, Optional, cast

import numpy as np

import faiss

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


class FAISSPatternMemory:
    """FAISS-based pattern memory for storing and recalling trading experiences."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dimension = config.get("vector_dimension", 64)
        self.index_path = Path(config.get("index_path", "data/memory/faiss_index"))
        self.metadata_path = Path(config.get("metadata_path", "data/memory/metadata.json"))

        # Create directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize FAISS index
        self.index: Optional[faiss.Index] = None
        self.metadata: dict[str, Any] = {}
        self.memory_counter = 0

        self._initialize_index()
        self._load_metadata()

    def _initialize_index(self) -> None:
        """Initialize the FAISS index."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index with {self.index.ntotal} entries")
        else:
            # Create new index with L2 distance
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created new FAISS index")

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} metadata entries")

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, default=str)

    def add_experience(self, vector: np.ndarray, metadata: dict[str, Any]) -> str:
        """Add a new experience to memory."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} != {self.dimension}")

        # Normalize vector
        vector = np.asarray(vector, dtype=np.float32)
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)

        # Add to FAISS index
        if self.index is None:
            self._initialize_index()
        assert self.index is not None
        self.index.add(vector.reshape(1, -1))

        # Create memory entry
        memory_id = f"memory_{self.memory_counter}"
        self.memory_counter += 1

        # Store metadata
        self.metadata[memory_id] = {
            "vector": vector.tolist(),
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat(),
            "index_position": self.index.ntotal - 1,
        }

        # Save metadata
        self._save_metadata()

        # Save index
        faiss.write_index(self.index, str(self.index_path))

        logger.info(f"Added experience to memory: {memory_id}")
        return memory_id

    def search_similar(self, query_vector: np.ndarray, k: int = 10) -> list[dict[str, Any]]:
        """Search for similar experiences."""
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} != {self.dimension}")

        # Normalize query vector
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if np.linalg.norm(query_vector) > 0:
            query_vector = query_vector / np.linalg.norm(query_vector)

        # Search FAISS index
        if self.index is None:
            return []
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        distances = cast(np.ndarray, distances)
        indices = cast(np.ndarray, indices)

        # Get results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue

            # Find memory entry by index position
            memory_id = None
            for mid, data in self.metadata.items():
                if data.get("index_position") == idx:
                    memory_id = mid
                    break

            if memory_id:
                results.append(
                    {
                        "memory_id": memory_id,
                        "distance": float(distance),
                        "metadata": self.metadata[memory_id]["metadata"],
                        "timestamp": self.metadata[memory_id]["timestamp"],
                    }
                )

        return results

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_memories": int(self.index.ntotal) if self.index is not None else 0,
            "dimension": self.dimension,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "memory_counter": self.memory_counter,
        }

    def clear_memory(self) -> None:
        """Clear all memories."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = {}
        self.memory_counter = 0

        # Remove files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()

        logger.info("Cleared all memories")

    def get_recent_memories(self, count: int = 100) -> list[dict[str, Any]]:
        """Get the most recent memories."""
        # Sort by timestamp
        sorted_memories = sorted(
            self.metadata.items(), key=lambda x: x[1]["timestamp"], reverse=True
        )

        return [
            {"memory_id": mid, "metadata": data["metadata"], "timestamp": data["timestamp"]}
            for mid, data in sorted_memories[:count]
        ]
