"""
Enhanced Pattern Memory with Active Recall

Provides long-term memory storage and retrieval for trading contexts,
enabling the system to recall similar historical situations.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from src.core.event_bus import EventBus
from src.core.state_store import StateStore

logger = logging.getLogger(__name__)


@dataclass
class PatternMemoryEntry:
    """A single memory entry with context and outcome."""

    timestamp: datetime
    latent_vector: np.ndarray
    market_context: dict[str, object]
    trading_outcome: dict[str, object]
    metadata: dict[str, object]


# Backward-compat alias to preserve legacy import name without duplicate ClassDef
MemoryEntry = PatternMemoryEntry


class PatternMemory:
    """
    Long-term memory system for trading patterns.

    Features:
    - Stores historical trading contexts
    - Provides similarity-based recall
    - Tracks outcomes for pattern learning
    - Automatic memory management
    """

    def __init__(self, event_bus: EventBus, state_store: StateStore):
        self.event_bus = event_bus
        self.state_store = state_store
        self._memory_key = "emp:pattern_memory"
        self._max_memory_size = 10000
        self._similarity_threshold = 0.7
        self._memory: List[PatternMemoryEntry] = []

    async def initialize(self) -> None:
        """Initialize pattern memory from storage."""
        await self._load_memory()

    async def store_context(
        self,
        latent_vector: np.ndarray,
        market_context: dict[str, object],
        trading_outcome: dict[str, object],
        metadata: Optional[dict[str, object]] = None,
    ) -> None:
        """Store a new trading context in memory."""
        entry = MemoryEntry(
            timestamp=datetime.utcnow(),
            latent_vector=latent_vector,
            market_context=market_context,
            trading_outcome=trading_outcome,
            metadata=metadata or {},
        )

        self._memory.append(entry)

        # Keep memory size manageable
        if len(self._memory) > self._max_memory_size:
            self._memory = self._memory[-self._max_memory_size :]

        # Persist to storage
        await self._save_memory()

        logger.debug(f"Stored new memory entry, total: {len(self._memory)}")

    async def find_similar_contexts(
        self,
        query_vector: np.ndarray,
        max_results: int = 5,
        time_window: Optional[timedelta] = None,
    ) -> List[Tuple[float, PatternMemoryEntry]]:
        """
        Find similar historical contexts.

        Args:
            query_vector: The latent vector to compare against
            max_results: Maximum number of similar contexts to return
            time_window: Only consider contexts within this time window

        Returns:
            List of (similarity_score, memory_entry) tuples
        """
        if len(self._memory) == 0:
            return []

        # Filter by time window if provided
        candidates = self._memory
        if time_window:
            cutoff = datetime.utcnow() - time_window
            candidates = [m for m in self._memory if m.timestamp >= cutoff]

        if len(candidates) == 0:
            return []

        # Calculate similarities
        similarities = []
        for entry in candidates:
            similarity = self._calculate_similarity(query_vector, entry.latent_vector)
            if similarity >= self._similarity_threshold:
                similarities.append((similarity, entry))

        # Sort by similarity (descending) and limit results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:max_results]

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1.shape != vec2.shape:
            return 0.0

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def get_pattern_statistics(self) -> dict[str, object]:
        """Get statistics about stored patterns."""
        if not self._memory:
            return {"total_patterns": 0, "time_span": None, "average_outcome": None}

        # Calculate time span
        timestamps = [m.timestamp for m in self._memory]
        time_span = max(timestamps) - min(timestamps)

        # Calculate average outcome
        outcomes = [m.trading_outcome.get("pnl", 0) for m in self._memory]
        avg_outcome = float(np.mean([float(cast(Any, x)) for x in outcomes])) if outcomes else 0

        return {
            "total_patterns": len(self._memory),
            "time_span": str(time_span),
            "average_outcome": float(avg_outcome),
            "recent_patterns": len(
                [m for m in self._memory if m.timestamp > datetime.utcnow() - timedelta(hours=24)]
            ),
        }

    async def _load_memory(self) -> None:
        """Load memory from Redis storage."""
        try:
            data = await self.state_store.get(self._memory_key)
            if data:
                memory_data = json.loads(data)
                self._memory = []

                for entry_data in memory_data:
                    entry = PatternMemoryEntry(
                        timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                        latent_vector=np.array(entry_data["latent_vector"]),
                        market_context=entry_data["market_context"],
                        trading_outcome=entry_data["trading_outcome"],
                        metadata=entry_data["metadata"],
                    )
                    self._memory.append(entry)

                logger.info(f"Loaded {len(self._memory)} memory entries")
            else:
                logger.info("No memory data found, starting fresh")

        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            self._memory = []

    async def _save_memory(self) -> None:
        """Save memory to Redis storage."""
        try:
            memory_data = []
            for entry in self._memory:
                memory_data.append(
                    {
                        "timestamp": entry.timestamp.isoformat(),
                        "latent_vector": entry.latent_vector.tolist(),
                        "market_context": entry.market_context,
                        "trading_outcome": entry.trading_outcome,
                        "metadata": entry.metadata,
                    }
                )

            await self.state_store.set(
                self._memory_key,
                json.dumps(memory_data),
                expire=86400,  # 24 hours
            )

        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    async def clear_memory(self) -> None:
        """Clear all stored patterns."""
        self._memory.clear()
        await self.state_store.delete(self._memory_key)
        logger.info("Pattern memory cleared")

    async def get_memory_context(self, query_vector: np.ndarray) -> Optional[dict[str, object]]:
        """
        Get enriched context with memory matches.

        Returns:
            Dictionary with memory context including similar patterns
        """
        similar_contexts = await self.find_similar_contexts(query_vector)

        if not similar_contexts:
            return None

        # Extract insights from similar contexts
        similar_outcomes = [entry.trading_outcome for _, entry in similar_contexts]
        avg_pnl = np.mean([o.get("pnl", 0) for o in similar_outcomes])
        win_rate = np.mean([o.get("win", 0) for o in similar_outcomes])

        return {
            "similar_patterns": len(similar_contexts),
            "average_pnl": float(avg_pnl),
            "win_rate": float(win_rate),
            "confidence": float(similar_contexts[0][0]) if similar_contexts else 0,
            "recent_patterns": len(
                [
                    s
                    for _, s in similar_contexts
                    if s.timestamp > datetime.utcnow() - timedelta(hours=24)
                ]
            ),
        }


# Global instance
_pattern_memory: Optional[PatternMemory] = None


async def get_pattern_memory(event_bus: EventBus, state_store: StateStore) -> PatternMemory:
    """Get or create global pattern memory instance."""
    global _pattern_memory
    if _pattern_memory is None:
        _pattern_memory = PatternMemory(event_bus, state_store)
        await _pattern_memory.initialize()
    return _pattern_memory
