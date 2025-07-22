"""
Enhanced Pattern Memory with FAISS Integration
Provides high-performance vector similarity search for real-time learning.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import faiss
from dataclasses import dataclass

from src.core.events import ContextPacket, LearningSignal
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry with vector and outcome."""
    timestamp: datetime
    vector: np.ndarray
    context: ContextPacket
    outcome_pnl: float
    trade_duration: float
    confidence: float
    metadata: Dict[str, Any]


class FAISSPatternMemory:
    """
    High-performance pattern memory using FAISS for similarity search.
    
    Features:
    - FAISS-based vector similarity search
    - Automatic memory management
    - Confidence-based filtering
    - Real-time updates
    """
    
    def __init__(self, state_store: StateStore, vector_dimension: int = 512):
        self.state_store = state_store
        self.vector_dimension = vector_dimension
        self.index = None
        self.outcome_store: Dict[int, MemoryEntry] = {}
        self._memory_key = "emp:faiss_memory"
        self._max_memory_size = 10000
        self._similarity_threshold = 0.7
        
        # Initialize FAISS index
        self._init_index()
        
    def _init_index(self):
        """Initialize FAISS index for cosine similarity."""
        self.index = faiss.IndexFlatIP(self.vector_dimension)  # Inner product for cosine similarity
        
    async def initialize(self) -> None:
        """Initialize pattern memory from storage."""
        await self._load_memory()
        
    async def add_experience(self, learning_signal: LearningSignal) -> None:
        """Store a new learning experience in memory."""
        try:
            # Convert latent vector to numpy array
            vector = np.array(learning_signal.triggering_context.latent_vec, dtype=np.float32)
            
            # Normalize vector for cosine similarity
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            
            # Create memory entry
            entry = MemoryEntry(
                timestamp=learning_signal.timestamp,
                vector=vector,
                context=learning_signal.triggering_context,
                outcome_pnl=float(learning_signal.outcome_pnl),
                trade_duration=learning_signal.trade_duration_seconds,
                confidence=float(learning_signal.confidence_of_outcome),
                metadata=learning_signal.metadata
            )
            
            # Add to FAISS index
            self.index.add(vector.reshape(1, -1))
            
            # Store outcome with index ID
            index_id = self.index.ntotal - 1
            self.outcome_store[index_id] = entry
            
            # Manage memory size
            if len(self.outcome_store) > self._max_memory_size:
                await self._cleanup_memory()
                
            # Persist to storage
            await self._save_memory()
            
            logger.debug(f"Stored new experience, total: {len(self.outcome_store)}")
            
        except Exception as e:
            logger.error(f"Error storing experience: {e}")
            
    async def find_similar_experiences(
        self,
        query_vector: List[float],
        max_results: int = 5,
        min_confidence: float = 0.5,
        time_window: Optional[timedelta] = None
    ) -> List[Tuple[float, MemoryEntry]]:
        """
        Find similar historical experiences.
        
        Args:
            query_vector: The query vector to search for
            max_results: Maximum number of similar experiences to return
            min_confidence: Minimum confidence threshold for results
            time_window: Only consider experiences within this time window
            
        Returns:
            List of (similarity_score, memory_entry) tuples
        """
        if not self.outcome_store:
            return []
            
        try:
            # Convert query vector to numpy array
            query = np.array(query_vector, dtype=np.float32)
            
            # Normalize query vector
            if np.linalg.norm(query) > 0:
                query = query / np.linalg.norm(query)
            
            # Search for similar vectors
            k = min(max_results, len(self.outcome_store))
            distances, indices = self.index.search(query.reshape(1, -1), k)
            
            similar_experiences = []
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx in self.outcome_store:
                    entry = self.outcome_store[idx]
                    
                    # Apply filters
                    if entry.confidence < min_confidence:
                        continue
                        
                    if time_window and entry.timestamp < datetime.utcnow() - time_window:
                        continue
                    
                    # Convert distance to similarity score (cosine similarity)
                    similarity = float(distance)
                    
                    similar_experiences.append((similarity, entry))
            
            # Sort by similarity (descending)
            similar_experiences.sort(key=lambda x: x[0], reverse=True)
            
            return similar_experiences
            
        except Exception as e:
            logger.error(f"Error finding similar experiences: {e}")
            return []
            
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored patterns."""
        if not self.outcome_store:
            return {
                'total_patterns': 0,
                'time_span': None,
                'average_outcome': None,
                'win_rate': None
            }
        
        # Calculate statistics
        timestamps = [entry.timestamp for entry in self.outcome_store.values()]
        outcomes = [entry.outcome_pnl for entry in self.outcome_store.values()]
        
        time_span = max(timestamps) - min(timestamps)
        avg_outcome = np.mean(outcomes)
        win_rate = len([o for o in outcomes if o > 0]) / len(outcomes)
        
        return {
            'total_patterns': len(self.outcome_store),
            'time_span': str(time_span),
            'average_outcome': float(avg_outcome),
            'win_rate': float(win_rate),
            'recent_patterns': len([
                e for e in self.outcome_store.values()
                if e.timestamp > datetime.utcnow() - timedelta(hours=24)
            ])
        }
        
    async def _cleanup_memory(self) -> None:
        """Clean up old memory entries to maintain size limits."""
        if len(self.outcome_store) <= self._max_memory_size:
            return
            
        # Remove oldest entries
        sorted_entries = sorted(
            self.outcome_store.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Keep only the most recent entries
        keep_count = int(self._max_memory_size * 0.8)
        to_remove = len(self.outcome_store) - keep_count
        
        # Remove from both index and store
        for idx, _ in sorted_entries[:to_remove]:
            del self.outcome_store[idx]
            
        # Rebuild index with remaining entries
        self._rebuild_index()
        
    def _rebuild_index(self) -> None:
        """Rebuild FAISS index with current memory entries."""
        self._init_index()
        
        vectors = []
        new_outcome_store = {}
        
        for new_idx, (old_idx, entry) in enumerate(self.outcome_store.items()):
            vectors.append(entry.vector)
            new_outcome_store[new_idx] = entry
            
        if vectors:
            vectors_array = np.array(vectors, dtype=np.float32)
            self.index.add(vectors_array)
            
        self.outcome_store = new_outcome_store
        
    async def _load_memory(self) -> None:
        """Load memory from Redis storage."""
        try:
            data = await self.state_store.get(self._memory_key)
            if data:
                memory_data = json.loads(data)
                
                # Rebuild index
                self._init_index()
                
                vectors = []
                self.outcome_store = {}
                
                for idx, entry_data in enumerate(memory_data):
                    vector = np.array(entry_data['vector'], dtype=np.float32)
                    vectors.append(vector)
                    
                    entry = MemoryEntry(
                        timestamp=datetime.fromisoformat(entry_data['timestamp']),
                        vector=vector,
                        context=ContextPacket(**entry_data['context']),
                        outcome_pnl=entry_data['outcome_pnl'],
                        trade_duration=entry_data['trade_duration'],
                        confidence=entry_data['confidence'],
                        metadata=entry_data['metadata']
                    )
                    
                    self.outcome_store[idx] = entry
                
                if vectors:
                    vectors_array = np.array(vectors, dtype=np.float32)
                    self.index.add(vectors_array)
                
                logger.info(f"Loaded {len(self.outcome_store)} memory entries")
            else:
                logger.info("No FAISS memory data found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS memory: {e}")
            self._init_index()
            
    async def _save_memory(self) -> None:
        """Save memory to Redis storage."""
        try:
            memory_data = []
            
            for idx, entry in self.outcome_store.items():
                memory_data.append({
                    'timestamp': entry.timestamp.isoformat(),
                    'vector': entry.vector.tolist(),
                    'context': entry.context.dict(),
                    'outcome_pnl': entry.outcome_pnl,
                    'trade_duration': entry.trade_duration,
                    'confidence': entry.confidence,
                    'metadata': entry.metadata
                })
            
            await self.state_store.set(
                self._memory_key,
                json.dumps(memory_data),
                expire=86400 * 7  # 7 days
            )
            
        except Exception as e:
            logger.error(f"Failed to save FAISS memory: {e}")
            
    async def clear_memory(self) -> None:
        """Clear all stored patterns."""
        self._init_index()
        self.outcome_store.clear()
        await self.state_store.delete(self._memory_key)
        logger.info("FAISS pattern memory cleared")
        
    def get_memory_size(self) -> int:
        """Get current memory size."""
        return len(self.outcome_store)
