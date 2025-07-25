#!/usr/bin/env python3
"""
FAISSPatternMemory - Epic 1: The Predator's Instinct
Upgraded memory system for storing and recalling trading experiences.
"""

import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """A single memory entry in the FAISS index."""
    vector: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime
    learning_signal_id: str
    
    @property
    def features(self) -> Dict[str, float]:
        """Get the features from metadata."""
        return self.metadata.get('features', {})
    
    @property
    def outcome(self) -> Dict[str, float]:
        """Get the outcome from metadata."""
        return self.metadata.get('outcome', {})

class FAISSPatternMemory:
    """FAISS-based pattern memory for storing and recalling trading experiences."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dimension = config.get('vector_dimension', 64)
        self.index_path = Path(config.get('index_path', 'data/memory/faiss_index'))
        self.metadata_path = Path(config.get('metadata_path', 'data/memory/metadata.json'))
        
        # Create directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.index = None
        self.metadata = {}
        self.memory_counter = 0
        
        self._initialize_index()
        self._load_metadata()
        
    def _initialize_index(self):
        """Initialize the FAISS index."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index with {self.index.ntotal} entries")
        else:
            # Create new index with L2 distance
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created new FAISS index")
            
    def _load_metadata(self):
        """Load metadata from disk."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} metadata entries")
            
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, default=str)
            
    def add_experience(self, vector: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Add a new experience to memory."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} != {self.dimension}")
            
        # Normalize vector
        vector = vector.astype(np.float32)
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
            
        # Add to FAISS index
        self.index.add(vector.reshape(1, -1))
        
        # Create memory entry
        memory_id = f"memory_{self.memory_counter}"
        self.memory_counter += 1
        
        # Store metadata
        self.metadata[memory_id] = {
            'vector': vector.tolist(),
            'metadata': metadata,
            'timestamp': datetime.utcnow().isoformat(),
            'index_position': self.index.ntotal - 1
        }
        
        # Save metadata
        self._save_metadata()
        
        # Save index
        faiss.write_index(self.index, str(self.index_path))
        
        logger.info(f"Added experience to memory: {memory_id}")
        return memory_id
    
    def search_similar(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar experiences."""
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} != {self.dimension}")
            
        # Normalize query vector
        query_vector = query_vector.astype(np.float32)
        if np.linalg.norm(query_vector) > 0:
            query_vector = query_vector / np.linalg.norm(query_vector)
            
        # Search FAISS index
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        # Get results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
                
            # Find memory entry by index position
            memory_id = None
            for mid, data in self.metadata.items():
                if data.get('index_position') == idx:
                    memory_id = mid
                    break
                    
            if memory_id:
                results.append({
                    'memory_id': memory_id,
                    'distance': float(distance),
                    'metadata': self.metadata[memory_id]['metadata'],
                    'timestamp': self.metadata[memory_id]['timestamp']
                })
                
        return results
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_memories': self.index.ntotal,
            'dimension': self.dimension,
            'index_path': str(self.index_path),
            'metadata_path': str(self.metadata_path),
            'memory_counter': self.memory_counter
        }
    
    def clear_memory(self):
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
        
    def get_recent_memories(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent memories."""
        # Sort by timestamp
        sorted_memories = sorted(
            self.metadata.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        return [
            {
                'memory_id': mid,
                'metadata': data['metadata'],
                'timestamp': data['timestamp']
            }
            for mid, data in sorted_memories[:count]
        ]
