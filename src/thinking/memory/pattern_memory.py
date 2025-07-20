"""
Pattern Memory - Ticket THINK-02
Cognitive memory system for pattern recognition
"""

import numpy as np
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PatternMemory:
    """
    Long-term memory system for market pattern recognition
    Stores and retrieves market patterns based on latent vectors
    """
    
    def __init__(self, dimension: int = 64, index_path: str = "data/pattern_memory.json"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Storage for patterns
        self.vectors = []  # Store vectors for retrieval
        self.metadata = []  # Store associated metadata
        
        # Load existing patterns if available
        self._load_patterns()
        
        logger.info(f"PatternMemory initialized with dimension {dimension}")
    
    def _load_patterns(self) -> None:
        """Load existing patterns from disk"""
        try:
            if self.index_path.exists():
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                    self.vectors = [np.array(v) for v in data.get('vectors', [])]
                    self.metadata = data.get('metadata', [])
                
                logger.info(f"Loaded pattern memory with {len(self.vectors)} patterns")
            else:
                logger.info("No existing pattern memory found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load pattern memory: {e}")
            self.vectors = []
            self.metadata = []
    
    def _save_patterns(self) -> None:
        """Save patterns to disk"""
        try:
            data = {
                'vectors': [v.tolist() for v in self.vectors],
                'metadata': self.metadata,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.index_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Pattern memory saved to disk")
        except Exception as e:
            logger.error(f"Failed to save pattern memory: {e}")
    
    def add_pattern(self, vector: List[float], metadata: Dict[str, Any]) -> None:
        """
        Add a pattern to memory
        
        Args:
            vector: Latent vector representing the pattern
            metadata: Associated metadata for the pattern
        """
        try:
            # Convert to numpy array
            vector_np = np.array(vector, dtype=np.float32)
            
            # Ensure vector has correct dimension
            if len(vector_np) != self.dimension:
                logger.warning(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector_np)}")
                return
            
            # Store vector and metadata
            self.vectors.append(vector_np)
            self.metadata.append({
                'timestamp': datetime.now().isoformat(),
                **metadata
            })
            
            logger.debug(f"Added pattern to memory. Total patterns: {len(self.vectors)}")
            
            # Save periodically (every 100 additions)
            if len(self.vectors) % 100 == 0:
                self._save_patterns()
                
        except Exception as e:
            logger.error(f"Error adding pattern to memory: {e}")
    
    def find_similar_patterns(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Find k most similar patterns to the query vector
        
        Args:
            query_vector: Vector to search for
            k: Number of similar patterns to return
            
        Returns:
            List of similar patterns with distances and metadata
        """
        try:
            if len(self.vectors) == 0:
                logger.warning("No patterns in memory")
                return []
            
            if len(self.vectors) < k:
                k = len(self.vectors)
            
            # Convert query to numpy array
            query = np.array(query_vector, dtype=np.float32)
            
            # Calculate distances using L2 norm
            distances = []
            for i, vec in enumerate(self.vectors):
                distance = float(np.linalg.norm(query - vec))
                distances.append((distance, i))
            
            # Sort by distance and take top k
            distances.sort(key=lambda x: x[0])
            
            # Build results
            results = []
            for distance, idx in distances[:k]:
                result = {
                    'distance': distance,
                    'index': idx,
                    'metadata': self.metadata[idx],
                    'vector': self.vectors[idx].tolist()
                }
                results.append(result)
            
            logger.debug(f"Found {len(results)} similar patterns")
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the pattern memory"""
        return {
            'total_patterns': len(self.vectors),
            'dimension': self.dimension,
            'last_updated': datetime.now().isoformat()
        }
    
    def clear_memory(self) -> None:
        """Clear all patterns from memory"""
        try:
            self.vectors = []
            self.metadata = []
            
            # Remove saved file
            if self.index_path.exists():
                self.index_path.unlink()
                
            logger.info("Pattern memory cleared")
        except Exception as e:
            logger.error(f"Error clearing pattern memory: {e}")
    
    def save_memory(self) -> None:
        """Explicitly save memory to disk"""
        self._save_patterns()
    
    def get_recent_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent patterns added to memory"""
        recent_count = min(limit, len(self.metadata))
        recent_patterns = []
        
        for i in range(recent_count):
            idx = len(self.metadata) - recent_count + i
            if idx >= 0:
                pattern = {
                    'metadata': self.metadata[idx],
                    'vector': self.vectors[idx].tolist()
                }
                recent_patterns.append(pattern)
        
        return recent_patterns


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    def test_pattern_memory():
        # Create pattern memory
        memory = PatternMemory(dimension=4)
        
        # Test data
        test_patterns = [
            {
                'vector': [0.1, 0.2, 0.3, 0.4],
                'metadata': {'symbol': 'EURUSD', 'price': 1.1000, 'volume': 1000}
            },
            {
                'vector': [0.1, 0.2, 0.3, 0.45],
                'metadata': {'symbol': 'EURUSD', 'price': 1.1005, 'volume': 1100}
            },
            {
                'vector': [0.1, 0.2, 0.3, 0.5],
                'metadata': {'symbol': 'EURUSD', 'price': 1.1010, 'volume': 1200}
            }
        ]
        
        # Add patterns to memory
        for pattern in test_patterns:
            memory.add_pattern(pattern['vector'], pattern['metadata'])
        
        # Test similarity search
        query_vector = [0.1, 0.2, 0.3, 0.42]
        similar = memory.find_similar_patterns(query_vector, k=2)
        
        print("Memory stats:", memory.get_memory_stats())
        print("Similar patterns found:", len(similar))
        for result in similar:
            print(f"Distance: {result['distance']}, Metadata: {result['metadata']}")
        
        # Test recent patterns
        recent = memory.get_recent_patterns(5)
        print("Recent patterns:", len(recent))
        
        # Save memory
        memory.save_memory()
        
        # Test loading
        new_memory = PatternMemory(dimension=4)
        print("Loaded memory stats:", new_memory.get_memory_stats())
    
    test_pattern_memory()
