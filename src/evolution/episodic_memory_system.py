#!/usr/bin/env python3
"""
Episodic Memory System - Phase 2E
==================================

Advanced memory system for context-aware evolution and meta-learning.
Enables the evolution engine to learn from historical patterns and adapt
mutation strategies based on market regimes.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MarketEpisode:
    """Represents a significant market event or regime"""
    episode_id: str
    start_time: datetime
    end_time: datetime
    regime_type: str  # 'BULL', 'BEAR', 'VOLATILE', 'CRISIS', 'RANGE'
    volatility: float
    trend_strength: float
    volume_anomaly: float
    success_patterns: List[str]  # Genome patterns that worked well
    failure_patterns: List[str]  # Genome patterns that failed
    optimal_parameters: Dict[str, float]


@dataclass
class EvolutionMemory:
    """Memory of evolution performance in different contexts"""
    memory_id: str
    timestamp: datetime
    market_regime: str
    genome_pattern: str
    fitness_score: float
    survival_time: int
    adaptation_success: float
    context_features: Dict[str, float]


class EpisodicMemorySystem:
    """Advanced memory system for context-aware evolution"""
    
    def __init__(self, db_path: str = "episodic_memory.db"):
        self.db_path = db_path
        self.connection = None
        self._init_database()
        
    def _init_database(self):
        """Initialize the episodic memory database"""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        
        # Create episodes table
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS market_episodes (
                episode_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                regime_type TEXT,
                volatility REAL,
                trend_strength REAL,
                volume_anomaly REAL,
                success_patterns TEXT,
                failure_patterns TEXT,
                optimal_parameters TEXT
            )
        ''')
        
        # Create evolution memory table
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS evolution_memory (
                memory_id TEXT PRIMARY KEY,
                timestamp TEXT,
                market_regime TEXT,
                genome_pattern TEXT,
                fitness_score REAL,
                survival_time INTEGER,
                adaptation_success REAL,
                context_features TEXT
            )
        ''')
        
        self.connection.commit()
    
    def record_episode(self, episode: MarketEpisode):
        """Record a market episode to memory"""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO market_episodes 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            episode.episode_id,
            episode.start_time.isoformat(),
            episode.end_time.isoformat(),
            episode.regime_type,
            episode.volatility,
            episode.trend_strength,
            episode.volume_anomaly,
            json.dumps(episode.success_patterns),
            json.dumps(episode.failure_patterns),
            json.dumps(episode.optimal_parameters)
        ))
        self.connection.commit()
    
    def record_evolution_memory(self, memory: EvolutionMemory):
        """Record evolution performance to memory"""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO evolution_memory 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.memory_id,
            memory.timestamp.isoformat(),
            memory.market_regime,
            memory.genome_pattern,
            memory.fitness_score,
            memory.survival_time,
            memory.adaptation_success,
            json.dumps(memory.context_features)
        ))
        self.connection.commit()
    
    def get_similar_episodes(self, current_context: Dict[str, float], 
                           threshold: float = 0.7) -> List[MarketEpisode]:
        """Find episodes similar to current market context"""
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM market_episodes ORDER BY start_time DESC LIMIT 100')
        
        similar_episodes = []
        for row in cursor.fetchall():
            episode = MarketEpisode(
                episode_id=row['episode_id'],
                start_time=datetime.fromisoformat(row['start_time']),
                end_time=datetime.fromisoformat(row['end_time']),
                regime_type=row['regime_type'],
                volatility=row['volatility'],
                trend_strength=row['trend_strength'],
                volume_anomaly=row['volume_anomaly'],
                success_patterns=json.loads(row['success_patterns']),
                failure_patterns=json.loads(row['failure_patterns']),
                optimal_parameters=json.loads(row['optimal_parameters'])
            )
            
            # Calculate similarity score
            similarity = self._calculate_context_similarity(
                current_context,
                {
                    'volatility': episode.volatility,
                    'trend_strength': episode.trend_strength,
                    'volume_anomaly': episode.volume_anomaly
                }
            )
            
            if similarity >= threshold:
                similar_episodes.append(episode)
        
        return similar_episodes
    
    def get_optimal_parameters(self, regime_type: str) -> Dict[str, float]:
        """Get optimal evolution parameters for a given regime"""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT optimal_parameters FROM market_episodes 
            WHERE regime_type = ? ORDER BY end_time DESC LIMIT 5
        ''', (regime_type,))
        
        parameters_list = []
        for row in cursor.fetchall():
            params = json.loads(row['optimal_parameters'])
            parameters_list.append(params)
        
        if not parameters_list:
            return self._get_default_parameters()
        
        # Average the parameters from similar regimes
        averaged_params = {}
        for key in parameters_list[0].keys():
            values = [p[key] for p in parameters_list if key in p]
            averaged_params[key] = np.mean(values) if values else 0.5
        
        return averaged_params
    
    def get_success_patterns(self, regime_type: str) -> List[str]:
        """Get genome patterns that succeeded in similar regimes"""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT success_patterns FROM market_episodes 
            WHERE regime_type = ? ORDER BY end_time DESC LIMIT 10
        ''', (regime_type,))
        
        patterns = []
        for row in cursor.fetchall():
            patterns.extend(json.loads(row['success_patterns']))
        
        return list(set(patterns))  # Remove duplicates
    
    def get_context_aware_mutation_rate(self, current_context: Dict[str, float]) -> float:
        """Calculate context-aware mutation rate"""
        similar_episodes = self.get_similar_episodes(current_context)
        
        if not similar_episodes:
            return 0.3  # Default mutation rate
        
        # Calculate average adaptation success
        cursor = self.connection.cursor()
        episode_ids = [e.episode_id for e in similar_episodes]
        
        success_rates = []
        for episode_id in episode_ids:
            cursor.execute('''
                SELECT adaptation_success FROM evolution_memory 
                WHERE market_regime = ? ORDER BY timestamp DESC LIMIT 1
            ''', (episode_id,))
            
            row = cursor.fetchone()
            if row and row['adaptation_success']:
                success_rates.append(row['adaptation_success'])
        
        if not success_rates:
            return 0.3
        
        # Lower mutation rate if similar contexts had high success
        avg_success = np.mean(success_rates)
        return max(0.1, min(0.5, 0.3 * (2 - avg_success)))
    
    def _calculate_context_similarity(self, ctx1: Dict[str, float], 
                                    ctx2: Dict[str, float]) -> float:
        """Calculate similarity between two market contexts"""
        if not ctx1 or not ctx2:
            return 0.0
        
        # Normalize values to 0-1 range
        normalized_ctx1 = self._normalize_context(ctx1)
        normalized_ctx2 = self._normalize_context(ctx2)
        
        # Calculate cosine similarity
        dot_product = sum(normalized_ctx1[k] * normalized_ctx2[k] 
                         for k in normalized_ctx1.keys() 
                         if k in normalized_ctx2)
        
        magnitude1 = np.sqrt(sum(v**2 for v in normalized_ctx1.values()))
        magnitude2 = np.sqrt(sum(v**2 for v in normalized_ctx2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _normalize_context(self, context: Dict[str, float]) -> Dict[str, float]:
        """Normalize context values to 0-1 range"""
        normalized = {}
        for key, value in context.items():
            # Simple min-max normalization based on reasonable ranges
            ranges = {
                'volatility': (0, 0.1),
                'trend_strength': (-1, 1),
                'volume_anomaly': (0, 5)
            }
            
            min_val, max_val = ranges.get(key, (0, 1))
            normalized[key] = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        
        return normalized
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default evolution parameters"""
        return {
            'mutation_rate': 0.3,
            'crossover_rate': 0.7,
            'selection_pressure': 2.0,
            'diversity_threshold': 0.1,
            'convergence_threshold': 0.001
        }
    
    def analyze_market_regime(self, market_data: Dict[str, float]) -> str:
        """Analyze current market regime based on data"""
        volatility = market_data.get('volatility', 0)
        trend_strength = market_data.get('trend_strength', 0)
        volume_anomaly = market_data.get('volume_anomaly', 0)
        
        if volatility > 0.05:
            return 'CRISIS'
        elif volatility > 0.02:
            return 'VOLATILE'
        elif abs(trend_strength) > 0.7:
            return 'BULL' if trend_strength > 0 else 'BEAR'
        else:
            return 'RANGE'
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of stored memories"""
        cursor = self.connection.cursor()
        
        # Episode summary
        cursor.execute('SELECT COUNT(*) as episode_count FROM market_episodes')
        episode_count = cursor.fetchone()['episode_count']
        
        cursor.execute('SELECT regime_type, COUNT(*) as count FROM market_episodes GROUP BY regime_type')
        regime_counts = dict(cursor.fetchall())
        
        # Memory summary
        cursor.execute('SELECT COUNT(*) as memory_count FROM evolution_memory')
        memory_count = cursor.fetchone()['memory_count']
        
        cursor.execute('SELECT AVG(fitness_score) as avg_fitness FROM evolution_memory')
        avg_fitness = cursor.fetchone()['avg_fitness'] or 0
        
        return {
            'total_episodes': episode_count,
            'regime_distribution': regime_counts,
            'total_memories': memory_count,
            'average_fitness': avg_fitness,
            'database_path': self.db_path
        }
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class ContextAwareEvolutionEngine:
    """Evolution engine enhanced with episodic memory"""
    
    def __init__(self, episodic_memory: EpisodicMemorySystem):
        self.memory = episodic_memory
        
    def get_context_aware_parameters(self, market_context: Dict[str, float]) -> Dict[str, float]:
        """Get evolution parameters based on market context"""
        regime = self.memory.analyze_market_regime(market_context)
        
        # Get optimal parameters for this regime
        optimal_params = self.memory.get_optimal_parameters(regime)
        
        # Adjust mutation rate based on context similarity
        mutation_rate = self.memory.get_context_aware_mutation_rate(market_context)
        optimal_params['mutation_rate'] = mutation_rate
        
        # Get successful patterns
        success_patterns = self.memory.get_success_patterns(regime)
        
        return {
            'parameters': optimal_params,
            'success_patterns': success_patterns,
            'regime': regime
        }
    
    def record_evolution_outcome(self, market_context: Dict[str, float], 
                               genome_pattern: str, fitness_score: float,
                               survival_time: int, adaptation_success: float):
        """Record evolution outcome to memory"""
        regime = self.memory.analyze_market_regime(market_context)
        
        memory = EvolutionMemory(
            memory_id=f"{datetime.now().isoformat()}_{genome_pattern}",
            timestamp=datetime.now(),
            market_regime=regime,
            genome_pattern=genome_pattern,
            fitness_score=fitness_score,
            survival_time=survival_time,
            adaptation_success=adaptation_success,
            context_features=market_context
        )
        
        self.memory.record_evolution_memory(memory)


# Example usage and testing
async def test_episodic_memory():
    """Test the episodic memory system"""
    memory = EpisodicMemorySystem(":memory:")  # Use in-memory DB for testing
    
    # Record some test episodes
    episode = MarketEpisode(
        episode_id="test_2020_crash",
        start_time=datetime(2020, 2, 20),
        end_time=datetime(2020, 3, 23),
        regime_type="CRISIS",
        volatility=0.08,
        trend_strength=-0.9,
        volume_anomaly=3.5,
        success_patterns=["pattern_1", "pattern_2"],
        failure_patterns=["pattern_3", "pattern_4"],
        optimal_parameters={"mutation_rate": 0.1, "crossover_rate": 0.8}
    )
    
    memory.record_episode(episode)
    
    # Test context-aware parameters
    context = {"volatility": 0.07, "trend_strength": -0.8, "volume_anomaly": 3.0}
    engine = ContextAwareEvolutionEngine(memory)
    
    params = engine.get_context_aware_parameters(context)
    print("Context-aware parameters:", params)
    
    # Record evolution outcome
    engine.record_evolution_outcome(
        market_context=context,
        genome_pattern="test_pattern",
        fitness_score=0.85,
        survival_time=100,
        adaptation_success=0.9
    )
    
    summary = memory.get_memory_summary()
    print("Memory summary:", summary)
    
    memory.close()


if __name__ == "__main__":
    asyncio.run(test_episodic_memory())
