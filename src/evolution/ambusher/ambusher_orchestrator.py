#!/usr/bin/env python3
"""
AmbusherOrchestrator - Epic 2: Evolving "The Ambusher"
Main orchestrator for the ambusher evolution system.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.evolution.engine import EvolutionEngine  # consolidated engine

AmbusherGenome = Dict[str, Any]  # simplified alias for compatibility
try:
    from src.evolution.ambusher.ambusher_fitness import AmbusherFitnessFunction
except Exception:  # pragma: no cover
    class AmbusherFitnessFunction:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            pass

logger = logging.getLogger(__name__)

class AmbusherOrchestrator:
    """Main orchestrator for ambusher evolution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.genetic_engine = EvolutionEngine()
        self.fitness_function = AmbusherFitnessFunction(config.get('fitness', {}))
        
        # State tracking
        self.is_active = False
        self.current_genome: Optional[AmbusherGenome] = None
        self.evolution_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            'evolutions_completed': 0,
            'best_fitness': 0.0,
            'total_trades': 0,
            'total_pnl': 0.0
        }
        
        # Paths
        self.genome_path = Path(config.get('genome_path', 'data/evolution/ambusher_genome.json'))
        self.history_path = Path(config.get('history_path', 'data/evolution/ambusher_history.json'))
        
        # Create directories
        self.genome_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def start(self):
        """Start the ambusher orchestrator."""
        logger.info("Starting Ambusher Orchestrator...")
        self.is_active = True
        
        # Load existing genome
        if self.genome_path.exists():
            self.current_genome = self.genetic_engine.load_genome(str(self.genome_path))
            logger.info("Loaded existing ambusher genome")
        else:
            logger.info("No existing genome found, will create new one")
            
    async def stop(self):
        """Stop the ambusher orchestrator."""
        logger.info("Stopping Ambusher Orchestrator...")
        self.is_active = False
        
    async def evolve_strategy(self, market_data: Dict[str, Any], trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve a new ambush strategy."""
        if not self.is_active:
            logger.warning("Ambusher Orchestrator is not active")
            return {}
            
        logger.info("Starting ambusher evolution...")
        
        # Run consolidated evolution engine (placeholder behavior)
        _ = self.genetic_engine.evolve()
        best_genome = {"params": {}, "metadata": {"ts": datetime.utcnow().isoformat()}}
        summary = {"best_fitness": 0.0}
        
        # Save genome
        with open(self.genome_path, 'w') as f:
            json.dump(best_genome, f)
        
        # Update current genome
        self.current_genome = best_genome
        
        # Update metrics
        self.performance_metrics['evolutions_completed'] += 1
        self.performance_metrics['best_fitness'] = summary['best_fitness']
        
        # Save evolution history
        evolution_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': summary,
            'metrics': self.performance_metrics.copy()
        }
        self.evolution_history.append(evolution_record)
        
        # Save history
        with open(self.history_path, 'w') as f:
            json.dump(self.evolution_history, f, indent=2)
            
        logger.info(f"Ambusher evolution completed. Best fitness: {summary['best_fitness']:.4f}")
        
        return {
            'genome': best_genome,
            'fitness': summary['best_fitness'],
            'evolution_record': evolution_record
        }
        
    def get_current_strategy(self) -> Optional[Dict[str, Any]]:
        """Get the current ambush strategy."""
        if self.current_genome:
            if hasattr(self.current_genome, 'to_dict'):
                return self.current_genome.to_dict()
            return self.current_genome
        return None
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()
        
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get evolution history."""
        return self.evolution_history.copy()
        
    async def reset(self):
        """Reset the ambusher orchestrator."""
        logger.info("Resetting Ambusher Orchestrator...")
        self.current_genome = None
        self.evolution_history.clear()
        self.performance_metrics = {
            'evolutions_completed': 0,
            'best_fitness': 0.0,
            'total_trades': 0,
            'total_pnl': 0.0
        }
        
        # Remove files
        if self.genome_path.exists():
            self.genome_path.unlink()
        if self.history_path.exists():
            self.history_path.unlink()
            
        logger.info("Ambusher Orchestrator reset complete")
        
    def should_evolve(self, current_fitness: float, threshold: float = 0.7) -> bool:
        """Determine if evolution should be triggered."""
        return current_fitness < threshold
        
    def update_trade_metrics(self, trade_data: Dict[str, Any]):
        """Update trade-based metrics."""
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['total_pnl'] += trade_data.get('pnl', 0.0)
