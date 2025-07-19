"""
EMP Population Manager v1.1

Manages genome populations, their lifecycle, and provides
population-level operations for the genetic engine.
"""

import asyncio
import logging
import random
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass

from src.genome.models.genome import DecisionGenome
from src.core.exceptions import EvolutionException
from src.core.event_bus import event_bus

logger = logging.getLogger(__name__)


@dataclass
class PopulationMetrics:
    """Metrics for a genome population."""
    size: int
    average_fitness: float
    best_fitness: float
    worst_fitness: float
    diversity_score: float
    age_distribution: Dict[str, int]
    fitness_distribution: Dict[str, int]
    generation_count: int


class PopulationManager:
    """Manages genome populations and their lifecycle."""
    
    def __init__(self, max_population_size: int = 100, archive_size: int = 50):
        self.max_population_size = max_population_size
        self.archive_size = archive_size
        self.population: List[DecisionGenome] = []
        self.archive: List[DecisionGenome] = []
        self.generation_count = 0
        self.total_genomes_created = 0
        self.total_genomes_evaluated = 0
        
        # Tracking sets
        self._genome_ids: Set[str] = set()
        self._fitness_history: Dict[str, List[float]] = {}
        
        logger.info(f"Population Manager initialized with max size {max_population_size}")
        
    async def add_genomes(self, genomes: List[DecisionGenome]) -> bool:
        """Add genomes to the population."""
        try:
            added_count = 0
            
            for genome in genomes:
                if genome.genome_id not in self._genome_ids:
                    # Validate genome
                    if not genome.validate():
                        logger.warning(f"Invalid genome {genome.genome_id} rejected")
                        continue
                        
                    # Add to population
                    self.population.append(genome)
                    self._genome_ids.add(genome.genome_id)
                    self.total_genomes_created += 1
                    added_count += 1
                    
                    # Track fitness history
                    self._fitness_history[genome.genome_id] = []
                    
                    # Emit genome added event
                    await event_bus.publish('population.genome.added', {
                        'genome_id': genome.genome_id,
                        'population_size': len(self.population)
                    })
                    
            # Maintain population size
            if len(self.population) > self.max_population_size:
                await self._trim_population()
                
            logger.info(f"Added {added_count} genomes to population")
            return added_count > 0
            
        except Exception as e:
            raise EvolutionException(f"Error adding genomes: {e}")
            
    async def remove_genomes(self, genome_ids: List[str]) -> int:
        """Remove genomes from the population."""
        try:
            removed_count = 0
            
            for genome_id in genome_ids:
                if genome_id in self._genome_ids:
                    # Remove from population
                    self.population = [g for g in self.population if g.genome_id != genome_id]
                    self._genome_ids.remove(genome_id)
                    removed_count += 1
                    
                    # Remove from fitness history
                    if genome_id in self._fitness_history:
                        del self._fitness_history[genome_id]
                        
                    # Emit genome removed event
                    await event_bus.publish('population.genome.removed', {
                        'genome_id': genome_id,
                        'population_size': len(self.population)
                    })
                    
            logger.info(f"Removed {removed_count} genomes from population")
            return removed_count
            
        except Exception as e:
            raise EvolutionException(f"Error removing genomes: {e}")
            
    async def update_fitness(self, genome_id: str, fitness_score: float) -> bool:
        """Update fitness score for a genome."""
        try:
            if genome_id not in self._genome_ids:
                return False
                
            # Find genome and update fitness
            for genome in self.population:
                if genome.genome_id == genome_id:
                    genome.fitness_score = fitness_score
                    genome.generation = self.generation_count
                    
                    # Track fitness history
                    if genome_id in self._fitness_history:
                        self._fitness_history[genome_id].append(fitness_score)
                    
                    self.total_genomes_evaluated += 1
                    
                    # Emit fitness updated event
                    await event_bus.publish('population.fitness.updated', {
                        'genome_id': genome_id,
                        'fitness_score': fitness_score,
                        'generation': self.generation_count
                    })
                    
                    return True
                    
            return False
            
        except Exception as e:
            raise EvolutionException(f"Error updating fitness: {e}")
            
    async def select_elite(self, elite_size: int) -> List[DecisionGenome]:
        """Select elite genomes based on fitness."""
        try:
            # Sort population by fitness
            sorted_population = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
            
            # Select elite
            elite = sorted_population[:elite_size]
            
            logger.info(f"Selected {len(elite)} elite genomes")
            return elite
            
        except Exception as e:
            raise EvolutionException(f"Error selecting elite: {e}")
            
    async def select_tournament(self, tournament_size: int, num_winners: int = 1) -> List[DecisionGenome]:
        """Select genomes using tournament selection."""
        try:
            winners = []
            
            for _ in range(num_winners):
                if len(self.population) < tournament_size:
                    # If population is smaller than tournament size, return random selection
                    tournament = self.population
                else:
                    # Select random tournament
                    tournament = random.sample(self.population, tournament_size)
                    
                # Select winner (highest fitness)
                winner = max(tournament, key=lambda g: g.fitness_score)
                winners.append(winner)
                
            return winners
            
        except Exception as e:
            raise EvolutionException(f"Error in tournament selection: {e}")
            
    async def archive_best_genomes(self, archive_count: Optional[int] = None) -> int:
        """Archive the best genomes for preservation."""
        try:
            count = archive_count or self.archive_size
            
            # Get best genomes
            sorted_population = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
            best_genomes = sorted_population[:count]
            
            # Add to archive
            archived_count = 0
            for genome in best_genomes:
                if genome.genome_id not in [g.genome_id for g in self.archive]:
                    self.archive.append(genome)
                    archived_count += 1
                    
            # Maintain archive size
            if len(self.archive) > self.archive_size:
                self.archive = sorted(self.archive, key=lambda g: g.fitness_score, reverse=True)[:self.archive_size]
                
            logger.info(f"Archived {archived_count} best genomes")
            return archived_count
            
        except Exception as e:
            raise EvolutionException(f"Error archiving genomes: {e}")
            
    async def inject_archive_genomes(self, injection_count: int) -> int:
        """Inject archived genomes back into population."""
        try:
            if not self.archive:
                return 0
                
            # Select random archived genomes
            if len(self.archive) <= injection_count:
                selected = self.archive.copy()
            else:
                selected = random.sample(self.archive, injection_count)
                
            # Add to population
            injected_count = 0
            for genome in selected:
                if genome.genome_id not in self._genome_ids:
                    # Create copy with new ID
                    new_genome = DecisionGenome.from_dict(genome.to_dict())
                    new_genome.genome_id = f"archive_{genome.genome_id}_{datetime.now().timestamp()}"
                    new_genome.parent_ids = [genome.genome_id]
                    
                    self.population.append(new_genome)
                    self._genome_ids.add(new_genome.genome_id)
                    self._fitness_history[new_genome.genome_id] = []
                    injected_count += 1
                    
            logger.info(f"Injected {injected_count} archived genomes")
            return injected_count
            
        except Exception as e:
            raise EvolutionException(f"Error injecting archived genomes: {e}")
            
    async def _trim_population(self):
        """Trim population to maintain size limit."""
        try:
            if len(self.population) <= self.max_population_size:
                return
                
            # Sort by fitness and keep best
            sorted_population = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
            self.population = sorted_population[:self.max_population_size]
            
            # Update tracking sets
            self._genome_ids = {g.genome_id for g in self.population}
            
            # Clean up fitness history
            removed_ids = set(self._fitness_history.keys()) - self._genome_ids
            for genome_id in removed_ids:
                del self._fitness_history[genome_id]
                
            logger.info(f"Trimmed population to {len(self.population)} genomes")
            
        except Exception as e:
            raise EvolutionException(f"Error trimming population: {e}")
            
    def get_population_metrics(self) -> PopulationMetrics:
        """Get comprehensive population metrics."""
        try:
            if not self.population:
                return PopulationMetrics(
                    size=0,
                    average_fitness=0.0,
                    best_fitness=0.0,
                    worst_fitness=0.0,
                    diversity_score=0.0,
                    age_distribution={},
                    fitness_distribution={},
                    generation_count=self.generation_count
                )
                
            # Calculate basic metrics
            fitness_scores = [g.fitness_score for g in self.population]
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness = max(fitness_scores)
            worst_fitness = min(fitness_scores)
            
            # Calculate diversity
            diversity_score = self._calculate_diversity()
            
            # Age distribution
            age_distribution = {}
            for genome in self.population:
                age = self.generation_count - genome.generation
                age_key = f"gen_{age}"
                age_distribution[age_key] = age_distribution.get(age_key, 0) + 1
                
            # Fitness distribution
            fitness_distribution = {}
            for score in fitness_scores:
                if score >= 0.9:
                    bucket = "0.9-1.0"
                elif score >= 0.8:
                    bucket = "0.8-0.9"
                elif score >= 0.7:
                    bucket = "0.7-0.8"
                elif score >= 0.6:
                    bucket = "0.6-0.7"
                elif score >= 0.5:
                    bucket = "0.5-0.6"
                else:
                    bucket = "0.0-0.5"
                    
                fitness_distribution[bucket] = fitness_distribution.get(bucket, 0) + 1
                
            return PopulationMetrics(
                size=len(self.population),
                average_fitness=average_fitness,
                best_fitness=best_fitness,
                worst_fitness=worst_fitness,
                diversity_score=diversity_score,
                age_distribution=age_distribution,
                fitness_distribution=fitness_distribution,
                generation_count=self.generation_count
            )
            
        except Exception as e:
            raise EvolutionException(f"Error calculating population metrics: {e}")
            
    def _calculate_diversity(self) -> float:
        """Calculate population diversity score."""
        if len(self.population) < 2:
            return 0.0
            
        # Calculate diversity based on fitness variance
        fitness_scores = [g.fitness_score for g in self.population]
        mean_fitness = sum(fitness_scores) / len(fitness_scores)
        variance = sum((score - mean_fitness) ** 2 for score in fitness_scores) / len(fitness_scores)
        
        return min(variance, 1.0)  # Normalize to [0, 1]
        
    def get_best_genome(self) -> Optional[DecisionGenome]:
        """Get the best genome in the population."""
        if not self.population:
            return None
            
        return max(self.population, key=lambda g: g.fitness_score)
        
    def get_genome_by_id(self, genome_id: str) -> Optional[DecisionGenome]:
        """Get genome by ID."""
        for genome in self.population:
            if genome.genome_id == genome_id:
                return genome
        return None
        
    def get_fitness_history(self, genome_id: str) -> List[float]:
        """Get fitness history for a genome."""
        return self._fitness_history.get(genome_id, [])
        
    def increment_generation(self):
        """Increment generation counter."""
        self.generation_count += 1
        
    def get_summary(self) -> Dict[str, Any]:
        """Get population summary."""
        metrics = self.get_population_metrics()
        
        return {
            'population_size': len(self.population),
            'archive_size': len(self.archive),
            'generation_count': self.generation_count,
            'total_genomes_created': self.total_genomes_created,
            'total_genomes_evaluated': self.total_genomes_evaluated,
            'best_fitness': metrics.best_fitness,
            'average_fitness': metrics.average_fitness,
            'diversity_score': metrics.diversity_score,
            'age_distribution': metrics.age_distribution,
            'fitness_distribution': metrics.fitness_distribution
        } 