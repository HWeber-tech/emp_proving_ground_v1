"""
Evolution Engine for the EMP Proving Ground system.

This module provides the evolutionary decision system:
- EvolutionEngine: Main evolution engine
- DecisionGenome: Decision tree genome
- FitnessEvaluator: Fitness evaluation system
"""

import logging
import random
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


@dataclass
class TradingAction:
    """Represents a trading action decision"""
    action_type: ActionType
    size_factor: float = 1.0
    confidence_threshold: float = 0.5
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NodeType(Enum):
    CONDITION = "condition"
    ACTION = "action"


@dataclass
class DecisionNode:
    """Node in the decision tree genome"""
    node_id: str
    node_type: NodeType
    dimension: Optional[str] = None
    operator: Optional[str] = None
    threshold: Optional[float] = None
    action: Optional[TradingAction] = None
    left_child: Optional["DecisionNode"] = None
    right_child: Optional["DecisionNode"] = None
    creation_generation: int = 0
    usage_count: int = 0
    success_rate: float = 0.0


class DecisionGenome:
    """Represents a decision tree genome for trading decisions"""
    
    def __init__(self, max_depth: int = 10, max_nodes: int = 100):
        self.genome_id = str(uuid.uuid4())
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.root = None
        self.node_count = 0
        self.generation = 0
        self.fitness = 0.0
        
        self._initialize_random_tree()
        logger.info(f"Created decision genome {self.genome_id}")
    
    def _initialize_random_tree(self):
        """Initialize with a random decision tree"""
        self.root = self._create_random_node(0)
        self.node_count = self._count_nodes(self.root)
    
    def _create_random_node(self, depth: int) -> DecisionNode:
        """Create a random decision node"""
        if depth >= self.max_depth or self.node_count >= self.max_nodes:
            action_type = random.choice(list(ActionType))
            action = TradingAction(
                action_type=action_type,
                size_factor=random.uniform(0.5, 2.0),
                confidence_threshold=random.uniform(0.3, 0.8)
            )
            
            return DecisionNode(
                node_id=str(uuid.uuid4()),
                node_type=NodeType.ACTION,
                action=action,
                creation_generation=self.generation
            )
        
        if random.random() < 0.7:
            dimension = random.choice(["why", "how", "what", "when", "anomaly"])
            operator = random.choice([">", "<", ">=", "<=", "=="])
            threshold = random.uniform(-1.0, 1.0)
            
            node = DecisionNode(
                node_id=str(uuid.uuid4()),
                node_type=NodeType.CONDITION,
                dimension=dimension,
                operator=operator,
                threshold=threshold,
                creation_generation=self.generation
            )
            
            node.left_child = self._create_random_node(depth + 1)
            node.right_child = self._create_random_node(depth + 1)
            
            return node
        else:
            action_type = random.choice(list(ActionType))
            action = TradingAction(
                action_type=action_type,
                size_factor=random.uniform(0.5, 2.0),
                confidence_threshold=random.uniform(0.3, 0.8)
            )
            
            return DecisionNode(
                node_id=str(uuid.uuid4()),
                node_type=NodeType.ACTION,
                action=action,
                creation_generation=self.generation
            )
    
    def _count_nodes(self, node: DecisionNode) -> int:
        """Count total nodes in tree"""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left_child) + self._count_nodes(node.right_child)
    
    def decide(self, sensory_reading) -> Optional[TradingAction]:
        """Make a trading decision based on sensory reading"""
        if self.root is None:
            return None
        
        return self._traverse_tree(self.root, sensory_reading)
    
    def _traverse_tree(self, node: DecisionNode, sensory_reading) -> Optional[TradingAction]:
        """Traverse the decision tree"""
        if node is None:
            return None
        
        if node.node_type == NodeType.ACTION:
            node.usage_count += 1
            return node.action
        
        elif node.node_type == NodeType.CONDITION:
            value = self._get_dimension_value(node.dimension, sensory_reading)
            if value is None:
                return None
            
            condition_met = self._evaluate_condition(value, node.operator, node.threshold)
            node.usage_count += 1
            
            if condition_met:
                return self._traverse_tree(node.left_child, sensory_reading)
            else:
                return self._traverse_tree(node.right_child, sensory_reading)
        
        return None
    
    def _get_dimension_value(self, dimension: str, sensory_reading) -> Optional[float]:
        """Get value for a specific dimension"""
        if dimension == "why":
            return sensory_reading.why_score
        elif dimension == "how":
            return sensory_reading.how_score
        elif dimension == "what":
            return sensory_reading.what_score
        elif dimension == "when":
            return sensory_reading.when_score
        elif dimension == "anomaly":
            return sensory_reading.anomaly_score
        else:
            return None
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a condition"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.01
        else:
            return False
    
    def mutate(self, mutation_rate: float = 0.1) -> "DecisionGenome":
        """Create a mutated copy of this genome"""
        new_genome = DecisionGenome(self.max_depth, self.max_nodes)
        new_genome.root = self._copy_and_mutate_node(self.root, mutation_rate)
        new_genome.node_count = self._count_nodes(new_genome.root)
        new_genome.generation = self.generation + 1
        
        return new_genome
    
    def _copy_and_mutate_node(self, node: DecisionNode, mutation_rate: float) -> DecisionNode:
        """Copy and potentially mutate a node"""
        if node is None:
            return None
        
        new_node = DecisionNode(
            node_id=str(uuid.uuid4()),
            node_type=node.node_type,
            dimension=node.dimension,
            operator=node.operator,
            threshold=node.threshold,
            action=node.action,
            creation_generation=self.generation + 1
        )
        
        if random.random() < mutation_rate:
            self._mutate_node(new_node, mutation_rate)
        
        new_node.left_child = self._copy_and_mutate_node(node.left_child, mutation_rate)
        new_node.right_child = self._copy_and_mutate_node(node.right_child, mutation_rate)
        
        return new_node
    
    def _mutate_node(self, node: DecisionNode, mutation_rate: float):
        """Mutate a single node"""
        if node.node_type == NodeType.CONDITION:
            if random.random() < 0.3:
                node.dimension = random.choice(["why", "how", "what", "when", "anomaly"])
            if random.random() < 0.3:
                node.operator = random.choice([">", "<", ">=", "<=", "=="])
            if random.random() < 0.3:
                node.threshold = random.uniform(-1.0, 1.0)
        
        elif node.node_type == NodeType.ACTION:
            if random.random() < 0.3:
                node.action.action_type = random.choice(list(ActionType))
            if random.random() < 0.3:
                node.action.size_factor = random.uniform(0.5, 2.0)
            if random.random() < 0.3:
                node.action.confidence_threshold = random.uniform(0.3, 0.8)
    
    def get_complexity(self) -> Dict[str, int]:
        """Get complexity metrics"""
        return {
            'total_nodes': self.node_count,
            'max_depth': self._get_depth(self.root),
            'action_nodes': self._count_action_nodes(self.root),
            'condition_nodes': self._count_condition_nodes(self.root)
        }
    
    def _get_depth(self, node: DecisionNode) -> int:
        """Get depth of tree"""
        if node is None:
            return 0
        return 1 + max(self._get_depth(node.left_child), self._get_depth(node.right_child))
    
    def _count_action_nodes(self, node: DecisionNode) -> int:
        """Count action nodes"""
        if node is None:
            return 0
        count = 1 if node.node_type == NodeType.ACTION else 0
        return count + self._count_action_nodes(node.left_child) + self._count_action_nodes(node.right_child)
    
    def _count_condition_nodes(self, node: DecisionNode) -> int:
        """Count condition nodes"""
        if node is None:
            return 0
        count = 1 if node.node_type == NodeType.CONDITION else 0
        return count + self._count_condition_nodes(node.left_child) + self._count_condition_nodes(node.right_child)


@dataclass
class EvolutionConfig:
    """Configuration for evolution engine"""
    population_size: int = 500
    elite_ratio: float = 0.1
    crossover_ratio: float = 0.6
    mutation_ratio: float = 0.3
    mutation_rate: float = 0.1
    max_stagnation: int = 20
    complexity_penalty: float = 0.01
    min_fitness_improvement: float = 0.001


@dataclass
class GenerationStats:
    """Statistics for a generation"""
    generation: int
    population_size: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    diversity_score: float
    stagnation_count: int
    elite_count: int
    new_genomes: int
    complexity_stats: Dict[str, float]


class EvolutionEngine:
    """Main evolution engine for evolving trading strategies"""
    
    def __init__(self, config: EvolutionConfig, fitness_evaluator):
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.population: List[DecisionGenome] = []
        self.best_genome: Optional[DecisionGenome] = None
        self.best_fitness: float = float('-inf')
        self.generation = 0
        self.stagnation_count = 0
        self.generation_history: List[GenerationStats] = []
        
        logger.info("Initialized evolution engine")
    
    def initialize_population(self, seed: Optional[int] = None) -> bool:
        """Initialize the population with random genomes"""
        try:
            if seed is not None:
                random.seed(seed)
            
            self.population = []
            for _ in range(self.config.population_size):
                genome = DecisionGenome()
                self.population.append(genome)
            
            logger.info(f"Initialized population with {len(self.population)} genomes")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            return False
    
    def evolve_generation(self) -> GenerationStats:
        """Evolve one generation"""
        try:
            self._evaluate_population()
            self._update_best_genome()
            self._check_stagnation()
            self._create_next_generation()
            
            stats = self._calculate_generation_stats()
            self.generation_history.append(stats)
            self.generation += 1
            
            logger.info(f"Completed generation {self.generation}, best fitness: {self.best_fitness:.4f}")
            return stats
            
        except Exception as e:
            logger.error(f"Error evolving generation: {e}")
            return GenerationStats(
                generation=self.generation,
                population_size=len(self.population),
                best_fitness=0.0,
                average_fitness=0.0,
                worst_fitness=0.0,
                diversity_score=0.0,
                stagnation_count=self.stagnation_count,
                elite_count=0,
                new_genomes=0,
                complexity_stats={}
            )
    
    def _evaluate_population(self):
        """Evaluate fitness of all genomes in population"""
        for genome in self.population:
            try:
                fitness = self.fitness_evaluator(genome)
                genome.fitness = fitness
            except Exception as e:
                logger.warning(f"Error evaluating genome {genome.genome_id}: {e}")
                genome.fitness = 0.0
    
    def _update_best_genome(self):
        """Update the best genome found so far"""
        for genome in self.population:
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome
    
    def _check_stagnation(self):
        """Check if evolution has stagnated"""
        if len(self.generation_history) < 2:
            return
        
        current_best = self.generation_history[-1].best_fitness
        previous_best = self.generation_history[-2].best_fitness
        improvement = current_best - previous_best
        
        if improvement < self.config.min_fitness_improvement:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
    
    def _create_next_generation(self):
        """Create the next generation"""
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        elite_count = int(self.config.population_size * self.config.elite_ratio)
        crossover_count = int(self.config.population_size * self.config.crossover_ratio)
        mutation_count = self.config.population_size - elite_count - crossover_count
        
        new_population = []
        new_population.extend(self.population[:elite_count])
        
        # Create offspring through crossover and mutation
        for _ in range(crossover_count):
            parent = random.choice(self.population[:elite_count * 2])
            mutated = parent.mutate(self.config.mutation_rate)
            new_population.append(mutated)
        
        for _ in range(mutation_count):
            parent = random.choice(self.population[:elite_count * 2])
            mutated = parent.mutate(self.config.mutation_rate)
            new_population.append(mutated)
        
        self.population = new_population[:self.config.population_size]
    
    def _calculate_generation_stats(self) -> GenerationStats:
        """Calculate statistics for current generation"""
        if not self.population:
            return GenerationStats(
                generation=self.generation,
                population_size=0,
                best_fitness=0.0,
                average_fitness=0.0,
                worst_fitness=0.0,
                diversity_score=0.0,
                stagnation_count=self.stagnation_count,
                elite_count=0,
                new_genomes=0,
                complexity_stats={}
            )
        
        fitnesses = [g.fitness for g in self.population]
        
        complexity_stats = {}
        if self.population:
            complexities = [g.get_complexity() for g in self.population]
            for key in complexities[0].keys():
                values = [c[key] for c in complexities]
                complexity_stats[f"avg_{key}"] = sum(values) / len(values)
                complexity_stats[f"max_{key}"] = max(values)
        
        return GenerationStats(
            generation=self.generation,
            population_size=len(self.population),
            best_fitness=max(fitnesses),
            average_fitness=sum(fitnesses) / len(fitnesses),
            worst_fitness=min(fitnesses),
            diversity_score=self._calculate_diversity(),
            stagnation_count=self.stagnation_count,
            elite_count=int(len(self.population) * self.config.elite_ratio),
            new_genomes=len(self.population),
            complexity_stats=complexity_stats
        )
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 0.0
        
        fitnesses = [g.fitness for g in self.population]
        mean_fitness = sum(fitnesses) / len(fitnesses)
        variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        
        return variance
    
    def get_population_summary(self) -> Dict:
        """Get summary of current population"""
        if not self.population:
            return {}
        
        fitnesses = [g.fitness for g in self.population]
        
        return {
            'population_size': len(self.population),
            'best_fitness': max(fitnesses),
            'average_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': min(fitnesses),
            'generation': self.generation,
            'stagnation_count': self.stagnation_count
        }
    
    def get_best_genomes(self, count: int = 10) -> List[DecisionGenome]:
        """Get the best genomes from current population"""
        sorted_population = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        return sorted_population[:count]


class FitnessEvaluator:
    """Evaluates fitness of decision genomes"""
    
    def __init__(self, data_storage, evaluation_period_days: int = 30):
        self.data_storage = data_storage
        self.evaluation_period_days = evaluation_period_days
    
    def __call__(self, genome: DecisionGenome) -> float:
        """Evaluate fitness of a genome"""
        # Placeholder implementation
        return random.uniform(0.0, 1.0) 