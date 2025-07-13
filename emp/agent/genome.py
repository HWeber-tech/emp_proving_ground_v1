"""
DecisionGenome: Evolutionary decision tree for trading strategies.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import random
import uuid

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
    size_factor: float = 1.0  # Multiplier for base position size
    confidence_threshold: float = 0.5  # Minimum confidence required
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class NodeType(Enum):
    CONDITION = "condition"
    ACTION = "action"
    COMPOSITE = "composite"

@dataclass
class DecisionNode:
    """Node in the decision tree genome"""
    node_id: str
    node_type: NodeType
    
    # For condition nodes
    dimension: Optional[str] = None  # "why", "how", "what", "when", "anomaly"
    operator: Optional[str] = None   # ">", "<", ">=", "<=", "=="
    threshold: Optional[float] = None
    
    # For action nodes
    action: Optional[TradingAction] = None
    
    # Tree structure
    left_child: Optional["DecisionNode"] = None
    right_child: Optional["DecisionNode"] = None
    
    # Metadata
    creation_generation: int = 0
    usage_count: int = 0
    success_rate: float = 0.0

class DecisionGenome:
    """
    Evolutionary decision tree genome for trading strategies.
    
    The genome represents a decision tree that processes sensory readings
    and outputs trading actions based on learned patterns.
    """
    
    def __init__(self, max_depth: int = 10, max_nodes: int = 100):
        self.genome_id = str(uuid.uuid4())
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.root: Optional[DecisionNode] = None
        self.generation = 0
        self.fitness_score = 0.0
        self.complexity_penalty = 0.0
        
        # Performance tracking
        self.total_decisions = 0
        self.successful_decisions = 0
        self.decision_history: List[Dict] = []
        
        # Initialize random tree
        self._initialize_random_tree()
        
        logger.info(f"Initialized DecisionGenome {self.genome_id}")
    
    def _initialize_random_tree(self):
        """Initialize the genome with a random decision tree."""
        self.root = self._create_random_node(0)
    
    def _create_random_node(self, depth: int) -> DecisionNode:
        """Create a random decision node."""
        node_id = str(uuid.uuid4())
        
        # Determine node type based on depth and random chance
        if depth >= self.max_depth:
            node_type = NodeType.ACTION
        elif depth < 2:
            node_type = NodeType.CONDITION
        else:
            node_type = random.choice([NodeType.CONDITION, NodeType.ACTION])
        
        node = DecisionNode(
            node_id=node_id,
            node_type=node_type,
            creation_generation=self.generation
        )
        
        if node_type == NodeType.CONDITION:
            # Create condition node
            node.dimension = random.choice(["why", "how", "what", "when", "anomaly"])
            node.operator = random.choice([">", "<", ">=", "<=", "=="])
            node.threshold = random.uniform(0.0, 1.0)
            
            # Create children
            if depth < self.max_depth - 1:
                node.left_child = self._create_random_node(depth + 1)
                node.right_child = self._create_random_node(depth + 1)
        
        elif node_type == NodeType.ACTION:
            # Create action node
            action_type = random.choice(list(ActionType))
            size_factor = random.uniform(0.1, 2.0)
            confidence_threshold = random.uniform(0.3, 0.8)
            
            node.action = TradingAction(
                action_type=action_type,
                size_factor=size_factor,
                confidence_threshold=confidence_threshold
            )
        
        return node
    
    def decide(self, sensory_reading) -> Optional[TradingAction]:
        """
        Make a trading decision based on sensory reading.
        
        Args:
            sensory_reading: SensoryReading from the cortex
            
        Returns:
            TradingAction or None if no decision
        """
        if not self.root:
            return None
        
        decision = self._traverse_tree(self.root, sensory_reading)
        
        if decision:
            # Record decision
            self.total_decisions += 1
            self.decision_history.append({
                'timestamp': sensory_reading.timestamp,
                'action': decision.action_type.value,
                'size_factor': decision.size_factor,
                'confidence': sensory_reading.confidence,
                'sensory_scores': {
                    'why': sensory_reading.why_score,
                    'how': sensory_reading.how_score,
                    'what': sensory_reading.what_score,
                    'when': sensory_reading.when_score,
                    'anomaly': sensory_reading.anomaly_score
                }
            })
        
        return decision
    
    def _traverse_tree(self, node: DecisionNode, sensory_reading) -> Optional[TradingAction]:
        """Traverse the decision tree to find the appropriate action."""
        if not node:
            return None
        
        # Update usage count
        node.usage_count += 1
        
        if node.node_type == NodeType.ACTION:
            # Check confidence threshold
            if node.action and sensory_reading.confidence >= node.action.confidence_threshold:
                return node.action
            else:
                return None
        
        elif node.node_type == NodeType.CONDITION:
            # Evaluate condition
            if node.dimension is None or node.operator is None or node.threshold is None:
                return None
                
            value = self._get_dimension_value(node.dimension, sensory_reading)
            if value is None:
                return None
            
            condition_met = self._evaluate_condition(value, node.operator, node.threshold)
            
            # Traverse appropriate child
            if condition_met:
                return self._traverse_tree(node.left_child, sensory_reading) if node.left_child else None
            else:
                return self._traverse_tree(node.right_child, sensory_reading) if node.right_child else None
        
        return None
    
    def _get_dimension_value(self, dimension: str, sensory_reading) -> Optional[float]:
        """Get the value for a specific dimension from sensory reading."""
        dimension_map = {
            'why': sensory_reading.why_score,
            'how': sensory_reading.how_score,
            'what': sensory_reading.what_score,
            'when': sensory_reading.when_score,
            'anomaly': sensory_reading.anomaly_score
        }
        
        return dimension_map.get(dimension)
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a condition."""
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
        """
        Create a mutated copy of this genome.
        
        Args:
            mutation_rate: Probability of mutation per node
            
        Returns:
            New mutated DecisionGenome
        """
        new_genome = DecisionGenome(max_depth=self.max_depth, max_nodes=self.max_nodes)
        new_genome.genome_id = str(uuid.uuid4())
        new_genome.generation = self.generation + 1
        if self.root:
            new_genome.root = self._copy_and_mutate_node(self.root, mutation_rate)
        
        return new_genome
    
    def _copy_and_mutate_node(self, node: DecisionNode, mutation_rate: float) -> DecisionNode:
        """Copy a node and potentially mutate it."""
        if not node:
            return None
        
        # Create copy
        new_node = DecisionNode(
            node_id=str(uuid.uuid4()),
            node_type=node.node_type,
            dimension=node.dimension,
            operator=node.operator,
            threshold=node.threshold,
            creation_generation=node.creation_generation,
            usage_count=node.usage_count,
            success_rate=node.success_rate
        )
        
        # Copy action if present
        if node.action:
            new_node.action = TradingAction(
                action_type=node.action.action_type,
                size_factor=node.action.size_factor,
                confidence_threshold=node.action.confidence_threshold,
                stop_loss=node.action.stop_loss,
                take_profit=node.action.take_profit,
                metadata=node.action.metadata.copy()
            )
        
        # Mutate with probability
        if random.random() < mutation_rate:
            self._mutate_node(new_node, mutation_rate)
        
        # Copy children
        if node.left_child:
            new_node.left_child = self._copy_and_mutate_node(node.left_child, mutation_rate)
        if node.right_child:
            new_node.right_child = self._copy_and_mutate_node(node.right_child, mutation_rate)
        
        return new_node
    
    def _mutate_node(self, node: DecisionNode, mutation_rate: float):
        """Mutate a single node."""
        if node.node_type == NodeType.CONDITION:
            # Mutate condition parameters
            mutation_type = random.choice(['dimension', 'operator', 'threshold'])
            
            if mutation_type == 'dimension':
                node.dimension = random.choice(["why", "how", "what", "when", "anomaly"])
            elif mutation_type == 'operator':
                node.operator = random.choice([">", "<", ">=", "<=", "=="])
            elif mutation_type == 'threshold':
                node.threshold = random.uniform(0.0, 1.0)
        
        elif node.node_type == NodeType.ACTION:
            # Mutate action parameters
            if node.action is None:
                # Create a new action if none exists
                node.action = TradingAction(
                    action_type=random.choice(list(ActionType)),
                    size_factor=random.uniform(0.1, 2.0),
                    confidence_threshold=random.uniform(0.3, 0.8)
                )
            else:
                mutation_type = random.choice(['action_type', 'size_factor', 'confidence_threshold'])
                
                if mutation_type == 'action_type':
                    node.action.action_type = random.choice(list(ActionType))
                elif mutation_type == 'size_factor':
                    node.action.size_factor = random.uniform(0.1, 2.0)
                elif mutation_type == 'confidence_threshold':
                    node.action.confidence_threshold = random.uniform(0.3, 0.8)
    
    def crossover(self, other: "DecisionGenome") -> Tuple["DecisionGenome", "DecisionGenome"]:
        """
        Perform crossover with another genome.
        
        Args:
            other: Another DecisionGenome
            
        Returns:
            Tuple of two new DecisionGenome offspring
        """
        # Create offspring
        offspring1 = DecisionGenome(max_depth=self.max_depth, max_nodes=self.max_nodes)
        offspring2 = DecisionGenome(max_depth=self.max_depth, max_nodes=self.max_nodes)
        
        offspring1.genome_id = str(uuid.uuid4())
        offspring2.genome_id = str(uuid.uuid4())
        offspring1.generation = max(self.generation, other.generation) + 1
        offspring2.generation = max(self.generation, other.generation) + 1
        
        # Perform crossover
        if self.root and other.root:
            offspring1.root, offspring2.root = self._crossover_nodes(self.root, other.root)
        
        return offspring1, offspring2
    
    def _crossover_nodes(self, node1: DecisionNode, node2: DecisionNode) -> Tuple[DecisionNode, DecisionNode]:
        """Perform crossover between two nodes."""
        # Create copies
        new_node1 = self._copy_node(node1)
        new_node2 = self._copy_node(node2)
        
        # Ensure we have valid nodes
        if new_node1 is None or new_node2 is None:
            # Fallback: create simple action nodes
            new_node1 = DecisionNode(
                node_id=str(uuid.uuid4()),
                node_type=NodeType.ACTION,
                action=TradingAction(action_type=ActionType.HOLD)
            )
            new_node2 = DecisionNode(
                node_id=str(uuid.uuid4()),
                node_type=NodeType.ACTION,
                action=TradingAction(action_type=ActionType.HOLD)
            )
        
        # Randomly swap subtrees
        if random.random() < 0.5:
            # Swap left children
            new_node1.left_child, new_node2.left_child = new_node2.left_child, new_node1.left_child
        
        if random.random() < 0.5:
            # Swap right children
            new_node1.right_child, new_node2.right_child = new_node2.right_child, new_node1.right_child
        
        return new_node1, new_node2
    
    def _copy_node(self, node: DecisionNode) -> Optional[DecisionNode]:
        """Create a deep copy of a node."""
        if not node:
            return None
        
        new_node = DecisionNode(
            node_id=str(uuid.uuid4()),
            node_type=node.node_type,
            dimension=node.dimension,
            operator=node.operator,
            threshold=node.threshold,
            creation_generation=node.creation_generation,
            usage_count=node.usage_count,
            success_rate=node.success_rate
        )
        
        if node.action:
            new_node.action = TradingAction(
                action_type=node.action.action_type,
                size_factor=node.action.size_factor,
                confidence_threshold=node.action.confidence_threshold,
                stop_loss=node.action.stop_loss,
                take_profit=node.action.take_profit,
                metadata=node.action.metadata.copy()
            )
        
        return new_node
    
    def get_complexity(self) -> Dict[str, int]:
        """Get complexity metrics for the genome."""
        if not self.root:
            return {'nodes': 0, 'depth': 0, 'leaves': 0}
        
        nodes = self._count_nodes(self.root)
        depth = self._get_depth(self.root)
        leaves = self._count_leaves(self.root)
        
        return {
            'nodes': nodes,
            'depth': depth,
            'leaves': leaves
        }
    
    def _count_nodes(self, node: DecisionNode) -> int:
        """Count total nodes in the tree."""
        if not node:
            return 0
        
        count = 1
        if node.left_child:
            count += self._count_nodes(node.left_child)
        if node.right_child:
            count += self._count_nodes(node.right_child)
        
        return count
    
    def _get_depth(self, node: DecisionNode) -> int:
        """Get the depth of the tree."""
        if not node:
            return 0
        
        left_depth = self._get_depth(node.left_child) if node.left_child else 0
        right_depth = self._get_depth(node.right_child) if node.right_child else 0
        
        return max(left_depth, right_depth) + 1
    
    def _count_leaves(self, node: DecisionNode) -> int:
        """Count leaf nodes (action nodes)."""
        if not node:
            return 0
        
        if node.node_type == NodeType.ACTION:
            return 1
        
        count = 0
        if node.left_child:
            count += self._count_leaves(node.left_child)
        if node.right_child:
            count += self._count_leaves(node.right_child)
        
        return count
    
    def get_decision_path(self, sensory_reading) -> List[str]:
        """Get the decision path for a sensory reading."""
        path = []
        if self.root:
            self._trace_path(self.root, sensory_reading, path)
        return path
    
    def _trace_path(self, node: DecisionNode, sensory_reading, path: List[str]):
        """Trace the decision path through the tree."""
        if not node:
            return
        
        if node.node_type == NodeType.CONDITION:
            if node.dimension is None or node.operator is None or node.threshold is None:
                return
                
            value = self._get_dimension_value(node.dimension, sensory_reading)
            if value is None:
                return
                
            condition_met = self._evaluate_condition(value, node.operator, node.threshold)
            
            path.append(f"{node.dimension} {node.operator} {node.threshold:.3f} -> {'True' if condition_met else 'False'}")
            
            if condition_met:
                if node.left_child:
                    self._trace_path(node.left_child, sensory_reading, path)
            else:
                if node.right_child:
                    self._trace_path(node.right_child, sensory_reading, path)
        
        elif node.node_type == NodeType.ACTION:
            if node.action:
                path.append(f"ACTION: {node.action.action_type.value}")
    
    def update_success_rate(self, success: bool):
        """Update the success rate of the genome."""
        if success:
            self.successful_decisions += 1
        
        if self.total_decisions > 0:
            self.fitness_score = self.successful_decisions / self.total_decisions
    
    def to_dict(self) -> Dict:
        """Convert genome to dictionary representation."""
        return {
            'genome_id': self.genome_id,
            'generation': self.generation,
            'fitness_score': self.fitness_score,
            'complexity': self.get_complexity(),
            'total_decisions': self.total_decisions,
            'successful_decisions': self.successful_decisions,
            'root': self._node_to_dict(self.root) if self.root else None
        }
    
    def _node_to_dict(self, node: DecisionNode) -> Dict:
        """Convert a node to dictionary representation."""
        if not node:
            return None
        
        node_dict = {
            'node_id': node.node_id,
            'node_type': node.node_type.value,
            'creation_generation': node.creation_generation,
            'usage_count': node.usage_count,
            'success_rate': node.success_rate
        }
        
        if node.node_type == NodeType.CONDITION:
            node_dict.update({
                'dimension': node.dimension,
                'operator': node.operator,
                'threshold': node.threshold
            })
        
        elif node.node_type == NodeType.ACTION and node.action:
            node_dict['action'] = {
                'action_type': node.action.action_type.value,
                'size_factor': node.action.size_factor,
                'confidence_threshold': node.action.confidence_threshold,
                'stop_loss': node.action.stop_loss,
                'take_profit': node.action.take_profit
            }
        
        # Add children
        if node.left_child:
            node_dict['left_child'] = self._node_to_dict(node.left_child)
        if node.right_child:
            node_dict['right_child'] = self._node_to_dict(node.right_child)
        
        return node_dict
    
    def from_dict(self, data: Dict):
        """Load genome from dictionary representation."""
        self.genome_id = data.get('genome_id', str(uuid.uuid4()))
        self.generation = data.get('generation', 0)
        self.fitness_score = data.get('fitness_score', 0.0)
        self.total_decisions = data.get('total_decisions', 0)
        self.successful_decisions = data.get('successful_decisions', 0)
        
        if data.get('root'):
            self.root = self._node_from_dict(data['root'])
    
    def _node_from_dict(self, data: Dict) -> Optional[DecisionNode]:
        """Create a node from dictionary representation."""
        if not data:
            return None
        
        node = DecisionNode(
            node_id=data['node_id'],
            node_type=NodeType(data['node_type']),
            creation_generation=data.get('creation_generation', 0),
            usage_count=data.get('usage_count', 0),
            success_rate=data.get('success_rate', 0.0)
        )
        
        if node.node_type == NodeType.CONDITION:
            node.dimension = data.get('dimension')
            node.operator = data.get('operator')
            node.threshold = data.get('threshold')
        
        elif node.node_type == NodeType.ACTION and data.get('action'):
            action_data = data['action']
            node.action = TradingAction(
                action_type=ActionType(action_data['action_type']),
                size_factor=action_data.get('size_factor', 1.0),
                confidence_threshold=action_data.get('confidence_threshold', 0.5),
                stop_loss=action_data.get('stop_loss'),
                take_profit=action_data.get('take_profit')
            )
        
        # Load children
        if data.get('left_child'):
            node.left_child = self._node_from_dict(data['left_child'])
        if data.get('right_child'):
            node.right_child = self._node_from_dict(data['right_child'])
        
        return node 