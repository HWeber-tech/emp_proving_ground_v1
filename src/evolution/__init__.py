"""
EMP Evolution Layer v1.1

The Evolution Layer orchestrates the genetic programming and evolution of
trading strategies. It manages populations, selection, variation, and
evaluation to drive continuous improvement and adaptation.

Architecture:
- engine/: Genetic engine and population management
- selection/: Selection algorithms (tournament, fitness proportionate)
- variation/: Crossover, mutation, and recombination operators
- evaluation/: Fitness evaluation and backtesting
- meta/: Meta-evolution for self-improving evolution
"""

from .engine import *
from .selection import *
from .variation import *
from .evaluation import *
from .meta import *

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Evolution Layer - Genetic Programming and Evolution" 