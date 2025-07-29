"""
EMP Genome Layer v1.1

The Genome Layer contains the genetic encoding and decoding mechanisms for
trading strategies, risk parameters, and timing preferences. It provides
the genetic foundation for the Adaptive Core's evolution.

Architecture:
- encoders/: Strategy, risk, and timing encoders
- decoders/: Strategy decoders for execution
- models/: Genome data models and structures
"""

from .encoders import *
from .decoders import *
from .models import *

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Genome Layer - Genetic Encoding and Decoding" 
