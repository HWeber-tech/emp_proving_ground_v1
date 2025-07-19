"""
EMP Sensory Layer v1.1

The Sensory Layer is responsible for raw market perception and signal processing.
It acts as the "eyes and ears" of the system, processing raw market data into
perceived signals that can be consumed by the Thinking Layer.

Architecture:
- organs/: Specialized sensory organs for different data types
- integration/: Sensory cortex for cross-modal integration
- calibration/: Calibration and validation of sensory inputs
- models/: Data models for perception
"""

from .organs import *
from .integration import *
from .calibration import *
from .models import *

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Sensory Layer - Market Perception and Signal Processing"

