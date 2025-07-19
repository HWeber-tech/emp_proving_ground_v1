"""
EMP Simulation Envelope v1.1

The simulation layer responsible for market simulation, stress testing,
adversarial testing, and fitness evaluation in the EMP Ultimate Architecture v1.1.

This layer owns all simulation functions including:
- Market simulation and backtesting
- Stress testing and scenario analysis
- Adversarial testing and robustness validation
- Fitness evaluation and scoring
- Validation and verification systems
"""

from .market_simulator import *
from .stress_tester import *
from .adversarial import *
from .validators import *

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Simulation envelope for testing and validation" 