"""
EMP Simulation Envelope v1.1

The Simulation Envelope provides the adversarial proving ground for testing
strategies against various market conditions, stress scenarios, and regime changes.
It ensures robustness and adaptability before live deployment.

Architecture:
- market_simulator.py: Historical and synthetic market simulation
- stress_tester.py: Edge cases and black swan scenarios
- regime_tester.py: Market regime transitions and volatility shifts
- adversarial_engine.py: Adversarial testing and validation
- validators/: Reality checks and validation logic
"""

from .market_simulator import MarketSimulator
from .stress_tester import StressTester
from .regime_tester import RegimeTester
from .adversarial_engine import AdversarialEngine

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Simulation Envelope - Adversarial Proving Ground" 