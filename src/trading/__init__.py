"""
EMP Trading Layer v1.1

The Trading Layer handles strategy execution, order management, risk management,
and performance monitoring. It receives decisions from the Adaptive Core and
executes them in the live market environment.

Architecture:
- strategies/: Strategy registry and evolved strategies
- execution/: Order and position management
- risk/: Live risk management and position sizing
- monitoring/: Performance tracking and health monitoring
- integration/: External broker integration (cTrader)
"""

from .strategies import *
from .execution import *
from .risk import *
from .monitoring import *
from .integration import *

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Trading Layer - Strategy Execution and Order Management" 