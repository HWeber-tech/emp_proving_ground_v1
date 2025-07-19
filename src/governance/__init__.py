"""
EMP Governance Layer v1.1

The Governance Layer provides oversight, control, and audit capabilities for
the entire EMP system. It includes fitness definition management, strategy
registry, configuration vault, and human approval gateways.

Architecture:
- fitness_store.py: Fitness definition management
- strategy_registry.py: Strategy registration and approval
- audit_logger.py: Comprehensive audit logging
- config_vault.py: Secure configuration management
- human_gateway.py: Human oversight and approval
"""

from .fitness_store import FitnessStore
from .strategy_registry import StrategyRegistry
from .audit_logger import AuditLogger
from .config_vault import ConfigVault
from .human_gateway import HumanGateway

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Governance Layer - System Oversight and Control" 