"""
EMP Core Layer v1.1

The Core Layer provides cross-cutting concerns, interfaces, and foundational
components used throughout the EMP system. It defines the contracts and
abstractions that enable layer communication and system integration.

Architecture:
- events.py: Event definitions and types
- event_bus.py: Core event bus interface
- interfaces.py: Layer interfaces and contracts
- exceptions.py: System-wide exception types
- configuration.py: Configuration management
"""

from .events import *
from .event_bus import EventBus
from .interfaces import *
from .exceptions import *
from .configuration import Configuration

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Core Layer - Cross-cutting Concerns and Interfaces" 