"""
EMP Operational Backbone v1.1

The Operational Backbone provides the infrastructure and operational support
for the entire EMP system. It includes state management, event bus, health
monitoring, and container orchestration.

Architecture:
- state_store.py: Redis-based state management
- event_bus.py: NATS/Kafka event bus
- health_monitor.py: System health monitoring
- metrics_collector.py: Performance metrics collection
- container/: Docker and Kubernetes configuration
"""

from .state_store import StateStore
from .event_bus import EventBus
from .health_monitor import HealthMonitor
from .metrics_collector import MetricsCollector

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Operational Backbone - Infrastructure and Operations" 